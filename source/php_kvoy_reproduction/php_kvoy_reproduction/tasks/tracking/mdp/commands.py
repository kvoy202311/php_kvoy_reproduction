from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .climb_progress import bounded_episode_progress_increment
from .motion_data import (
    MultiMotionAdaptiveSampler,
    adaptive_failure_mask,
    advance_motion_frames,
    advance_motion_frames_with_final_hold,
    apply_forced_motion_starts,
    deterministic_motion_starts,
    load_motion_dataset,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.motion_root_body_index = (
            0 if self.cfg.root_body_name is None else self.cfg.body_names.index(self.cfg.root_body_name)
        )
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = load_motion_dataset(
            motion_file=self.cfg.motion_file,
            motion_dir=self.cfg.motion_dir,
            body_indexes=self.body_indexes,
            device=self.device,
        )
        if self.motion.joint_count != len(self.robot.joint_names):
            raise ValueError(
                f"Motion data contains {self.motion.joint_count} joints, but robot '{cfg.asset_name}' has "
                f"{len(self.robot.joint_names)} joints."
            )
        if self.motion.body_count != len(self.robot.body_names):
            raise ValueError(
                f"Motion data contains {self.motion.body_count} bodies, but robot '{cfg.asset_name}' has "
                f"{len(self.robot.body_names)} bodies."
            )
        if self.motion.joint_names is not None and self.motion.joint_names != tuple(self.robot.joint_names):
            raise ValueError(
                "Motion joint_names do not exactly match the simulator joint order. Reconvert the motion for this "
                "robot; silently reordering reference tensors during training is not allowed."
            )
        if self.motion.body_names is not None and self.motion.body_names != tuple(self.robot.body_names):
            raise ValueError(
                "Motion body_names do not exactly match the simulator body order. Reconvert the motion for this "
                "robot; silently reordering reference tensors during training is not allowed."
            )
        env_fps = 1.0 / (env.cfg.decimation * env.cfg.sim.dt)
        if abs(self.motion.fps - env_fps) > 1.0e-6:
            raise ValueError(
                f"Motion data runs at {self.motion.fps:g} Hz, but the environment runs at {env_fps:g} Hz "
                f"(decimation={env.cfg.decimation}, sim.dt={env.cfg.sim.dt:g}). Resample the motion data or "
                "change the environment timing so one policy step advances exactly one motion frame."
            )
        if self.cfg.motion_end_hold_time_s < 0.0:
            raise ValueError(
                f"motion_end_hold_time_s must be non-negative, got {self.cfg.motion_end_hold_time_s}."
            )
        self.motion_end_hold_steps = int(round(self.cfg.motion_end_hold_time_s * env_fps))
        if self.cfg.motion_sampling_mode not in ("random", "fixed", "round_robin"):
            raise ValueError(
                "motion_sampling_mode must be 'random', 'fixed', or 'round_robin', "
                f"got {self.cfg.motion_sampling_mode!r}."
            )
        self.reference_transform_asset: RigidObject | None = None
        if self.cfg.reference_transform_asset_name is not None:
            reference_asset = env.scene[self.cfg.reference_transform_asset_name]
            if not isinstance(reference_asset, RigidObject):
                raise TypeError(
                    f"Reference transform asset '{self.cfg.reference_transform_asset_name}' must be a RigidObject, "
                    f"got {type(reference_asset).__name__}."
                )
            self.reference_transform_asset = reference_asset

        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.previous_motion_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.motion_resample_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_switch_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.motion_final_hold_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.episode_started_at_motion_beginning = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._has_sampled_motion = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Stateful buffers used by the bounded climb-progress shaping term.
        # Each maximum records the best physical progress reached in the
        # episode. Contact loss/recovery therefore cannot repeatedly reward
        # the same progress.
        self._climb_progress_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._climb_approach_max_potential = torch.zeros(self.num_envs, device=self.device)
        self._climb_lift_max_potential = torch.zeros(self.num_envs, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.motion_sampler = MultiMotionAdaptiveSampler(
            motion_start_idx=self.motion.motion_start_idx,
            motion_end_idx=self.motion.motion_end_idx,
            env_fps=env_fps,
            device=self.device,
            adaptive_kernel_size=self.cfg.adaptive_kernel_size,
            adaptive_lambda=self.cfg.adaptive_lambda,
            adaptive_uniform_ratio=self.cfg.adaptive_uniform_ratio,
            adaptive_alpha=self.cfg.adaptive_alpha,
            motion_signatures=self.motion.motion_signatures,
        )

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset command state, including the climb-progress potential baseline."""

        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._climb_progress_initialized[env_ids] = False
        self._climb_approach_max_potential[env_ids] = 0.0
        self._climb_lift_max_potential[env_ids] = 0.0
        return extras

    def climb_progress_deltas(
        self,
        approach_potential: torch.Tensor,
        lift_potential: torch.Tensor,
        max_delta_per_step: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reset-safe, non-repeatable progress increments.

        The first sample after reset establishes the episode baseline and has
        zero shaping reward. Thereafter only a new episode maximum can create
        reward. Backing away or losing contact cannot reset the maximum or farm
        repeated reward.
        """

        expected_shape = (self.num_envs,)
        if approach_potential.shape != expected_shape or lift_potential.shape != expected_shape:
            raise ValueError(
                "climb progress potentials must have shape "
                f"{expected_shape}, got {approach_potential.shape} and {lift_potential.shape}."
            )

        approach_delta, approach_max = bounded_episode_progress_increment(
            approach_potential,
            self._climb_progress_initialized,
            self._climb_approach_max_potential,
            max_delta_per_step,
        )
        lift_delta, lift_max = bounded_episode_progress_increment(
            lift_potential,
            self._climb_progress_initialized,
            self._climb_lift_max_potential,
            max_delta_per_step,
        )
        self._climb_approach_max_potential.copy_(approach_max)
        self._climb_lift_max_potential.copy_(lift_max)
        self._climb_progress_initialized.fill_(True)
        return approach_delta, lift_delta

    @property
    def source_joint_pos(self) -> torch.Tensor:
        """Joint positions read directly from the immutable motion dataset."""

        return self.motion.joint_pos[self.time_steps]

    @property
    def final_hold_progress(self) -> torch.Tensor:
        """Normalized progress through the extra final-frame hold."""

        if self.motion_end_hold_steps <= 0:
            return torch.zeros(self.num_envs, dtype=self.motion.joint_pos.dtype, device=self.device)
        return torch.clamp(
            self.motion_final_hold_count.to(dtype=self.motion.joint_pos.dtype) / self.motion_end_hold_steps,
            min=0.0,
            max=1.0,
        )

    @property
    def joint_pos(self) -> torch.Tensor:
        """Observable joint target taken directly from the immutable motion data.

        The final-frame hold must preserve one kinematically consistent
        reference. Moving only the joint target toward the articulation
        default while body targets remain at the NPZ final frame creates two
        incompatible objectives and makes a settled robot move again.
        """

        return self.source_joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        positions = self.motion.body_pos_w[self.time_steps]
        if self.reference_transform_asset is None:
            return positions + self._env.scene.env_origins[:, None, :]

        yaw_delta = yaw_quat(self.reference_transform_asset.data.root_quat_w)
        nominal_xy = torch.tensor(
            self.cfg.reference_transform_nominal_xy, dtype=positions.dtype, device=self.device
        )
        delta = torch.zeros_like(positions)
        delta[..., :2] = positions[..., :2] - nominal_xy
        rotated_delta = quat_apply(yaw_delta[:, None, :].expand(-1, positions.shape[1], -1), delta)

        transformed = positions + self._env.scene.env_origins[:, None, :]
        transformed[..., :2] = self.reference_transform_asset.data.root_pos_w[:, None, :2] + rotated_delta[..., :2]
        return transformed

    @property
    def body_quat_w(self) -> torch.Tensor:
        orientations = self.motion.body_quat_w[self.time_steps]
        if self.reference_transform_asset is None:
            return orientations
        yaw_delta = yaw_quat(self.reference_transform_asset.data.root_quat_w)
        yaw_delta = yaw_delta[:, None, :].expand(-1, orientations.shape[1], -1)
        return quat_mul(yaw_delta, orientations)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        velocities = self.motion.body_lin_vel_w[self.time_steps]
        if self.reference_transform_asset is None:
            return velocities
        yaw_delta = yaw_quat(self.reference_transform_asset.data.root_quat_w)
        return quat_apply(yaw_delta[:, None, :].expand(-1, velocities.shape[1], -1), velocities)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        velocities = self.motion.body_ang_vel_w[self.time_steps]
        if self.reference_transform_asset is None:
            return velocities
        yaw_delta = yaw_quat(self.reference_transform_asset.data.root_quat_w)
        return quat_apply(yaw_delta[:, None, :].expand(-1, velocities.shape[1], -1), velocities)

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _sample_motion_indices(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cfg.motion_sampling_mode in ("fixed", "round_robin"):
            return deterministic_motion_starts(
                env_ids,
                self.motion.motion_start_idx,
                mode=self.cfg.motion_sampling_mode,
                fixed_motion_id=self.cfg.fixed_motion_id,
            )

        if self.cfg.use_adaptive_sampling:
            extra_failure_masks = tuple(
                self._env.termination_manager.get_term(term_name)[env_ids]
                for term_name in self.cfg.adaptive_failure_term_names
            )
            episode_failed = adaptive_failure_mask(
                self._env.termination_manager.terminated[env_ids],
                extra_failure_masks,
            )
            if torch.any(episode_failed):
                failed_env_ids = env_ids[episode_failed]
                self.motion_sampler.record_failures(
                    self.motion_ids[failed_env_ids],
                    self.time_steps[failed_env_ids],
                )
            motion_ids, time_steps = self.motion_sampler.sample(len(env_ids))
            for metric_name, metric_value in self.motion_sampler.get_metrics().items():
                self.metrics[metric_name][:] = metric_value
        else:
            motion_ids, time_steps = self.motion_sampler.sample_uniform(len(env_ids))

        if self.cfg.start_at_motion_beginning:
            force_start_mask = torch.ones(len(env_ids), dtype=torch.bool, device=self.device)
        elif self.cfg.random_phase_env_mask_attr is not None:
            if self.reference_transform_asset is None:
                raise RuntimeError(
                    "random_phase_env_mask_attr requires reference_transform_asset_name to be configured."
                )
            random_phase_allowed = getattr(
                self.reference_transform_asset,
                self.cfg.random_phase_env_mask_attr,
                None,
            )
            if random_phase_allowed is None:
                raise RuntimeError(
                    f"The reference asset does not expose the random-phase mask "
                    f"{self.cfg.random_phase_env_mask_attr!r}."
                )
            random_phase_allowed = random_phase_allowed.to(device=self.device, dtype=torch.bool)
            if random_phase_allowed.shape != (self.num_envs,):
                raise RuntimeError(
                    f"Random-phase mask {self.cfg.random_phase_env_mask_attr!r} must have shape "
                    f"({self.num_envs},), got {random_phase_allowed.shape}."
                )
            force_start_mask = ~random_phase_allowed[env_ids]
        else:
            force_start_mask = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)

        time_steps = apply_forced_motion_starts(
            motion_ids,
            time_steps,
            self.motion.motion_start_idx,
            force_start_mask,
        )
        return motion_ids, time_steps

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        old_motion_ids = self.motion_ids[env_ids].clone()
        had_motion = self._has_sampled_motion[env_ids]
        sampled_motion_ids, sampled_time_steps = self._sample_motion_indices(env_ids)
        self.previous_motion_ids[env_ids] = torch.where(had_motion, old_motion_ids, -1)
        self.motion_resample_count[env_ids] += had_motion.long()
        self.motion_switch_count[env_ids] += (had_motion & (old_motion_ids != sampled_motion_ids)).long()
        self.motion_ids[env_ids] = sampled_motion_ids
        self.time_steps[env_ids] = sampled_time_steps
        self.episode_started_at_motion_beginning[env_ids] = sampled_time_steps == self.motion.motion_start_idx[
            sampled_motion_ids
        ]
        self.motion_finished[env_ids] = False
        self.motion_final_hold_count[env_ids] = 0
        self._has_sampled_motion[env_ids] = True

        root_pos = self.body_pos_w[:, self.motion_root_body_index].clone()
        root_ori = self.body_quat_w[:, self.motion_root_body_index].clone()
        root_lin_vel = self.body_lin_vel_w[:, self.motion_root_body_index].clone()
        root_ang_vel = self.body_ang_vel_w[:, self.motion_root_body_index].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        final_frames = self.motion.motion_end_idx[self.motion_ids] - 1
        was_on_final_frame = self.time_steps >= final_frames
        if self.cfg.terminate_on_motion_end:
            self.time_steps, self.motion_final_hold_count, motion_completed = advance_motion_frames_with_final_hold(
                self.motion_ids,
                self.time_steps,
                self.motion.motion_end_idx,
                self.motion_final_hold_count,
                self.motion_end_hold_steps,
            )
            self.motion_finished |= motion_completed
        else:
            self.time_steps, _ = advance_motion_frames(self.motion_ids, self.time_steps, self.motion.motion_end_idx)
            self._resample_command(torch.where(was_on_final_frame)[0])

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.motion_sampler.update()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str | None = None
    motion_dir: str | None = None
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    root_body_name: str | None = None

    motion_sampling_mode: Literal["random", "fixed", "round_robin"] = "random"
    fixed_motion_id: int = 0
    start_at_motion_beginning: bool = False
    use_adaptive_sampling: bool = True
    terminate_on_motion_end: bool = False
    """Whether reaching a clip's final frame requests an episode boundary.

    A matching termination term must consume :attr:`MotionCommand.motion_finished`.
    This setting is independent of ``start_at_motion_beginning``: training may
    reset from a random phase, while deterministic evaluation can start at the
    first frame.
    """

    motion_end_hold_time_s: float = 0.0
    """Additional time to hold the final reference frame before ending a clip."""

    adaptive_failure_term_names: tuple[str, ...] = ()
    """Timeout/termination term masks that should also update adaptive phase statistics."""

    random_phase_env_mask_attr: str | None = None
    """Reference-asset boolean-mask attribute identifying safe random-phase environments."""

    reference_transform_asset_name: str | None = None
    reference_transform_nominal_xy: tuple[float, float] = (0.0, 0.0)

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
