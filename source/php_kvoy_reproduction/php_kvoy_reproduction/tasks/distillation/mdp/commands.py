"""Routed locomotion, climb and down-roll commands with observable transitions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import math
from typing import Literal

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_inv,
    quat_mul,
    wrap_to_pi,
    yaw_quat,
)

from php_kvoy_reproduction.distillation.skill_routing import (
    CLIMB_SKILL_ID,
    DOWN_ROLL_SKILL_ID,
    LOCOMOTION_SKILL_ID,
    NUM_SKILLS,
    balanced_skill_ids,
    down_roll_transition_ready,
    platform_height_teacher_confidence,
    platform_reference_center_offsets,
)
from php_kvoy_reproduction.tasks.tracking.mdp.motion_data import load_motion_dataset
from php_kvoy_reproduction.tasks.tracking.mdp.obstacle import get_climb_box_sizes
from php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry import oriented_box_local_xy


_DIRECT_STAGE = 0
_APPROACH_STAGE = 1
_SETTLE_STAGE = 2
_MOTION_STAGE = 3
_POST_LOCOMOTION_STAGE = 4


class _CombinedMotionDataset:
    """Read-only concatenation of two validated motion datasets."""

    _FRAME_FIELDS = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )

    def __init__(self, climb, down_roll) -> None:
        for field in ("fps", "joint_count", "body_count", "joint_names", "body_names"):
            if getattr(climb, field) != getattr(down_roll, field):
                raise ValueError(f"climb and down-roll motion datasets disagree on {field}")
        self.fps = climb.fps
        self.joint_count = climb.joint_count
        self.body_count = climb.body_count
        self.joint_names = climb.joint_names
        self.body_names = climb.body_names
        self.climb_motion_count = int(climb.num_motions)
        self.down_roll_motion_count = int(down_roll.num_motions)
        self.num_motions = self.climb_motion_count + self.down_roll_motion_count
        if self.climb_motion_count <= 0 or self.down_roll_motion_count <= 0:
            raise ValueError("both routed motion datasets must contain at least one clip")

        frame_offset = int(climb.joint_pos.shape[0])
        for field in self._FRAME_FIELDS:
            setattr(self, field, torch.cat((getattr(climb, field), getattr(down_roll, field)), dim=0))
        self.motion_lengths = torch.cat((climb.motion_lengths, down_roll.motion_lengths), dim=0)
        self.motion_start_idx = torch.cat(
            (climb.motion_start_idx, down_roll.motion_start_idx + frame_offset), dim=0
        )
        self.motion_end_idx = torch.cat(
            (climb.motion_end_idx, down_roll.motion_end_idx + frame_offset), dim=0
        )
        self.motion_random_start_end_idx = torch.cat(
            (
                climb.motion_random_start_end_idx,
                down_roll.motion_random_start_end_idx + frame_offset,
            ),
            dim=0,
        )
        self.climb_motion_ids = torch.arange(self.climb_motion_count, device=self.joint_pos.device)
        self.down_roll_motion_ids = torch.arange(
            self.climb_motion_count,
            self.num_motions,
            device=self.joint_pos.device,
        )


class MultiSkillCommand(CommandTerm):
    """Own all route, reference, transition and reset state for three skills.

    Environments keep one balanced episode family for their entire lifetime.
    This makes transition exposure independent of the very different episode
    lengths.  A configurable subset of climb/down-roll resets additionally
    traverses observable locomotion-to-motion and motion-to-locomotion stages;
    the remaining resets retain uniform atomic-motion phase coverage.
    """

    cfg: "MultiSkillCommandCfg"

    def __init__(self, cfg: "MultiSkillCommandCfg", env) -> None:
        super().__init__(cfg, env)
        if tuple(cfg.skill_names) != ("locomotion", "climb", "down_roll"):
            raise ValueError("skill_names must be exactly ('locomotion', 'climb', 'down_roll')")
        if (
            len(cfg.platform_size) != 3
            or not all(math.isfinite(value) and value > 0.0 for value in cfg.platform_size)
        ):
            raise ValueError("platform_size must contain three finite positive dimensions")
        if (
            not math.isfinite(cfg.platform_height)
            or cfg.platform_height <= 0.0
            or cfg.platform_size[2] != cfg.platform_height
        ):
            raise ValueError("platform_size z and platform_height must be the same positive value")
        if not 0.0 <= cfg.locomotion_standing_fraction < 1.0:
            raise ValueError("locomotion_standing_fraction must lie in [0, 1)")
        if not 0.0 < cfg.locomotion_speed_range[0] <= cfg.locomotion_speed_range[1]:
            raise ValueError("locomotion_speed_range must be positive and ordered")
        if cfg.locomotion_command_resampling_time_s <= 0.0:
            raise ValueError("locomotion command resampling time must be positive")
        if cfg.gait_cycle <= 0.0:
            raise ValueError("gait_cycle must be positive")
        if not 0.0 <= cfg.composed_episode_fraction <= 1.0:
            raise ValueError("composed_episode_fraction must lie in [0, 1]")
        if not 0.0 < cfg.approach_distance_range[0] <= cfg.approach_distance_range[1]:
            raise ValueError("approach_distance_range must be positive and ordered")
        if cfg.approach_switch_distance <= 0.0:
            raise ValueError("approach_switch_distance must be positive")
        if cfg.composed_locomotion_speed <= 0.0:
            raise ValueError("composed_locomotion_speed must be positive")
        for name, duration_range in (
            ("transition_settle_time_range_s", cfg.transition_settle_time_range_s),
            ("post_locomotion_time_range_s", cfg.post_locomotion_time_range_s),
            ("top_locomotion_time_range_s", cfg.top_locomotion_time_range_s),
        ):
            if (
                len(duration_range) != 2
                or not all(math.isfinite(value) for value in duration_range)
                or not 0.0 <= duration_range[0] <= duration_range[1]
            ):
                raise ValueError(f"{name} must be non-negative and ordered")
        if cfg.student_termination_relaxation_iterations <= 0:
            raise ValueError("student_termination_relaxation_iterations must be positive")
        if cfg.student_termination_final_scale < 1.0:
            raise ValueError("student_termination_final_scale must be at least one")
        if (
            not math.isfinite(cfg.teacher_height_full_tolerance)
            or not math.isfinite(cfg.teacher_height_zero_tolerance)
            or cfg.teacher_height_full_tolerance < 0.0
            or cfg.teacher_height_zero_tolerance <= cfg.teacher_height_full_tolerance
        ):
            raise ValueError("teacher height tolerances must satisfy 0 <= full < zero")
        if not 0.0 <= cfg.motion_tracking_termination_min_confidence <= 1.0:
            raise ValueError("motion_tracking_termination_min_confidence must lie in [0, 1]")
        if (
            not math.isfinite(cfg.top_locomotion_settle_time_s)
            or cfg.top_locomotion_settle_time_s < 0.0
        ):
            raise ValueError("top_locomotion_settle_time_s must be non-negative")
        if (
            len(cfg.down_roll_edge_distance_range) != 2
            or not all(math.isfinite(value) for value in cfg.down_roll_edge_distance_range)
            or cfg.down_roll_edge_distance_range[0] > cfg.down_roll_edge_distance_range[1]
        ):
            raise ValueError("down_roll_edge_distance_range must be finite and ordered")
        if not math.isfinite(cfg.down_roll_lateral_margin) or cfg.down_roll_lateral_margin < 0.0:
            raise ValueError("down_roll_lateral_margin must be finite and non-negative")
        if (
            not math.isfinite(cfg.down_roll_minimum_forward_speed)
            or cfg.down_roll_minimum_forward_speed < 0.0
        ):
            raise ValueError("down_roll_minimum_forward_speed must be finite and non-negative")
        if (
            not math.isfinite(cfg.down_roll_maximum_heading_error)
            or not 0.0 < cfg.down_roll_maximum_heading_error <= math.pi
        ):
            raise ValueError("down_roll_maximum_heading_error must lie in (0, pi]")
        if (
            not math.isfinite(cfg.down_roll_maximum_gravity_xy_norm)
            or not 0.0 < cfg.down_roll_maximum_gravity_xy_norm <= 1.0
        ):
            raise ValueError("down_roll_maximum_gravity_xy_norm must lie in (0, 1]")

        self.robot: Articulation = env.scene[cfg.asset_name]
        platform = env.scene[cfg.platform_asset_name]
        if not isinstance(platform, RigidObject):
            raise TypeError(f"{cfg.platform_asset_name!r} must name a RigidObject")
        self.platform: RigidObject = platform
        self.platform_pos_w = self.platform.data.root_pos_w.clone()
        self.platform_quat_w = self.platform.data.root_quat_w.clone()
        # Geometry randomization is a prestartup-only event.  Cache its exact
        # collider dimensions once on the simulation device instead of moving
        # CPU metadata to the GPU every time a reward or transition queries it.
        self._platform_sizes = get_climb_box_sizes(
            self.platform,
            base_size=cfg.platform_size,
            device=self.device,
        )

        robot_body_ids = self.robot.find_bodies(cfg.body_names, preserve_order=True)[0]
        self.body_indexes = torch.as_tensor(robot_body_ids, device=self.device, dtype=torch.long)
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        root_name = cfg.anchor_body_name if cfg.root_body_name is None else cfg.root_body_name
        self.motion_root_body_index = cfg.body_names.index(root_name)

        climb = load_motion_dataset(
            motion_file=cfg.climb_motion_file,
            motion_dir=cfg.climb_motion_dir,
            body_indexes=self.body_indexes,
            device=self.device,
            exclude_repeated_terminal_frames_from_random_starts=True,
        )
        down_roll = load_motion_dataset(
            motion_file=cfg.down_roll_motion_file,
            motion_dir=cfg.down_roll_motion_dir,
            body_indexes=self.body_indexes,
            device=self.device,
            exclude_repeated_terminal_frames_from_random_starts=True,
        )
        self.motion = _CombinedMotionDataset(climb, down_roll)
        expected_fps = 1.0 / env.step_dt
        if abs(self.motion.fps - expected_fps) > 1.0e-6:
            raise ValueError(
                f"routed motion data is {self.motion.fps:g} Hz but the environment is {expected_fps:g} Hz"
            )
        if self.motion.joint_count != len(self.robot.joint_names):
            raise ValueError("routed motion joint count does not match the articulation")
        if self.motion.body_count != len(self.robot.body_names):
            raise ValueError("routed motion body count does not match the articulation")
        if self.motion.joint_names is not None and self.motion.joint_names != tuple(self.robot.joint_names):
            raise ValueError("routed motion joint order does not exactly match the articulation")
        if self.motion.body_names is not None and self.motion.body_names != tuple(self.robot.body_names):
            raise ValueError("routed motion body order does not exactly match the articulation")

        # Episode families are assigned exactly once and remain balanced even
        # though locomotion episodes are much longer than atomic motion clips.
        self.episode_skill_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.skill_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motion_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.motion_finished = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.composed_episode = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_complete = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.transition_stage = torch.full(
            (self.num_envs,), _DIRECT_STAGE, device=self.device, dtype=torch.int8
        )
        self.transition_time_left = torch.zeros(self.num_envs, device=self.device)
        self.transition_stage_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.transition_target_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.last_motion_skill_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.top_locomotion_released = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.training_iteration = 0
        self.episode_started_at_motion_beginning = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.world_command = torch.zeros(self.num_envs, 2, device=self.device)
        self.heading_target = self.robot.data.heading_w.clone()
        self.locomotion_command_time_left = torch.zeros(self.num_envs, device=self.device)
        self.gait_time = torch.zeros(self.num_envs, device=self.device)
        self.gait_phase = torch.zeros(self.num_envs, 2, device=self.device)
        self.phase_ratio = torch.tensor(cfg.gait_air_ratios, device=self.device).repeat(self.num_envs, 1)
        self.phase_offset = torch.tensor(cfg.gait_phase_offsets, device=self.device).repeat(self.num_envs, 1)
        self._skill_cursor = 0

        tracked_count = len(cfg.body_names)
        self.body_pos_relative_w = torch.zeros(self.num_envs, tracked_count, 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, tracked_count, 4, device=self.device)
        self.body_quat_relative_w[..., 0] = 1.0
        self.metrics["route_locomotion"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["route_climb"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["route_down_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_anchor_pos_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_anchor_ori_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["teacher_confidence"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["forward_edge_distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["platform_length"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["platform_width"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["platform_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Return the 58-D tracking command expected by climb/down teachers."""

        return torch.cat((self.joint_pos, self.joint_vel), dim=1)

    @property
    def locomotion_mask(self) -> torch.Tensor:
        return self.skill_ids == LOCOMOTION_SKILL_ID

    @property
    def climb_mask(self) -> torch.Tensor:
        return self.skill_ids == CLIMB_SKILL_ID

    @property
    def down_roll_mask(self) -> torch.Tensor:
        return self.skill_ids == DOWN_ROLL_SKILL_ID

    @property
    def motion_mask(self) -> torch.Tensor:
        return self.skill_ids != LOCOMOTION_SKILL_ID

    def mask_for_skill(self, skill: str | int) -> torch.Tensor:
        if isinstance(skill, str):
            try:
                skill = self.cfg.skill_names.index(skill)
            except ValueError as exc:
                raise ValueError(f"unknown skill name {skill!r}") from exc
        if isinstance(skill, bool) or not isinstance(skill, int) or not 0 <= skill < NUM_SKILLS:
            raise ValueError(f"skill must identify one of {self.cfg.skill_names}")
        return self.skill_ids == skill

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        return super().reset(env_ids)

    def _normalize_env_ids(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            if env_ids != slice(None):
                raise ValueError("only slice(None) is supported for command reset")
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        ids = self._normalize_env_ids(env_ids)
        if ids.numel() == 0:
            return
        if self.cfg.forced_skill_id is None:
            sampled = self.episode_skill_ids[ids].clone()
            unassigned = sampled < 0
            if torch.any(unassigned):
                assigned, self._skill_cursor = balanced_skill_ids(
                    int(torch.count_nonzero(unassigned).item()),
                    cursor=self._skill_cursor,
                    device=self.device,
                )
                sampled[unassigned] = assigned
                self.episode_skill_ids[ids[unassigned]] = assigned
        else:
            if not 0 <= self.cfg.forced_skill_id < NUM_SKILLS:
                raise ValueError(f"forced_skill_id must lie in [0, {NUM_SKILLS})")
            sampled = torch.full_like(ids, self.cfg.forced_skill_id)
            self.episode_skill_ids[ids] = sampled
        self.skill_ids[ids] = sampled
        self.motion_finished[ids] = False
        self.episode_complete[ids] = False
        self.composed_episode[ids] = False
        self.transition_stage[ids] = _DIRECT_STAGE
        self.transition_time_left[ids] = 0.0
        self.transition_stage_elapsed[ids] = 0.0
        self.last_motion_skill_ids[ids] = -1
        self.top_locomotion_released[ids] = False
        self.gait_time[ids] = 0.0
        self.gait_phase[ids] = self.phase_offset[ids]

        locomotion_ids = ids[sampled == LOCOMOTION_SKILL_ID]
        climb_ids = ids[sampled == CLIMB_SKILL_ID]
        down_ids = ids[sampled == DOWN_ROLL_SKILL_ID]
        self._reset_platform(ids, sampled)
        if locomotion_ids.numel() > 0:
            self._reset_locomotion(locomotion_ids)
        for routed_ids, skill_id in (
            (climb_ids, CLIMB_SKILL_ID),
            (down_ids, DOWN_ROLL_SKILL_ID),
        ):
            if routed_ids.numel() == 0:
                continue
            nominal_geometry = self._nominal_geometry_mask(routed_ids)
            compose = (~nominal_geometry) | (
                torch.rand(routed_ids.numel(), device=self.device)
                < self.cfg.composed_episode_fraction
            )
            direct_ids = routed_ids[~compose]
            composed_ids = routed_ids[compose]
            if direct_ids.numel() > 0:
                self._reset_motion_skill(direct_ids, skill_id)
            if composed_ids.numel() > 0:
                self._reset_motion_skill(
                    composed_ids,
                    skill_id,
                    start_at_beginning=True,
                    write_robot_state=skill_id == DOWN_ROLL_SKILL_ID,
                    align_reset_height_to_platform=skill_id == DOWN_ROLL_SKILL_ID,
                )
                self.composed_episode[composed_ids] = True
                if skill_id == CLIMB_SKILL_ID:
                    self._begin_climb_approach(composed_ids)
                else:
                    self._begin_down_roll_settle(composed_ids)
        self._refresh_relative_reference()
        motion_ids = ids[self.motion_mask[ids]]
        if motion_ids.numel() > 0:
            # Articulation data is refreshed only after CommandManager.reset
            # returns.  The just-written source state is exact, so seed the
            # first post-reset teacher observation with the source reference
            # instead of the preceding episode's stale body buffers.
            self.body_pos_relative_w[motion_ids] = self.body_pos_w[motion_ids]
            self.body_quat_relative_w[motion_ids] = self.body_quat_w[motion_ids]

    def _reset_platform(self, env_ids: torch.Tensor, route_ids: torch.Tensor) -> None:
        sizes = self.platform_sizes[env_ids]
        local_center = torch.tensor(self.cfg.platform_center, device=self.device).repeat(env_ids.numel(), 1)
        # Hold the climb entry edge fixed while length grows in the forward
        # direction.  This preserves the source climb geometry and makes only
        # the distance to the far edge variable.
        local_center[:, 0] += 0.5 * (sizes[:, 0] - self.cfg.platform_size[0])
        local_center[:, :2] += torch.tensor(self.cfg.platform_xy_offset, device=self.device)
        local_center[:, 2] = 0.5 * sizes[:, 2]
        hidden = route_ids == LOCOMOTION_SKILL_ID
        local_center[hidden, 2] = self.cfg.hidden_platform_center_z
        positions = local_center + self._env.scene.env_origins[env_ids]
        orientations = torch.zeros(env_ids.numel(), 4, device=self.device)
        orientations[:, 0] = 1.0
        self.platform_pos_w[env_ids] = positions
        self.platform_quat_w[env_ids] = orientations
        self.platform.write_root_pose_to_sim(
            torch.cat((positions, orientations), dim=1), env_ids=env_ids
        )
        self.platform.write_root_velocity_to_sim(
            torch.zeros(env_ids.numel(), 6, device=self.device), env_ids=env_ids
        )

    def _reset_locomotion(self, env_ids: torch.Tensor) -> None:
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._env.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        # Articulation data buffers are not refreshed until the reset forward
        # pass.  Use the root quaternion that was actually written here rather
        # than ``robot.data.heading_w`` from the preceding episode.  Otherwise
        # a zero-speed route can receive an unobservable stale heading target.
        root_forward = quat_apply(
            root_state[:, 3:7],
            torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(env_ids.numel(), -1),
        )
        reset_heading = torch.atan2(root_forward[:, 1], root_forward[:, 0])
        self._sample_locomotion_command(env_ids, current_heading=reset_heading)
        self.time_steps[env_ids] = 0
        self.motion_ids[env_ids] = 0
        self.episode_started_at_motion_beginning[env_ids] = True

    def _candidate_motion_ids(self, skill_id: int) -> torch.Tensor:
        return self.motion.climb_motion_ids if skill_id == CLIMB_SKILL_ID else self.motion.down_roll_motion_ids

    def _reset_motion_skill(
        self,
        env_ids: torch.Tensor,
        skill_id: int,
        *,
        start_at_beginning: bool | None = None,
        write_robot_state: bool = True,
        align_reset_height_to_platform: bool = False,
    ) -> None:
        candidates = self._candidate_motion_ids(skill_id)
        if self.cfg.motion_sampling_mode == "fixed":
            local_id = self.cfg.fixed_motion_id
            if not 0 <= local_id < candidates.numel():
                raise ValueError(
                    f"fixed_motion_id={local_id} is outside the selected skill's {candidates.numel()} clips"
                )
            motion_ids = candidates[local_id].expand(env_ids.numel())
        elif self.cfg.motion_sampling_mode == "round_robin":
            motion_ids = candidates[env_ids % candidates.numel()]
        elif self.cfg.motion_sampling_mode == "random":
            sampled = torch.randint(candidates.numel(), (env_ids.numel(),), device=self.device)
            motion_ids = candidates[sampled]
        else:
            raise ValueError("motion_sampling_mode must be 'random', 'fixed', or 'round_robin'")

        starts = self.motion.motion_start_idx[motion_ids]
        if start_at_beginning is None:
            start_at_beginning = self.cfg.start_at_motion_beginning
        if start_at_beginning:
            time_steps = starts
        else:
            ends = self.motion.motion_random_start_end_idx[motion_ids]
            widths = (ends - starts).clamp_min(1)
            time_steps = starts + torch.floor(torch.rand(env_ids.numel(), device=self.device) * widths).long()
        self.motion_ids[env_ids] = motion_ids
        self.time_steps[env_ids] = time_steps
        self.episode_started_at_motion_beginning[env_ids] = time_steps == starts
        self.world_command[env_ids] = torch.tensor(
            self.cfg.motion_world_command, device=self.device, dtype=self.world_command.dtype
        )
        self.heading_target[env_ids] = torch.atan2(
            self.world_command[env_ids, 1], self.world_command[env_ids, 0]
        )
        self.transition_stage[env_ids] = _MOTION_STAGE
        if write_robot_state:
            root_pos = self.body_pos_w[env_ids, self.motion_root_body_index]
            root_quat = self.body_quat_w[env_ids, self.motion_root_body_index]
            root_lin_vel = self.body_lin_vel_w[env_ids, self.motion_root_body_index]
            root_ang_vel = self.body_ang_vel_w[env_ids, self.motion_root_body_index]
            if align_reset_height_to_platform:
                root_pos = root_pos.clone()
                root_pos[:, 2] += self.platform_sizes[env_ids, 2] - self.cfg.platform_height
            joint_pos = self.motion.joint_pos[time_steps]
            joint_vel = self.motion.joint_vel[time_steps]
            limits = self.robot.data.soft_joint_pos_limits[env_ids]
            joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
            self.robot.write_root_state_to_sim(
                torch.cat((root_pos, root_quat, root_lin_vel, root_ang_vel), dim=1),
                env_ids=env_ids,
            )

    def _motion_direction_w(self, env_ids: torch.Tensor) -> torch.Tensor:
        platform_yaw = yaw_quat(self.platform_quat_w[env_ids])
        local_forward = torch.zeros(env_ids.numel(), 3, device=self.device)
        local_forward[:, 0] = 1.0
        return quat_apply(platform_yaw, local_forward)[:, :2]

    def _set_fixed_world_command(
        self,
        env_ids: torch.Tensor,
        command: torch.Tensor,
        *,
        current_heading: torch.Tensor | None = None,
    ) -> None:
        if command.shape != (env_ids.numel(), 2):
            raise ValueError("fixed world command must have shape [num_envs, 2]")
        if current_heading is None:
            current_heading = self.robot.data.heading_w[env_ids]
        elif current_heading.shape != (env_ids.numel(),):
            raise ValueError("current_heading must have one value per environment")
        self.world_command[env_ids] = command
        moving = torch.linalg.vector_norm(command, dim=1) > 1.0e-6
        target = torch.atan2(command[:, 1], command[:, 0])
        self.heading_target[env_ids] = torch.where(moving, target, current_heading)
        self.locomotion_command_time_left[env_ids] = float("inf")

    def _begin_climb_approach(self, env_ids: torch.Tensor) -> None:
        target_pos = self.body_pos_w[env_ids, self.motion_root_body_index].clone()
        target_quat = self.body_quat_w[env_ids, self.motion_root_body_index].clone()
        direction = self._motion_direction_w(env_ids)
        distance = torch.empty(env_ids.numel(), device=self.device).uniform_(
            *self.cfg.approach_distance_range
        )
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] = target_pos
        root_state[:, :2] -= direction * distance[:, None]
        root_state[:, 3:7] = target_quat
        root_state[:, 7:] = 0.0
        joint_pos = self.motion.joint_pos[self.time_steps[env_ids]]
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clamp(joint_pos, limits[..., 0], limits[..., 1])
        self.robot.write_joint_state_to_sim(
            joint_pos,
            torch.zeros_like(joint_pos),
            env_ids=env_ids,
        )
        self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.transition_target_xy[env_ids] = target_pos[:, :2]
        self.skill_ids[env_ids] = LOCOMOTION_SKILL_ID
        self.transition_stage[env_ids] = _APPROACH_STAGE
        speed = torch.full(
            (env_ids.numel(), 1),
            self.cfg.composed_locomotion_speed,
            device=self.device,
        )
        self._set_fixed_world_command(env_ids, direction * speed)

    def _sample_transition_duration(
        self,
        env_ids: torch.Tensor,
        duration_range: tuple[float, float],
    ) -> torch.Tensor:
        return torch.empty(env_ids.numel(), device=self.device).uniform_(*duration_range)

    def _begin_down_roll_settle(self, env_ids: torch.Tensor) -> None:
        root_quat = self.body_quat_w[env_ids, self.motion_root_body_index]
        root_forward = quat_apply(
            root_quat,
            torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(env_ids.numel(), -1),
        )
        reset_heading = torch.atan2(root_forward[:, 1], root_forward[:, 0])
        self.skill_ids[env_ids] = LOCOMOTION_SKILL_ID
        self.transition_stage[env_ids] = _SETTLE_STAGE
        self.transition_time_left[env_ids] = self._sample_transition_duration(
            env_ids,
            self.cfg.transition_settle_time_range_s,
        )
        self.transition_stage_elapsed[env_ids] = 0.0
        self._set_fixed_world_command(
            env_ids,
            torch.zeros(env_ids.numel(), 2, device=self.device),
            current_heading=reset_heading,
        )

    def _begin_motion_stage(
        self,
        env_ids: torch.Tensor,
        skill_ids: torch.Tensor | None = None,
    ) -> None:
        if skill_ids is None:
            skill_ids = self.episode_skill_ids[env_ids]
        if skill_ids.shape != (env_ids.numel(),):
            raise ValueError("motion-stage skill_ids must contain one route per environment")
        if torch.any((skill_ids != CLIMB_SKILL_ID) & (skill_ids != DOWN_ROLL_SKILL_ID)):
            raise ValueError("motion stage must route only climb or down-roll")
        self.skill_ids[env_ids] = skill_ids
        self.last_motion_skill_ids[env_ids] = skill_ids
        self.transition_stage[env_ids] = _MOTION_STAGE
        self.transition_stage_elapsed[env_ids] = 0.0
        self.motion_finished[env_ids] = False
        direction = self._motion_direction_w(env_ids)
        speed = float(math.hypot(*self.cfg.motion_world_command))
        self._set_fixed_world_command(env_ids, direction * speed)

    def _begin_post_locomotion(
        self,
        env_ids: torch.Tensor,
        completed_skill_ids: torch.Tensor,
    ) -> None:
        if completed_skill_ids.shape != (env_ids.numel(),):
            raise ValueError("completed_skill_ids must contain one route per environment")
        self.skill_ids[env_ids] = LOCOMOTION_SKILL_ID
        self.last_motion_skill_ids[env_ids] = completed_skill_ids
        self.transition_stage[env_ids] = _POST_LOCOMOTION_STAGE
        self.transition_stage_elapsed[env_ids] = 0.0
        self.top_locomotion_released[env_ids] = completed_skill_ids != CLIMB_SKILL_ID
        climb = completed_skill_ids == CLIMB_SKILL_ID
        down_roll = completed_skill_ids == DOWN_ROLL_SKILL_ID
        if torch.any(climb):
            climb_ids = env_ids[climb]
            self.transition_time_left[climb_ids] = self._sample_transition_duration(
                climb_ids,
                self.cfg.top_locomotion_time_range_s,
            )
        if torch.any(down_roll):
            down_ids = env_ids[down_roll]
            self.transition_time_left[down_ids] = self._sample_transition_duration(
                down_ids,
                self.cfg.post_locomotion_time_range_s,
            )
        direction = self._motion_direction_w(env_ids)
        speed = torch.where(
            completed_skill_ids == DOWN_ROLL_SKILL_ID,
            torch.full((env_ids.numel(),), self.cfg.composed_locomotion_speed, device=self.device),
            torch.zeros(env_ids.numel(), device=self.device),
        )
        self._set_fixed_world_command(env_ids, direction * speed[:, None])

    def _start_down_roll_from_edge(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._reset_motion_skill(
            env_ids,
            DOWN_ROLL_SKILL_ID,
            start_at_beginning=True,
            write_robot_state=False,
        )
        routes = torch.full(
            (env_ids.numel(),), DOWN_ROLL_SKILL_ID, device=self.device, dtype=torch.long
        )
        self._begin_motion_stage(env_ids, routes)

    def _down_roll_ready(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sizes = self.platform_sizes[env_ids]
        root_positions = self.robot.data.root_pos_w[env_ids]
        local_xy = oriented_box_local_xy(
            root_positions[:, None, :],
            self.platform_pos_w[env_ids],
            self.platform_quat_w[env_ids],
        ).squeeze(1)
        forward_edge_distance = 0.5 * sizes[:, 0] - local_xy[:, 0]
        direction = self._motion_direction_w(env_ids)
        command_forward_speed = torch.sum(self.world_command[env_ids] * direction, dim=1)
        platform_heading = torch.atan2(direction[:, 1], direction[:, 0])
        heading_error = wrap_to_pi(platform_heading - self.robot.data.heading_w[env_ids])
        gravity_xy_norm = torch.linalg.vector_norm(
            self.robot.data.projected_gravity_b[env_ids, :2], dim=1
        )
        ready = down_roll_transition_ready(
            forward_edge_distance,
            local_xy[:, 1],
            0.5 * sizes[:, 1],
            command_forward_speed,
            heading_error,
            gravity_xy_norm,
            minimum_edge_distance=self.cfg.down_roll_edge_distance_range[0],
            maximum_edge_distance=self.cfg.down_roll_edge_distance_range[1],
            lateral_margin=self.cfg.down_roll_lateral_margin,
            minimum_forward_speed=self.cfg.down_roll_minimum_forward_speed,
            maximum_heading_error=self.cfg.down_roll_maximum_heading_error,
            maximum_gravity_xy_norm=self.cfg.down_roll_maximum_gravity_xy_norm,
        )
        return ready, forward_edge_distance

    def _nominal_geometry_mask(self, env_ids: torch.Tensor) -> torch.Tensor:
        stored = getattr(self.platform, "_climb_box_random_phase_env_mask", None)
        if stored is not None:
            return stored.to(device=self.device, dtype=torch.bool)[env_ids]
        nominal = torch.tensor(self.cfg.platform_size, device=self.device)
        return torch.all(torch.isclose(self.platform_sizes[env_ids], nominal, atol=1.0e-6, rtol=0.0), dim=1)

    def set_training_iteration(self, iteration: int) -> None:
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("training iteration must be a non-negative integer")
        self.training_iteration = iteration

    @property
    def student_termination_scale(self) -> float:
        progress = min(
            1.0,
            self.training_iteration / self.cfg.student_termination_relaxation_iterations,
        )
        return 1.0 + progress * (self.cfg.student_termination_final_scale - 1.0)

    def _sample_locomotion_command(
        self,
        env_ids: torch.Tensor,
        *,
        current_heading: torch.Tensor | None = None,
    ) -> None:
        if self.cfg.forced_world_command is None:
            speed = torch.empty(env_ids.numel(), device=self.device).uniform_(*self.cfg.locomotion_speed_range)
            angle = torch.empty(env_ids.numel(), device=self.device).uniform_(-math.pi, math.pi)
            standing = torch.rand(env_ids.numel(), device=self.device) < self.cfg.locomotion_standing_fraction
            speed[standing] = 0.0
            self.world_command[env_ids, 0] = speed * torch.cos(angle)
            self.world_command[env_ids, 1] = speed * torch.sin(angle)
        else:
            forced = torch.tensor(self.cfg.forced_world_command, device=self.device)
            if forced.shape != (2,) or not bool(torch.isfinite(forced).all()):
                raise ValueError("forced_world_command must contain two finite values")
            self.world_command[env_ids] = forced
            speed = torch.linalg.vector_norm(self.world_command[env_ids], dim=1)
            angle = torch.atan2(self.world_command[env_ids, 1], self.world_command[env_ids, 0])
            standing = speed <= 1.0e-6
        if current_heading is None:
            current_heading = self.robot.data.heading_w[env_ids]
        elif current_heading.shape != (env_ids.numel(),):
            raise ValueError("current_heading must have one value per locomotion environment")
        self.heading_target[env_ids] = torch.where(standing, current_heading, angle)
        self.locomotion_command_time_left[env_ids] = self.cfg.locomotion_command_resampling_time_s

    def _update_metrics(self) -> None:
        self.metrics["route_locomotion"][:] = self.locomotion_mask
        self.metrics["route_climb"][:] = self.climb_mask
        self.metrics["route_down_roll"][:] = self.down_roll_mask
        motion = self.motion_mask
        self.metrics["motion_anchor_pos_error"][:] = torch.where(
            motion,
            torch.linalg.vector_norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=1),
            0.0,
        )
        self.metrics["motion_anchor_ori_error"][:] = torch.where(
            motion,
            quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w),
            0.0,
        )
        self.metrics["teacher_confidence"][:] = self.teacher_valid
        self.metrics["platform_length"][:] = self.platform_sizes[:, 0]
        self.metrics["platform_width"][:] = self.platform_sizes[:, 1]
        self.metrics["platform_height"][:] = self.platform_sizes[:, 2]
        top_ids = torch.where(
            self.locomotion_mask & (self.last_motion_skill_ids == CLIMB_SKILL_ID)
        )[0]
        self.metrics["forward_edge_distance"][:] = 0.0
        if top_ids.numel() > 0:
            _, distance = self._down_roll_ready(top_ids)
            self.metrics["forward_edge_distance"][top_ids] = distance

    def _update_command(self) -> None:
        locomotion = self.locomotion_mask
        if torch.any(locomotion):
            self.locomotion_command_time_left[locomotion] -= self._env.step_dt
            pure_locomotion = locomotion & (self.episode_skill_ids == LOCOMOTION_SKILL_ID)
            resample = torch.where(
                pure_locomotion & (self.locomotion_command_time_left <= 0.0)
            )[0]
            if resample.numel() > 0:
                self._sample_locomotion_command(resample)
            self.gait_time[locomotion] += self._env.step_dt
            phase_time = self.gait_time[locomotion] / self.cfg.gait_cycle
            self.gait_phase[locomotion] = (
                phase_time.unsqueeze(1) + self.phase_offset[locomotion]
            ) % 1.0

        # Capture the motion set before any stage transition.  A freshly
        # activated motion must expose its first frame for one full action
        # interval rather than being advanced immediately in this update.
        motion_ids_to_advance = torch.where(self.motion_mask & ~self.motion_finished)[0]

        approach_ids = torch.where(
            self.composed_episode & (self.transition_stage == _APPROACH_STAGE)
        )[0]
        if approach_ids.numel() > 0:
            error = torch.linalg.vector_norm(
                self.robot.data.root_pos_w[approach_ids, :2]
                - self.transition_target_xy[approach_ids],
                dim=1,
            )
            reached = approach_ids[error <= self.cfg.approach_switch_distance]
            if reached.numel() > 0:
                self.transition_stage[reached] = _SETTLE_STAGE
                self.transition_time_left[reached] = self._sample_transition_duration(
                    reached,
                    self.cfg.transition_settle_time_range_s,
                )
                self._set_fixed_world_command(
                    reached,
                    torch.zeros(reached.numel(), 2, device=self.device),
                )

        settle_ids = torch.where(
            self.composed_episode & (self.transition_stage == _SETTLE_STAGE)
        )[0]
        if settle_ids.numel() > 0:
            self.transition_time_left[settle_ids] -= self._env.step_dt
            self.transition_stage_elapsed[settle_ids] += self._env.step_dt
            ready = settle_ids[self.transition_time_left[settle_ids] <= 0.0]
            if ready.numel() > 0:
                self._begin_motion_stage(ready)

        finished_ids = torch.where(
            self.composed_episode
            & self.motion_mask
            & self.motion_finished
            & (self.transition_stage == _MOTION_STAGE)
        )[0]
        if finished_ids.numel() > 0:
            completed_skills = self.skill_ids[finished_ids].clone()
            self._begin_post_locomotion(finished_ids, completed_skills)

        post_ids = torch.where(
            self.composed_episode & (self.transition_stage == _POST_LOCOMOTION_STAGE)
        )[0]
        if post_ids.numel() > 0:
            self.transition_stage_elapsed[post_ids] += self._env.step_dt
            top = self.last_motion_skill_ids[post_ids] == CLIMB_SKILL_ID
            previously_released = self.top_locomotion_released[post_ids].clone()
            release = top & ~self.top_locomotion_released[post_ids] & (
                self.transition_stage_elapsed[post_ids] >= self.cfg.top_locomotion_settle_time_s
            )
            if torch.any(release):
                release_ids = post_ids[release]
                direction = self._motion_direction_w(release_ids)
                speed = torch.full(
                    (release_ids.numel(), 1),
                    self.cfg.composed_locomotion_speed,
                    device=self.device,
                )
                self._set_fixed_world_command(release_ids, direction * speed)
                self.top_locomotion_released[release_ids] = True

            top_ready = top & self.top_locomotion_released[post_ids]
            start_down = torch.zeros(post_ids.numel(), device=self.device, dtype=torch.bool)
            if torch.any(top_ready):
                candidate_ids = post_ids[top_ready]
                ready, _ = self._down_roll_ready(candidate_ids)
                start_down[top_ready] = ready
                self._start_down_roll_from_edge(candidate_ids[ready])

            remaining = ~start_down
            remaining_ids = post_ids[remaining]
            # The sampled 2--4 s top-locomotion duration begins only after the
            # zero-command settle has completed.  Do not consume one control
            # step on the same update that publishes the forward command.
            countdown = remaining & (~top | previously_released)
            countdown_ids = post_ids[countdown]
            self.transition_time_left[countdown_ids] -= self._env.step_dt
            complete = remaining_ids[self.transition_time_left[remaining_ids] <= 0.0]
            self.episode_complete[complete] = True

        if motion_ids_to_advance.numel() > 0:
            final = self.motion.motion_end_idx[self.motion_ids[motion_ids_to_advance]] - 1
            next_steps = self.time_steps[motion_ids_to_advance] + 1
            completed = next_steps >= final
            self.time_steps[motion_ids_to_advance] = torch.minimum(next_steps, final)
            self.motion_finished[motion_ids_to_advance] |= completed
        self._refresh_relative_reference()

    def _refresh_relative_reference(self) -> None:
        anchor_pos = self.anchor_pos_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        anchor_quat = self.anchor_quat_w[:, None, :].expand(-1, len(self.cfg.body_names), -1)
        robot_anchor_pos = self.robot_anchor_pos_w[:, None, :].expand_as(anchor_pos)
        robot_anchor_quat = self.robot_anchor_quat_w[:, None, :].expand_as(anchor_quat)
        delta_pos = robot_anchor_pos.clone()
        delta_pos[..., 2] = anchor_pos[..., 2]
        delta_quat = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        target_quat = quat_mul(delta_quat, self.body_quat_w)
        target_pos = delta_pos + quat_apply(delta_quat, self.body_pos_w - anchor_pos)
        locomotion = self.locomotion_mask
        target_pos[locomotion] = self.robot_body_pos_w[locomotion]
        target_quat[locomotion] = self.robot_body_quat_w[locomotion]
        self.body_pos_relative_w = target_pos
        self.body_quat_relative_w = target_quat

    def _source_body_pos_w(self) -> torch.Tensor:
        positions = self.motion.body_pos_w[self.time_steps]
        transformed = positions + self._env.scene.env_origins[:, None, :]
        platform_quat = yaw_quat(self.platform_quat_w)
        nominal_xy = torch.tensor(self.cfg.platform_center[:2], device=self.device, dtype=positions.dtype)
        delta = torch.zeros_like(positions)
        delta[..., :2] = positions[..., :2] - nominal_xy
        rotated = quat_apply(platform_quat[:, None, :].expand(-1, positions.shape[1], -1), delta)
        climb_offset, down_roll_offset = platform_reference_center_offsets(
            self.platform_sizes[:, 0], nominal_length=self.cfg.platform_size[0]
        )
        reference_offset = torch.where(self.down_roll_mask, down_roll_offset, climb_offset)
        direction = self._motion_direction_w(torch.arange(self.num_envs, device=self.device))
        reference_center_xy = self.platform_pos_w[:, :2] + direction * reference_offset[:, None]
        transformed[..., :2] = reference_center_xy[:, None, :] + rotated[..., :2]
        return transformed

    def _source_body_quat_w(self) -> torch.Tensor:
        orientations = self.motion.body_quat_w[self.time_steps]
        platform_quat = yaw_quat(self.platform_quat_w)
        return quat_mul(platform_quat[:, None, :].expand(-1, orientations.shape[1], -1), orientations)

    @property
    def joint_pos(self) -> torch.Tensor:
        source = self.motion.joint_pos[self.time_steps]
        return torch.where(self.motion_mask[:, None], source, self.robot.data.default_joint_pos)

    @property
    def source_joint_pos(self) -> torch.Tensor:
        return self.joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        source = self.motion.joint_vel[self.time_steps]
        return torch.where(self.motion_mask[:, None], source, torch.zeros_like(source))

    @property
    def body_pos_w(self) -> torch.Tensor:
        source = self._source_body_pos_w()
        return torch.where(self.motion_mask[:, None, None], source, self.robot_body_pos_w)

    @property
    def body_quat_w(self) -> torch.Tensor:
        source = self._source_body_quat_w()
        return torch.where(self.motion_mask[:, None, None], source, self.robot_body_quat_w)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        source = self.motion.body_lin_vel_w[self.time_steps]
        platform_quat = yaw_quat(self.platform_quat_w)
        source = quat_apply(platform_quat[:, None, :].expand(-1, source.shape[1], -1), source)
        return torch.where(self.motion_mask[:, None, None], source, self.robot_body_lin_vel_w)

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        source = self.motion.body_ang_vel_w[self.time_steps]
        platform_quat = yaw_quat(self.platform_quat_w)
        source = quat_apply(platform_quat[:, None, :].expand(-1, source.shape[1], -1), source)
        return torch.where(self.motion_mask[:, None, None], source, self.robot_body_ang_vel_w)

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

    @property
    def locomotion_teacher_command_b(self) -> torch.Tensor:
        speed = torch.linalg.vector_norm(self.world_command, dim=1)
        heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
        yaw_rate = torch.clamp(
            self.cfg.heading_control_stiffness * heading_error,
            self.cfg.locomotion_teacher_ang_vel_range[0],
            self.cfg.locomotion_teacher_ang_vel_range[1],
        )
        command = torch.stack((speed, torch.zeros_like(speed), yaw_rate), dim=1)
        command[:, 0] = command[:, 0].clamp(*self.cfg.locomotion_teacher_lin_vel_x_range)
        return command

    @property
    def teacher_valid(self) -> torch.Tensor:
        """Continuous confidence inside each frozen teacher's verified scope."""

        valid = torch.zeros(self.num_envs, dtype=self.world_command.dtype, device=self.device)
        pure_locomotion = self.locomotion_mask & (
            self.episode_skill_ids == LOCOMOTION_SKILL_ID
        )
        valid[pure_locomotion] = 1.0
        motion = self.motion_mask
        anchor_z_error = torch.abs(self.anchor_pos_w[:, 2] - self.robot_anchor_pos_w[:, 2])
        reference_gravity = quat_apply_inverse(self.anchor_quat_w, self.robot.data.GRAVITY_VEC_W)
        robot_gravity = quat_apply_inverse(self.robot_anchor_quat_w, self.robot.data.GRAVITY_VEC_W)
        orientation_error = torch.abs(reference_gravity[:, 2] - robot_gravity[:, 2])
        end_effector_ids = torch.tensor(
            [self.cfg.body_names.index(name) for name in self.cfg.teacher_end_effector_names],
            device=self.device,
            dtype=torch.long,
        )
        ee_z_error = torch.abs(
            self.body_pos_relative_w[:, end_effector_ids, 2]
            - self.robot_body_pos_w[:, end_effector_ids, 2]
        ).amax(dim=1)
        inside_scope = (
            (anchor_z_error <= self.cfg.teacher_anchor_z_threshold)
            & (orientation_error <= self.cfg.teacher_orientation_threshold)
            & (ee_z_error <= self.cfg.teacher_end_effector_z_threshold)
        )
        valid[motion] = (
            self.motion_teacher_geometry_confidence[motion]
            * inside_scope[motion].to(dtype=valid.dtype)
        )
        return valid

    @property
    def platform_sizes(self) -> torch.Tensor:
        return self._platform_sizes

    @property
    def motion_teacher_geometry_confidence(self) -> torch.Tensor:
        return platform_height_teacher_confidence(
            self.platform_sizes[:, 2],
            nominal_height=self.cfg.platform_height,
            full_tolerance=self.cfg.teacher_height_full_tolerance,
            zero_tolerance=self.cfg.teacher_height_zero_tolerance,
        )

    @property
    def motion_tracking_termination_enabled(self) -> torch.Tensor:
        return self.motion_teacher_geometry_confidence >= (
            self.cfg.motion_tracking_termination_min_confidence
        )

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:  # noqa: ARG002
        return

    def _debug_vis_callback(self, event) -> None:  # noqa: ARG002
        return


@configclass
class MultiSkillCommandCfg(CommandTermCfg):
    class_type: type = MultiSkillCommand

    asset_name: str = "robot"
    platform_asset_name: str = "platform"
    skill_names: tuple[str, str, str] = ("locomotion", "climb", "down_roll")
    forced_skill_id: int | None = None

    climb_motion_file: str | None = None
    climb_motion_dir: str | None = None
    down_roll_motion_file: str | None = None
    down_roll_motion_dir: str | None = None
    anchor_body_name: str = MISSING
    root_body_name: str | None = None
    body_names: list[str] = MISSING

    motion_sampling_mode: Literal["random", "fixed", "round_robin"] = "random"
    fixed_motion_id: int = 0
    start_at_motion_beginning: bool = False
    motion_world_command: tuple[float, float] = (0.6, 0.0)

    platform_size: tuple[float, float, float] = (0.51, 0.80, 0.66)
    platform_center: tuple[float, float, float] = (-0.95, 0.0, 0.33)
    platform_xy_offset: tuple[float, float] = (0.0, 0.0)
    platform_height: float = 0.66
    hidden_platform_center_z: float = -1.0

    locomotion_speed_range: tuple[float, float] = (0.2, 1.0)
    locomotion_standing_fraction: float = 0.10
    forced_world_command: tuple[float, float] | None = None
    locomotion_command_resampling_time_s: float = 10.0
    locomotion_teacher_lin_vel_x_range: tuple[float, float] = (-0.6, 1.0)
    locomotion_teacher_ang_vel_range: tuple[float, float] = (-1.57, 1.57)
    heading_control_stiffness: float = 0.5
    gait_air_ratios: tuple[float, float] = (0.38, 0.38)
    gait_phase_offsets: tuple[float, float] = (0.38, 0.88)
    gait_cycle: float = 0.85

    # Half of motion-family resets retain uniform atomic phase sampling; the
    # other half explicitly covers observable teacher transitions.
    composed_episode_fraction: float = 0.5
    approach_distance_range: tuple[float, float] = (0.4, 1.0)
    approach_switch_distance: float = 0.08
    transition_settle_time_range_s: tuple[float, float] = (0.2, 0.5)
    post_locomotion_time_range_s: tuple[float, float] = (0.8, 1.2)
    top_locomotion_time_range_s: tuple[float, float] = (2.0, 4.0)
    top_locomotion_settle_time_s: float = 0.2
    composed_locomotion_speed: float = 0.6
    down_roll_edge_distance_range: tuple[float, float] = (-0.05, 0.48)
    down_roll_lateral_margin: float = 0.08
    down_roll_minimum_forward_speed: float = 0.1
    down_roll_maximum_heading_error: float = 0.35
    down_roll_maximum_gravity_xy_norm: float = 0.35

    # Match the PHP student curriculum: start at each frozen teacher's scope
    # and linearly relax failure thresholds by 2x over the DAgger curriculum.
    student_termination_relaxation_iterations: int = 10_000
    student_termination_final_scale: float = 2.0

    teacher_anchor_z_threshold: float = 0.25
    teacher_orientation_threshold: float = 0.8
    teacher_end_effector_z_threshold: float = 0.30
    teacher_height_full_tolerance: float = 0.005
    teacher_height_zero_tolerance: float = 0.04
    motion_tracking_termination_min_confidence: float = 0.5
    teacher_end_effector_names: tuple[str, ...] = (
        "l_ankle_x_link",
        "r_ankle_x_link",
        "l_wrist_z_link",
        "r_wrist_z_link",
    )

    # The command owns all within-episode transitions.  This large interval
    # prevents CommandTerm's generic timer from racing that state machine.
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    debug_vis: bool = False


__all__ = [
    "CLIMB_SKILL_ID",
    "DOWN_ROLL_SKILL_ID",
    "LOCOMOTION_SKILL_ID",
    "MultiSkillCommand",
    "MultiSkillCommandCfg",
    "balanced_skill_ids",
]
