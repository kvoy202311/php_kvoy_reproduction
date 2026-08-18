from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
import math
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_apply_inverse,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from .climb_progress import bounded_episode_progress_increment
from .obstacle import get_climb_box_sizes
from .obstacle_geometry import (
    advance_filtered_platform_contact_time,
    foot_sole_corners_world,
    terminal_platform_z_alignment,
    terminal_sole_support_plane_z,
)
from .platform_foot_support import (
    platform_foot_load_valid,
    platform_foot_support_settings,
    platform_foot_support_state,
)
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

        self._terminal_platform_alignment_ramp_steps = 0
        self._terminal_source_support_z: torch.Tensor | None = None
        self._initialize_terminal_platform_z_alignment()
        self._validate_terminal_physical_hold_budget()

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
        # Allocated lazily by the climb-only first-foothold reward.  This must
        # not reuse ContactSensor.current_contact_time because that timer
        # includes ground and other non-platform contacts.
        self._first_foothold_filtered_contact_time: torch.Tensor | None = None
        # This timer is intentionally separate from the first-foothold timer:
        # terminal rewards and success inspect both feet on every policy step,
        # while the first-foot term is limited to one arriving lead foot.
        self._platform_foot_filtered_contact_time: torch.Tensor | None = None
        self._platform_foot_filtered_contact_step = -1

        # Per-environment state for the one-way default-pose terminal mode.
        # The source motion remains authoritative through the dynamic climb.
        # Once verified dual-foot support is reached in the static tail, the
        # command exposes one smooth joint target from that source pose to the
        # articulation default, never two competing targets.
        self._terminal_default_pose_latched = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._terminal_default_pose_age_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._terminal_default_pose_start_joint_pos = torch.zeros(
            self.num_envs, self.motion.joint_count, dtype=self.motion.joint_pos.dtype, device=self.device
        )
        self._terminal_default_pose_transition_steps = 0
        self._terminal_default_pose_static_window_steps = 0
        self._terminal_default_anchor_to_sole_height: torch.Tensor | None = None
        self._terminal_default_pose_settings = None
        # Strict terminal rewards and success use this timer regardless of
        # whether the optional default-joint terminal mode is enabled.
        self._platform_foot_support_timer_enabled = False
        self._initialize_platform_foot_support_timer()
        self._initialize_terminal_default_pose_mode()
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

        reset_all = env_ids is None
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._climb_progress_initialized[env_ids] = False
        self._climb_approach_max_potential[env_ids] = 0.0
        self._climb_lift_max_potential[env_ids] = 0.0
        if self._first_foothold_filtered_contact_time is not None:
            self._first_foothold_filtered_contact_time[env_ids] = 0.0
        if self._platform_foot_filtered_contact_time is not None:
            self._platform_foot_filtered_contact_time[env_ids] = 0.0
        # The step stamp is global, not per environment.  Clearing it for an
        # asynchronous subset reset would let later reward terms advance the
        # still-live environments twice in the same policy step.
        if reset_all:
            self._platform_foot_filtered_contact_step = -1
        self._terminal_default_pose_latched[env_ids] = False
        self._terminal_default_pose_age_steps[env_ids] = 0
        self._terminal_default_pose_start_joint_pos[env_ids] = 0.0
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

    def advance_first_foothold_filtered_contact_time(
        self,
        active_platform_support: torch.Tensor,
        step_dt: float,
    ) -> torch.Tensor:
        """Advance per-foot contact duration using only filtered platform support.

        The first-foothold reward is evaluated once per policy step.  Its
        state belongs here, rather than in a free function, so every subset of
        environments is reset together with its motion command.
        """

        if active_platform_support.ndim != 2 or active_platform_support.shape[0] != self.num_envs:
            raise ValueError(
                "active_platform_support must have shape "
                f"({self.num_envs}, num_feet), got {active_platform_support.shape}."
            )
        expected_device = torch.device(self.device)
        if active_platform_support.device != expected_device:
            raise ValueError(
                "active_platform_support must be on the MotionCommand device "
                f"{expected_device}, got {active_platform_support.device}."
            )
        if (
            self._first_foothold_filtered_contact_time is None
            or self._first_foothold_filtered_contact_time.shape != active_platform_support.shape
        ):
            self._first_foothold_filtered_contact_time = torch.zeros(
                active_platform_support.shape,
                dtype=torch.float32,
                device=self.device,
            )
        self._first_foothold_filtered_contact_time = advance_filtered_platform_contact_time(
            self._first_foothold_filtered_contact_time,
            active_platform_support,
            step_dt=step_dt,
        )
        return self._first_foothold_filtered_contact_time

    def advance_platform_foot_filtered_contact_time(
        self,
        active_platform_support: torch.Tensor,
        step_dt: float,
    ) -> torch.Tensor:
        """Advance shared two-foot platform support time exactly once per step.

        Several reward terms and the success term inspect terminal support in
        one environment step.  A generic ``ContactSensor.current_contact_time``
        would include ground contacts, while independently advancing a custom
        timer in every reward would make 0.25 s of support appear in a single
        policy step.  The command owns this state and de-duplicates updates by
        Isaac Lab's global policy-step counter.
        """

        if active_platform_support.shape != (self.num_envs, 2):
            raise ValueError(
                "active_platform_support must have shape "
                f"{(self.num_envs, 2)}, got {active_platform_support.shape}."
            )
        if active_platform_support.device != torch.device(self.device):
            raise ValueError(
                "active_platform_support must be on the MotionCommand device "
                f"{self.device}, got {active_platform_support.device}."
            )
        if step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, got {step_dt}.")
        if self._platform_foot_filtered_contact_time is None:
            self._platform_foot_filtered_contact_time = torch.zeros(
                (self.num_envs, 2), dtype=torch.float32, device=self.device
            )

        common_step_counter = getattr(self._env, "common_step_counter", None)
        if common_step_counter is None:
            # Production Isaac Lab environments always expose this counter.
            # Keep lightweight test doubles usable while making the fallback
            # explicit rather than silently sharing stale time between calls.
            self._platform_foot_filtered_contact_time = advance_filtered_platform_contact_time(
                self._platform_foot_filtered_contact_time,
                active_platform_support,
                step_dt=step_dt,
            )
            return self._platform_foot_filtered_contact_time

        common_step_counter = int(common_step_counter)
        if common_step_counter != self._platform_foot_filtered_contact_step:
            self._platform_foot_filtered_contact_time = advance_filtered_platform_contact_time(
                self._platform_foot_filtered_contact_time,
                active_platform_support,
                step_dt=step_dt,
            )
            self._platform_foot_filtered_contact_step = common_step_counter
        return self._platform_foot_filtered_contact_time

    @property
    def platform_foot_filtered_contact_time(self) -> torch.Tensor:
        """Read the two-foot filtered-support timer owned by command update.

        Terminal reward and termination terms intentionally consume this
        buffer without advancing it.  That makes support duration independent
        of how many terms are configured and avoids test-double fallbacks that
        do not expose a global policy-step counter.
        """

        if self._platform_foot_filtered_contact_time is None:
            return torch.zeros((self.num_envs, 2), dtype=self.motion.joint_pos.dtype, device=self.device)
        return self._platform_foot_filtered_contact_time

    def _source_body_quat_w_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Return source body orientations after the platform yaw transform."""

        orientations = self.motion.body_quat_w[time_steps]
        if self.reference_transform_asset is None:
            return orientations
        yaw_delta = yaw_quat(self.reference_transform_asset.data.root_quat_w)
        return quat_mul(yaw_delta[:, None, :].expand(-1, orientations.shape[1], -1), orientations)

    def _source_body_pos_w_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        """Return source body positions after the platform x/y/yaw transform."""

        positions = self.motion.body_pos_w[time_steps]
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

    def _initialize_platform_foot_support_timer(self) -> None:
        """Validate the shared physical two-foot support timer configuration.

        Strict terminal rewards and success consume this timer even when the
        optional default-joint terminal handoff is disabled.  Keeping the
        ownership here prevents that optional mode from silently deciding
        whether contact duration exists at all.
        """

        params = self.cfg.terminal_default_pose_platform_support_params
        base_size = self.cfg.terminal_default_pose_base_size
        if params is None:
            if base_size is not None:
                raise ValueError(
                    "terminal_default_pose_base_size requires terminal_default_pose_platform_support_params."
                )
            return
        if self.reference_transform_asset is None:
            raise ValueError(
                "shared platform-foot support timing requires reference_transform_asset_name='platform'."
            )
        if base_size is None or len(base_size) != 3 or any(size <= 0.0 for size in base_size):
            raise ValueError(
                "shared platform-foot support timing requires a three-element positive "
                "terminal_default_pose_base_size."
            )
        if self.cfg.terminal_default_pose_min_upward_force <= 0.0:
            raise ValueError(
                "shared platform-foot support timing requires a positive terminal_default_pose_min_upward_force."
            )
        if self.cfg.terminal_default_pose_sole_height_tolerance <= 0.0:
            raise ValueError(
                "shared platform-foot support timing requires a positive "
                "terminal_default_pose_sole_height_tolerance."
            )
        settings = platform_foot_support_settings(params)
        if len(settings.foot_body_names) != 2:
            raise ValueError(
                "shared platform-foot support timing requires exactly two configured foot bodies, "
                f"got {settings.foot_body_names}."
            )
        missing_foot_names = [name for name in settings.foot_body_names if name not in self.cfg.body_names]
        if missing_foot_names:
            raise ValueError(
                "shared platform-foot support timing requires feet tracked by MotionCommand, "
                f"missing {missing_foot_names}."
            )
        self._platform_foot_support_timer_enabled = True

    def _validate_terminal_physical_hold_budget(self) -> None:
        """Ensure a short terminal hold can still reach strict success.

        This is deliberately independent of the optional default-q handoff.
        The climb task can keep the immutable expert terminal pose while still
        requiring the completed whole-body Z bridge, sustained dual-foot
        support, and a continuous stable-standing interval before the motion
        clip is allowed to end.
        """

        support_time_s = self.cfg.terminal_support_confirmation_time_s
        stable_time_s = self.cfg.terminal_stable_time_s
        for name, value in (
            ("terminal_support_confirmation_time_s", support_time_s),
            ("terminal_stable_time_s", stable_time_s),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}.")

        # Generic tracking tasks leave both values at zero and retain their
        # historical final-frame behavior.  A climb configuration that opts
        # into either physical window must reserve both time budgets after the
        # terminal Z alignment, because support may first become valid only at
        # the end of that alignment.
        if support_time_s == 0.0 and stable_time_s == 0.0:
            return
        support_steps = math.ceil(support_time_s / self._env.step_dt - 1.0e-9)
        stable_steps = math.ceil(stable_time_s / self._env.step_dt - 1.0e-9)
        required_terminal_hold_steps = self._terminal_platform_alignment_ramp_steps + support_steps + stable_steps
        if self.motion_end_hold_steps < required_terminal_hold_steps:
            raise ValueError(
                "motion_end_hold_time_s is too short for terminal platform alignment, "
                "support confirmation, and stable evaluation."
            )

    def _platform_foot_support_state(self):
        """Return the shared physical support state, or ``None`` when unused."""

        if not self._platform_foot_support_timer_enabled:
            return None
        params = self.cfg.terminal_default_pose_platform_support_params
        base_size = self.cfg.terminal_default_pose_base_size
        if params is None or base_size is None or self.reference_transform_asset is None:
            raise RuntimeError("Validated shared platform-foot support configuration became incomplete.")
        return platform_foot_support_state(
            self._env,
            self.robot,
            self.device,
            SceneEntityCfg(self.cfg.reference_transform_asset_name),
            base_size,
            params,
            min_upward_force=self.cfg.terminal_default_pose_min_upward_force,
            sole_height_tolerance=self.cfg.terminal_default_pose_sole_height_tolerance,
        )

    def _update_platform_foot_support_timer(self) -> None:
        """Advance strict two-foot platform-contact time once per policy step."""

        support = self._platform_foot_support_state()
        if support is not None:
            self.advance_platform_foot_filtered_contact_time(support.active_support, self._env.step_dt)

    def _initialize_terminal_default_pose_mode(self) -> None:
        """Validate and precompute the platform-relative default terminal mode."""

        if not self.cfg.terminal_default_pose_enabled:
            return
        if self.reference_transform_asset is None:
            raise ValueError("terminal default-pose mode requires reference_transform_asset_name='platform'.")
        if not self.cfg.terminate_on_motion_end:
            raise ValueError("terminal default-pose mode requires terminate_on_motion_end=True.")
        if self.cfg.terminal_default_pose_platform_support_params is None:
            raise ValueError("terminal default-pose mode requires terminal_default_pose_platform_support_params.")
        if self.cfg.terminal_default_pose_base_size is None:
            raise ValueError("terminal default-pose mode requires terminal_default_pose_base_size.")
        base_size = self.cfg.terminal_default_pose_base_size
        if len(base_size) != 3 or any(size <= 0.0 for size in base_size):
            raise ValueError(
                "terminal_default_pose_base_size must contain three positive values, "
                f"got {base_size}."
            )
        if self.cfg.terminal_default_pose_transition_time_s <= 0.0:
            raise ValueError(
                "terminal_default_pose_transition_time_s must be positive when terminal mode is enabled."
            )
        if self.cfg.terminal_default_pose_static_window_time_s <= 0.0:
            raise ValueError(
                "terminal_default_pose_static_window_time_s must be positive when terminal mode is enabled."
            )
        if self.cfg.terminal_default_pose_contact_time_s <= 0.0:
            raise ValueError(
                "terminal_default_pose_contact_time_s must be positive when terminal mode is enabled."
            )
        if self.cfg.terminal_default_pose_sole_height_tolerance <= 0.0:
            raise ValueError(
                "terminal_default_pose_sole_height_tolerance must be positive when terminal mode is enabled."
            )
        if self.cfg.terminal_default_pose_min_upward_force <= 0.0:
            raise ValueError(
                "terminal_default_pose_min_upward_force must be positive when terminal mode is enabled."
            )
        if not 0.0 <= self.cfg.terminal_default_pose_min_total_load_fraction <= 1.0:
            raise ValueError("terminal_default_pose_min_total_load_fraction must lie in [0, 1].")
        if not 0.0 <= self.cfg.terminal_default_pose_max_torso_tilt < math.pi:
            raise ValueError("terminal_default_pose_max_torso_tilt must lie in [0, pi).")
        if self.cfg.terminal_default_pose_max_root_linear_speed <= 0.0:
            raise ValueError(
                "terminal_default_pose_max_root_linear_speed must be positive when terminal mode is enabled."
            )
        if self.cfg.terminal_default_pose_max_root_angular_speed <= 0.0:
            raise ValueError(
                "terminal_default_pose_max_root_angular_speed must be positive when terminal mode is enabled."
            )
        if self.cfg.terminal_default_pose_reference_max_joint_speed < 0.0:
            raise ValueError(
                "terminal_default_pose_reference_max_joint_speed must be non-negative when terminal mode is enabled."
            )

        settings = platform_foot_support_settings(self.cfg.terminal_default_pose_platform_support_params)
        self._terminal_default_pose_settings = settings
        missing_foot_names = [name for name in settings.foot_body_names if name not in self.cfg.body_names]
        if missing_foot_names:
            raise ValueError(
                "terminal default-pose foot bodies must be tracked by MotionCommand, "
                f"missing {missing_foot_names}."
            )
        self._terminal_default_pose_transition_steps = max(
            1, math.ceil(self.cfg.terminal_default_pose_transition_time_s / self._env.step_dt - 1.0e-9)
        )
        self._terminal_default_pose_static_window_steps = max(
            1, math.ceil(self.cfg.terminal_default_pose_static_window_time_s / self._env.step_dt - 1.0e-9)
        )
        terminal_contact_steps = max(
            1, math.ceil(self.cfg.terminal_default_pose_contact_time_s / self._env.step_dt - 1.0e-9)
        )
        # In the worst case, both feet only make contact after the source
        # reference has completed its platform-height ramp.  Reserve the
        # alignment, contact confirmation, q transition, and final stable
        # window explicitly so a configuration cannot make success impossible
        # merely by shortening the final hold.
        required_terminal_hold_steps = (
            self._terminal_platform_alignment_ramp_steps
            + terminal_contact_steps
            + self._terminal_default_pose_transition_steps
            + self._terminal_default_pose_static_window_steps
        )
        if self.motion_end_hold_steps < required_terminal_hold_steps:
            raise ValueError(
                "motion_end_hold_time_s is too short for terminal platform alignment, contact confirmation, "
                "default-pose transition, and stable evaluation."
            )

        # The supplied default-start motion is deliberately checked instead of
        # assuming that every arbitrary source clip begins at the robot's
        # default joint pose.  This gives a physical platform-relative torso
        # height for the same default pose used as the terminal q target.
        source_start_joint_pos = self.motion.joint_pos[self.motion.motion_start_idx]
        default_joint_pos = self.robot.data.default_joint_pos
        if default_joint_pos.ndim != 2 or default_joint_pos.shape[1] != self.motion.joint_count:
            raise ValueError(
                "Robot default_joint_pos must have shape [num_envs, joint_count] for terminal default-pose mode, "
                f"got {tuple(default_joint_pos.shape)}."
            )
        if not torch.allclose(default_joint_pos, default_joint_pos[:1], atol=1.0e-6, rtol=1.0e-6):
            raise ValueError(
                "terminal default-pose mode requires one shared articulation default across environments; "
                "per-environment default-joint randomization would invalidate its precomputed standing geometry."
            )
        # Source has one initial pose per clip, while the articulation stores
        # one default pose per environment.  Compare every source start with
        # the shared nominal default, then broadcast the live per-environment
        # default only when forming the terminal target below.
        nominal_default_joint_pos = default_joint_pos[0]
        if not torch.allclose(source_start_joint_pos, nominal_default_joint_pos, atol=1.0e-4, rtol=1.0e-4):
            max_error = torch.max(torch.abs(source_start_joint_pos - nominal_default_joint_pos)).item()
            raise ValueError(
                "terminal default-pose mode requires default-start expert clips; "
                f"the largest source/default joint mismatch is {max_error:.6f} rad."
            )
        foot_body_ids = torch.tensor(
            [self.cfg.body_names.index(name) for name in settings.foot_body_names],
            dtype=torch.long,
            device=self.device,
        )
        sole_corners_b = torch.tensor(
            settings.sole_corners_b,
            dtype=self.motion.body_pos_w.dtype,
            device=self.device,
        )
        source_start_sole = foot_sole_corners_world(
            self.motion.body_pos_w[self.motion.motion_start_idx][:, foot_body_ids],
            self.motion.body_quat_w[self.motion.motion_start_idx][:, foot_body_ids],
            sole_corners_b,
        )
        source_start_sole_z = source_start_sole[..., 2].amin(dim=(1, 2))
        source_start_anchor_z = self.motion.body_pos_w[
            self.motion.motion_start_idx, self.motion_anchor_body_index, 2
        ]
        self._terminal_default_anchor_to_sole_height = source_start_anchor_z - source_start_sole_z

    def _terminal_default_static_tail(self) -> torch.Tensor:
        """Return the final stationary source window, excluding the initial stand."""

        if not self.cfg.terminal_default_pose_enabled:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        final_frames = self.motion.motion_end_idx[self.motion_ids] - 1
        window_start = final_frames - (self._terminal_default_pose_static_window_steps - 1)
        in_final_window = self.time_steps >= window_start
        source_speed = torch.max(torch.abs(self.motion.joint_vel[self.time_steps]), dim=1).values
        source_static = source_speed <= self.cfg.terminal_default_pose_reference_max_joint_speed
        return in_final_window & source_static

    def _terminal_default_pose_support_candidate(self) -> torch.Tensor:
        """Return a strict, physical candidate for entering terminal default pose."""

        if not self.cfg.terminal_default_pose_enabled:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        support = self._platform_foot_support_state()
        if support is None:
            raise RuntimeError("Terminal default-pose support configuration disappeared after initialization.")
        filtered_time = self.platform_foot_filtered_contact_time
        dual_foot_support = torch.all(
            support.active_support & (filtered_time >= self.cfg.terminal_default_pose_contact_time_s), dim=1
        )
        feet_carry_load = platform_foot_load_valid(
            support,
            self.robot,
            min_total_load_fraction=self.cfg.terminal_default_pose_min_total_load_fraction,
        )
        root_linear_speed = torch.linalg.vector_norm(self.robot_anchor_lin_vel_w, dim=-1)
        root_angular_speed = torch.linalg.vector_norm(self.robot_anchor_ang_vel_w, dim=-1)
        projected_gravity_b = quat_apply_inverse(self.robot_anchor_quat_w, self.robot.data.GRAVITY_VEC_W)
        upright = projected_gravity_b[:, 2] <= -math.cos(self.cfg.terminal_default_pose_max_torso_tilt)
        quiet_enough = (
            (root_linear_speed <= self.cfg.terminal_default_pose_max_root_linear_speed)
            & (root_angular_speed <= self.cfg.terminal_default_pose_max_root_angular_speed)
            & upright
        )
        # The source reference must first finish its smooth Z bridge to the
        # sampled platform. Otherwise, a low box would require real support
        # while body/anchor rewards still point at a floating source pose.
        return (
            self._terminal_default_static_tail()
            & self.terminal_platform_alignment_complete
            & dual_foot_support
            & feet_carry_load
            & quiet_enough
        )

    def _update_terminal_default_pose_mode(self) -> None:
        """Latch the default-pose transition only after real bilateral support."""

        if not self.cfg.terminal_default_pose_enabled:
            return
        candidate = self._terminal_default_pose_support_candidate()
        newly_latched = candidate & ~self._terminal_default_pose_latched
        if torch.any(newly_latched):
            # Capture the *real supported articulation pose*, not raw source
            # q. Starting at the actual pose makes alpha=0 reward/target-
            # continuous, so the default-q handoff cannot create an abrupt
            # joint-target jump after the physical support check.
            self._terminal_default_pose_start_joint_pos[newly_latched] = self.robot.data.joint_pos[
                newly_latched
            ].to(dtype=self.motion.joint_pos.dtype)
            self._terminal_default_pose_latched[newly_latched] = True
            self._terminal_default_pose_age_steps[newly_latched] = 0
        already_latched = self._terminal_default_pose_latched & ~newly_latched
        self._terminal_default_pose_age_steps[already_latched] += 1

    def _terminal_default_pose_alpha_and_rate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return smooth command interpolation progress and its time derivative."""

        dtype = self.motion.joint_pos.dtype
        if not self.cfg.terminal_default_pose_enabled:
            zero = torch.zeros(self.num_envs, dtype=dtype, device=self.device)
            return zero, zero
        progress = (
            self._terminal_default_pose_age_steps.to(dtype=dtype) / float(self._terminal_default_pose_transition_steps)
        ).clamp(min=0.0, max=1.0)
        alpha = progress * progress * (3.0 - 2.0 * progress)
        alpha = alpha * self._terminal_default_pose_latched.to(dtype=dtype)
        alpha_rate = (
            6.0
            * progress
            * (1.0 - progress)
            / (float(self._terminal_default_pose_transition_steps) * self._env.step_dt)
        ) * self._terminal_default_pose_latched.to(dtype=dtype)
        return alpha, alpha_rate

    def _initialize_terminal_platform_z_alignment(self) -> None:
        """Precompute each clip's physical terminal sole plane when enabled."""

        ramp_time_s = self.cfg.terminal_platform_alignment_ramp_time_s
        if not math.isfinite(ramp_time_s) or ramp_time_s < 0.0:
            raise ValueError(
                "terminal_platform_alignment_ramp_time_s must be finite and non-negative, "
                f"got {ramp_time_s}."
            )
        if ramp_time_s == 0.0:
            return
        if self.reference_transform_asset is None:
            raise ValueError(
                "terminal platform z alignment requires reference_transform_asset_name to name the platform."
            )
        if not self.cfg.terminate_on_motion_end:
            raise ValueError("terminal platform z alignment requires terminate_on_motion_end=True.")
        if self.cfg.terminal_platform_alignment_base_size is None:
            raise ValueError("terminal platform z alignment requires terminal_platform_alignment_base_size.")
        base_size = self.cfg.terminal_platform_alignment_base_size
        if len(base_size) != 3 or any(size <= 0.0 for size in base_size):
            raise ValueError(
                "terminal_platform_alignment_base_size must contain three positive values, "
                f"got {base_size}."
            )
        foot_body_names = tuple(self.cfg.terminal_platform_alignment_foot_body_names)
        if not foot_body_names:
            raise ValueError("terminal platform z alignment requires at least one configured foot body.")
        if len(set(foot_body_names)) != len(foot_body_names):
            raise ValueError("terminal platform z alignment foot body names must be unique.")
        missing_foot_names = [name for name in foot_body_names if name not in self.cfg.body_names]
        if missing_foot_names:
            raise ValueError(
                "terminal platform z alignment foot bodies must be tracked by MotionCommand, "
                f"missing {missing_foot_names}."
            )
        if not self.cfg.terminal_platform_alignment_sole_corners_b:
            raise ValueError("terminal platform z alignment requires non-empty sole corner samples.")
        if not math.isfinite(self.cfg.terminal_platform_alignment_clearance) or (
            self.cfg.terminal_platform_alignment_clearance < 0.0
        ):
            raise ValueError(
                "terminal_platform_alignment_clearance must be finite and non-negative, "
                f"got {self.cfg.terminal_platform_alignment_clearance}."
            )

        self._terminal_platform_alignment_ramp_steps = max(
            1, math.ceil(ramp_time_s / self._env.step_dt - 1.0e-9)
        )
        if self.motion_end_hold_steps <= self._terminal_platform_alignment_ramp_steps:
            raise ValueError(
                "motion_end_hold_time_s must leave at least one stationary policy step after terminal platform "
                "z alignment; increase the hold or reduce the alignment ramp."
            )

        foot_body_ids = torch.tensor(
            [self.cfg.body_names.index(name) for name in foot_body_names], dtype=torch.long, device=self.device
        )
        final_frames = self.motion.motion_end_idx - 1
        terminal_foot_positions = self.motion.body_pos_w[final_frames][:, foot_body_ids]
        terminal_foot_orientations = self.motion.body_quat_w[final_frames][:, foot_body_ids]
        sole_corners_b = torch.as_tensor(
            self.cfg.terminal_platform_alignment_sole_corners_b,
            dtype=terminal_foot_positions.dtype,
            device=self.device,
        )
        self._terminal_source_support_z = terminal_sole_support_plane_z(
            terminal_foot_positions,
            terminal_foot_orientations,
            sole_corners_b,
        )

    def _terminal_platform_z_alignment(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the configured final-hold platform-relative reference offset."""

        if self._terminal_source_support_z is None:
            zero = torch.zeros(self.num_envs, dtype=self.motion.body_pos_w.dtype, device=self.device)
            return zero, zero, torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if self.reference_transform_asset is None:
            raise RuntimeError("Terminal platform z alignment lost its configured reference asset.")
        base_size = self.cfg.terminal_platform_alignment_base_size
        if base_size is None:
            raise RuntimeError("Terminal platform z alignment lost its configured base size.")
        platform_sizes = get_climb_box_sizes(
            self.reference_transform_asset,
            base_size=base_size,
            device=self.device,
        )
        if platform_sizes.shape != (self.num_envs, 3):
            raise RuntimeError(
                "Terminal platform z alignment received invalid platform sizes: "
                f"expected {(self.num_envs, 3)}, got {platform_sizes.shape}."
            )
        # Motion clips are stored in the per-environment local frame, whereas
        # the platform root position is world-frame.  ``body_pos_w`` adds the
        # environment origin after this correction, so align both support
        # planes in the local frame here; otherwise a nonzero terrain origin
        # would be applied twice.
        platform_top_z = (
            self.reference_transform_asset.data.root_pos_w[:, 2]
            + 0.5 * platform_sizes[:, 2]
            - self._env.scene.env_origins[:, 2]
        )
        return terminal_platform_z_alignment(
            self._terminal_source_support_z[self.motion_ids],
            platform_top_z,
            self.motion_final_hold_count,
            ramp_steps=self._terminal_platform_alignment_ramp_steps,
            step_dt=self._env.step_dt,
            terminal_clearance=self.cfg.terminal_platform_alignment_clearance,
        )

    @property
    def source_joint_pos(self) -> torch.Tensor:
        """Joint positions read directly from the immutable motion dataset."""

        return self.motion.joint_pos[self.time_steps]

    @property
    def terminal_default_pose_latched(self) -> torch.Tensor:
        """Whether each environment has irrevocably entered terminal q mode."""

        if not self.cfg.terminal_default_pose_enabled:
            return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return self._terminal_default_pose_latched

    @property
    def terminal_default_pose_alpha(self) -> torch.Tensor:
        """Smooth [0, 1] progress from captured supported q to default q."""

        alpha, _ = self._terminal_default_pose_alpha_and_rate()
        return alpha

    @property
    def terminal_default_pose_complete(self) -> torch.Tensor:
        """Whether the terminal default-pose interpolation has completed."""

        if not self.cfg.terminal_default_pose_enabled:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return self._terminal_default_pose_latched & (
            self._terminal_default_pose_age_steps >= self._terminal_default_pose_transition_steps
        )

    @property
    def terminal_default_pose_static_tail(self) -> torch.Tensor:
        """Whether the immutable source reference is in its verified static tail.

        This exposes only a source-data fact.  Reward terms use it to provide
        a settling signal before the terminal q switch, while still disabling
        that signal during the moving default-q interpolation.
        """

        if not self.cfg.terminal_default_pose_enabled:
            # Without a q handoff, only the fixed final source frame is a
            # terminal stationary target.  Returning all ones here would make
            # an optional consumer mistake the moving end of the expert clip
            # for a static tail merely because default-q mode is disabled.
            final_frames = self.motion.motion_end_idx[self.motion_ids] - 1
            return self.time_steps >= final_frames
        return self._terminal_default_static_tail()

    @property
    def terminal_default_pose_expert_tracking_factor(self) -> torch.Tensor:
        """Scale for source-body/anchor objectives during terminal q takeover.

        The switch is intentionally discrete at the *mode* boundary, rather
        than a reward cross-fade. At the latch boundary the sole joint target
        is the current physically supported articulation pose, so turning off
        source-body objectives cannot pull the robot back toward a stale
        reference placement. Every subsequent step has exactly one pose
        objective: the smooth command q trajectory to the articulation
        default. This avoids asking the policy to satisfy a stale expert body
        pose and a default-pose joint target at the same time.
        """

        if not self.cfg.terminal_default_pose_enabled:
            return torch.ones(self.num_envs, dtype=self.motion.joint_pos.dtype, device=self.device)
        return (~self._terminal_default_pose_latched).to(dtype=self.motion.joint_pos.dtype)

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
    def terminal_platform_alignment_complete(self) -> torch.Tensor:
        """Whether the terminal reference Z bridge has finished."""

        if self._terminal_source_support_z is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return self.motion_final_hold_count >= self._terminal_platform_alignment_ramp_steps

    @property
    def joint_pos(self) -> torch.Tensor:
        """Return one joint target, smoothly switching only after real support.

        Before latching this is exactly the immutable source motion. After
        latching it is the sole q objective: a smooth interpolation from the
        captured real supported pose to ``ELF3_CFG.init_state.joint_pos``.
        """

        if not self.cfg.terminal_default_pose_enabled:
            return self.source_joint_pos
        alpha, _ = self._terminal_default_pose_alpha_and_rate()
        default_joint_pos = self.robot.data.default_joint_pos.to(dtype=self.source_joint_pos.dtype)
        terminal_target = torch.lerp(self._terminal_default_pose_start_joint_pos, default_joint_pos, alpha[:, None])
        return torch.where(self._terminal_default_pose_latched[:, None], terminal_target, self.source_joint_pos)

    @property
    def joint_vel(self) -> torch.Tensor:
        if not self.cfg.terminal_default_pose_enabled:
            return self.motion.joint_vel[self.time_steps]
        _, alpha_rate = self._terminal_default_pose_alpha_and_rate()
        default_joint_pos = self.robot.data.default_joint_pos.to(dtype=self.motion.joint_vel.dtype)
        terminal_velocity = (default_joint_pos - self._terminal_default_pose_start_joint_pos) * alpha_rate[:, None]
        return torch.where(
            self._terminal_default_pose_latched[:, None], terminal_velocity, self.motion.joint_vel[self.time_steps]
        )

    @property
    def body_pos_w(self) -> torch.Tensor:
        transformed = self._source_body_pos_w_at(self.time_steps)
        transformed = transformed.clone()
        terminal_z_offset, _, _ = self._terminal_platform_z_alignment()
        transformed[..., 2] += terminal_z_offset[:, None]
        return transformed

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._source_body_quat_w_at(self.time_steps)

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        velocities = self.motion.body_lin_vel_w[self.time_steps]
        if self.reference_transform_asset is None:
            transformed = velocities
        else:
            yaw_delta = yaw_quat(self.reference_transform_asset.data.root_quat_w)
            transformed = quat_apply(yaw_delta[:, None, :].expand(-1, velocities.shape[1], -1), velocities)
        _, terminal_z_velocity, _ = self._terminal_platform_z_alignment()
        transformed[..., 2] += terminal_z_velocity[:, None]
        return transformed

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
    def terminal_default_anchor_pos_w(self) -> torch.Tensor:
        """Platform-relative anchor-height target for the default standing q.

        Only the z component is meaningful to the final-standing classifier.
        It uses the start clip's default standing anchor-to-sole distance and
        the *sampled* platform top, so a 0.60 m and a 0.70 m box receive the
        same natural default pose without translating an expert terminal body.
        """

        source_anchor = self.anchor_pos_w.clone()
        if not self.cfg.terminal_default_pose_enabled:
            return source_anchor
        if self.reference_transform_asset is None or self._terminal_default_anchor_to_sole_height is None:
            raise RuntimeError("Terminal default anchor target is missing validated platform geometry.")
        base_size = self.cfg.terminal_default_pose_base_size
        if base_size is None:
            raise RuntimeError("Terminal default anchor target is missing its platform base size.")
        sizes = get_climb_box_sizes(self.reference_transform_asset, base_size=base_size, device=self.device)
        platform_top_z = self.reference_transform_asset.data.root_pos_w[:, 2] + 0.5 * sizes[:, 2]
        default_anchor_z = platform_top_z + self._terminal_default_anchor_to_sole_height[self.motion_ids]
        alpha = self.terminal_default_pose_alpha
        source_anchor[:, 2] = torch.lerp(source_anchor[:, 2], default_anchor_z, alpha)
        return source_anchor

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
        if self._first_foothold_filtered_contact_time is not None:
            self._first_foothold_filtered_contact_time[env_ids] = 0.0
        if self._platform_foot_filtered_contact_time is not None:
            self._platform_foot_filtered_contact_time[env_ids] = 0.0
        self._terminal_default_pose_latched[env_ids] = False
        self._terminal_default_pose_age_steps[env_ids] = 0
        self._terminal_default_pose_start_joint_pos[env_ids] = 0.0
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

        # The command owns the strict platform-contact timer, so it advances
        # exactly once per policy step even when default-q handoff is disabled.
        # Isaac Lab evaluates reward and termination terms before
        # CommandManager.compute, therefore those terms consume the completed
        # timer from the preceding physics step; current contact loss is still
        # rejected immediately by their current force/geometry checks.
        self._update_platform_foot_support_timer()
        self._update_terminal_default_pose_mode()

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

    # Optional climb-only terminal correction.  During the extra final-frame
    # hold, all reference bodies translate together so the source's physical
    # sole plane meets the sampled platform top.  Joints and orientations stay
    # exactly equal to the source motion.
    terminal_platform_alignment_foot_body_names: tuple[str, ...] = ()
    terminal_platform_alignment_sole_corners_b: tuple[tuple[float, float, float], ...] = ()
    terminal_platform_alignment_base_size: tuple[float, float, float] | None = None
    terminal_platform_alignment_clearance: float = 0.0
    terminal_platform_alignment_ramp_time_s: float = 0.0

    # Optional physical final-hold budget, independent of default-q mode.
    # When either duration is non-zero, the command validates that the held
    # final frame has enough time for the Z bridge, dual-foot confirmation,
    # and continuous stable-standing evaluation.
    terminal_support_confirmation_time_s: float = 0.0
    terminal_stable_time_s: float = 0.0

    # One-way terminal mode for climb clips with a verified static tail.  It
    # latches only after two real platform-supported feet carry sufficient
    # load, then replaces the source joint target with one smooth path to the
    # articulation's configured default pose.
    terminal_default_pose_enabled: bool = False
    terminal_default_pose_platform_support_params: dict[str, object] | None = None
    terminal_default_pose_base_size: tuple[float, float, float] | None = None
    terminal_default_pose_transition_time_s: float = 0.0
    terminal_default_pose_static_window_time_s: float = 0.0
    terminal_default_pose_contact_time_s: float = 0.0
    terminal_default_pose_sole_height_tolerance: float = 0.0
    terminal_default_pose_min_upward_force: float = 0.0
    terminal_default_pose_min_total_load_fraction: float = 0.0
    terminal_default_pose_max_torso_tilt: float = 0.0
    terminal_default_pose_max_root_linear_speed: float = 0.0
    terminal_default_pose_max_root_angular_speed: float = 0.0
    terminal_default_pose_reference_max_joint_speed: float = 0.0

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
