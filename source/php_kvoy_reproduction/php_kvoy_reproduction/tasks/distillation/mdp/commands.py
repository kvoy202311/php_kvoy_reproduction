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
    approach_transition_status,
    balanced_skill_ids,
    climb_settle_geometry_ready,
    down_roll_settle_geometry_ready,
    down_roll_transition_ready,
    motion_boundary_alignment_ready,
    planar_command_speed_valid,
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
    the remaining resets mix complete atomic clips with uniform phase coverage.
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
        if (
            len(cfg.motion_world_command) != 2
            or not all(math.isfinite(value) for value in cfg.motion_world_command)
            or math.hypot(*cfg.motion_world_command) <= 0.0
        ):
            raise ValueError("motion_world_command must contain one finite non-zero planar command")
        if cfg.forced_world_command is not None and (
            len(cfg.forced_world_command) != 2
            or not all(math.isfinite(value) for value in cfg.forced_world_command)
        ):
            raise ValueError("forced_world_command must contain two finite values")
        if not 0.0 <= cfg.locomotion_standing_fraction < 1.0:
            raise ValueError("locomotion_standing_fraction must lie in [0, 1)")
        if not 0.0 < cfg.locomotion_speed_range[0] <= cfg.locomotion_speed_range[1]:
            raise ValueError("locomotion_speed_range must be positive and ordered")
        if (
            len(cfg.locomotion_teacher_lin_vel_x_range) != 2
            or not all(math.isfinite(value) for value in cfg.locomotion_teacher_lin_vel_x_range)
            or cfg.locomotion_teacher_lin_vel_x_range[0]
            > cfg.locomotion_teacher_lin_vel_x_range[1]
            or cfg.locomotion_teacher_lin_vel_x_range[1] <= 0.0
        ):
            raise ValueError(
                "locomotion_teacher_lin_vel_x_range must be finite, ordered, and allow positive speed"
            )
        if cfg.locomotion_speed_range[1] > cfg.locomotion_teacher_lin_vel_x_range[1]:
            raise ValueError(
                "locomotion_speed_range maximum must not exceed the locomotion teacher maximum"
            )
        maximum_requested_speed = cfg.locomotion_speed_range[1]
        if math.hypot(*cfg.motion_world_command) > maximum_requested_speed:
            raise ValueError(
                "motion_world_command speed must not exceed locomotion_speed_range maximum"
            )
        if (
            cfg.forced_world_command is not None
            and math.hypot(*cfg.forced_world_command) > maximum_requested_speed
        ):
            raise ValueError(
                "forced_world_command speed must not exceed locomotion_speed_range maximum"
            )
        if cfg.locomotion_command_resampling_time_s <= 0.0:
            raise ValueError("locomotion command resampling time must be positive")
        if not isinstance(cfg.locked_command_resampling_enabled, bool):
            raise ValueError("locked_command_resampling_enabled must be a boolean")
        if (
            len(cfg.locked_command_resampling_time_range_s) != 2
            or not all(
                math.isfinite(value) and value > 0.0
                for value in cfg.locked_command_resampling_time_range_s
            )
            or cfg.locked_command_resampling_time_range_s[0]
            > cfg.locked_command_resampling_time_range_s[1]
        ):
            raise ValueError("locked command resampling time range must be positive and ordered")
        if cfg.gait_cycle <= 0.0:
            raise ValueError("gait_cycle must be positive")
        if not 0.0 <= cfg.composed_episode_fraction <= 1.0:
            raise ValueError("composed_episode_fraction must lie in [0, 1]")
        if not 0.0 < cfg.approach_distance_range[0] <= cfg.approach_distance_range[1]:
            raise ValueError("approach_distance_range must be positive and ordered")
        if cfg.approach_switch_distance <= 0.0:
            raise ValueError("approach_switch_distance must be positive")
        for name, value in (
            ("approach_lateral_tolerance", cfg.approach_lateral_tolerance),
            ("approach_maximum_overshoot", cfg.approach_maximum_overshoot),
            ("approach_timeout_s", cfg.approach_timeout_s),
            ("transition_maximum_settle_time_s", cfg.transition_maximum_settle_time_s),
            ("transition_maximum_joint_position_rms", cfg.transition_maximum_joint_position_rms),
            ("transition_maximum_joint_speed_rms", cfg.transition_maximum_joint_speed_rms),
            ("transition_maximum_gravity_xy_norm", cfg.transition_maximum_gravity_xy_norm),
            ("climb_maximum_heading_error", cfg.climb_maximum_heading_error),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if cfg.climb_maximum_heading_error > math.pi:
            raise ValueError("climb_maximum_heading_error must not exceed pi")
        if not 0.0 <= cfg.atomic_motion_start_at_beginning_fraction <= 1.0:
            raise ValueError("atomic_motion_start_at_beginning_fraction must lie in [0, 1]")
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
            not math.isfinite(cfg.post_motion_command_release_time_s)
            or cfg.post_motion_command_release_time_s < 0.0
        ):
            raise ValueError("post_motion_command_release_time_s must be non-negative")
        if (
            not math.isfinite(cfg.post_climb_contact_grace_time_s)
            or cfg.post_climb_contact_grace_time_s < 0.0
        ):
            raise ValueError("post_climb_contact_grace_time_s must be non-negative")
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
        self.pending_motion_skill_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.transition_failed = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.last_motion_skill_ids = torch.full(
            (self.num_envs,), -1, device=self.device, dtype=torch.long
        )
        self.post_motion_command_released = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.training_iteration = 0
        self.episode_started_at_motion_beginning = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_reached_motion = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_completed_climb = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_reached_top_locomotion = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.episode_started_down_roll = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        # Keep the live deployment request separate from the privileged
        # training controller.  The Actor always observes the latest request,
        # while ``world_command`` may deliberately stop the locomotion teacher
        # during a committed climb/down-roll boundary and resumes the latest
        # request only after that motion has been released safely.
        self.requested_world_command = torch.zeros(self.num_envs, 2, device=self.device)
        self.world_command = torch.zeros(self.num_envs, 2, device=self.device)
        self.heading_target = self.robot.data.heading_w.clone()
        self.locomotion_command_time_left = torch.zeros(self.num_envs, device=self.device)
        self.locked_command_time_left = torch.full(
            (self.num_envs,), float("inf"), device=self.device
        )
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
        for name in (
            "stage_approach",
            "stage_settle",
            "stage_motion",
            "stage_post_locomotion",
            "approach_longitudinal_error",
            "approach_lateral_error",
            "settle_joint_position_rms",
            "settle_joint_speed_rms",
            "settle_gravity_xy_norm",
            "settle_geometry_ready",
            "settle_kinematic_scope_valid",
            "motion_control_locked",
            "command_suppressed",
            "requested_command_speed",
            "active_command_speed",
            "post_motion_command_released",
            "transition_failed",
            "motion_started_at_beginning",
            "episode_reached_motion",
            "episode_completed_climb",
            "episode_reached_top_locomotion",
            "episode_started_down_roll",
        ):
            self.metrics[name] = torch.zeros(self.num_envs, device=self.device)

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

    @property
    def motion_control_locked(self) -> torch.Tensor:
        """Whether a committed motion temporarily owns whole-body control.

        Approach remains ordinary command-responsive locomotion.  Locking
        begins at the boundary settle, covers the complete climb/down-roll
        clip, and ends only after the short post-motion safety release.
        """

        settling = self.transition_stage == _SETTLE_STAGE
        post_release = (self.transition_stage == _POST_LOCOMOTION_STAGE) & (
            ~self.post_motion_command_released
        )
        return settling | self.motion_mask | post_release

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
        self.pending_motion_skill_ids[ids] = -1
        self.transition_failed[ids] = False
        self.last_motion_skill_ids[ids] = -1
        self.post_motion_command_released[ids] = False
        self.locked_command_time_left[ids] = float("inf")
        self.episode_reached_motion[ids] = False
        self.episode_completed_climb[ids] = False
        self.episode_reached_top_locomotion[ids] = False
        self.episode_started_down_roll[ids] = False
        self.gait_time[ids] = 0.0
        self.gait_phase[ids] = self.phase_offset[ids]

        locomotion_ids = ids[sampled == LOCOMOTION_SKILL_ID]
        climb_ids = ids[sampled == CLIMB_SKILL_ID]
        down_ids = ids[sampled == DOWN_ROLL_SKILL_ID]
        self._reset_platform(ids, sampled)
        motion_route_ids = ids[sampled != LOCOMOTION_SKILL_ID]
        if motion_route_ids.numel() > 0:
            self._initialize_motion_requested_command(motion_route_ids)
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
                self._reset_atomic_motion(direct_ids, skill_id)
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

    def _reset_atomic_motion(self, env_ids: torch.Tensor, skill_id: int) -> None:
        """Mix full-clip starts with uniform phase coverage for atomic skills."""

        self.episode_reached_motion[env_ids] = True
        if skill_id == DOWN_ROLL_SKILL_ID:
            self.episode_started_down_roll[env_ids] = True
        if self.cfg.start_at_motion_beginning:
            self._reset_motion_skill(env_ids, skill_id, start_at_beginning=True)
        else:
            full_start = torch.rand(env_ids.numel(), device=self.device) < (
                self.cfg.atomic_motion_start_at_beginning_fraction
            )
            if torch.any(full_start):
                self._reset_motion_skill(env_ids[full_start], skill_id, start_at_beginning=True)
            if torch.any(~full_start):
                self._reset_motion_skill(env_ids[~full_start], skill_id, start_at_beginning=False)
        # Atomic clips terminate before returning to locomotion, so immediately
        # randomizing the still-visible deployment request teaches command
        # invariance without disrupting a composed transition curriculum.
        self._begin_motion_command_lock(env_ids, randomize_immediately=True)

    def _reset_gait_phase(self, env_ids: torch.Tensor) -> None:
        self.gait_time[env_ids] = 0.0
        self.gait_phase[env_ids] = self.phase_offset[env_ids]

    def _initialize_motion_requested_command(self, env_ids: torch.Tensor) -> None:
        """Publish the initial live deployment request for a motion-family episode."""

        values = (
            self.cfg.motion_world_command
            if self.cfg.forced_world_command is None
            else self.cfg.forced_world_command
        )
        self.requested_world_command[env_ids] = torch.tensor(
            values,
            device=self.device,
            dtype=self.requested_world_command.dtype,
        )

    def _sample_requested_command_values(self, count: int) -> torch.Tensor:
        """Sample realistic world-frame deployment requests without applying them."""

        if count < 0:
            raise ValueError("requested command sample count must be non-negative")
        if count == 0:
            return torch.empty(0, 2, device=self.device)
        if self.cfg.forced_world_command is not None:
            forced = torch.tensor(
                self.cfg.forced_world_command,
                device=self.device,
                dtype=self.requested_world_command.dtype,
            )
            return forced.expand(count, -1).clone()
        speed = torch.empty(count, device=self.device).uniform_(*self.cfg.locomotion_speed_range)
        angle = torch.empty(count, device=self.device).uniform_(-math.pi, math.pi)
        standing = torch.rand(count, device=self.device) < self.cfg.locomotion_standing_fraction
        speed[standing] = 0.0
        return torch.stack((speed * torch.cos(angle), speed * torch.sin(angle)), dim=1)

    def _sync_requested_locomotion_command(self, env_ids: torch.Tensor) -> None:
        """Apply the latest public request to command-responsive locomotion."""

        if env_ids.numel() == 0:
            return
        self._set_fixed_world_command(
            env_ids,
            self.requested_world_command[env_ids].clone(),
        )

    def set_requested_world_command(
        self,
        env_ids: Sequence[int] | slice,
        command: torch.Tensor | Sequence[float],
    ) -> None:
        """Update the live deployment request and apply it only when unlocked.

        Hardware may continue publishing joystick commands during climb or
        down-roll.  Those values remain visible to the Student and are retained
        here, but they cannot interrupt a committed motion.  The newest value is
        applied automatically when locomotion is released again.
        """

        ids = self._normalize_env_ids(env_ids)
        values = torch.as_tensor(
            command,
            device=self.device,
            dtype=self.requested_world_command.dtype,
        )
        if values.shape == (2,):
            values = values.expand(ids.numel(), -1)
        if values.shape != (ids.numel(), 2):
            raise ValueError("requested world command must have shape [2] or [num_envs, 2]")
        if not bool(torch.isfinite(values).all()):
            raise ValueError("requested world command must contain only finite values")
        valid_speed = planar_command_speed_valid(
            values,
            maximum_speed=self.cfg.locomotion_speed_range[1],
        )
        if not bool(torch.all(valid_speed)):
            raise ValueError(
                "requested world command speed must not exceed "
                f"{self.cfg.locomotion_speed_range[1]:g} m/s"
            )
        self.requested_world_command[ids] = values
        responsive = self.locomotion_mask[ids] & ~self.motion_control_locked[ids]
        self._sync_requested_locomotion_command(ids[responsive])

    def _begin_motion_command_lock(
        self,
        env_ids: torch.Tensor,
        *,
        randomize_immediately: bool,
    ) -> None:
        """Start nuisance-command scheduling for a committed motion."""

        if env_ids.numel() == 0:
            return
        # A new settle/motion commitment supersedes any release state left by
        # the preceding skill (notably climb -> top locomotion -> down-roll).
        self.post_motion_command_released[env_ids] = False
        if not self.cfg.locked_command_resampling_enabled or self.cfg.forced_world_command is not None:
            self.locked_command_time_left[env_ids] = float("inf")
            return
        if randomize_immediately:
            self.requested_world_command[env_ids] = self._sample_requested_command_values(
                env_ids.numel()
            )
        self.locked_command_time_left[env_ids] = self._sample_transition_duration(
            env_ids,
            self.cfg.locked_command_resampling_time_range_s,
        )

    def _update_locked_requested_commands(self) -> None:
        """Change only Actor-visible requests while committed motion owns control."""

        locked = self.motion_control_locked
        if (
            not self.cfg.locked_command_resampling_enabled
            or self.cfg.forced_world_command is not None
            or not torch.any(locked)
        ):
            return
        self.locked_command_time_left[locked] -= self._env.step_dt
        due = torch.where(locked & (self.locked_command_time_left <= 0.0))[0]
        if due.numel() == 0:
            return
        self.requested_world_command[due] = self._sample_requested_command_values(due.numel())
        self.locked_command_time_left[due] = self._sample_transition_duration(
            due,
            self.cfg.locked_command_resampling_time_range_s,
        )

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
        """Set only the privileged locomotion-control command.

        The Student-visible deployment request intentionally remains unchanged
        across approach, settle, motion and post-motion stage transitions.
        """

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
        self.pending_motion_skill_ids[env_ids] = CLIMB_SKILL_ID
        self.transition_stage[env_ids] = _APPROACH_STAGE
        self.transition_stage_elapsed[env_ids] = 0.0
        self._reset_gait_phase(env_ids)
        root_forward = quat_apply(
            root_state[:, 3:7],
            torch.tensor((1.0, 0.0, 0.0), device=self.device).expand(env_ids.numel(), -1),
        )
        reset_heading = torch.atan2(root_forward[:, 1], root_forward[:, 0])
        self._set_fixed_world_command(
            env_ids,
            self.requested_world_command[env_ids].clone(),
            current_heading=reset_heading,
        )

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
        self.pending_motion_skill_ids[env_ids] = DOWN_ROLL_SKILL_ID
        self.transition_stage[env_ids] = _SETTLE_STAGE
        self.transition_time_left[env_ids] = self._sample_transition_duration(
            env_ids,
            self.cfg.transition_settle_time_range_s,
        )
        self.transition_stage_elapsed[env_ids] = 0.0
        self._reset_gait_phase(env_ids)
        self._set_fixed_world_command(
            env_ids,
            torch.zeros(env_ids.numel(), 2, device=self.device),
            current_heading=reset_heading,
        )
        self._begin_motion_command_lock(env_ids, randomize_immediately=False)

    def _begin_motion_stage(
        self,
        env_ids: torch.Tensor,
        skill_ids: torch.Tensor | None = None,
    ) -> None:
        if skill_ids is None:
            skill_ids = self.pending_motion_skill_ids[env_ids]
        if skill_ids.shape != (env_ids.numel(),):
            raise ValueError("motion-stage skill_ids must contain one route per environment")
        if torch.any((skill_ids != CLIMB_SKILL_ID) & (skill_ids != DOWN_ROLL_SKILL_ID)):
            raise ValueError("motion stage must route only climb or down-roll")
        self.skill_ids[env_ids] = skill_ids
        self.last_motion_skill_ids[env_ids] = skill_ids
        self.transition_stage[env_ids] = _MOTION_STAGE
        self.transition_stage_elapsed[env_ids] = 0.0
        self.motion_finished[env_ids] = False
        self.pending_motion_skill_ids[env_ids] = -1
        self.episode_reached_motion[env_ids] = True
        down_roll = skill_ids == DOWN_ROLL_SKILL_ID
        self.episode_started_down_roll[env_ids[down_roll]] = True
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
        self._reset_gait_phase(env_ids)
        self.post_motion_command_released[env_ids] = False
        climb = completed_skill_ids == CLIMB_SKILL_ID
        down_roll = completed_skill_ids == DOWN_ROLL_SKILL_ID
        if torch.any(climb):
            climb_ids = env_ids[climb]
            self.episode_completed_climb[climb_ids] = True
            self.episode_reached_top_locomotion[climb_ids] = True
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
        # Both skills retain control for one short safety interval.  A joystick
        # request may keep changing during this interval, but it is applied only
        # when the post-motion release gate opens below.
        self._set_fixed_world_command(
            env_ids,
            torch.zeros(env_ids.numel(), 2, device=self.device),
        )

    def _begin_down_roll_from_edge_settle(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._reset_motion_skill(
            env_ids,
            DOWN_ROLL_SKILL_ID,
            start_at_beginning=True,
            write_robot_state=False,
        )
        self.skill_ids[env_ids] = LOCOMOTION_SKILL_ID
        self.pending_motion_skill_ids[env_ids] = DOWN_ROLL_SKILL_ID
        self.transition_stage[env_ids] = _SETTLE_STAGE
        self.transition_time_left[env_ids] = self._sample_transition_duration(
            env_ids,
            self.cfg.transition_settle_time_range_s,
        )
        self.transition_stage_elapsed[env_ids] = 0.0
        self._reset_gait_phase(env_ids)
        self._set_fixed_world_command(
            env_ids,
            torch.zeros(env_ids.numel(), 2, device=self.device),
        )
        self._begin_motion_command_lock(env_ids, randomize_immediately=False)

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
        command_heading = torch.atan2(
            self.world_command[env_ids, 1],
            self.world_command[env_ids, 0],
        )
        command_heading_error = wrap_to_pi(platform_heading - command_heading)
        body_heading_error = wrap_to_pi(
            platform_heading - self.robot.data.heading_w[env_ids]
        )
        gravity_xy_norm = torch.linalg.vector_norm(
            self.robot.data.projected_gravity_b[env_ids, :2], dim=1
        )
        ready = down_roll_transition_ready(
            forward_edge_distance,
            local_xy[:, 1],
            0.5 * sizes[:, 1],
            command_forward_speed,
            command_heading_error,
            body_heading_error,
            gravity_xy_norm,
            minimum_edge_distance=self.cfg.down_roll_edge_distance_range[0],
            maximum_edge_distance=self.cfg.down_roll_edge_distance_range[1],
            lateral_margin=self.cfg.down_roll_lateral_margin,
            minimum_forward_speed=self.cfg.down_roll_minimum_forward_speed,
            maximum_heading_error=self.cfg.down_roll_maximum_heading_error,
            maximum_gravity_xy_norm=self.cfg.down_roll_maximum_gravity_xy_norm,
        )
        return ready, forward_edge_distance

    def _approach_errors(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        direction = self._motion_direction_w(env_ids)
        delta = self.transition_target_xy[env_ids] - self.robot.data.root_pos_w[env_ids, :2]
        longitudinal = torch.sum(delta * direction, dim=1)
        lateral_direction = torch.stack((-direction[:, 1], direction[:, 0]), dim=1)
        lateral = torch.sum(delta * lateral_direction, dim=1)
        return longitudinal, lateral

    def _motion_boundary_alignment(
        self,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_joint_pos = self.motion.joint_pos[self.time_steps[env_ids]]
        joint_position_rms = torch.sqrt(
            torch.mean(torch.square(self.robot.data.joint_pos[env_ids] - target_joint_pos), dim=1)
        )
        # This is a settle gate, not a velocity-tracking reward.  Measure the
        # robot's absolute joint speed so it cannot enter a new frozen teacher
        # merely by matching a non-zero first-frame velocity while still
        # moving quickly.
        joint_speed_rms = torch.sqrt(
            torch.mean(torch.square(self.robot.data.joint_vel[env_ids]), dim=1)
        )
        gravity_xy_norm = torch.linalg.vector_norm(
            self.robot.data.projected_gravity_b[env_ids, :2], dim=1
        )
        ready = motion_boundary_alignment_ready(
            joint_position_rms,
            joint_speed_rms,
            gravity_xy_norm,
            maximum_joint_position_rms=self.cfg.transition_maximum_joint_position_rms,
            maximum_joint_speed_rms=self.cfg.transition_maximum_joint_speed_rms,
            maximum_gravity_xy_norm=self.cfg.transition_maximum_gravity_xy_norm,
        )
        return ready, joint_position_rms, joint_speed_rms, gravity_xy_norm

    def _settle_geometry_ready(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Check the pending teacher's physical entrance at the end of settle."""

        pending = self.pending_motion_skill_ids[env_ids]
        if torch.any((pending != CLIMB_SKILL_ID) & (pending != DOWN_ROLL_SKILL_ID)):
            raise RuntimeError("settle environments must have a pending climb or down-roll skill")
        ready = torch.zeros(env_ids.numel(), device=self.device, dtype=torch.bool)

        climb = pending == CLIMB_SKILL_ID
        if torch.any(climb):
            climb_ids = env_ids[climb]
            longitudinal, lateral = self._approach_errors(climb_ids)
            direction = self._motion_direction_w(climb_ids)
            platform_heading = torch.atan2(direction[:, 1], direction[:, 0])
            heading_error = wrap_to_pi(platform_heading - self.robot.data.heading_w[climb_ids])
            ready[climb] = climb_settle_geometry_ready(
                longitudinal,
                lateral,
                heading_error,
                switch_distance=self.cfg.approach_switch_distance,
                maximum_overshoot=self.cfg.approach_maximum_overshoot,
                lateral_tolerance=self.cfg.approach_lateral_tolerance,
                maximum_heading_error=self.cfg.climb_maximum_heading_error,
            )

        down_roll = pending == DOWN_ROLL_SKILL_ID
        if torch.any(down_roll):
            down_ids = env_ids[down_roll]
            sizes = self.platform_sizes[down_ids]
            local_xy = oriented_box_local_xy(
                self.robot.data.root_pos_w[down_ids, None, :],
                self.platform_pos_w[down_ids],
                self.platform_quat_w[down_ids],
            ).squeeze(1)
            forward_edge_distance = 0.5 * sizes[:, 0] - local_xy[:, 0]
            direction = self._motion_direction_w(down_ids)
            platform_heading = torch.atan2(direction[:, 1], direction[:, 0])
            heading_error = wrap_to_pi(platform_heading - self.robot.data.heading_w[down_ids])
            ready[down_roll] = down_roll_settle_geometry_ready(
                forward_edge_distance,
                local_xy[:, 1],
                0.5 * sizes[:, 1],
                heading_error,
                minimum_edge_distance=self.cfg.down_roll_edge_distance_range[0],
                maximum_edge_distance=self.cfg.down_roll_edge_distance_range[1],
                lateral_margin=self.cfg.down_roll_lateral_margin,
                maximum_heading_error=self.cfg.down_roll_maximum_heading_error,
            )
        return ready

    def motion_kinematic_scope_valid(
        self,
        env_ids: torch.Tensor,
        skill_ids: torch.Tensor,
        *,
        threshold_scale: float = 1.0,
    ) -> torch.Tensor:
        """Return frozen-motion kinematic validity for explicit routed skills."""

        if env_ids.ndim != 1 or skill_ids.shape != env_ids.shape:
            raise ValueError("motion scope env_ids and skill_ids must share one-dimensional shape")
        if torch.any((skill_ids != CLIMB_SKILL_ID) & (skill_ids != DOWN_ROLL_SKILL_ID)):
            raise ValueError("motion scope accepts only climb and down-roll skill IDs")
        if not math.isfinite(threshold_scale) or threshold_scale < 1.0:
            raise ValueError("motion scope threshold_scale must be finite and at least one")
        if env_ids.numel() == 0:
            return torch.empty(0, device=self.device, dtype=torch.bool)

        source_pos = self._source_body_pos_w_for(env_ids, skill_ids)
        source_quat = self._source_body_quat_w_for(env_ids)
        source_anchor_pos = source_pos[:, self.motion_anchor_body_index]
        source_anchor_quat = source_quat[:, self.motion_anchor_body_index]
        robot_anchor_pos = self.robot_anchor_pos_w[env_ids]
        robot_anchor_quat = self.robot_anchor_quat_w[env_ids]
        anchor_z_error = torch.abs(source_anchor_pos[:, 2] - robot_anchor_pos[:, 2])
        reference_gravity = quat_apply_inverse(source_anchor_quat, self.robot.data.GRAVITY_VEC_W[env_ids])
        robot_gravity = quat_apply_inverse(robot_anchor_quat, self.robot.data.GRAVITY_VEC_W[env_ids])
        orientation_error = torch.abs(reference_gravity[:, 2] - robot_gravity[:, 2])
        end_effector_ids = torch.tensor(
            [self.cfg.body_names.index(name) for name in self.cfg.teacher_end_effector_names],
            device=self.device,
            dtype=torch.long,
        )
        end_effector_z_error = torch.abs(
            source_pos[:, end_effector_ids, 2]
            - self.robot_body_pos_w[env_ids][:, end_effector_ids, 2]
        ).amax(dim=1)
        return (
            (anchor_z_error <= self.cfg.teacher_anchor_z_threshold * threshold_scale)
            & (orientation_error <= self.cfg.teacher_orientation_threshold * threshold_scale)
            & (end_effector_z_error <= self.cfg.teacher_end_effector_z_threshold * threshold_scale)
        )

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
            self.requested_world_command[env_ids, 0] = speed * torch.cos(angle)
            self.requested_world_command[env_ids, 1] = speed * torch.sin(angle)
        else:
            forced = torch.tensor(self.cfg.forced_world_command, device=self.device)
            if forced.shape != (2,) or not bool(torch.isfinite(forced).all()):
                raise ValueError("forced_world_command must contain two finite values")
            self.requested_world_command[env_ids] = forced
            speed = torch.linalg.vector_norm(self.requested_world_command[env_ids], dim=1)
            angle = torch.atan2(
                self.requested_world_command[env_ids, 1],
                self.requested_world_command[env_ids, 0],
            )
            standing = speed <= 1.0e-6
        self.world_command[env_ids] = self.requested_world_command[env_ids]
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
        self.metrics["stage_approach"][:] = self.transition_stage == _APPROACH_STAGE
        self.metrics["stage_settle"][:] = self.transition_stage == _SETTLE_STAGE
        self.metrics["stage_motion"][:] = self.transition_stage == _MOTION_STAGE
        self.metrics["stage_post_locomotion"][:] = self.transition_stage == _POST_LOCOMOTION_STAGE
        self.metrics["approach_longitudinal_error"][:] = 0.0
        self.metrics["approach_lateral_error"][:] = 0.0
        approach_ids = torch.where(self.transition_stage == _APPROACH_STAGE)[0]
        if approach_ids.numel() > 0:
            longitudinal, lateral = self._approach_errors(approach_ids)
            self.metrics["approach_longitudinal_error"][approach_ids] = longitudinal
            self.metrics["approach_lateral_error"][approach_ids] = torch.abs(lateral)
        self.metrics["settle_joint_position_rms"][:] = 0.0
        self.metrics["settle_joint_speed_rms"][:] = 0.0
        self.metrics["settle_gravity_xy_norm"][:] = 0.0
        self.metrics["settle_geometry_ready"][:] = 0.0
        self.metrics["settle_kinematic_scope_valid"][:] = 0.0
        settle_ids = torch.where(self.transition_stage == _SETTLE_STAGE)[0]
        if settle_ids.numel() > 0:
            _, pose_rms, speed_rms, gravity_norm = self._motion_boundary_alignment(settle_ids)
            self.metrics["settle_joint_position_rms"][settle_ids] = pose_rms
            self.metrics["settle_joint_speed_rms"][settle_ids] = speed_rms
            self.metrics["settle_gravity_xy_norm"][settle_ids] = gravity_norm
            self.metrics["settle_geometry_ready"][settle_ids] = self._settle_geometry_ready(
                settle_ids
            )
            self.metrics["settle_kinematic_scope_valid"][settle_ids] = (
                self.motion_kinematic_scope_valid(
                    settle_ids,
                    self.pending_motion_skill_ids[settle_ids],
                )
            )
        locked = self.motion_control_locked
        requested_speed = torch.linalg.vector_norm(self.requested_world_command, dim=1)
        active_speed = torch.linalg.vector_norm(self.world_command, dim=1)
        self.metrics["motion_control_locked"][:] = locked
        self.metrics["requested_command_speed"][:] = requested_speed
        self.metrics["active_command_speed"][:] = active_speed
        self.metrics["command_suppressed"][:] = locked & (
            torch.linalg.vector_norm(
                self.requested_world_command - self.world_command,
                dim=1,
            )
            > 1.0e-4
        )
        self.metrics["post_motion_command_released"][:] = self.post_motion_command_released
        self.metrics["transition_failed"][:] = self.transition_failed
        self.metrics["motion_started_at_beginning"][:] = self.episode_started_at_motion_beginning
        self.metrics["episode_reached_motion"][:] = self.episode_reached_motion
        self.metrics["episode_completed_climb"][:] = self.episode_completed_climb
        self.metrics["episode_reached_top_locomotion"][:] = self.episode_reached_top_locomotion
        self.metrics["episode_started_down_roll"][:] = self.episode_started_down_roll
        top_ids = torch.where(
            self.locomotion_mask & (self.last_motion_skill_ids == CLIMB_SKILL_ID)
        )[0]
        self.metrics["forward_edge_distance"][:] = 0.0
        if top_ids.numel() > 0:
            _, distance = self._down_roll_ready(top_ids)
            self.metrics["forward_edge_distance"][top_ids] = distance

    def _update_command(self) -> None:
        # Update the public joystick request first.  While a motion owns
        # control this deliberately changes only the Actor input; the active
        # teacher command remains locked until post-motion release.
        self._update_locked_requested_commands()

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
            self.transition_stage_elapsed[approach_ids] += self._env.step_dt
            longitudinal, lateral = self._approach_errors(approach_ids)
            ready_mask, failed_mask = approach_transition_status(
                longitudinal,
                lateral,
                self.transition_stage_elapsed[approach_ids],
                switch_distance=self.cfg.approach_switch_distance,
                lateral_tolerance=self.cfg.approach_lateral_tolerance,
                maximum_overshoot=self.cfg.approach_maximum_overshoot,
                timeout=self.cfg.approach_timeout_s,
            )
            reached = approach_ids[ready_mask]
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
                self.transition_stage_elapsed[reached] = 0.0
                self._begin_motion_command_lock(reached, randomize_immediately=False)
            failed = approach_ids[failed_mask]
            self.transition_failed[failed] = True

        settle_ids = torch.where(
            self.composed_episode & (self.transition_stage == _SETTLE_STAGE)
        )[0]
        if settle_ids.numel() > 0:
            self.transition_time_left[settle_ids] -= self._env.step_dt
            self.transition_stage_elapsed[settle_ids] += self._env.step_dt
            minimum_elapsed = self.transition_time_left[settle_ids] <= 0.0
            alignment_ready, _, _, _ = self._motion_boundary_alignment(settle_ids)
            geometry_ready = self._settle_geometry_ready(settle_ids)
            kinematic_scope_valid = self.motion_kinematic_scope_valid(
                settle_ids,
                self.pending_motion_skill_ids[settle_ids],
            )
            transition_ready = alignment_ready & geometry_ready & kinematic_scope_valid
            ready = settle_ids[minimum_elapsed & transition_ready]
            if ready.numel() > 0:
                self._begin_motion_stage(ready)
            failed = settle_ids[
                (~transition_ready)
                & (self.transition_stage_elapsed[settle_ids] >= self.cfg.transition_maximum_settle_time_s)
            ]
            self.transition_failed[failed] = True

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
            previously_released = self.post_motion_command_released[post_ids].clone()
            release = ~self.post_motion_command_released[post_ids] & (
                self.transition_stage_elapsed[post_ids]
                >= self.cfg.post_motion_command_release_time_s
            )
            if torch.any(release):
                release_ids = post_ids[release]
                self._sync_requested_locomotion_command(release_ids)
                self.post_motion_command_released[release_ids] = True
                self.locked_command_time_left[release_ids] = float("inf")

            top_ready = top & self.post_motion_command_released[post_ids]
            start_down = torch.zeros(post_ids.numel(), device=self.device, dtype=torch.bool)
            if torch.any(top_ready):
                candidate_ids = post_ids[top_ready]
                ready, _ = self._down_roll_ready(candidate_ids)
                start_down[top_ready] = ready
                self._begin_down_roll_from_edge_settle(candidate_ids[ready])

            remaining = ~start_down
            remaining_ids = post_ids[remaining]
            # The post-motion duration begins only after the zero-command
            # safety interval.  Do not consume one control step on the same
            # update that publishes the latest deployment request.
            countdown = remaining & previously_released
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

    def _source_body_pos_w_for(
        self,
        env_ids: torch.Tensor,
        skill_ids: torch.Tensor,
    ) -> torch.Tensor:
        if env_ids.ndim != 1 or skill_ids.shape != env_ids.shape:
            raise ValueError("source body env_ids and skill_ids must share one-dimensional shape")
        if torch.any((skill_ids != CLIMB_SKILL_ID) & (skill_ids != DOWN_ROLL_SKILL_ID)):
            raise ValueError("source body transforms accept only climb and down-roll skill IDs")
        positions = self.motion.body_pos_w[self.time_steps[env_ids]]
        transformed = positions + self._env.scene.env_origins[env_ids, None, :]
        platform_quat = yaw_quat(self.platform_quat_w[env_ids])
        nominal_xy = torch.tensor(self.cfg.platform_center[:2], device=self.device, dtype=positions.dtype)
        delta = torch.zeros_like(positions)
        delta[..., :2] = positions[..., :2] - nominal_xy
        rotated = quat_apply(platform_quat[:, None, :].expand(-1, positions.shape[1], -1), delta)
        climb_offset, down_roll_offset = platform_reference_center_offsets(
            self.platform_sizes[env_ids, 0], nominal_length=self.cfg.platform_size[0]
        )
        reference_offset = torch.where(
            skill_ids == DOWN_ROLL_SKILL_ID,
            down_roll_offset,
            climb_offset,
        )
        direction = self._motion_direction_w(env_ids)
        reference_center_xy = self.platform_pos_w[env_ids, :2] + direction * reference_offset[:, None]
        transformed[..., :2] = reference_center_xy[:, None, :] + rotated[..., :2]
        return transformed

    def _source_body_pos_w(self) -> torch.Tensor:
        env_ids = torch.arange(self.num_envs, device=self.device)
        source_skills = torch.where(
            self.down_roll_mask,
            DOWN_ROLL_SKILL_ID,
            CLIMB_SKILL_ID,
        )
        return self._source_body_pos_w_for(env_ids, source_skills)

    def _source_body_quat_w_for(self, env_ids: torch.Tensor) -> torch.Tensor:
        if env_ids.ndim != 1:
            raise ValueError("source body quaternion env_ids must be one-dimensional")
        orientations = self.motion.body_quat_w[self.time_steps[env_ids]]
        platform_quat = yaw_quat(self.platform_quat_w[env_ids])
        return quat_mul(platform_quat[:, None, :].expand(-1, orientations.shape[1], -1), orientations)

    def _source_body_quat_w(self) -> torch.Tensor:
        return self._source_body_quat_w_for(torch.arange(self.num_envs, device=self.device))

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
        # The locomotion teacher remains the valid supervisor during every
        # physically locomoting stage: pure walking, climb approach, boundary
        # settle, top traversal, and post-skill recovery.  Restricting labels
        # to the pure-locomotion episode family left all composed transitions
        # without an imitation gradient even though they use the same frozen
        # locomotion observation/action contract.
        valid[self.locomotion_mask] = 1.0
        motion_ids = torch.where(self.motion_mask)[0]
        if motion_ids.numel() > 0:
            inside_scope = self.motion_kinematic_scope_valid(
                motion_ids,
                self.skill_ids[motion_ids],
            )
            valid[motion_ids] = (
                self.motion_teacher_geometry_confidence[motion_ids]
                * inside_scope.to(dtype=valid.dtype)
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

    @property
    def locomotion_contact_termination_enabled(self) -> torch.Tensor:
        post_climb_grace = (
            self.composed_episode
            & (self.transition_stage == _POST_LOCOMOTION_STAGE)
            & (self.last_motion_skill_ids == CLIMB_SKILL_ID)
            & (self.transition_stage_elapsed < self.cfg.post_climb_contact_grace_time_s)
        )
        return ~post_climb_grace

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
    # Within direct atomic episodes, explicitly train complete execution from
    # frame zero while retaining random-phase recovery coverage.
    atomic_motion_start_at_beginning_fraction: float = 0.5
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
    # During a committed climb/down-roll, these resamples change only the
    # Actor-visible deployment request.  Frozen motion teachers remain
    # command-independent, teaching the Student to finish an irreversible
    # skill even if the hardware joystick keeps moving.
    locked_command_resampling_enabled: bool = True
    locked_command_resampling_time_range_s: tuple[float, float] = (1.0, 2.0)
    locomotion_teacher_lin_vel_x_range: tuple[float, float] = (-0.6, 1.0)
    locomotion_teacher_ang_vel_range: tuple[float, float] = (-1.57, 1.57)
    heading_control_stiffness: float = 0.5
    gait_air_ratios: tuple[float, float] = (0.38, 0.38)
    gait_phase_offsets: tuple[float, float] = (0.38, 0.88)
    gait_cycle: float = 0.85

    # Fraction of motion-family episodes that execute the continuous
    # locomotion -> expert -> locomotion route.  The remainder use the direct
    # atomic start mixture configured above.
    composed_episode_fraction: float = 0.5
    approach_distance_range: tuple[float, float] = (0.4, 1.0)
    approach_switch_distance: float = 0.08
    approach_lateral_tolerance: float = 0.20
    approach_maximum_overshoot: float = 0.20
    approach_timeout_s: float = 3.0
    transition_settle_time_range_s: tuple[float, float] = (0.2, 0.5)
    transition_maximum_settle_time_s: float = 1.0
    # The checked-in four-clip datasets have at most about 0.289 rad RMS
    # between a default-like locomotion stance and a motion boundary pose.
    # This leaves a small physical-policy margin without admitting an
    # unrelated posture.
    transition_maximum_joint_position_rms: float = 0.35
    transition_maximum_joint_speed_rms: float = 1.0
    transition_maximum_gravity_xy_norm: float = 0.35
    climb_maximum_heading_error: float = 0.35
    post_locomotion_time_range_s: tuple[float, float] = (0.8, 1.2)
    top_locomotion_time_range_s: tuple[float, float] = (2.0, 4.0)
    post_motion_command_release_time_s: float = 0.2
    post_climb_contact_grace_time_s: float = 0.2
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
