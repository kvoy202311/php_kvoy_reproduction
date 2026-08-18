from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from php_kvoy_reproduction.tasks.tracking.mdp.commands import MotionCommand
from php_kvoy_reproduction.tasks.tracking.mdp.motion_data import motion_clip_timeout_mask
from php_kvoy_reproduction.tasks.tracking.mdp.obstacle import get_climb_box_sizes, points_inside_oriented_box_xy
from php_kvoy_reproduction.tasks.tracking.mdp.platform_foot_support import (
    platform_foot_load_valid,
    platform_foot_support_state,
)
from php_kvoy_reproduction.tasks.tracking.mdp.rewards import _get_body_indexes


def _expert_reference_termination_active(command: MotionCommand) -> torch.Tensor:
    """Disable stale expert-error terminations after terminal q takeover."""

    latched = getattr(command, "terminal_default_pose_latched", None)
    if latched is None:
        return torch.ones(command.time_steps.shape, dtype=torch.bool, device=command.time_steps.device)
    if latched.shape != command.time_steps.shape:
        raise RuntimeError(
            "terminal_default_pose_latched must match command time_steps, "
            f"got {latched.shape} and {command.time_steps.shape}."
        )
    return ~latched.to(dtype=torch.bool)


def _terminal_default_pose_complete(command: MotionCommand) -> torch.Tensor:
    """Return terminal q completion, defaulting to true for generic commands."""

    complete = getattr(command, "terminal_default_pose_complete", None)
    if complete is None:
        return torch.ones(command.time_steps.shape, dtype=torch.bool, device=command.time_steps.device)
    if complete.shape != command.time_steps.shape:
        raise RuntimeError(
            "terminal_default_pose_complete must match command time_steps, "
            f"got {complete.shape} and {command.time_steps.shape}."
        )
    return complete.to(dtype=torch.bool)


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _expert_reference_termination_active(command) & (
        torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold
    )


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _expert_reference_termination_active(command) & (
        torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    )


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return _expert_reference_termination_active(command) & (
        (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
    )


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return _expert_reference_termination_active(command) & torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return _expert_reference_termination_active(command) & torch.any(error > threshold, dim=-1)


def motion_clip_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Request an episode boundary after the final frame of a motion clip.

    Configure this term with ``time_out=True``.  The clip boundary is an
    external data boundary rather than a physical failure, so PPO should cut
    the rollout while retaining value bootstrapping.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.cfg.terminate_on_motion_end:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Physical-failure terms must be configured before this timeout term.  The
    # masks are kept mutually exclusive because RSL-RL bootstraps every timeout,
    # including a timeout that coincides with a true termination.
    return motion_clip_timeout_mask(command.motion_finished, env.termination_manager.terminated)


def _expert_joint_position_rms(command: MotionCommand) -> torch.Tensor:
    """Return all-joint RMS error to the immutable source pose."""

    robot_joint_pos = getattr(command, "robot_joint_pos", None)
    if robot_joint_pos is None:
        reference = command.robot_joint_vel
        return torch.zeros(reference.shape[0], dtype=reference.dtype, device=reference.device)
    target_joint_pos = getattr(command, "source_joint_pos", None)
    if target_joint_pos is None:
        target_joint_pos = getattr(command, "joint_pos", None)
    if target_joint_pos is None:
        # Generic test doubles and non-tracking callers may not expose an
        # expert command.  Keep this optional climb-only condition neutral for
        # those callers; production MotionCommand always exposes source q.
        return torch.zeros(robot_joint_pos.shape[0], dtype=robot_joint_pos.dtype, device=robot_joint_pos.device)
    if target_joint_pos.shape != robot_joint_pos.shape:
        raise RuntimeError(
            "Expert and robot joint-position tensors must have the same shape, "
            f"got {target_joint_pos.shape} and {robot_joint_pos.shape}."
        )
    return torch.sqrt(torch.mean(torch.square(robot_joint_pos - target_joint_pos), dim=1))


def _climb_standing_conditions(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_range: tuple[float, float],
    min_foot_contact_force: float,
    min_foot_contact_time: float,
    max_root_height_error: float,
    max_root_linear_speed: float,
    max_root_angular_speed: float,
    max_joint_speed: float,
    max_torso_tilt: float,
    platform_support_params: Mapping[str, object] | None = None,
    sole_height_tolerance: float | None = None,
    min_total_load_fraction: float = 0.0,
    max_default_joint_pos_rms: float | None = None,
    max_expert_joint_pos_rms: float | None = None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not foot_body_names:
        raise ValueError("foot_body_names must contain at least one body.")
    if footprint_inset < 0.0:
        raise ValueError(f"footprint_inset must be non-negative, got {footprint_inset}.")
    if foot_height_range[0] > foot_height_range[1]:
        raise ValueError(f"foot_height_range must be ordered, got {foot_height_range}.")
    for name, value in (
        ("min_foot_contact_force", min_foot_contact_force),
        ("min_foot_contact_time", min_foot_contact_time),
        ("max_root_height_error", max_root_height_error),
        ("max_root_linear_speed", max_root_linear_speed),
        ("max_root_angular_speed", max_root_angular_speed),
        ("max_joint_speed", max_joint_speed),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    if not 0.0 <= max_torso_tilt < math.pi:
        raise ValueError(f"max_torso_tilt must lie in [0, pi), got {max_torso_tilt}.")
    if not 0.0 <= min_total_load_fraction <= 1.0:
        raise ValueError(
            "min_total_load_fraction must lie in [0, 1], "
            f"got {min_total_load_fraction}."
        )
    if max_default_joint_pos_rms is not None and max_default_joint_pos_rms <= 0.0:
        raise ValueError(
            "max_default_joint_pos_rms must be positive when provided, "
            f"got {max_default_joint_pos_rms}."
        )
    if max_expert_joint_pos_rms is not None and max_expert_joint_pos_rms <= 0.0:
        raise ValueError(
            "max_expert_joint_pos_rms must be positive when provided, "
            f"got {max_expert_joint_pos_rms}."
        )

    # The strict climb path deliberately uses the two filtered platform
    # sensors rather than aggregate robot contact.  The latter includes ground,
    # walls and wrists, so it cannot establish that a foot is actually bearing
    # on the box top.  The contact-time buffer is command-owned and is read
    # only here: reward/termination evaluations must never advance it.
    if platform_support_params is not None:
        if sole_height_tolerance is None or sole_height_tolerance <= 0.0:
            raise ValueError(
                "Strict climb standing conditions require a positive sole_height_tolerance."
            )
        support = platform_foot_support_state(
            env,
            command.robot,
            command.device,
            platform_cfg,
            base_size,
            platform_support_params,
            min_upward_force=min_foot_contact_force,
            sole_height_tolerance=sole_height_tolerance,
        )
        if tuple(foot_body_names) != support.settings.foot_body_names:
            raise ValueError(
                "Strict climb standing conditions require foot_body_names to exactly match "
                "platform_support_params['foot_body_names']."
            )
        filtered_contact_time = getattr(command, "platform_foot_filtered_contact_time", None)
        if filtered_contact_time is None:
            raise RuntimeError(
                "Strict climb standing conditions require command-owned filtered platform-foot contact time; "
                "it must be advanced by MotionCommand update, not by terminations."
            )
        if filtered_contact_time.shape != support.active_support.shape:
            raise RuntimeError(
                "platform_foot_filtered_contact_time must match strict platform-foot support shape, "
                f"got {filtered_contact_time.shape} and {support.active_support.shape}."
            )

        # ``sole_geometry_valid`` enforces the actual sole footprint, including
        # the user-approved maximum 5 cm heel overhang.  In particular, this is
        # not an ankle-origin-in-box approximation.
        inside = support.sole_geometry_valid
        height_valid = support.sole_plane_height_error.abs() <= sole_height_tolerance
        contact_valid = support.active_support & (filtered_contact_time >= min_foot_contact_time)
        load_valid = platform_foot_load_valid(
            support,
            command.robot,
            min_total_load_fraction=min_total_load_fraction,
        )
    else:
        # Generic/backward-compatible path.  Existing non-climb tasks may not
        # expose the per-foot filtered sensors or command-owned contact timer.
        platform: RigidObject = env.scene[platform_cfg.name]
        sizes = get_climb_box_sizes(platform, base_size=base_size, device=platform.device)
        footprint_sizes = sizes.clone()
        footprint_sizes[:, :2] -= 2.0 * footprint_inset
        if torch.any(footprint_sizes[:, :2] <= 0.0):
            raise ValueError("footprint_inset leaves a non-positive platform footprint.")
        foot_body_ids = torch.tensor(
            [command.robot.body_names.index(name) for name in foot_body_names],
            dtype=torch.long,
            device=command.device,
        )
        foot_positions = command.robot.data.body_pos_w[:, foot_body_ids]
        inside = points_inside_oriented_box_xy(
            foot_positions,
            platform.data.root_pos_w,
            platform.data.root_quat_w,
            footprint_sizes,
        )
        platform_top = platform.data.root_pos_w[:, None, 2] + 0.5 * sizes[:, None, 2]
        relative_height = foot_positions[..., 2] - platform_top
        height_valid = (relative_height >= foot_height_range[0]) & (relative_height <= foot_height_range[1])

        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        if contact_sensor.data.net_forces_w is None or contact_sensor.data.current_contact_time is None:
            raise RuntimeError("The climb contact sensor must provide net forces and current contact time.")
        foot_contact_forces = torch.linalg.vector_norm(
            contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids],
            dim=-1,
        )
        foot_contact_times = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids]
        contact_valid = (foot_contact_forces >= min_foot_contact_force) & (
            foot_contact_times >= min_foot_contact_time
        )
        load_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

    projected_gravity_b = math_utils.quat_apply_inverse(
        command.robot_anchor_quat_w,
        command.robot.data.GRAVITY_VEC_W,
    )
    upright = projected_gravity_b[:, 2] <= -math.cos(max_torso_tilt)
    terminal_anchor_pos_w = getattr(command, "terminal_default_anchor_pos_w", None)
    if terminal_anchor_pos_w is None:
        target_anchor_pos_w = command.anchor_pos_w
    else:
        if terminal_anchor_pos_w.shape != command.anchor_pos_w.shape:
            raise RuntimeError(
                "terminal_default_anchor_pos_w must match anchor_pos_w, "
                f"got {terminal_anchor_pos_w.shape} and {command.anchor_pos_w.shape}."
            )
        target_anchor_pos_w = terminal_anchor_pos_w
    root_height_valid = (
        torch.abs(target_anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]) <= max_root_height_error
    )
    root_linear_speed_valid = (
        torch.linalg.vector_norm(command.robot_anchor_lin_vel_w, dim=-1) <= max_root_linear_speed
    )
    root_angular_speed_valid = (
        torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=-1) <= max_root_angular_speed
    )
    joint_speed_valid = torch.max(torch.abs(command.robot_joint_vel), dim=1).values <= max_joint_speed
    default_joint_pos_rms = torch.sqrt(
        torch.mean(torch.square(command.robot_joint_pos - command.robot.data.default_joint_pos), dim=1)
    )
    if max_default_joint_pos_rms is None:
        default_joint_pos_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    else:
        default_joint_pos_valid = default_joint_pos_rms <= max_default_joint_pos_rms
    conditions = {
        "feet_inside": torch.all(inside, dim=1),
        "foot_height_valid": torch.all(height_valid, dim=1),
        "foot_contact_valid": torch.all(contact_valid, dim=1),
        "foot_load_valid": load_valid,
        "upright": upright,
        "root_height_valid": root_height_valid,
        "root_linear_speed_valid": root_linear_speed_valid,
        "root_angular_speed_valid": root_angular_speed_valid,
        "joint_speed_valid": joint_speed_valid,
        "default_joint_pos_valid": default_joint_pos_valid,
    }
    if max_expert_joint_pos_rms is not None:
        conditions["expert_joint_pos_valid"] = (
            _expert_joint_position_rms(command) <= max_expert_joint_pos_rms
        )
    return conditions, default_joint_pos_rms


def _terminal_platform_alignment_complete(command: MotionCommand) -> torch.Tensor:
    """Return the optional final-reference alignment state, defaulting to true."""

    completed = getattr(command, "terminal_platform_alignment_complete", None)
    if completed is None:
        return torch.ones_like(command.time_steps, dtype=torch.bool)
    if completed.shape != command.time_steps.shape:
        raise RuntimeError(
            "terminal_platform_alignment_complete must match command time_steps, "
            f"got {completed.shape} and {command.time_steps.shape}."
        )
    return completed.to(dtype=torch.bool)


class motion_end_success(ManagerTermBase):
    """Classify a clip as successful only after a continuous stable final stand.

    Success requires real bilateral platform load, quiet motion and a broad
    all-joint RMS bound around the immutable final expert pose.  The bound does
    not require exact imitation; it rejects only terminal configurations such
    as crossed legs or strongly folded/raised arms that otherwise satisfy the
    pose-independent stability checks.
    """

    _METRIC_PREFIX = "final_standing_"
    _JOINT_GROUP_TOKENS = {
        "arm": ("shoulder", "elbow", "wrist"),
        "waist": ("waist",),
        "leg": ("hip", "knee", "ankle"),
    }
    _WRIST_BODY_NAMES = ("l_wrist_z_link", "r_wrist_z_link")

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._longest_stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._required_stable_steps = max(1, math.ceil(float(cfg.params["min_stable_time"]) / env.step_dt - 1.0e-9))

        command: MotionCommand = env.command_manager.get_term(cfg.params["command_name"])
        self._expert_joint_pos_gate_enabled = cfg.params.get("max_expert_joint_pos_rms") is not None
        if self._expert_joint_pos_gate_enabled and getattr(command.cfg, "terminal_default_pose_enabled", False):
            raise ValueError(
                "motion_end_success cannot combine max_expert_joint_pos_rms with "
                "terminal_default_pose_enabled: terminal default-q handoff replaces the source expert q, "
                "so a final source-q RMS success gate would be contradictory. Disable one of these options."
            )
        self._joint_group_ids = {
            group_name: torch.tensor(
                [
                    joint_id
                    for joint_id, joint_name in enumerate(command.robot.joint_names)
                    if any(token in joint_name for token in name_tokens)
                ],
                dtype=torch.long,
                device=env.device,
            )
            for group_name, name_tokens in self._JOINT_GROUP_TOKENS.items()
        }
        empty_groups = [name for name, ids in self._joint_group_ids.items() if ids.numel() == 0]
        if empty_groups:
            raise RuntimeError(f"No robot joints found for diagnostic groups: {empty_groups}.")

        contact_sensor: ContactSensor = env.scene.sensors[cfg.params["contact_sensor_cfg"].name]
        wrist_body_ids, wrist_body_names = contact_sensor.find_bodies(
            list(self._WRIST_BODY_NAMES), preserve_order=True
        )
        if tuple(wrist_body_names) != self._WRIST_BODY_NAMES:
            raise RuntimeError(
                "The contact sensor must expose wrist bodies in the requested order; "
                f"expected {self._WRIST_BODY_NAMES}, got {tuple(wrist_body_names)}."
            )
        self._wrist_contact_body_ids = wrist_body_ids

        metric_names = (
            "feet_inside",
            "foot_height_valid",
            "foot_contact_valid",
            "foot_load_valid",
            "upright",
            "root_height_valid",
            "root_linear_speed_valid",
            "root_angular_speed_valid",
            "joint_speed_valid",
            "default_joint_pos_valid",
            "terminal_alignment_complete",
            "terminal_default_pose_complete",
            "final_frame_fraction",
            "max_joint_speed",
            "joint_speed_rms",
            "joints_over_0_5",
            "joints_over_0_75",
            "joints_over_1_0",
            "joints_over_2_0",
            "root_angular_speed",
            "arm_max_joint_speed",
            "waist_max_joint_speed",
            "leg_max_joint_speed",
            "left_wrist_contact_force",
            "right_wrist_contact_force",
            "left_wrist_contact_time",
            "right_wrist_contact_time",
            "default_joint_pos_rms",
            "stable_time",
            "longest_stable_time",
            "random_phase_allowed",
        )
        if self._expert_joint_pos_gate_enabled:
            metric_names += ("expert_joint_pos_valid", "expert_joint_pos_rms")
        for name in metric_names:
            command.metrics.setdefault(self._METRIC_PREFIX + name, torch.zeros(env.num_envs, device=env.device))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stable_steps[env_ids] = 0
        self._longest_stable_steps[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        platform_cfg: SceneEntityCfg,
        contact_sensor_cfg: SceneEntityCfg,
        base_size: tuple[float, float, float],
        foot_body_names: list[str],
        footprint_inset: float,
        foot_height_range: tuple[float, float],
        min_foot_contact_force: float,
        min_foot_contact_time: float,
        max_root_height_error: float,
        max_root_linear_speed: float,
        max_root_angular_speed: float,
        max_joint_speed: float,
        max_torso_tilt: float,
        min_stable_time: float,
        platform_support_params: Mapping[str, object] | None = None,
        sole_height_tolerance: float | None = None,
        min_total_load_fraction: float = 0.0,
        max_default_joint_pos_rms: float | None = None,
        max_expert_joint_pos_rms: float | None = None,
    ) -> torch.Tensor:
        if min_stable_time <= 0.0:
            raise ValueError(f"min_stable_time must be positive, got {min_stable_time}.")

        command: MotionCommand = env.command_manager.get_term(command_name)
        if not command.cfg.terminate_on_motion_end:
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        conditions, default_joint_pos_rms = _climb_standing_conditions(
            env,
            command,
            platform_cfg,
            contact_sensor_cfg,
            base_size,
            foot_body_names,
            footprint_inset,
            foot_height_range,
            min_foot_contact_force,
            min_foot_contact_time,
            max_root_height_error,
            max_root_linear_speed,
            max_root_angular_speed,
            max_joint_speed,
            max_torso_tilt,
            platform_support_params,
            sole_height_tolerance,
            min_total_load_fraction,
            max_default_joint_pos_rms,
            max_expert_joint_pos_rms,
        )
        expert_joint_pos_rms = (
            _expert_joint_position_rms(command) if self._expert_joint_pos_gate_enabled else None
        )
        final_frames = command.motion.motion_end_idx[command.motion_ids] - 1
        at_final_frame = command.time_steps >= final_frames
        alignment_complete = _terminal_platform_alignment_complete(command)
        terminal_default_pose_complete = _terminal_default_pose_complete(command)
        # The q target intentionally moves during the default-pose transition;
        # its samples cannot count toward the required quiet final stand.
        eligible_final_frame = at_final_frame & alignment_complete & terminal_default_pose_complete
        standing_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for value in conditions.values():
            standing_valid &= value
        valid_final_stand = eligible_final_frame & standing_valid
        self._stable_steps = torch.where(valid_final_stand, self._stable_steps + 1, 0)
        self._longest_stable_steps = torch.maximum(self._longest_stable_steps, self._stable_steps)

        absolute_joint_speed = torch.abs(command.robot_joint_vel)
        max_joint_speed_value = torch.max(absolute_joint_speed, dim=1).values
        joint_speed_rms = torch.sqrt(torch.mean(torch.square(command.robot_joint_vel), dim=1))
        root_angular_speed = torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=1)

        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        if contact_sensor.data.net_forces_w is None or contact_sensor.data.current_contact_time is None:
            raise RuntimeError("The climb contact sensor must provide net forces and current contact time.")
        wrist_contact_forces = torch.linalg.vector_norm(
            contact_sensor.data.net_forces_w[:, self._wrist_contact_body_ids], dim=-1
        )
        wrist_contact_times = contact_sensor.data.current_contact_time[:, self._wrist_contact_body_ids]

        for name, value in conditions.items():
            command.metrics[self._METRIC_PREFIX + name].copy_((eligible_final_frame & value).float())
        command.metrics[self._METRIC_PREFIX + "terminal_alignment_complete"].copy_(
            (at_final_frame & alignment_complete).float()
        )
        command.metrics[self._METRIC_PREFIX + "terminal_default_pose_complete"].copy_(
            (at_final_frame & terminal_default_pose_complete).float()
        )
        command.metrics[self._METRIC_PREFIX + "default_joint_pos_rms"].copy_(
            torch.where(eligible_final_frame, default_joint_pos_rms, torch.zeros_like(default_joint_pos_rms))
        )
        if expert_joint_pos_rms is not None:
            command.metrics[self._METRIC_PREFIX + "expert_joint_pos_rms"].copy_(
                torch.where(eligible_final_frame, expert_joint_pos_rms, torch.zeros_like(expert_joint_pos_rms))
            )
        command.metrics[self._METRIC_PREFIX + "stable_time"].copy_(self._stable_steps.float() * env.step_dt)
        final_float = eligible_final_frame.to(dtype=max_joint_speed_value.dtype)
        command.metrics[self._METRIC_PREFIX + "final_frame_fraction"].copy_(final_float)
        command.metrics[self._METRIC_PREFIX + "max_joint_speed"].copy_(final_float * max_joint_speed_value)
        command.metrics[self._METRIC_PREFIX + "joint_speed_rms"].copy_(final_float * joint_speed_rms)
        for threshold_name, threshold in (
            ("0_5", 0.5),
            ("0_75", 0.75),
            ("1_0", 1.0),
            ("2_0", 2.0),
        ):
            joints_over_threshold = torch.count_nonzero(absolute_joint_speed > threshold, dim=1)
            command.metrics[self._METRIC_PREFIX + f"joints_over_{threshold_name}"].copy_(
                final_float * joints_over_threshold.to(dtype=final_float.dtype)
            )
        command.metrics[self._METRIC_PREFIX + "root_angular_speed"].copy_(
            final_float * root_angular_speed
        )
        for group_name, joint_ids in self._joint_group_ids.items():
            group_max_speed = torch.max(absolute_joint_speed[:, joint_ids], dim=1).values
            command.metrics[self._METRIC_PREFIX + f"{group_name}_max_joint_speed"].copy_(
                final_float * group_max_speed
            )
        for wrist_index, side in enumerate(("left", "right")):
            command.metrics[self._METRIC_PREFIX + f"{side}_wrist_contact_force"].copy_(
                final_float * wrist_contact_forces[:, wrist_index]
            )
            command.metrics[self._METRIC_PREFIX + f"{side}_wrist_contact_time"].copy_(
                final_float * wrist_contact_times[:, wrist_index]
            )
        command.metrics[self._METRIC_PREFIX + "longest_stable_time"].copy_(
            self._longest_stable_steps.float() * env.step_dt
        )
        platform = env.scene[platform_cfg.name]
        random_phase_mask = getattr(platform, "_climb_box_random_phase_env_mask", None)
        if random_phase_mask is None:
            # Old externally configured environments may still expose only the
            # historical alias.  Its semantics were always the safe
            # random-phase subset, even when its name implied geometry.
            random_phase_mask = getattr(platform, "_climb_box_nominal_geometry_mask", None)
        if random_phase_mask is None:
            raise RuntimeError("Platform does not expose the random-phase environment diagnostic mask.")
        command.metrics[self._METRIC_PREFIX + "random_phase_allowed"].copy_(
            random_phase_mask.to(device=env.device, dtype=max_joint_speed_value.dtype)
        )

        continuously_stable = self._stable_steps >= self._required_stable_steps
        clip_timeout = motion_clip_timeout_mask(command.motion_finished, env.termination_manager.terminated)
        return clip_timeout & continuously_stable


def motion_end_failure(
    env: ManagerBasedRLEnv,
    command_name: str,
    success_term_name: str,
) -> torch.Tensor:
    """Classify a completed clip as a platform-standing failure."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.cfg.terminate_on_motion_end:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    clip_timeout = motion_clip_timeout_mask(command.motion_finished, env.termination_manager.terminated)
    # The success term is configured immediately before this term.  Reusing
    # its result guarantees that completed clips are partitioned exactly once.
    successful = env.termination_manager.get_term(success_term_name)
    return clip_timeout & ~successful
