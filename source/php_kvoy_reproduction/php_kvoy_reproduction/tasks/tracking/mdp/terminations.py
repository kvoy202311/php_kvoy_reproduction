from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_error_magnitude

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

from php_kvoy_reproduction.tasks.tracking.mdp.commands import MotionCommand
from php_kvoy_reproduction.tasks.tracking.mdp.motion_data import motion_clip_boundary_mask
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


def motion_clip_end(
    env: ManagerBasedRLEnv,
    command_name: str,
    classified_term_names: Sequence[str] = (),
) -> torch.Tensor:
    """Request a truncated episode boundary after a motion clip's final frame.

    Configure normal clip-completion terms with ``time_out=True``.  The source
    ends in a repeated stationary reference, so truncation preserves its
    continuing value and prevents an artificial zero-value terminal cliff.
    Physical tracking failures remain separate true terminations.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.cfg.terminate_on_motion_end:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Physical failures and the more specific success/failure classifiers must
    # be configured first.  Timeout classifiers do not enter the manager's
    # ``terminated`` tensor, so include their raw term masks explicitly to
    # keep every boundary label mutually exclusive.
    claimed = env.termination_manager.terminated.clone()
    for term_name in classified_term_names:
        term_value = env.termination_manager.get_term(term_name)
        if term_value.shape != claimed.shape or term_value.dtype != torch.bool:
            raise RuntimeError(
                f"Classified clip-boundary term {term_name!r} must be a bool tensor with shape "
                f"{claimed.shape}, got {term_value.shape} and {term_value.dtype}."
            )
        claimed |= term_value
    return motion_clip_boundary_mask(command.motion_finished, claimed)


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
        height_valid = support.sole_surface_valid
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


def _episode_started_at_motion_beginning(command: MotionCommand) -> torch.Tensor:
    """Return which episodes contain the complete authored motion.

    Random-phase episodes remain useful for local policy learning, but a clip
    that starts inside the terminal window may not have enough elapsed time to
    satisfy the contact and contiguous-stability durations.  Such a partial
    trial must therefore be neither a terminal-quality success nor a
    terminal-quality failure; the generic clip-boundary terminal still ends it.
    """

    started = getattr(command, "episode_started_at_motion_beginning", None)
    if not isinstance(started, torch.Tensor):
        raise RuntimeError(
            "Terminal quality classification requires "
            "MotionCommand.episode_started_at_motion_beginning to be a tensor."
        )
    if started.shape != command.time_steps.shape:
        raise RuntimeError(
            "episode_started_at_motion_beginning must match command time_steps, "
            f"got {started.shape} and {command.time_steps.shape}."
        )
    if started.device != command.time_steps.device:
        raise RuntimeError(
            "episode_started_at_motion_beginning and command time_steps must share a device, "
            f"got {started.device} and {command.time_steps.device}."
        )
    return started.to(dtype=torch.bool)


def _resolve_terminal_quality_joint_groups(
    command: MotionCommand,
    joint_groups: Mapping[str, Sequence[str]],
    group_pose_rms_thresholds: Mapping[str, float],
    group_pose_max_thresholds: Mapping[str, float],
    group_velocity_rms_thresholds: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    """Validate and resolve a complete, disjoint terminal joint partition."""

    if not isinstance(joint_groups, Mapping) or not joint_groups:
        raise ValueError("joint_groups must be a non-empty mapping of group names to joint names.")
    group_names = tuple(joint_groups)
    threshold_mappings = (
        ("group_pose_rms_thresholds", group_pose_rms_thresholds),
        ("group_pose_max_thresholds", group_pose_max_thresholds),
        ("group_velocity_rms_thresholds", group_velocity_rms_thresholds),
    )
    for mapping_name, thresholds in threshold_mappings:
        if not isinstance(thresholds, Mapping) or set(thresholds) != set(group_names):
            raise ValueError(f"{mapping_name} keys must exactly match joint_groups keys.")
        invalid = {
            name: thresholds[name]
            for name in group_names
            if not math.isfinite(float(thresholds[name])) or float(thresholds[name]) <= 0.0
        }
        if invalid:
            raise ValueError(f"Every {mapping_name} value must be positive; got {invalid}.")

    robot_joint_names = tuple(command.robot.joint_names)
    name_to_id = {name: joint_id for joint_id, name in enumerate(robot_joint_names)}
    if len(name_to_id) != len(robot_joint_names):
        raise RuntimeError("Robot joint names must be unique for terminal quality checks.")

    resolved: dict[str, torch.Tensor] = {}
    assigned_names: set[str] = set()
    for group_name, names_value in joint_groups.items():
        if isinstance(names_value, (str, bytes)):
            raise ValueError(
                f"Terminal joint group {group_name!r} must be a sequence of joint names, not a string."
            )
        names = tuple(names_value)
        if not names:
            raise ValueError(f"Terminal joint group {group_name!r} must not be empty.")
        local_duplicates = sorted({name for name in names if names.count(name) > 1})
        if local_duplicates:
            raise ValueError(
                f"Terminal joint group {group_name!r} contains repeated joints: {local_duplicates}."
            )
        repeated = sorted(assigned_names.intersection(names))
        if repeated:
            raise ValueError(f"Terminal joint groups must be disjoint; repeated joints: {repeated}.")
        unknown = [name for name in names if name not in name_to_id]
        if unknown:
            raise RuntimeError(
                f"Terminal joint group {group_name!r} contains unknown robot joints: {unknown}."
            )
        assigned_names.update(names)
        resolved[group_name] = torch.tensor(
            [name_to_id[name] for name in names],
            dtype=torch.long,
            device=command.robot_joint_vel.device,
        )

    missing = sorted(set(robot_joint_names).difference(assigned_names))
    if missing:
        raise ValueError(
            "joint_groups must cover every robot joint exactly once; "
            f"unassigned joints: {missing}."
        )
    return resolved


class motion_end_success(ManagerTermBase):
    """Classify a clip only after a contiguous, high-quality expert tail.

    No post-clip hold is required.  Quality samples accumulate inside the
    authored static tail, while the episode still ends immediately at the
    source boundary.  A bad sample clears the counter, so a robot that was
    briefly stable and then departs from the expert pose cannot be accepted.
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
        required_params = (
            "command_name",
            "contact_sensor_cfg",
            "min_stable_time",
            "static_window_time_s",
            "reference_max_joint_speed",
            "joint_groups",
            "group_pose_rms_thresholds",
            "group_pose_max_thresholds",
            "group_velocity_rms_thresholds",
            "max_torso_orientation_error",
        )
        missing_params = [name for name in required_params if name not in cfg.params]
        if missing_params:
            raise ValueError(
                "motion_end_success is missing required quality parameters: "
                f"{missing_params}."
            )
        if env.step_dt <= 0.0:
            raise ValueError(f"env.step_dt must be positive, got {env.step_dt}.")
        min_stable_time = float(cfg.params["min_stable_time"])
        static_window_time_s = float(cfg.params["static_window_time_s"])
        reference_max_joint_speed = float(cfg.params["reference_max_joint_speed"])
        max_torso_orientation_error = float(cfg.params["max_torso_orientation_error"])
        if not math.isfinite(min_stable_time) or min_stable_time <= 0.0:
            raise ValueError(f"min_stable_time must be positive, got {min_stable_time}.")
        if not math.isfinite(static_window_time_s) or static_window_time_s <= 0.0:
            raise ValueError(f"static_window_time_s must be positive, got {static_window_time_s}.")
        if not math.isfinite(reference_max_joint_speed) or reference_max_joint_speed < 0.0:
            raise ValueError(
                "reference_max_joint_speed must be non-negative, "
                f"got {reference_max_joint_speed}."
            )
        if not math.isfinite(max_torso_orientation_error) or not 0.0 <= max_torso_orientation_error <= math.pi:
            raise ValueError(
                "max_torso_orientation_error must lie in [0, pi], "
                f"got {max_torso_orientation_error}."
            )

        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._longest_stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._required_stable_steps = max(1, math.ceil(min_stable_time / env.step_dt - 1.0e-9))
        self._static_window_steps = max(1, math.ceil(static_window_time_s / env.step_dt - 1.0e-9))
        if self._required_stable_steps > self._static_window_steps:
            raise ValueError(
                "min_stable_time cannot exceed the available static_window_time_s when "
                "motion_end_hold_time_s is zero; "
                f"got {min_stable_time} s and {static_window_time_s} s."
            )

        command: MotionCommand = env.command_manager.get_term(cfg.params["command_name"])
        motion_end_hold_time_s = getattr(command.cfg, "motion_end_hold_time_s", None)
        if motion_end_hold_time_s is not None and float(motion_end_hold_time_s) != 0.0:
            raise ValueError(
                "Tail-window motion_end_success requires motion_end_hold_time_s=0; "
                f"got {motion_end_hold_time_s}."
            )
        if getattr(command.cfg, "terminal_default_pose_enabled", False):
            raise ValueError(
                "Tail-window motion_end_success tracks the immutable source expert pose and cannot "
                "be combined with terminal_default_pose_enabled."
            )
        self._expert_joint_pos_gate_enabled = cfg.params.get("max_expert_joint_pos_rms") is not None
        self._quality_joint_group_ids = _resolve_terminal_quality_joint_groups(
            command,
            cfg.params["joint_groups"],
            cfg.params["group_pose_rms_thresholds"],
            cfg.params["group_pose_max_thresholds"],
            cfg.params["group_velocity_rms_thresholds"],
        )
        exempt_groups = tuple(cfg.params.get("expert_pose_exempt_groups", ()))
        unknown_exempt_groups = sorted(set(exempt_groups).difference(self._quality_joint_group_ids))
        if unknown_exempt_groups:
            raise ValueError(
                "expert_pose_exempt_groups contains names absent from joint_groups: "
                f"{unknown_exempt_groups}."
            )
        if len(set(exempt_groups)) != len(exempt_groups):
            raise ValueError("expert_pose_exempt_groups must not contain duplicate names.")
        self._expert_pose_exempt_groups = frozenset(exempt_groups)
        self._diagnostic_joint_group_ids = {
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
        empty_groups = [name for name, ids in self._diagnostic_joint_group_ids.items() if ids.numel() == 0]
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
            "static_tail",
            "reference_static",
            "final_frame_fraction",
            "max_joint_speed",
            "joint_speed_rms",
            "joints_over_0_5",
            "joints_over_0_75",
            "joints_over_1_0",
            "joints_over_2_0",
            "root_linear_speed",
            "root_angular_speed",
            "torso_orientation_valid",
            "torso_orientation_error",
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
        for group_name in self._quality_joint_group_ids:
            metric_names += (
                f"{group_name}_pose_rms_valid",
                f"{group_name}_pose_max_valid",
                f"{group_name}_velocity_rms_valid",
                f"{group_name}_pose_rms",
                f"{group_name}_max_pose_error",
                f"{group_name}_joint_speed_rms",
            )
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
        static_window_time_s: float,
        reference_max_joint_speed: float,
        joint_groups: Mapping[str, Sequence[str]],
        group_pose_rms_thresholds: Mapping[str, float],
        group_pose_max_thresholds: Mapping[str, float],
        group_velocity_rms_thresholds: Mapping[str, float],
        max_torso_orientation_error: float,
        expert_pose_exempt_groups: Sequence[str] = (),
        platform_support_params: Mapping[str, object] | None = None,
        sole_height_tolerance: float | None = None,
        min_total_load_fraction: float = 0.0,
        max_default_joint_pos_rms: float | None = None,
        max_expert_joint_pos_rms: float | None = None,
    ) -> torch.Tensor:
        if not math.isfinite(min_stable_time) or min_stable_time <= 0.0:
            raise ValueError(f"min_stable_time must be positive, got {min_stable_time}.")
        if not math.isfinite(static_window_time_s) or static_window_time_s <= 0.0:
            raise ValueError(f"static_window_time_s must be positive, got {static_window_time_s}.")
        if not math.isfinite(reference_max_joint_speed) or reference_max_joint_speed < 0.0:
            raise ValueError(
                "reference_max_joint_speed must be non-negative, "
                f"got {reference_max_joint_speed}."
            )
        if not math.isfinite(max_torso_orientation_error) or not 0.0 <= max_torso_orientation_error <= math.pi:
            raise ValueError(
                "max_torso_orientation_error must lie in [0, pi], "
                f"got {max_torso_orientation_error}."
            )
        runtime_required_steps = max(1, math.ceil(min_stable_time / env.step_dt - 1.0e-9))
        runtime_window_steps = max(1, math.ceil(static_window_time_s / env.step_dt - 1.0e-9))
        if runtime_required_steps != self._required_stable_steps or runtime_window_steps != self._static_window_steps:
            raise ValueError(
                "motion_end_success duration parameters changed after construction; "
                "recreate the environment after changing min_stable_time or static_window_time_s."
            )
        if tuple(joint_groups) != tuple(self._quality_joint_group_ids):
            raise ValueError(
                "motion_end_success joint_groups changed after construction; recreate the environment."
            )
        runtime_exempt_groups = tuple(expert_pose_exempt_groups)
        if len(set(runtime_exempt_groups)) != len(runtime_exempt_groups):
            raise ValueError("expert_pose_exempt_groups must not contain duplicate names.")
        if frozenset(runtime_exempt_groups) != self._expert_pose_exempt_groups:
            raise ValueError(
                "motion_end_success expert_pose_exempt_groups changed after construction; "
                "recreate the environment."
            )

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

        robot_joint_pos = command.robot_joint_pos
        robot_joint_vel = command.robot_joint_vel
        source_joint_pos = getattr(command, "source_joint_pos", None)
        if source_joint_pos is None:
            raise RuntimeError("motion_end_success requires the immutable source_joint_pos tensor.")
        if source_joint_pos.shape != robot_joint_pos.shape or robot_joint_vel.shape != robot_joint_pos.shape:
            raise RuntimeError(
                "Source position, robot position, and robot velocity tensors must have the same shape, "
                f"got {source_joint_pos.shape}, {robot_joint_pos.shape}, and {robot_joint_vel.shape}."
            )
        if hasattr(command.motion, "joint_vel"):
            source_joint_vel = command.motion.joint_vel[command.time_steps]
        else:
            source_joint_vel = command.joint_vel
        if source_joint_vel.shape != robot_joint_vel.shape:
            raise RuntimeError(
                "Source and robot joint-velocity tensors must have the same shape, "
                f"got {source_joint_vel.shape} and {robot_joint_vel.shape}."
            )

        final_frames = command.motion.motion_end_idx[command.motion_ids] - 1
        static_window_start = final_frames - (self._static_window_steps - 1)
        in_static_tail = (command.time_steps >= static_window_start) & (command.time_steps <= final_frames)
        at_final_frame = command.time_steps >= final_frames
        alignment_complete = _terminal_platform_alignment_complete(command)
        terminal_default_pose_complete = _terminal_default_pose_complete(command)
        tail_observation = in_static_tail & alignment_complete & terminal_default_pose_complete
        reference_speed = torch.max(torch.abs(source_joint_vel), dim=1).values
        reference_static = reference_speed <= reference_max_joint_speed

        joint_position_error = robot_joint_pos - source_joint_pos
        group_pose_rms: dict[str, torch.Tensor] = {}
        group_pose_max: dict[str, torch.Tensor] = {}
        group_velocity_rms: dict[str, torch.Tensor] = {}
        group_pose_rms_valid: dict[str, torch.Tensor] = {}
        group_pose_max_valid: dict[str, torch.Tensor] = {}
        group_velocity_rms_valid: dict[str, torch.Tensor] = {}
        grouped_quality_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for group_name, joint_ids in self._quality_joint_group_ids.items():
            group_error = joint_position_error[:, joint_ids]
            group_velocity = robot_joint_vel[:, joint_ids]
            pose_rms = torch.sqrt(torch.mean(torch.square(group_error), dim=1))
            pose_max = torch.max(torch.abs(group_error), dim=1).values
            velocity_rms = torch.sqrt(torch.mean(torch.square(group_velocity), dim=1))
            expert_pose_required = group_name not in self._expert_pose_exempt_groups
            pose_rms_valid = (
                pose_rms <= float(group_pose_rms_thresholds[group_name])
                if expert_pose_required
                else torch.ones_like(pose_rms, dtype=torch.bool)
            )
            pose_max_valid = (
                pose_max <= float(group_pose_max_thresholds[group_name])
                if expert_pose_required
                else torch.ones_like(pose_max, dtype=torch.bool)
            )
            velocity_rms_valid = velocity_rms <= float(group_velocity_rms_thresholds[group_name])
            group_pose_rms[group_name] = pose_rms
            group_pose_max[group_name] = pose_max
            group_velocity_rms[group_name] = velocity_rms
            group_pose_rms_valid[group_name] = pose_rms_valid
            group_pose_max_valid[group_name] = pose_max_valid
            group_velocity_rms_valid[group_name] = velocity_rms_valid
            grouped_quality_valid &= pose_rms_valid & pose_max_valid & velocity_rms_valid

        absolute_joint_speed = torch.abs(robot_joint_vel)
        max_joint_speed_value = torch.max(absolute_joint_speed, dim=1).values
        joint_speed_rms = torch.sqrt(torch.mean(torch.square(robot_joint_vel), dim=1))
        global_joint_speed_valid = max_joint_speed_value <= max_joint_speed
        root_linear_speed = torch.linalg.vector_norm(command.robot_anchor_lin_vel_w, dim=1)
        root_angular_speed = torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=1)
        root_motion_valid = (root_linear_speed <= max_root_linear_speed) & (
            root_angular_speed <= max_root_angular_speed
        )
        torso_orientation_error = quat_error_magnitude(
            command.anchor_quat_w,
            command.robot_anchor_quat_w,
        )
        torso_orientation_valid = torso_orientation_error <= max_torso_orientation_error

        standing_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for value in conditions.values():
            standing_valid &= value
        valid_tail_sample = (
            tail_observation
            & reference_static
            & standing_valid
            & grouped_quality_valid
            & global_joint_speed_valid
            & root_motion_valid
            & torso_orientation_valid
        )
        self._stable_steps = torch.where(valid_tail_sample, self._stable_steps + 1, 0)
        self._longest_stable_steps = torch.maximum(self._longest_stable_steps, self._stable_steps)

        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        if contact_sensor.data.net_forces_w is None or contact_sensor.data.current_contact_time is None:
            raise RuntimeError("The climb contact sensor must provide net forces and current contact time.")
        wrist_contact_forces = torch.linalg.vector_norm(
            contact_sensor.data.net_forces_w[:, self._wrist_contact_body_ids], dim=-1
        )
        wrist_contact_times = contact_sensor.data.current_contact_time[:, self._wrist_contact_body_ids]

        for name, value in conditions.items():
            command.metrics[self._METRIC_PREFIX + name].copy_((tail_observation & value).float())
        command.metrics[self._METRIC_PREFIX + "terminal_alignment_complete"].copy_(
            (in_static_tail & alignment_complete).float()
        )
        command.metrics[self._METRIC_PREFIX + "terminal_default_pose_complete"].copy_(
            (in_static_tail & terminal_default_pose_complete).float()
        )
        command.metrics[self._METRIC_PREFIX + "static_tail"].copy_(in_static_tail.float())
        command.metrics[self._METRIC_PREFIX + "reference_static"].copy_(
            (in_static_tail & reference_static).float()
        )
        command.metrics[self._METRIC_PREFIX + "default_joint_pos_rms"].copy_(
            torch.where(tail_observation, default_joint_pos_rms, torch.zeros_like(default_joint_pos_rms))
        )
        if expert_joint_pos_rms is not None:
            command.metrics[self._METRIC_PREFIX + "expert_joint_pos_rms"].copy_(
                torch.where(tail_observation, expert_joint_pos_rms, torch.zeros_like(expert_joint_pos_rms))
            )
        command.metrics[self._METRIC_PREFIX + "stable_time"].copy_(self._stable_steps.float() * env.step_dt)
        final_eligible = at_final_frame & alignment_complete & terminal_default_pose_complete
        final_float = final_eligible.to(dtype=max_joint_speed_value.dtype)
        tail_float = tail_observation.to(dtype=max_joint_speed_value.dtype)
        command.metrics[self._METRIC_PREFIX + "final_frame_fraction"].copy_(final_float)
        command.metrics[self._METRIC_PREFIX + "max_joint_speed"].copy_(tail_float * max_joint_speed_value)
        command.metrics[self._METRIC_PREFIX + "joint_speed_rms"].copy_(tail_float * joint_speed_rms)
        for threshold_name, threshold in (
            ("0_5", 0.5),
            ("0_75", 0.75),
            ("1_0", 1.0),
            ("2_0", 2.0),
        ):
            joints_over_threshold = torch.count_nonzero(absolute_joint_speed > threshold, dim=1)
            command.metrics[self._METRIC_PREFIX + f"joints_over_{threshold_name}"].copy_(
                tail_float * joints_over_threshold.to(dtype=tail_float.dtype)
            )
        command.metrics[self._METRIC_PREFIX + "root_linear_speed"].copy_(
            tail_float * root_linear_speed
        )
        command.metrics[self._METRIC_PREFIX + "root_angular_speed"].copy_(
            tail_float * root_angular_speed
        )
        command.metrics[self._METRIC_PREFIX + "torso_orientation_valid"].copy_(
            (tail_observation & torso_orientation_valid).float()
        )
        command.metrics[self._METRIC_PREFIX + "torso_orientation_error"].copy_(
            tail_float * torso_orientation_error
        )
        for group_name, joint_ids in self._diagnostic_joint_group_ids.items():
            group_max_speed = torch.max(absolute_joint_speed[:, joint_ids], dim=1).values
            command.metrics[self._METRIC_PREFIX + f"{group_name}_max_joint_speed"].copy_(
                tail_float * group_max_speed
            )
        for group_name in self._quality_joint_group_ids:
            command.metrics[self._METRIC_PREFIX + f"{group_name}_pose_rms_valid"].copy_(
                (tail_observation & group_pose_rms_valid[group_name]).float()
            )
            command.metrics[self._METRIC_PREFIX + f"{group_name}_pose_max_valid"].copy_(
                (tail_observation & group_pose_max_valid[group_name]).float()
            )
            command.metrics[self._METRIC_PREFIX + f"{group_name}_velocity_rms_valid"].copy_(
                (tail_observation & group_velocity_rms_valid[group_name]).float()
            )
            command.metrics[self._METRIC_PREFIX + f"{group_name}_pose_rms"].copy_(
                tail_float * group_pose_rms[group_name]
            )
            command.metrics[self._METRIC_PREFIX + f"{group_name}_max_pose_error"].copy_(
                tail_float * group_pose_max[group_name]
            )
            command.metrics[self._METRIC_PREFIX + f"{group_name}_joint_speed_rms"].copy_(
                tail_float * group_velocity_rms[group_name]
            )
        for wrist_index, side in enumerate(("left", "right")):
            command.metrics[self._METRIC_PREFIX + f"{side}_wrist_contact_force"].copy_(
                tail_float * wrist_contact_forces[:, wrist_index]
            )
            command.metrics[self._METRIC_PREFIX + f"{side}_wrist_contact_time"].copy_(
                tail_float * wrist_contact_times[:, wrist_index]
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
        complete_trial = _episode_started_at_motion_beginning(command)
        clip_boundary = motion_clip_boundary_mask(command.motion_finished, env.termination_manager.terminated)
        return clip_boundary & complete_trial & at_final_frame & continuously_stable


def motion_end_failure(
    env: ManagerBasedRLEnv,
    command_name: str,
    success_term_name: str,
) -> torch.Tensor:
    """Classify a completed clip as a platform-standing failure."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    if not command.cfg.terminate_on_motion_end:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    clip_boundary = motion_clip_boundary_mask(command.motion_finished, env.termination_manager.terminated)
    # Only episodes that started at the authored first frame are eligible for
    # end-quality classification.  A random-phase episode can begin too close
    # to the boundary to accumulate the required contact/stability time; if it
    # were recorded as an adaptive failure, sampling would be biased toward
    # unlearnable one- or two-step terminal starts.
    complete_trial = _episode_started_at_motion_beginning(command)
    # The success term is configured immediately before this term.  Reusing
    # its result partitions every eligible completed clip exactly once.
    successful = env.termination_manager.get_term(success_term_name)
    return clip_boundary & complete_trial & ~successful
