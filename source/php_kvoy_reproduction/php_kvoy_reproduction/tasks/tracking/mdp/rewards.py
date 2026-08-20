from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import torch
from typing import TYPE_CHECKING, NamedTuple

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from php_kvoy_reproduction.tasks.tracking.mdp.commands import MotionCommand
from php_kvoy_reproduction.tasks.tracking.mdp.joint_settling import joint_settling_score
from php_kvoy_reproduction.tasks.tracking.mdp.obstacle import get_climb_box_sizes, points_inside_oriented_box_xy
from php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry import (
    filtered_platform_contact_score,
    filtered_platform_force_score,
    first_foothold_reference_gate,
    foot_sole_corners_world,
    foothold_precontact_score,
    foothold_safety_score,
    foothold_safety_violation,
    sole_surface_alignment_score,
    sole_surface_shaping_score,
    sole_top_height_score,
)
from php_kvoy_reproduction.tasks.tracking.mdp.platform_foot_support import (
    platform_foot_load_score,
    platform_foot_support_score,
    platform_foot_support_state,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _terminal_expert_tracking_factor(command: MotionCommand) -> torch.Tensor:
    """Return the source-reference weight during default-pose takeover.

    The command exposes this factor only for the climb terminal mode.  The
    fallback keeps generic tracking tasks and lightweight unit-test doubles
    behaviorally unchanged.
    """

    dtype = command.joint_pos.dtype if hasattr(command, "joint_pos") else torch.float32
    device = command.time_steps.device
    factor = getattr(command, "terminal_default_pose_expert_tracking_factor", None)
    if factor is None:
        return torch.ones(command.time_steps.shape, dtype=dtype, device=device)
    if factor.shape != command.time_steps.shape:
        raise RuntimeError(
            "terminal_default_pose_expert_tracking_factor must match command time_steps, "
            f"got {factor.shape} and {command.time_steps.shape}."
        )
    return factor.to(dtype=dtype).clamp(min=0.0, max=1.0)


def _terminal_default_pose_latched_gate(command: MotionCommand) -> torch.Tensor:
    """Return one after the command has switched to its sole terminal q target.

    The terminal default-pose reward is climb-specific.  A generic command
    that lacks the one-way mode must not accidentally receive this reward.
    """

    dtype = command.joint_pos.dtype if hasattr(command, "joint_pos") else torch.float32
    device = command.time_steps.device
    latched = getattr(command, "terminal_default_pose_latched", None)
    if latched is None:
        return torch.zeros(command.time_steps.shape, dtype=dtype, device=device)
    if latched.shape != command.time_steps.shape:
        raise RuntimeError(
            "terminal_default_pose_latched must match command time_steps, "
            f"got {latched.shape} and {command.time_steps.shape}."
        )
    return latched.to(dtype=dtype)


def _terminal_default_pose_complete_gate(command: MotionCommand) -> torch.Tensor:
    """Return one only once the terminal q target has reached default."""

    dtype = command.joint_pos.dtype if hasattr(command, "joint_pos") else torch.float32
    device = command.time_steps.device
    complete = getattr(command, "terminal_default_pose_complete", None)
    if complete is None:
        return torch.ones(command.time_steps.shape, dtype=dtype, device=device)
    if complete.shape != command.time_steps.shape:
        raise RuntimeError(
            "terminal_default_pose_complete must match command time_steps, "
            f"got {complete.shape} and {command.time_steps.shape}."
        )
    return complete.to(dtype=dtype)


def _terminal_stationary_target_gate(command: MotionCommand) -> torch.Tensor:
    """Enable settling only when the current terminal target is stationary.

    Before terminal default-pose mode latches, the immutable source static
    tail is a single, stationary expert target.  During the subsequent q
    interpolation the target deliberately moves, so high-weight stability
    rewards must be off.  Once interpolation completes, default q is again a
    single stationary target.  This gives a useful pre-latch settling bridge
    without ever combining expert and default pose objectives.
    """

    dtype = command.joint_pos.dtype if hasattr(command, "joint_pos") else torch.float32
    device = command.time_steps.device
    complete = _terminal_default_pose_complete_gate(command).to(dtype=torch.bool)
    latched = getattr(command, "terminal_default_pose_latched", None)
    static_tail = getattr(command, "terminal_default_pose_static_tail", None)
    # Disabling default-q means that there is no interpolation to complete;
    # its compatibility ``complete=True`` value must not be mistaken for a
    # stationary expert target.  The immutable source is guaranteed fixed at
    # its final frame, which is the only phase eligible for terminal settling
    # in this mode.  Keep the generic/test-double fallback below when a motion
    # command does not expose the required source-frame metadata.
    default_q_enabled = getattr(getattr(command, "cfg", None), "terminal_default_pose_enabled", None)
    if default_q_enabled is False:
        motion_end_idx = getattr(getattr(command, "motion", None), "motion_end_idx", None)
        motion_ids = getattr(command, "motion_ids", None)
        if motion_end_idx is not None and motion_ids is not None:
            final_frames = motion_end_idx[motion_ids] - 1
            if final_frames.shape != command.time_steps.shape:
                raise RuntimeError(
                    "terminal motion final frames must match command time_steps, "
                    f"got {final_frames.shape} and {command.time_steps.shape}."
                )
            return (command.time_steps >= final_frames).to(dtype=dtype)
    # Keep generic commands and lightweight test doubles on their historical
    # behavior: they only become eligible once an exposed completion gate is
    # true.  Production MotionCommand exposes both state tensors.
    if latched is None or static_tail is None:
        return complete.to(dtype=dtype)
    if latched.shape != command.time_steps.shape:
        raise RuntimeError(
            "terminal_default_pose_latched must match command time_steps, "
            f"got {latched.shape} and {command.time_steps.shape}."
        )
    if static_tail.shape != command.time_steps.shape:
        raise RuntimeError(
            "terminal_default_pose_static_tail must match command time_steps, "
            f"got {static_tail.shape} and {command.time_steps.shape}."
        )
    stationary_source = ~latched.to(dtype=torch.bool) & static_tail.to(dtype=torch.bool)
    return (stationary_source | complete).to(dtype=dtype, device=device)


class _FirstFootholdSettings(NamedTuple):
    """Validated first-foot support parameters shared by reward terms."""

    foot_body_names: tuple[str, ...]
    platform_contact_sensor_names: tuple[str, ...]
    sole_corners_b: tuple[tuple[float, float, float], ...]
    approach_side: float
    max_heel_overhang: float
    min_forefoot_inside: float
    far_edge_margin: float
    lateral_margin: float
    foot_height_std: float
    surface_height_std: float
    surface_height_tolerance: float
    surface_tilt_scale: float
    surface_height_scale: float
    precontact_approach_distance: float
    precontact_height_std: float
    reference_activation_distance: float
    reference_activation_inside: float
    reference_release_distance: float
    reference_release_inside: float
    phase_start: float
    phase_ramp: float
    phase_end: float
    phase_fade: float
    min_upward_force: float
    contact_time_scale: float


class _FirstFootholdState(NamedTuple):
    """Geometry and gates shared by first-foot reward/tracking calculations."""

    settings: _FirstFootholdSettings
    sole_corners_w: torch.Tensor
    platform: RigidObject
    sizes: torch.Tensor
    reference_gate: torch.Tensor
    reference_lead_mask: torch.Tensor
    precontact_scores: torch.Tensor


class _FirstFootholdTrackingGates(NamedTuple):
    """Separate source fades before and after verified physical top contact."""

    precontact: torch.Tensor
    physical_contact: torch.Tensor


def _first_foothold_settings(params: Mapping[str, object]) -> _FirstFootholdSettings:
    """Parse the compact first-foot configuration passed by ``RewTerm``."""

    required_keys = (
        "foot_body_names",
        "platform_contact_sensor_names",
        "sole_corners_b",
        "approach_side",
        "max_heel_overhang",
        "min_forefoot_inside",
        "far_edge_margin",
        "lateral_margin",
        "foot_height_std",
        "surface_height_std",
        "surface_height_tolerance",
        "surface_tilt_scale",
        "surface_height_scale",
        "precontact_approach_distance",
        "precontact_height_std",
        "reference_activation_distance",
        "reference_activation_inside",
        "reference_release_distance",
        "reference_release_inside",
        "phase_start",
        "phase_ramp",
        "phase_end",
        "phase_fade",
        "min_upward_force",
        "contact_time_scale",
    )
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise ValueError(f"first_foothold_params is missing required keys: {missing}.")

    raw_foot_names = params["foot_body_names"]
    raw_sensor_names = params["platform_contact_sensor_names"]
    raw_sole_corners = params["sole_corners_b"]
    if not isinstance(raw_foot_names, (list, tuple)) or len(raw_foot_names) != 2:
        raise ValueError("first_foothold_params['foot_body_names'] must contain exactly left and right foot names.")
    if not isinstance(raw_sensor_names, (list, tuple)) or len(raw_sensor_names) != len(raw_foot_names):
        raise ValueError(
            "first_foothold_params['platform_contact_sensor_names'] must match foot_body_names one-to-one."
        )
    if not isinstance(raw_sole_corners, (list, tuple)) or len(raw_sole_corners) < 4:
        raise ValueError("first_foothold_params['sole_corners_b'] must contain at least four sole samples.")

    foot_body_names = tuple(str(name) for name in raw_foot_names)
    platform_contact_sensor_names = tuple(str(name) for name in raw_sensor_names)
    if len(set(foot_body_names)) != len(foot_body_names):
        raise ValueError("first_foothold_params['foot_body_names'] must not contain duplicates.")
    if len(set(platform_contact_sensor_names)) != len(platform_contact_sensor_names):
        raise ValueError("first_foothold_params['platform_contact_sensor_names'] must not contain duplicates.")

    sole_corners: list[tuple[float, float, float]] = []
    for corner in raw_sole_corners:
        if not isinstance(corner, (list, tuple)) or len(corner) != 3:
            raise ValueError("Every first-foot sole corner must contain exactly three coordinates.")
        sole_corners.append((float(corner[0]), float(corner[1]), float(corner[2])))

    return _FirstFootholdSettings(
        foot_body_names=foot_body_names,
        platform_contact_sensor_names=platform_contact_sensor_names,
        sole_corners_b=tuple(sole_corners),
        approach_side=float(params["approach_side"]),
        max_heel_overhang=float(params["max_heel_overhang"]),
        min_forefoot_inside=float(params["min_forefoot_inside"]),
        far_edge_margin=float(params["far_edge_margin"]),
        lateral_margin=float(params["lateral_margin"]),
        foot_height_std=float(params["foot_height_std"]),
        surface_height_std=float(params["surface_height_std"]),
        surface_height_tolerance=float(params["surface_height_tolerance"]),
        surface_tilt_scale=float(params["surface_tilt_scale"]),
        surface_height_scale=float(params["surface_height_scale"]),
        precontact_approach_distance=float(params["precontact_approach_distance"]),
        precontact_height_std=float(params["precontact_height_std"]),
        reference_activation_distance=float(params["reference_activation_distance"]),
        reference_activation_inside=float(params["reference_activation_inside"]),
        reference_release_distance=float(params["reference_release_distance"]),
        reference_release_inside=float(params["reference_release_inside"]),
        phase_start=float(params["phase_start"]),
        phase_ramp=float(params["phase_ramp"]),
        phase_end=float(params["phase_end"]),
        phase_fade=float(params["phase_fade"]),
        min_upward_force=float(params["min_upward_force"]),
        contact_time_scale=float(params["contact_time_scale"]),
    )


def _named_body_ids(
    body_names: list[str],
    requested_names: tuple[str, ...],
    device: torch.device,
    *,
    context: str,
) -> torch.Tensor:
    missing = [name for name in requested_names if name not in body_names]
    if missing:
        raise RuntimeError(f"{context} references body names absent from the command/robot: {missing}.")
    return torch.tensor([body_names.index(name) for name in requested_names], dtype=torch.long, device=device)


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return _terminal_expert_tracking_factor(command) * torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return _terminal_expert_tracking_factor(command) * torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def _terminal_platform_contact_gate(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    platform_support_params: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Return a smooth gate that requires both feet to contact the platform."""

    per_foot_scores = _platform_foot_contact_scores(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
    )
    # The minimum prevents one well-supported foot from masking a missing or
    # poorly supported second foot.
    return per_foot_scores.amin(dim=1).clamp(min=0.0, max=1.0)


def _apply_first_foothold_body_tracking_weights(
    weights: torch.Tensor,
    command: MotionCommand,
    body_indexes: list[int],
    first_foothold_gates: _FirstFootholdTrackingGates | None,
    first_foothold_foot_body_names: tuple[str, ...] | None,
    first_foothold_body_weights: Mapping[str, float] | None,
    first_foothold_precontact_body_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Apply safe pre-contact and physical-contact source-weight floors."""

    if first_foothold_gates is None:
        if first_foothold_foot_body_names is not None or first_foothold_body_weights is not None:
            raise ValueError("First-foothold body weights/names require first_foothold_gates.")
        return weights
    if first_foothold_foot_body_names is None or first_foothold_body_weights is None:
        raise ValueError("first_foothold_gates requires both foot names and body weights.")
    expected_shape = (weights.shape[0], len(first_foothold_foot_body_names))
    if first_foothold_gates.precontact.shape != expected_shape:
        raise ValueError(
            "first_foothold_gates.precontact must have shape "
            f"{expected_shape}, got {first_foothold_gates.precontact.shape}."
        )
    if first_foothold_gates.physical_contact.shape != expected_shape:
        raise ValueError(
            "first_foothold_gates.physical_contact must have shape "
            f"{expected_shape}, got {first_foothold_gates.physical_contact.shape}."
        )

    tracked_names = command.cfg.body_names
    unknown_names = set(first_foothold_body_weights).difference(tracked_names)
    if unknown_names:
        raise ValueError(f"First-foothold body weights contain untracked body names: {sorted(unknown_names)}.")
    if first_foothold_precontact_body_weights is not None:
        unknown_precontact_names = set(first_foothold_precontact_body_weights).difference(tracked_names)
        if unknown_precontact_names:
            raise ValueError(
                "First-foothold pre-contact body weights contain untracked body names: "
                f"{sorted(unknown_precontact_names)}."
            )
    side_to_foot_index: dict[str, int] = {}
    for foot_index, foot_name in enumerate(first_foothold_foot_body_names):
        side, separator, _ = foot_name.partition("_")
        if not separator or side in side_to_foot_index:
            raise ValueError(
                "First-foothold foot body names must be unique left/right-style names such as l_ankle_x_link."
            )
        side_to_foot_index[side] = foot_index

    for local_index, body_index in enumerate(body_indexes):
        body_name = tracked_names[body_index]
        contact_target_weight = first_foothold_body_weights.get(body_name)
        if contact_target_weight is None:
            continue
        precontact_target_weight = (
            contact_target_weight
            if first_foothold_precontact_body_weights is None
            else first_foothold_precontact_body_weights.get(body_name, contact_target_weight)
        )
        if not 0.0 <= contact_target_weight <= 1.0:
            raise ValueError(
                "First-foothold physical-contact tracking weight for "
                f"'{body_name}' must be in [0, 1], got {contact_target_weight}."
            )
        if not 0.0 <= precontact_target_weight <= 1.0:
            raise ValueError(
                "First-foothold pre-contact tracking weight for "
                f"'{body_name}' must be in [0, 1], got {precontact_target_weight}."
            )
        side, separator, _ = body_name.partition("_")
        if not separator or side not in side_to_foot_index:
            raise ValueError(
                f"First-foothold tracked body '{body_name}' does not match a configured first-foot side."
            )
        foot_index = side_to_foot_index[side]
        precontact_weight = 1.0 - (1.0 - precontact_target_weight) * first_foothold_gates.precontact[
            :, foot_index
        ]
        contact_weight = 1.0 - (1.0 - contact_target_weight) * first_foothold_gates.physical_contact[
            :, foot_index
        ]
        first_weight = torch.minimum(precontact_weight, contact_weight)
        # Multiple fades must combine as the weaker active objective, never as
        # a product.  This preserves whichever terminal or arrival objective
        # is intentionally weaker instead of multiplying two gates together.
        weights[:, local_index] = torch.minimum(weights[:, local_index], first_weight)
    return weights


def _terminal_weighted_body_error_exp(
    error: torch.Tensor,
    command: MotionCommand,
    body_indexes: list[int],
    terminal_body_indexes: list[int],
    terminal_body_weight: float,
    terminal_gate: torch.Tensor,
    std: float,
    first_foothold_gates: _FirstFootholdTrackingGates | None = None,
    first_foothold_foot_body_names: tuple[str, ...] | None = None,
    first_foothold_body_weights: Mapping[str, float] | None = None,
    first_foothold_precontact_body_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Fade selected tracking objectives during first support and final contact."""

    if not math.isfinite(std) or std <= 0.0:
        raise ValueError(f"std must be positive and finite, got {std}.")
    weights = _terminal_body_tracking_weights(
        error,
        command,
        body_indexes,
        terminal_body_indexes,
        terminal_body_weight,
        terminal_gate,
        first_foothold_gates,
        first_foothold_foot_body_names,
        first_foothold_body_weights,
        first_foothold_precontact_body_weights,
    )
    # Keep the original body-count normalization. This genuinely weakens the
    # ankle objective instead of renormalizing the remaining body terms.
    return torch.exp(-(error * weights).mean(dim=-1) / std**2)


def _terminal_body_tracking_weights(
    error: torch.Tensor,
    command: MotionCommand,
    body_indexes: list[int],
    terminal_body_indexes: list[int],
    terminal_body_weight: float,
    terminal_gate: torch.Tensor,
    first_foothold_gates: _FirstFootholdTrackingGates | None = None,
    first_foothold_foot_body_names: tuple[str, ...] | None = None,
    first_foothold_body_weights: Mapping[str, float] | None = None,
    first_foothold_precontact_body_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Build per-body source-tracking weights without reducing error axes."""

    if not 0.0 <= terminal_body_weight <= 1.0:
        raise ValueError(f"terminal_body_weight must be in [0, 1], got {terminal_body_weight}.")
    if error.ndim != 2 or error.shape[1] != len(body_indexes):
        raise ValueError(
            f"error must have shape [N, {len(body_indexes)}], got {tuple(error.shape)}."
        )
    if terminal_gate.shape != (error.shape[0],):
        raise ValueError(f"terminal_gate must have shape {(error.shape[0],)}, got {terminal_gate.shape}.")
    terminal_set = set(terminal_body_indexes)
    terminal_mask = torch.tensor(
        [index in terminal_set for index in body_indexes], dtype=error.dtype, device=error.device
    )
    weights = 1.0 - (1.0 - terminal_body_weight) * terminal_gate[:, None] * terminal_mask[None, :]
    weights = _apply_first_foothold_body_tracking_weights(
        weights,
        command,
        body_indexes,
        first_foothold_gates,
        first_foothold_foot_body_names,
        first_foothold_body_weights,
        first_foothold_precontact_body_weights,
    )
    return weights


def _climb_terminal_gate(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    terminal_window_time_s: float,
    platform_support_params: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Fade terminal objectives in only near the clip end after contact."""

    return _final_phase_gate(command, terminal_window_time_s, env.step_dt) * _terminal_platform_contact_gate(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
    )


def climb_motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str],
    terminal_body_names: list[str],
    terminal_body_weight: float,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    terminal_window_time_s: float,
    first_foothold_params: Mapping[str, object] | None = None,
    first_foothold_body_weights: Mapping[str, float] | None = None,
    platform_support_params: Mapping[str, object] | None = None,
    terminal_vertical_body_weight: float | None = None,
    first_foothold_vertical_body_weights: Mapping[str, float] | None = None,
    first_foothold_vertical_precontact_body_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Track body XY while allowing physical terrain to own an arriving foot's Z."""

    if not math.isfinite(std) or std <= 0.0:
        raise ValueError(f"std must be positive and finite, got {std}.")
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    terminal_body_indexes = _get_body_indexes(command, terminal_body_names)
    position_error = torch.square(
        command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]
    )
    horizontal_error = position_error[..., :2].sum(dim=-1)
    vertical_error = position_error[..., 2]
    terminal_gate = _climb_terminal_gate(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        terminal_window_time_s,
        platform_support_params,
    )
    first_foothold_gates: _FirstFootholdTrackingGates | None = None
    first_foothold_foot_body_names: tuple[str, ...] | None = None
    if (
        first_foothold_params is not None
        or first_foothold_body_weights is not None
        or first_foothold_vertical_body_weights is not None
        or first_foothold_vertical_precontact_body_weights is not None
    ):
        if first_foothold_params is None or first_foothold_body_weights is None:
            raise ValueError(
                "Position tracking requires first_foothold_params and horizontal body weights "
                "whenever either first-foothold axis is configured."
            )
        first_foothold_gates, settings = _first_foothold_tracking_gates(
            env,
            command,
            platform_cfg,
            base_size,
            first_foothold_params,
        )
        first_foothold_foot_body_names = settings.foot_body_names
    horizontal_weights = _terminal_body_tracking_weights(
        horizontal_error,
        command,
        body_indexes,
        terminal_body_indexes,
        terminal_body_weight,
        terminal_gate,
        first_foothold_gates,
        first_foothold_foot_body_names,
        first_foothold_body_weights,
    )
    vertical_terminal_weight = (
        terminal_body_weight if terminal_vertical_body_weight is None else terminal_vertical_body_weight
    )
    vertical_first_weights = (
        first_foothold_body_weights
        if first_foothold_vertical_body_weights is None
        else first_foothold_vertical_body_weights
    )
    vertical_weights = _terminal_body_tracking_weights(
        vertical_error,
        command,
        body_indexes,
        terminal_body_indexes,
        vertical_terminal_weight,
        terminal_gate,
        first_foothold_gates,
        first_foothold_foot_body_names,
        vertical_first_weights,
        first_foothold_vertical_precontact_body_weights,
    )
    # Preserve the historical normalization over tracked bodies.  Only the
    # selected feet differ by axis: expert XY remains active while source Z is
    # handed to the physical sole-to-platform objective.
    weighted_error = horizontal_error * horizontal_weights + vertical_error * vertical_weights
    score = torch.exp(-weighted_error.mean(dim=-1) / std**2)
    return _terminal_expert_tracking_factor(command) * score


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def climb_motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str],
    terminal_body_names: list[str],
    terminal_body_weight: float,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    terminal_window_time_s: float,
    first_foothold_params: Mapping[str, object] | None = None,
    first_foothold_body_weights: Mapping[str, float] | None = None,
    first_foothold_precontact_body_weights: Mapping[str, float] | None = None,
    platform_support_params: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Track body orientations while freeing an arriving ankle from the reference."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    terminal_body_indexes = _get_body_indexes(command, terminal_body_names)
    error = quat_error_magnitude(
        command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes]
    ) ** 2
    terminal_gate = _climb_terminal_gate(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        terminal_window_time_s,
        platform_support_params,
    )
    first_foothold_gates: _FirstFootholdTrackingGates | None = None
    first_foothold_foot_body_names: tuple[str, ...] | None = None
    if (
        first_foothold_params is not None
        or first_foothold_body_weights is not None
        or first_foothold_precontact_body_weights is not None
    ):
        if first_foothold_params is None or first_foothold_body_weights is None:
            raise ValueError(
                "Orientation tracking requires both first_foothold_params and first_foothold_body_weights."
            )
        first_foothold_gates, settings = _first_foothold_tracking_gates(
            env,
            command,
            platform_cfg,
            base_size,
            first_foothold_params,
        )
        first_foothold_foot_body_names = settings.foot_body_names
    score = _terminal_weighted_body_error_exp(
        error,
        command,
        body_indexes,
        terminal_body_indexes,
        terminal_body_weight,
        terminal_gate,
        std,
        first_foothold_gates,
        first_foothold_foot_body_names,
        first_foothold_body_weights,
        first_foothold_precontact_body_weights,
    )
    return _terminal_expert_tracking_factor(command) * score


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def _first_foothold_weighted_body_error_exp(
    error: torch.Tensor,
    command: MotionCommand,
    body_indexes: list[int],
    std: float,
    first_foothold_gates: _FirstFootholdTrackingGates,
    first_foothold_foot_body_names: tuple[str, ...],
    first_foothold_body_weights: Mapping[str, float],
) -> torch.Tensor:
    """Apply only the first-support fade to a body-wise tracking error."""

    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    weights = torch.ones_like(error)
    weights = _apply_first_foothold_body_tracking_weights(
        weights,
        command,
        body_indexes,
        first_foothold_gates,
        first_foothold_foot_body_names,
        first_foothold_body_weights,
    )
    return torch.exp(-(error * weights).mean(dim=-1) / std**2)


def climb_motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str],
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    first_foothold_params: Mapping[str, object],
    first_foothold_body_weights: Mapping[str, float],
) -> torch.Tensor:
    """Track body linear velocity while allowing an arriving ankle to settle."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    first_foothold_gates, settings = _first_foothold_tracking_gates(
        env,
        command,
        platform_cfg,
        base_size,
        first_foothold_params,
    )
    score = _first_foothold_weighted_body_error_exp(
        error,
        command,
        body_indexes,
        std,
        first_foothold_gates,
        settings.foot_body_names,
        first_foothold_body_weights,
    )
    return _terminal_expert_tracking_factor(command) * score


def climb_motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    body_names: list[str],
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    first_foothold_params: Mapping[str, object],
    first_foothold_body_weights: Mapping[str, float],
) -> torch.Tensor:
    """Track body angular velocity while allowing the support ankle to settle."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    first_foothold_gates, settings = _first_foothold_tracking_gates(
        env,
        command,
        platform_cfg,
        base_size,
        first_foothold_params,
    )
    score = _first_foothold_weighted_body_error_exp(
        error,
        command,
        body_indexes,
        std,
        first_foothold_gates,
        settings.foot_body_names,
        first_foothold_body_weights,
    )
    return _terminal_expert_tracking_factor(command) * score


def _smoothstep_window(value: torch.Tensor, start: float, end: float) -> torch.Tensor:
    """Smooth cubic gate which is zero before ``start`` and one after ``end``."""

    if not 0.0 <= start < end <= 1.0:
        raise ValueError(f"phase window must satisfy 0 <= start < end <= 1, got {(start, end)}.")
    x = ((value - start) / (end - start)).clamp(min=0.0, max=1.0)
    return x * x * (3.0 - 2.0 * x)


def _motion_phase(command: MotionCommand) -> torch.Tensor:
    """Return normalized local reference phase for every environment."""

    starts = command.motion.motion_start_idx[command.motion_ids]
    lengths = command.motion.motion_lengths[command.motion_ids].to(dtype=torch.float32)
    return ((command.time_steps - starts).to(dtype=torch.float32) / (lengths - 1.0).clamp_min(1.0)).clamp(
        min=0.0, max=1.0
    )


def _platform_local_x(points_w: torch.Tensor, platform: RigidObject) -> torch.Tensor:
    """Transform world points to the platform's yaw-aligned local x coordinate."""

    w, x, y, z = platform.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    delta = points_w - platform.data.root_pos_w[:, None, :]
    return torch.cos(yaw)[:, None] * delta[..., 0] + torch.sin(yaw)[:, None] * delta[..., 1]


def _platform_local_y(points_w: torch.Tensor, platform: RigidObject) -> torch.Tensor:
    """Transform world points to the platform's yaw-aligned local y coordinate."""

    w, x, y, z = platform.data.root_quat_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    delta = points_w - platform.data.root_pos_w[:, None, :]
    return -torch.sin(yaw)[:, None] * delta[..., 0] + torch.cos(yaw)[:, None] * delta[..., 1]


def _first_foothold_state(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    first_foothold_params: Mapping[str, object],
) -> _FirstFootholdState:
    """Build platform-relative first-foot geometry without changing the NPZ target."""

    settings = _first_foothold_settings(first_foothold_params)
    platform: RigidObject = env.scene[platform_cfg.name]
    sizes = get_climb_box_sizes(platform, base_size=base_size, device=platform.device)

    # ``body_pos_w`` is the immutable expert target transformed only by the
    # current platform xy/yaw.  It is intentionally used here instead of the
    # torso-reanchored body_pos_relative_w, which would drift with the robot.
    reference_foot_ids = _named_body_ids(
        command.cfg.body_names,
        settings.foot_body_names,
        command.device,
        context="First-foothold reference gate",
    )
    reference_foot_positions = command.body_pos_w[:, reference_foot_ids]
    reference_gate, reference_lead_mask = first_foothold_reference_gate(
        reference_foot_positions,
        platform.data.root_pos_w,
        platform.data.root_quat_w,
        sizes,
        _motion_phase(command),
        approach_side=settings.approach_side,
        reference_activation_distance=settings.reference_activation_distance,
        reference_activation_inside=settings.reference_activation_inside,
        reference_release_distance=settings.reference_release_distance,
        reference_release_inside=settings.reference_release_inside,
        phase_start=settings.phase_start,
        phase_ramp=settings.phase_ramp,
        phase_end=settings.phase_end,
        phase_fade=settings.phase_fade,
    )

    actual_foot_ids = _named_body_ids(
        command.robot.body_names,
        settings.foot_body_names,
        command.device,
        context="First-foothold physical geometry",
    )
    target_device = torch.device(command.device)
    cached_sole_corners = getattr(command, "_first_foothold_sole_corners_b", None)
    if (
        cached_sole_corners is None
        or cached_sole_corners.device != target_device
        or cached_sole_corners.dtype != command.robot.data.body_pos_w.dtype
    ):
        cached_sole_corners = torch.tensor(
            settings.sole_corners_b,
            dtype=command.robot.data.body_pos_w.dtype,
            device=command.device,
        )
        command._first_foothold_sole_corners_b = cached_sole_corners
    sole_corners_w = foot_sole_corners_world(
        command.robot.data.body_pos_w[:, actual_foot_ids],
        command.robot.data.body_quat_w[:, actual_foot_ids],
        cached_sole_corners,
    )
    precontact_scores = foothold_precontact_score(
        sole_corners_w,
        platform.data.root_pos_w,
        platform.data.root_quat_w,
        sizes,
        approach_side=settings.approach_side,
        approach_distance=settings.precontact_approach_distance,
        height_std=settings.precontact_height_std,
    )
    return _FirstFootholdState(
        settings=settings,
        sole_corners_w=sole_corners_w,
        platform=platform,
        sizes=sizes,
        reference_gate=reference_gate,
        reference_lead_mask=reference_lead_mask,
        precontact_scores=precontact_scores,
    )


def _filtered_platform_foot_contact_scores(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    settings: _FirstFootholdSettings,
    safe_top_support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return sustained filtered support and its instantaneous force quality.

    ContactSensor's built-in contact timer includes every counterpart of an
    ankle.  The duration here is instead advanced only while this ankle has
    sufficient *filtered* upward force from ClimbPlatform and a safe sole is
    at the platform top.
    """

    expected_shape = (env.num_envs, len(settings.platform_contact_sensor_names))
    if safe_top_support.shape != expected_shape:
        raise ValueError(f"safe_top_support must have shape {expected_shape}, got {safe_top_support.shape}.")

    platform_forces: list[torch.Tensor] = []
    for sensor_name in settings.platform_contact_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError(
                f"First-foothold sensor '{sensor_name}' must configure filter_prim_paths_expr for ClimbPlatform."
            )
        if force_matrix.ndim != 4 or force_matrix.shape[0] != env.num_envs or force_matrix.shape[1] != 1:
            raise RuntimeError(
                f"First-foothold sensor '{sensor_name}' must contain exactly one ankle body; "
                f"got force matrix shape {tuple(force_matrix.shape)}."
            )
        if force_matrix.shape[2] == 0:
            raise RuntimeError(
                f"First-foothold sensor '{sensor_name}' resolved no ClimbPlatform filter bodies."
            )

        # ContactSensor reports the force exerted on the ankle by its filtered
        # counterpart.  A +z component is top support; a side strike has no
        # such component and cannot satisfy the first-foot reward.
        platform_forces.append(force_matrix[:, 0].sum(dim=1))

    platform_forces_w = torch.stack(platform_forces, dim=1)
    force_score = filtered_platform_force_score(
        platform_forces_w,
        min_upward_force=settings.min_upward_force,
    )
    continuous_support = (platform_forces_w[..., 2] >= settings.min_upward_force) & safe_top_support
    filtered_contact_time = command.advance_first_foothold_filtered_contact_time(
        continuous_support,
        env.step_dt,
    )
    sustained_score = filtered_platform_contact_score(
        platform_forces_w,
        filtered_contact_time,
        min_upward_force=settings.min_upward_force,
        contact_time_scale=settings.contact_time_scale,
    )
    return sustained_score, force_score


def _first_foothold_tracking_gates(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    first_foothold_params: Mapping[str, object],
) -> tuple[_FirstFootholdTrackingGates, _FirstFootholdSettings]:
    """Fade only configured source-foot objectives near the physical surface.

    The terrain-adaptive surface reward replaces the expert ankle pitch/roll
    during this gate.  Hip, knee and torso terms remain fully constrained by
    configuration, so the arriving foot is never an unconstrained airborne
    end effector.  A permissive forefoot contact keeps the gate active while
    the heel is being lowered toward complete-sole support.
    """

    state = _first_foothold_state(env, command, platform_cfg, base_size, first_foothold_params)
    support = platform_foot_support_state(
        env,
        command.robot,
        command.device,
        platform_cfg,
        base_size,
        first_foothold_params,
        min_upward_force=state.settings.min_upward_force,
        sole_height_tolerance=state.settings.foot_height_std,
    )
    if support.settings.foot_body_names != state.settings.foot_body_names:
        raise RuntimeError("First-foothold physical support feet do not match the reference-foot ordering.")
    phase = _motion_phase(command)
    phase_in = _smoothstep_window(
        phase,
        state.settings.phase_start,
        state.settings.phase_start + state.settings.phase_ramp,
    )
    phase_out = 1.0 - _smoothstep_window(
        phase,
        state.settings.phase_end,
        state.settings.phase_end + state.settings.phase_fade,
    )
    # Every foot that physically approaches the top receives the same terrain
    # handoff.  Reference-leading-foot selection remains exclusive to the
    # first-support bonus; using it here would leave the second arriving foot
    # stuck with the same bad expert pitch/roll.
    precontact_gate = phase_in[:, None] * phase_out[:, None] * state.precontact_scores
    # Once either foot physically reaches the platform, retain the adaptive
    # handoff even after the reference's first-foot window closes.  Otherwise
    # the bad source pitch/roll would be restored exactly while the second foot
    # is arriving and could pull a settled heel back off the surface.
    physical_contact_gate = support.contact_support.to(dtype=state.reference_gate.dtype)
    return _FirstFootholdTrackingGates(precontact_gate, physical_contact_gate), state.settings


def _first_foothold_sole_safety(
    state: _FirstFootholdState,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return footprint, transient-contact, and complete-sole surface facts."""

    geometry_score, valid_footprint = foothold_safety_score(
        state.sole_corners_w,
        state.platform.data.root_pos_w,
        state.platform.data.root_quat_w,
        state.sizes,
        approach_side=state.settings.approach_side,
        max_heel_overhang=state.settings.max_heel_overhang,
        min_forefoot_inside=state.settings.min_forefoot_inside,
        far_edge_margin=state.settings.far_edge_margin,
        lateral_margin=state.settings.lateral_margin,
    )
    height_score = sole_top_height_score(
        state.sole_corners_w,
        state.platform.data.root_pos_w,
        state.sizes,
        height_std=state.settings.foot_height_std,
    )
    surface_score, surface_valid, _ = sole_surface_alignment_score(
        state.sole_corners_w,
        state.platform.data.root_pos_w,
        state.sizes,
        height_std=state.settings.surface_height_std,
        height_tolerance=state.settings.surface_height_tolerance,
    )
    return geometry_score, valid_footprint, height_score, surface_score, surface_valid


def first_foothold_surface_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    first_foothold_params: Mapping[str, object],
    yaw_std: float,
) -> torch.Tensor:
    """Align the arriving sole to the sampled box top without copying bad expert tilt.

    Expert phase and each foot's yaw remain authoritative.  Sole Z, pitch and
    roll are instead scored against the physical platform surface for whichever
    foot is actually approaching it.  The pre-contact gate gives the policy a
    dense correction before impact and covers both the first and second foot.
    """

    if not math.isfinite(yaw_std) or yaw_std <= 0.0:
        raise ValueError(f"yaw_std must be positive and finite, got {yaw_std}.")
    command: MotionCommand = env.command_manager.get_term(command_name)
    state = _first_foothold_state(env, command, platform_cfg, base_size, first_foothold_params)
    surface_score, _, _ = sole_surface_shaping_score(
        state.sole_corners_w,
        state.platform.data.root_pos_w,
        state.sizes,
        tilt_scale=state.settings.surface_tilt_scale,
        height_scale=state.settings.surface_height_scale,
    )
    foot_ids = _named_body_ids(
        command.cfg.body_names,
        state.settings.foot_body_names,
        command.device,
        context="First-foothold adaptive surface heading",
    )
    reference_yaw = math_utils.yaw_quat(command.body_quat_relative_w[:, foot_ids])
    actual_yaw = math_utils.yaw_quat(command.robot_body_quat_w[:, foot_ids])
    yaw_error = quat_error_magnitude(reference_yaw, actual_yaw)
    yaw_score = torch.rsqrt(1.0 + torch.square(yaw_error / yaw_std))
    phase = _motion_phase(command)
    phase_in = _smoothstep_window(
        phase,
        state.settings.phase_start,
        state.settings.phase_start + state.settings.phase_ramp,
    )
    phase_out = 1.0 - _smoothstep_window(
        phase,
        state.settings.phase_end,
        state.settings.phase_end + state.settings.phase_fade,
    )
    gates = phase_in[:, None] * phase_out[:, None] * state.precontact_scores
    per_foot_score = gates * surface_score * yaw_score
    # Preserve a unit maximum whether one or both feet are simultaneously in
    # the arrival band, without amplifying a barely active gate.
    normalization = gates.sum(dim=1).clamp_min(1.0)
    return per_foot_score.sum(dim=1) / normalization


def first_foothold_support_quality(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    first_foothold_params: Mapping[str, object],
) -> torch.Tensor:
    """Reward a safe, sustained first foot support on the sampled platform.

    The selected foot follows the reference's leading left/right side, which
    automatically handles mirrored clips.  The reward is confined to that
    first-foot transfer and therefore cannot be farmed by the other foot in
    the initial or final standing phases.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    state = _first_foothold_state(env, command, platform_cfg, base_size, first_foothold_params)
    geometry_score, valid_footprint, _, surface_score, surface_valid = _first_foothold_sole_safety(state)
    platform_top = state.platform.data.root_pos_w[:, None, 2] + 0.5 * state.sizes[:, None, 2]
    near_platform_top = torch.abs(state.sole_corners_w[..., 2] - platform_top[:, :, None]).amin(dim=-1)
    # The same physical height scale that scores contact defines when filtered
    # force may accumulate support time.  This excludes a prior ground or box-
    # side contact from satisfying the sustained-top-contact requirement.
    safe_top_support = (
        (state.reference_gate[:, None] > 0.0)
        & state.reference_lead_mask.to(dtype=torch.bool)
        & valid_footprint
        & surface_valid
        & (near_platform_top <= state.settings.foot_height_std)
    )
    platform_contact_score, platform_force_score = _filtered_platform_foot_contact_scores(
        env,
        command,
        state.settings,
        safe_top_support,
    )
    safety_violation = foothold_safety_violation(
        state.sole_corners_w,
        state.platform.data.root_pos_w,
        state.platform.data.root_quat_w,
        state.sizes,
        approach_side=state.settings.approach_side,
        max_heel_overhang=state.settings.max_heel_overhang,
        min_forefoot_inside=state.settings.min_forefoot_inside,
        far_edge_margin=state.settings.far_edge_margin,
        lateral_margin=state.settings.lateral_margin,
    )
    # Above the exact safe boundary, a top-surface contact is actively worse
    # than returning to the valid footprint.  The positive term remains zero
    # there, so the requested 5 cm heel limit is both a hard reward gate and a
    # directional behavior constraint rather than merely a missing bonus.
    per_foot_quality = state.precontact_scores * (
        geometry_score
        * surface_score
        * surface_valid.to(dtype=surface_score.dtype)
        * platform_contact_score
        - safety_violation * platform_force_score
    )
    return state.reference_gate * torch.sum(state.reference_lead_mask * per_foot_quality, dim=1)


def _climb_platform_support_score(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    platform: RigidObject,
    sizes: torch.Tensor,
    support_body_names: list[str],
    contact_sensor_cfg: SceneEntityCfg,
    support_xy_margin: float,
    support_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
) -> torch.Tensor:
    """Smoothly detect a foot/hand physically supported by the platform."""

    if not support_body_names:
        raise ValueError("support_body_names must contain at least one body name.")
    if support_xy_margin < 0.0 or support_height_std <= 0.0:
        raise ValueError("support_xy_margin must be non-negative and support_height_std must be positive.")
    if min_contact_force <= 0.0 or contact_time_scale <= 0.0:
        raise ValueError("min_contact_force and contact_time_scale must be positive.")
    body_ids = torch.tensor(
        [command.robot.body_names.index(name) for name in support_body_names],
        dtype=torch.long,
        device=command.device,
    )
    positions = command.robot.data.body_pos_w[:, body_ids]
    support_sizes = sizes.clone()
    support_sizes[:, :2] += 2.0 * support_xy_margin
    inside = points_inside_oriented_box_xy(
        positions,
        platform.data.root_pos_w,
        platform.data.root_quat_w,
        support_sizes,
    ).to(dtype=positions.dtype)
    platform_top = platform.data.root_pos_w[:, None, 2] + 0.5 * sizes[:, None, 2]
    height_score = torch.exp(-0.5 * torch.square((positions[..., 2] - platform_top) / support_height_std))
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    if contact_sensor_cfg.body_ids is None:
        raise RuntimeError("The climb progress reward requires resolved contact sensor body_ids.")
    if contact_sensor.data.net_forces_w is None or contact_sensor.data.current_contact_time is None:
        raise RuntimeError("The climb progress contact sensor must provide net forces and current contact time.")
    if len(contact_sensor_cfg.body_ids) != len(support_body_names):
        raise RuntimeError(
            "The climb progress contact sensor body selection must match support_body_names: "
            f"{len(contact_sensor_cfg.body_ids)} != {len(support_body_names)}."
        )
    contact_force = torch.linalg.vector_norm(
        contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids], dim=-1
    )
    contact_time = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids]
    force_score = 1.0 - torch.exp(-contact_force / min_contact_force)
    time_score = (contact_time / contact_time_scale).clamp(min=0.0, max=1.0)
    return (inside * height_score * force_score * time_score).clamp(min=0.0, max=1.0)


def climb_platform_progress(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    support_body_names: list[str],
    lift_body_names: list[str],
    contact_sensor_cfg: SceneEntityCfg,
    approach_distance: float = 0.45,
    approach_lateral_margin: float = 0.15,
    support_xy_margin: float = 0.12,
    support_height_std: float = 0.12,
    lift_height_window: float = 0.25,
    min_contact_force: float = 10.0,
    contact_time_scale: float = 0.25,
    approach_side: float = -1.0,
    approach_phase_end: float = 0.70,
    lift_phase_start: float = 0.30,
    lift_phase_end: float = 0.75,
    approach_weight: float = 0.65,
    lift_weight: float = 0.35,
    max_delta_per_step: float = 0.05,
    first_foothold_params: Mapping[str, object] | None = None,
    platform_support_params: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Give conservative progress shaping for the physical climb geometry.

    The approach term saturates at the platform's near edge, so it cannot
    reward driving through the box.  The lift term uses the *sampled* platform
    top height and is gated by an actual foot-support score and
    a broad expert-motion phase window. Both terms reward only newly reached
    episode-best progress. Their historical maxima are reset on every
    episode, so losing and regaining the same contact cannot farm reward. This
    provides a small directional hint without replacing motion tracking or
    encouraging a ballistic jump.
    """

    if approach_distance <= 0.0 or approach_lateral_margin <= 0.0 or lift_height_window <= 0.0:
        raise ValueError("approach_distance, approach_lateral_margin, and lift_height_window must be positive.")
    if not lift_body_names:
        raise ValueError("lift_body_names must contain at least one body name.")
    if len(lift_body_names) != len(support_body_names):
        raise ValueError(
            "lift_body_names and support_body_names must have the same length so each height signal "
            "can be paired with its own contact gate."
        )
    if approach_side not in (-1.0, 1.0):
        raise ValueError(f"approach_side must be -1.0 or 1.0, got {approach_side}.")
    if not 0.15 < approach_phase_end <= 1.0:
        raise ValueError(f"approach_phase_end must lie in (0.15, 1], got {approach_phase_end}.")
    if not 0.0 <= lift_phase_start < lift_phase_end < 1.0:
        raise ValueError(
            f"lift phase window must satisfy 0 <= start < end < 1, got {(lift_phase_start, lift_phase_end)}."
        )
    if approach_weight < 0.0 or lift_weight < 0.0 or approach_weight + lift_weight <= 0.0:
        raise ValueError("approach_weight and lift_weight must be non-negative and not both zero.")
    if env.step_dt <= 0.0:
        raise ValueError(f"env.step_dt must be positive, got {env.step_dt}.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    platform: RigidObject = env.scene[platform_cfg.name]
    sizes = get_climb_box_sizes(platform, base_size=base_size, device=platform.device)

    # Approach: distance to the near edge on the expert's approach side.  The
    # ELF3 clips approach from negative platform-local x; yaw randomization is
    # handled by the local-coordinate transform above.  The finite horizon
    # keeps distant motion from producing a persistent incentive, and the
    # potential is exactly saturated at the edge (and inside the box).
    anchor_local_x = _platform_local_x(command.robot_anchor_pos_w[:, None, :], platform).squeeze(1)
    anchor_local_y = _platform_local_y(command.robot_anchor_pos_w[:, None, :], platform).squeeze(1)
    edge_local_x = approach_side * 0.5 * sizes[:, 0]
    distance_to_edge = (approach_side * (anchor_local_x - edge_local_x)).clamp(min=0.0)
    approach_fraction = 1.0 - (distance_to_edge / approach_distance).clamp(min=0.0, max=1.0)
    approach_potential = approach_fraction * approach_fraction * (3.0 - 2.0 * approach_fraction)
    lateral_distance = (anchor_local_y.abs() - 0.5 * sizes[:, 1]).clamp(min=0.0)
    lateral_fraction = 1.0 - (lateral_distance / approach_lateral_margin).clamp(min=0.0, max=1.0)
    lateral_gate = lateral_fraction * lateral_fraction * (3.0 - 2.0 * lateral_fraction)
    approach_potential = approach_potential * lateral_gate

    # Lift: use the highest configured foot, but only after a real support body
    # is close to the sampled platform top.  The lower bound is below the top;
    # values above the top are clipped, so jumping higher cannot earn more.
    lift_ids = torch.tensor(
        [command.robot.body_names.index(name) for name in lift_body_names],
        dtype=torch.long,
        device=command.device,
    )
    lift_positions = command.robot.data.body_pos_w[:, lift_ids]
    platform_top = platform.data.root_pos_w[:, 2] + 0.5 * sizes[:, 2]
    lift_progress = (
        (lift_positions[..., 2] - (platform_top[:, None] - lift_height_window)) / lift_height_window
    ).clamp(min=0.0, max=1.0)
    if platform_support_params is not None:
        support = platform_foot_support_state(
            env,
            command.robot,
            command.device,
            platform_cfg,
            base_size,
            platform_support_params,
            min_upward_force=min_contact_force,
            sole_height_tolerance=support_height_std,
        )
        if (
            tuple(support_body_names) != support.settings.foot_body_names
            or tuple(lift_body_names) != support.settings.foot_body_names
        ):
            raise ValueError(
                "Strict climb progress requires support_body_names and lift_body_names to exactly match the two "
                "platform-support feet; hands may assist physically but cannot earn progress credit."
            )
        filtered_time = getattr(command, "platform_foot_filtered_contact_time", None)
        if filtered_time is None:
            raise RuntimeError(
                "Strict climb progress requires command-owned filtered support time; it must be advanced by "
                "MotionCommand update, not by the reward."
            )
        support_scores = platform_foot_support_score(
            support,
            filtered_time,
            min_upward_force=min_contact_force,
            contact_time_scale=contact_time_scale,
            sole_height_tolerance=support_height_std,
        )
    else:
        support_scores = _climb_platform_support_score(
            env,
            command,
            platform,
            sizes,
            support_body_names,
            contact_sensor_cfg,
            support_xy_margin,
            support_height_std,
            min_contact_force,
            contact_time_scale,
        )
    # Legacy generic callers may contain ankles alongside other support
    # bodies.  Tighten those ankle entries with the real sole geometry.  The
    # strict path above already uses that geometry directly.
    if platform_support_params is None and first_foothold_params is not None:
        first_foothold_state = _first_foothold_state(
            env,
            command,
            platform_cfg,
            base_size,
            first_foothold_params,
        )
        foot_geometry_score, _, _, foot_surface_score, foot_surface_valid = _first_foothold_sole_safety(
            first_foothold_state
        )
        foot_support_score = (
            foot_geometry_score * foot_surface_score * foot_surface_valid.to(dtype=foot_surface_score.dtype)
        )
        for foot_index, foot_name in enumerate(first_foothold_state.settings.foot_body_names):
            if foot_name in support_body_names:
                support_index = support_body_names.index(foot_name)
                support_scores[:, support_index] *= foot_support_score[:, foot_index]

    # A body can contribute height only when that same body is physically
    # supported. This prevents one planted foot from rewarding an unsupported
    # jump of the other foot or a hand.
    lift_potential = (lift_progress * support_scores).amax(dim=1)

    phase = _motion_phase(command)
    approach_gate = 1.0 - _smoothstep_window(phase, approach_phase_end - 0.15, approach_phase_end)
    # Keep the progress hint out of the final standing/hold window.  The gate
    # ramps in at the beginning of the lift phase, remains active through the
    # configured end, then fades out smoothly over 0.15 normalized phase.
    lift_gate_in = _smoothstep_window(phase, lift_phase_start, min(lift_phase_start + 0.15, lift_phase_end))
    lift_gate_out = 1.0 - _smoothstep_window(phase, lift_phase_end, min(lift_phase_end + 0.15, 1.0))
    lift_gate = lift_gate_in * lift_gate_out
    approach_delta, lift_delta = command.climb_progress_deltas(
        approach_potential,
        lift_potential,
        max_delta_per_step,
    )
    approach_delta = approach_gate * approach_delta
    lift_delta = lift_gate * lift_delta
    normalization = approach_weight + lift_weight
    # RewardManager multiplies every term by env.step_dt.  Convert the
    # per-step progress increment back to a rate. Since each historical maximum
    # only increases from its reset baseline toward at most one, the cumulative
    # shaping contribution remains bounded even if contact is repeatedly lost.
    return (approach_weight * approach_delta + lift_weight * lift_delta) / (normalization * env.step_dt)


def final_default_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
) -> torch.Tensor:
    """Optional diagnostic reward for default-pose proximity in the final hold.

    The task currently assigns this term zero weight. It remains available for
    controlled experiments, but it does not alter the immutable NPZ command or
    determine whether a climb is functionally successful.
    """

    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.motion_end_hold_steps <= 0:
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    target_joint_pos = asset.data.default_joint_pos[:, joint_ids]
    robot_joint_pos = asset.data.joint_pos[:, joint_ids]
    hold_progress = command.final_hold_progress.to(dtype=robot_joint_pos.dtype)
    mean_squared_error = torch.mean(torch.square(robot_joint_pos - target_joint_pos), dim=1)

    # Scaling by progress confines this optional objective to the extra hold.
    return hold_progress * torch.exp(-mean_squared_error / std**2)


def climb_final_default_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
) -> torch.Tensor:
    """Apply a small default-pose regularizer only during contacted hold."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.motion_end_hold_steps <= 0:
        return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    target_joint_pos = asset.data.default_joint_pos[:, joint_ids]
    robot_joint_pos = asset.data.joint_pos[:, joint_ids]
    contact_gate = _terminal_platform_contact_gate(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
    )
    hold_progress = command.final_hold_progress.to(dtype=robot_joint_pos.dtype)
    mean_squared_error = torch.mean(torch.square(robot_joint_pos - target_joint_pos), dim=1)
    return hold_progress * contact_gate * torch.exp(-mean_squared_error / std**2)


def terminal_default_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    std: float,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    platform_support_params: Mapping[str, object] | None = None,
    min_total_load_fraction: float = 0.0,
) -> torch.Tensor:
    """Track the command's single smooth terminal q target over all joints.

    This is deliberately *not* a direct reward to ``default_joint_pos``.  In
    the 0.8 s transition the sole target is ``command.joint_pos``, which moves
    continuously from the latched real supported q to default. The reward
    starts when that one-way target mode latches (including its source-q
    boundary frame), and receives strict bilateral platform/load gating, so a
    hand-supported hover cannot earn default-pose credit.
    """

    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    command: MotionCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or len(joint_ids) == 0:
        raise RuntimeError("The terminal default-pose reward requires resolved non-empty joint_ids.")
    terminal_mode_active = _terminal_default_pose_latched_gate(command)
    target_joint_pos = command.joint_pos[:, joint_ids]
    robot_joint_pos = asset.data.joint_pos[:, joint_ids]
    mean_squared_error = torch.mean(torch.square(robot_joint_pos - target_joint_pos), dim=1)
    two_foot_support = _terminal_platform_contact_gate(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
    )
    foot_load_score = _platform_foot_load_score(
        env,
        command,
        platform_cfg,
        base_size,
        platform_support_params,
        min_contact_force=min_contact_force,
        foot_height_std=foot_height_std,
        min_total_load_fraction=min_total_load_fraction,
    )
    return terminal_mode_active * two_foot_support * foot_load_score * torch.exp(-mean_squared_error / std**2)


def _final_phase_gate(command: MotionCommand, window_time_s: float, step_dt: float) -> torch.Tensor:
    """Ramp a reward on during the final reference-motion window."""

    if window_time_s <= 0.0:
        raise ValueError(f"window_time_s must be positive, got {window_time_s}.")
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}.")

    final_frames = command.motion.motion_end_idx[command.motion_ids] - 1
    window_steps = max(1, math.ceil(window_time_s / step_dt))
    start_frame = final_frames - (window_steps - 1)
    progress = (command.time_steps - start_frame).to(dtype=torch.float32)
    progress = progress / float(max(1, window_steps - 1))
    progress = progress.clamp(min=0.0, max=1.0)
    return torch.maximum(progress, command.final_hold_progress.to(dtype=progress.dtype))


def _expert_static_tail_gate(
    command: MotionCommand,
    reference_max_joint_speed: float,
    static_window_time_s: float,
    step_dt: float,
    ramp_time_s: float | None = None,
) -> torch.Tensor:
    """Return a smooth gate for the stationary tail of the reference clip.

    A reference motion can contain a static pose at its beginning as well as
    at its end.  The final-window requirement deliberately excludes the
    initial stand, while the reference-speed check makes the gate robust to a
    short moving transition inside that window.  During MotionCommand's extra
    final-frame hold, :func:`_final_phase_gate` remains one.
    """

    if not math.isfinite(reference_max_joint_speed) or reference_max_joint_speed < 0.0:
        raise ValueError(
            "reference_max_joint_speed must be finite and non-negative, "
            f"got {reference_max_joint_speed}."
        )
    if not math.isfinite(static_window_time_s) or static_window_time_s <= 0.0:
        raise ValueError(f"static_window_time_s must be finite and positive, got {static_window_time_s}.")
    if not math.isfinite(step_dt) or step_dt <= 0.0:
        raise ValueError(f"step_dt must be finite and positive, got {step_dt}.")
    if ramp_time_s is None:
        ramp_time_s = static_window_time_s
    if not math.isfinite(ramp_time_s) or ramp_time_s <= 0.0:
        raise ValueError(f"ramp_time_s must be finite and positive, got {ramp_time_s}.")
    if ramp_time_s > static_window_time_s:
        raise ValueError(
            "ramp_time_s cannot exceed static_window_time_s, "
            f"got {ramp_time_s} and {static_window_time_s}."
        )

    final_frames = command.motion.motion_end_idx[command.motion_ids] - 1
    window_steps = max(1, math.ceil(static_window_time_s / step_dt))
    ramp_steps = max(1, math.ceil(ramp_time_s / step_dt))
    start_frames = final_frames - (window_steps - 1)
    ramp_progress = (command.time_steps - start_frames).to(dtype=torch.float32)
    ramp_progress = ramp_progress / float(max(1, ramp_steps - 1))
    final_window_ramp = ramp_progress.clamp(min=0.0, max=1.0)
    final_window_ramp = torch.maximum(
        final_window_ramp,
        command.final_hold_progress.to(dtype=final_window_ramp.dtype),
    )

    reference_speed = torch.max(torch.abs(command.joint_vel), dim=1).values
    reference_is_static = reference_speed <= reference_max_joint_speed
    return final_window_ramp * reference_is_static.to(
        dtype=command.joint_pos.dtype
    )


def _validate_terminal_joint_groups(
    command: MotionCommand,
    joint_groups: Mapping[str, Sequence[str]],
    group_stds: Mapping[str, float],
    group_weights: Mapping[str, float],
) -> dict[str, torch.Tensor]:
    """Resolve and validate disjoint terminal joint groups in robot order."""

    if not joint_groups:
        raise ValueError("joint_groups must contain at least one group.")
    group_names = tuple(joint_groups)
    if set(group_stds) != set(group_names):
        raise ValueError("group_stds keys must exactly match joint_groups keys.")
    if set(group_weights) != set(group_names):
        raise ValueError("group_weights keys must exactly match joint_groups keys.")
    if any(
        not math.isfinite(float(group_stds[name])) or float(group_stds[name]) <= 0.0
        for name in group_names
    ):
        raise ValueError("Every terminal joint-group std must be positive.")
    if any(
        not math.isfinite(float(group_weights[name])) or float(group_weights[name]) < 0.0
        for name in group_names
    ):
        raise ValueError("Terminal joint-group weights must be non-negative.")
    if sum(float(group_weights[name]) for name in group_names) <= 0.0:
        raise ValueError("At least one terminal joint-group weight must be positive.")

    joint_names = tuple(command.robot.joint_names)
    name_to_id = {name: joint_id for joint_id, name in enumerate(joint_names)}
    if len(name_to_id) != len(joint_names):
        raise RuntimeError("Robot joint names must be unique for grouped terminal rewards.")

    group_signature = tuple((name, tuple(names)) for name, names in joint_groups.items())
    cached_groups = getattr(command, "_terminal_reward_joint_group_ids", None)
    if cached_groups is not None and cached_groups[0] == group_signature:
        return cached_groups[1]

    resolved: dict[str, torch.Tensor] = {}
    assigned_names: set[str] = set()
    device = command.robot_joint_pos.device
    for group_name, names in joint_groups.items():
        names = tuple(names)
        if not names:
            raise ValueError(f"Terminal joint group {group_name!r} must not be empty.")
        duplicates = assigned_names.intersection(names)
        if duplicates:
            raise ValueError(
                f"Terminal joint groups must be disjoint; repeated joints: {sorted(duplicates)}."
            )
        missing = [name for name in names if name not in name_to_id]
        if missing:
            raise RuntimeError(
                f"Terminal joint group {group_name!r} contains unknown robot joints: {missing}."
            )
        assigned_names.update(names)
        resolved[group_name] = torch.tensor(
            [name_to_id[name] for name in names], dtype=torch.long, device=device
        )
    # MotionCommand and its robot device/order are fixed for the lifetime of
    # an environment.  Cache CUDA index tensors instead of reallocating all
    # seven groups on every reward evaluation.
    command._terminal_reward_joint_group_ids = (group_signature, resolved)
    return resolved


class _GroupedJointScoreDetails(NamedTuple):
    """Aggregate score plus per-group diagnostics for a terminal joint objective."""

    aggregate: torch.Tensor
    rms: dict[str, torch.Tensor]
    maximum: dict[str, torch.Tensor]
    mean_score: dict[str, torch.Tensor]
    worst_score: dict[str, torch.Tensor]
    group_score: dict[str, torch.Tensor]


def _validate_score_exponent(score_exponent: float | None) -> float | None:
    """Validate an optional inverse-power exponent.

    ``None`` deliberately selects the legacy group-RMS inverse-quadratic
    calculation.  An explicit value enables per-joint inverse-power scores;
    in particular, ``0.5`` is the intended inverse-square-root heavy tail.
    """

    if score_exponent is None:
        return None
    if isinstance(score_exponent, bool):
        raise TypeError("score_exponent must be a positive finite scalar or None.")
    try:
        exponent = float(score_exponent)
    except (TypeError, ValueError) as exc:
        raise TypeError("score_exponent must be a positive finite scalar or None.") from exc
    if not math.isfinite(exponent) or exponent <= 0.0:
        raise ValueError(f"score_exponent must be positive and finite, got {score_exponent}.")
    return exponent


def _resolve_group_option(
    value: int | float | Mapping[str, int] | Mapping[str, float],
    group_names: Sequence[str],
    *,
    option_name: str,
) -> dict[str, int | float]:
    """Expand a scalar or exact-key mapping into one value per joint group."""

    if isinstance(value, Mapping):
        if set(value) != set(group_names):
            raise ValueError(f"{option_name} mapping keys must exactly match joint_groups keys.")
        return {name: value[name] for name in group_names}
    return {name: value for name in group_names}


def _terminal_group_scoring_options(
    joint_groups: Mapping[str, Sequence[str]],
    score_exponent: float | None,
    worst_joint_count: int | Mapping[str, int],
    worst_joint_weight: float | Mapping[str, float],
    group_aggregation: str,
) -> tuple[float | None, dict[str, int], dict[str, float]]:
    """Validate configurable within-group and across-group score aggregation."""

    exponent = _validate_score_exponent(score_exponent)
    group_names = tuple(joint_groups)
    raw_counts = _resolve_group_option(
        worst_joint_count, group_names, option_name="worst_joint_count"
    )
    raw_weights = _resolve_group_option(
        worst_joint_weight, group_names, option_name="worst_joint_weight"
    )
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for group_name, joint_names in joint_groups.items():
        count = raw_counts[group_name]
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(
                "worst_joint_count values must be integers, "
                f"got {count!r} for group {group_name!r}."
            )
        if count < 0 or count > len(joint_names):
            raise ValueError(
                f"worst_joint_count for group {group_name!r} must be in "
                f"[0, {len(joint_names)}], got {count}."
            )
        try:
            weight = float(raw_weights[group_name])
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "worst_joint_weight values must be finite scalars in [0, 1]."
            ) from exc
        if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError(
                f"worst_joint_weight for group {group_name!r} must be in [0, 1], got {weight}."
            )
        if weight > 0.0 and count == 0:
            raise ValueError(
                f"worst_joint_count for group {group_name!r} must be positive when "
                "worst_joint_weight is positive."
            )
        counts[group_name] = count
        weights[group_name] = weight

    if group_aggregation not in ("arithmetic", "harmonic"):
        raise ValueError(
            "group_aggregation must be either 'arithmetic' or 'harmonic', "
            f"got {group_aggregation!r}."
        )
    if exponent is None and any(counts.values()):
        raise ValueError(
            "An explicit score_exponent is required when worst_joint_count is non-zero."
        )
    return exponent, counts, weights


def _inverse_power_score(error: torch.Tensor, std: float, exponent: float) -> torch.Tensor:
    """Return ``(1 + (error / std)^2)^(-exponent)`` without exponential saturation."""

    base = 1.0 + torch.square(error / std)
    if exponent == 0.5:
        return torch.rsqrt(base)
    return torch.pow(base, -exponent)


def _aggregate_terminal_group_scores(
    group_scores: Mapping[str, torch.Tensor],
    group_weights: Mapping[str, float],
    group_aggregation: str,
    worst_group_weight: float = 0.0,
) -> torch.Tensor:
    """Combine group scores while retaining an explicit worst-group signal.

    ``worst_group_weight=0`` preserves the historical weighted aggregation.
    A positive value blends in the lowest anatomical-group score regardless
    of that group's normal averaging weight.  This prevents a single rotated
    leg or folded arm from being diluted by four already-correct groups.
    """

    if isinstance(worst_group_weight, bool):
        raise TypeError("worst_group_weight must be a finite scalar in [0, 1].")
    try:
        worst_weight = float(worst_group_weight)
    except (TypeError, ValueError) as exc:
        raise TypeError("worst_group_weight must be a finite scalar in [0, 1].") from exc
    if not math.isfinite(worst_weight) or not 0.0 <= worst_weight <= 1.0:
        raise ValueError(
            f"worst_group_weight must be finite and in [0, 1], got {worst_group_weight}."
        )

    group_names = tuple(group_scores)
    active_group_names = tuple(
        name for name in group_names if float(group_weights[name]) > 0.0
    )
    weight_sum = sum(float(group_weights[name]) for name in active_group_names)
    if group_aggregation == "arithmetic":
        weighted_score = sum(
            (float(group_weights[name]) / weight_sum) * group_scores[name]
            for name in active_group_names
        )
    else:
        # Inverse-power scores are strictly positive for finite errors.  The
        # tiny clamp only protects a numerical underflow/overflow edge and
        # preserves an exact score of one when every group is perfect.
        reference = group_scores[active_group_names[0]]
        denominator = torch.zeros_like(reference)
        tiny = torch.finfo(reference.dtype).tiny
        for name in active_group_names:
            weight = float(group_weights[name])
            denominator += weight / group_scores[name].clamp_min(tiny)
        weighted_score = weight_sum / denominator

    if worst_weight == 0.0:
        return weighted_score
    worst_group_score = torch.stack(
        [group_scores[name] for name in active_group_names], dim=1
    ).amin(dim=1)
    return (1.0 - worst_weight) * weighted_score + worst_weight * worst_group_score


def _grouped_joint_score_details(
    values: torch.Tensor,
    target: torch.Tensor,
    joint_groups: Mapping[str, Sequence[str]],
    group_ids: Mapping[str, torch.Tensor],
    group_stds: Mapping[str, float],
    group_weights: Mapping[str, float],
    *,
    score_exponent: float | None,
    worst_joint_count: int | Mapping[str, int],
    worst_joint_weight: float | Mapping[str, float],
    group_aggregation: str,
    worst_group_weight: float = 0.0,
) -> _GroupedJointScoreDetails:
    """Score grouped joint errors with legacy or explicit heavy-tail semantics."""

    exponent, worst_counts, worst_weights = _terminal_group_scoring_options(
        joint_groups,
        score_exponent,
        worst_joint_count,
        worst_joint_weight,
        group_aggregation,
    )
    rms: dict[str, torch.Tensor] = {}
    maximum: dict[str, torch.Tensor] = {}
    mean_scores: dict[str, torch.Tensor] = {}
    worst_scores: dict[str, torch.Tensor] = {}
    group_scores: dict[str, torch.Tensor] = {}
    for group_name, joint_ids in group_ids.items():
        absolute_error = torch.abs(values[:, joint_ids] - target[:, joint_ids])
        mean_squared_error = torch.mean(torch.square(absolute_error), dim=1)
        rms[group_name] = torch.sqrt(mean_squared_error)
        maximum[group_name] = torch.max(absolute_error, dim=1).values
        std = float(group_stds[group_name])

        if exponent is None:
            # Exact pre-existing behavior for callers that omit every new
            # scoring option: inverse quadratic of the group RMS.
            legacy_score = torch.reciprocal(1.0 + mean_squared_error / std**2)
            mean_scores[group_name] = legacy_score
            worst_scores[group_name] = legacy_score
            group_scores[group_name] = legacy_score
            continue

        joint_scores = _inverse_power_score(absolute_error, std, exponent)
        mean_score = torch.mean(joint_scores, dim=1)
        count = worst_counts[group_name]
        worst_score = (
            torch.topk(joint_scores, k=count, dim=1, largest=False).values.mean(dim=1)
            if count > 0
            else mean_score
        )
        worst_weight = worst_weights[group_name]
        mean_scores[group_name] = mean_score
        worst_scores[group_name] = worst_score
        group_scores[group_name] = (1.0 - worst_weight) * mean_score + worst_weight * worst_score

    aggregate = _aggregate_terminal_group_scores(
        group_scores,
        group_weights,
        group_aggregation,
        worst_group_weight,
    )
    return _GroupedJointScoreDetails(
        aggregate, rms, maximum, mean_scores, worst_scores, group_scores
    )


def _grouped_expert_joint_pose_score_details(
    command: MotionCommand,
    joint_groups: Mapping[str, Sequence[str]],
    group_stds: Mapping[str, float],
    group_weights: Mapping[str, float],
    *,
    score_exponent: float | None = None,
    worst_joint_count: int | Mapping[str, int] = 0,
    worst_joint_weight: float | Mapping[str, float] = 0.0,
    group_aggregation: str = "arithmetic",
    worst_group_weight: float = 0.0,
) -> _GroupedJointScoreDetails:
    """Return complete grouped expert-pose score diagnostics.

    Omitting ``score_exponent`` preserves the historical inverse quadratic of
    each group's RMS error.  Supplying an exponent scores every joint first;
    ``0.5`` selects the intended inverse-square-root heavy tail.  The optional
    worst-k blend exposes isolated shoulder, wrist, hip, or ankle failures
    that a plain group mean could otherwise dilute.
    """

    group_ids = _validate_terminal_joint_groups(command, joint_groups, group_stds, group_weights)
    robot_joint_pos = command.robot_joint_pos
    target_joint_pos = getattr(command, "source_joint_pos", None)
    if target_joint_pos is None:
        target_joint_pos = getattr(command, "joint_pos", None)
    if target_joint_pos is None or target_joint_pos.shape != robot_joint_pos.shape:
        target_shape = None if target_joint_pos is None else target_joint_pos.shape
        raise RuntimeError(
            "Grouped terminal pose reward requires matching expert and robot joint tensors, "
            f"got {target_shape} and {robot_joint_pos.shape}."
        )

    return _grouped_joint_score_details(
        robot_joint_pos,
        target_joint_pos,
        joint_groups,
        group_ids,
        group_stds,
        group_weights,
        score_exponent=score_exponent,
        worst_joint_count=worst_joint_count,
        worst_joint_weight=worst_joint_weight,
        group_aggregation=group_aggregation,
        worst_group_weight=worst_group_weight,
    )


def _grouped_expert_joint_pose_score(
    command: MotionCommand,
    joint_groups: Mapping[str, Sequence[str]],
    group_stds: Mapping[str, float],
    group_weights: Mapping[str, float],
    *,
    score_exponent: float | None = None,
    worst_joint_count: int | Mapping[str, int] = 0,
    worst_joint_weight: float | Mapping[str, float] = 0.0,
    group_aggregation: str = "arithmetic",
    worst_group_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return a grouped expert-pose score with its legacy RMS/max diagnostics."""

    details = _grouped_expert_joint_pose_score_details(
        command,
        joint_groups,
        group_stds,
        group_weights,
        score_exponent=score_exponent,
        worst_joint_count=worst_joint_count,
        worst_joint_weight=worst_joint_weight,
        group_aggregation=group_aggregation,
        worst_group_weight=worst_group_weight,
    )
    return details.aggregate, details.rms, details.maximum


def _grouped_actual_joint_speed_score_details(
    command: MotionCommand,
    joint_groups: Mapping[str, Sequence[str]],
    group_stds: Mapping[str, float],
    group_weights: Mapping[str, float],
    *,
    score_exponent: float | None = None,
    worst_joint_count: int | Mapping[str, int] = 0,
    worst_joint_weight: float | Mapping[str, float] = 0.0,
    group_aggregation: str = "arithmetic",
    worst_group_weight: float = 0.0,
) -> _GroupedJointScoreDetails:
    """Return complete grouped score diagnostics for measured joint speeds."""

    group_ids = _validate_terminal_joint_groups(command, joint_groups, group_stds, group_weights)
    robot_joint_vel = command.robot_joint_vel
    if robot_joint_vel.shape != command.robot_joint_pos.shape:
        raise RuntimeError(
            "Robot joint position and velocity tensors must have the same shape, "
            f"got {command.robot_joint_pos.shape} and {robot_joint_vel.shape}."
        )

    return _grouped_joint_score_details(
        robot_joint_vel,
        torch.zeros_like(robot_joint_vel),
        joint_groups,
        group_ids,
        group_stds,
        group_weights,
        score_exponent=score_exponent,
        worst_joint_count=worst_joint_count,
        worst_joint_weight=worst_joint_weight,
        group_aggregation=group_aggregation,
        worst_group_weight=worst_group_weight,
    )


def _grouped_actual_joint_speed_score(
    command: MotionCommand,
    joint_groups: Mapping[str, Sequence[str]],
    group_stds: Mapping[str, float],
    group_weights: Mapping[str, float],
    *,
    score_exponent: float | None = None,
    worst_joint_count: int | Mapping[str, int] = 0,
    worst_joint_weight: float | Mapping[str, float] = 0.0,
    group_aggregation: str = "arithmetic",
    worst_group_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return a grouped measured-speed score with legacy RMS/max diagnostics."""

    details = _grouped_actual_joint_speed_score_details(
        command,
        joint_groups,
        group_stds,
        group_weights,
        score_exponent=score_exponent,
        worst_joint_count=worst_joint_count,
        worst_joint_weight=worst_joint_weight,
        group_aggregation=group_aggregation,
        worst_group_weight=worst_group_weight,
    )
    return details.aggregate, details.rms, details.maximum


def _terminal_platform_alignment_gate(command: MotionCommand) -> torch.Tensor:
    """Return one only after the optional terminal platform Z bridge finishes.

    Older/generic motion commands do not configure the correction.  They are
    treated as already aligned so this climb-only gate remains backward
    compatible with the base tracking tasks and lightweight unit-test mocks.
    """

    completed = getattr(command, "terminal_platform_alignment_complete", None)
    if completed is None:
        return torch.ones_like(command.time_steps, dtype=torch.float32)
    if completed.shape != command.time_steps.shape:
        raise RuntimeError(
            "terminal_platform_alignment_complete must match command time_steps, "
            f"got {completed.shape} and {command.time_steps.shape}."
        )
    return completed.to(dtype=torch.float32)


def final_expert_upper_body_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    reference_max_joint_speed: float,
    static_window_time_s: float,
    std: float,
) -> torch.Tensor:
    """Track the stationary expert's waist and arms after two-foot contact.

    This is intentionally a reference-motion objective, not a default-pose
    regularizer.  The caller selects only the waist and arm joints through
    ``asset_cfg``; legs remain free to adapt to the randomized platform
    height through physical contact and the existing stability terms.
    """

    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or len(joint_ids) == 0:
        raise RuntimeError("The terminal expert upper-body reward requires resolved non-empty joint_ids.")

    target_joint_pos = command.joint_pos[:, joint_ids]
    robot_joint_pos = asset.data.joint_pos[:, joint_ids]
    mean_squared_error = torch.mean(torch.square(robot_joint_pos - target_joint_pos), dim=1)
    two_foot_contact = _terminal_platform_contact_gate(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
    )
    static_tail = _expert_static_tail_gate(
        command,
        reference_max_joint_speed,
        static_window_time_s,
        env.step_dt,
    )
    alignment_complete = _terminal_platform_alignment_gate(command)
    return alignment_complete * static_tail * two_foot_contact * torch.exp(-mean_squared_error / std**2)


def _platform_foot_contact_scores(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    platform_support_params: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Return one physical contact score per configured foot in ``[0, 1]``."""

    if not foot_body_names:
        raise ValueError("foot_body_names must contain at least one body.")
    if footprint_inset < 0.0:
        raise ValueError(f"footprint_inset must be non-negative, got {footprint_inset}.")
    for name, value in (
        ("foot_height_std", foot_height_std),
        ("min_contact_force", min_contact_force),
        ("contact_time_scale", contact_time_scale),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")

    if platform_support_params is not None:
        support = platform_foot_support_state(
            env,
            command.robot,
            command.device,
            platform_cfg,
            base_size,
            platform_support_params,
            min_upward_force=min_contact_force,
            sole_height_tolerance=foot_height_std,
        )
        if tuple(foot_body_names) != support.settings.foot_body_names:
            raise ValueError(
                "foot_body_names must exactly match platform_support_params['foot_body_names'] so all terminal "
                "terms share one physical support definition."
            )
        filtered_time = getattr(command, "platform_foot_filtered_contact_time", None)
        if filtered_time is None:
            raise RuntimeError(
                "Strict platform-foot scores require MotionCommand.platform_foot_filtered_contact_time; "
                "advance it once from command update, not from rewards."
            )
        return platform_foot_support_score(
            support,
            filtered_time,
            min_upward_force=min_contact_force,
            contact_time_scale=contact_time_scale,
            sole_height_tolerance=foot_height_std,
        )

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
    feet_inside = points_inside_oriented_box_xy(
        foot_positions,
        platform.data.root_pos_w,
        platform.data.root_quat_w,
        footprint_sizes,
    ).to(dtype=foot_positions.dtype)

    platform_top = platform.data.root_pos_w[:, None, 2] + 0.5 * sizes[:, None, 2]
    height_error = foot_positions[..., 2] - platform_top
    height_score = torch.exp(-0.5 * torch.square(height_error / foot_height_std))

    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
    if contact_sensor_cfg.body_ids is None:
        raise RuntimeError("The platform contact reward requires resolved contact sensor body_ids.")
    if contact_sensor.data.net_forces_w is None or contact_sensor.data.current_contact_time is None:
        raise RuntimeError("The platform contact sensor must provide net forces and current contact time.")
    contact_force = torch.linalg.vector_norm(
        contact_sensor.data.net_forces_w[:, contact_sensor_cfg.body_ids], dim=-1
    )
    contact_time = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids]
    force_score = 1.0 - torch.exp(-contact_force / min_contact_force)
    time_score = (contact_time / contact_time_scale).clamp(min=0.0, max=1.0)
    return feet_inside * height_score * force_score * time_score


def _platform_foot_load_score(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    platform_support_params: Mapping[str, object] | None,
    *,
    min_contact_force: float,
    foot_height_std: float,
    min_total_load_fraction: float,
) -> torch.Tensor:
    """Return foot-borne load share for strict terminal terms.

    A zero requested fraction remains neutral for generic/backward-compatible
    callers.  Climb passes 0.50, preventing the two feet from merely touching
    the platform while a wrist carries most of the robot.
    """

    if platform_support_params is None or min_total_load_fraction == 0.0:
        reference = command.joint_pos if hasattr(command, "joint_pos") else command.robot_joint_vel
        return torch.ones(command.time_steps.shape, dtype=reference.dtype, device=reference.device)
    support = platform_foot_support_state(
        env,
        command.robot,
        command.device,
        platform_cfg,
        base_size,
        platform_support_params,
        min_upward_force=min_contact_force,
        sole_height_tolerance=foot_height_std,
    )
    return platform_foot_load_score(
        support,
        command.robot,
        min_total_load_fraction=min_total_load_fraction,
    )


def _expert_joint_pose_score(command: MotionCommand, std: float) -> torch.Tensor:
    """Return a broad all-joint score around the immutable expert pose.

    The source joint configuration is invariant to the terminal whole-body Z
    translation, so it remains a consistent natural-pose target for every
    platform height.  A broad Gaussian is intentional: the policy may retain
    small physically useful deviations, but crossed legs or folded/raised
    arms cannot maximize a terminal settling objective merely by stopping.
    """

    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    robot_joint_pos = command.robot_joint_pos
    target_joint_pos = getattr(command, "source_joint_pos", None)
    if target_joint_pos is None:
        target_joint_pos = getattr(command, "joint_pos", None)
    if target_joint_pos is None:
        raise RuntimeError("The expert joint-pose score requires source_joint_pos or joint_pos.")
    if target_joint_pos.shape != robot_joint_pos.shape:
        raise RuntimeError(
            "Expert and robot joint-position tensors must have the same shape, "
            f"got {target_joint_pos.shape} and {robot_joint_pos.shape}."
        )
    mean_squared_error = torch.mean(torch.square(robot_joint_pos - target_joint_pos), dim=1)
    return torch.exp(-mean_squared_error / std**2)


def _terminal_expert_support_gate(
    env: ManagerBasedRLEnv,
    command: MotionCommand,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    platform_support_params: Mapping[str, object] | None,
    min_total_load_fraction: float,
    support_floor: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a soft terminal support gate and its strict physical score."""

    if not 0.0 <= support_floor <= 1.0:
        raise ValueError(f"support_floor must be in [0, 1], got {support_floor}.")
    per_foot_scores = _platform_foot_contact_scores(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
    )
    two_foot_support = per_foot_scores.amin(dim=1).clamp(min=0.0, max=1.0)
    foot_load_score = _platform_foot_load_score(
        env,
        command,
        platform_cfg,
        base_size,
        platform_support_params,
        min_contact_force=min_contact_force,
        foot_height_std=foot_height_std,
        min_total_load_fraction=min_total_load_fraction,
    ).clamp(min=0.0, max=1.0)
    strict_support = two_foot_support * foot_load_score
    # With a full floor the expert correction is intentionally independent of
    # support.  Return an exact constant instead of evaluating ``0 * score``:
    # IEEE arithmetic would otherwise propagate a diagnostic NaN into a reward
    # that is configured not to depend on that diagnostic at all.
    if support_floor == 1.0:
        soft_support = torch.ones_like(strict_support)
    else:
        soft_support = support_floor + (1.0 - support_floor) * strict_support
    return soft_support, strict_support


def _record_final_tail_group_metrics(
    command: MotionCommand,
    static_tail: torch.Tensor,
    *,
    strict_support: torch.Tensor | None = None,
    torso_orientation_error: torch.Tensor | None = None,
    torso_linear_velocity_error: torch.Tensor | None = None,
    torso_angular_velocity_error: torch.Tensor | None = None,
    torso_angular_speed: torch.Tensor | None = None,
    root_orientation_score: torch.Tensor | None = None,
    root_linear_velocity_score: torch.Tensor | None = None,
    root_angular_velocity_score: torch.Tensor | None = None,
    pose_quality: torch.Tensor | None = None,
    speed_quality: torch.Tensor | None = None,
    speed_pose_quality: torch.Tensor | None = None,
    speed_pose_modulation: torch.Tensor | None = None,
    pose_rms: Mapping[str, torch.Tensor] | None = None,
    pose_max: Mapping[str, torch.Tensor] | None = None,
    pose_mean_score: Mapping[str, torch.Tensor] | None = None,
    pose_worst_score: Mapping[str, torch.Tensor] | None = None,
    pose_group_score: Mapping[str, torch.Tensor] | None = None,
    speed_rms: Mapping[str, torch.Tensor] | None = None,
    speed_max: Mapping[str, torch.Tensor] | None = None,
    speed_mean_score: Mapping[str, torch.Tensor] | None = None,
    speed_worst_score: Mapping[str, torch.Tensor] | None = None,
    speed_group_score: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Publish read-only final-tail diagnostics when a command owns metrics."""

    metrics = getattr(command, "metrics", None)
    if metrics is None:
        return
    active = (static_tail > 0.0).to(dtype=static_tail.dtype)

    def record(name: str, value: torch.Tensor) -> None:
        metric = metrics.setdefault(name, torch.zeros_like(value))
        if metric.shape != value.shape:
            raise RuntimeError(
                f"Terminal diagnostic {name!r} has shape {metric.shape}, expected {value.shape}."
            )
        metric.copy_(active * value)

    if strict_support is not None:
        record("final_tail_strict_support", strict_support)
    if torso_orientation_error is not None:
        record("final_tail_torso_orientation_error", torso_orientation_error)
    if torso_linear_velocity_error is not None:
        record("final_tail_torso_linear_velocity_error", torso_linear_velocity_error)
    if torso_angular_velocity_error is not None:
        record("final_tail_torso_angular_velocity_error", torso_angular_velocity_error)
    if torso_angular_speed is not None:
        record("final_tail_torso_angular_speed", torso_angular_speed)
    if root_orientation_score is not None:
        record("final_tail_root_orientation_score", root_orientation_score)
    if root_linear_velocity_score is not None:
        record("final_tail_root_linear_velocity_score", root_linear_velocity_score)
    if root_angular_velocity_score is not None:
        record("final_tail_root_angular_velocity_score", root_angular_velocity_score)
    if pose_quality is not None:
        record("final_tail_pose_quality", pose_quality)
    if speed_quality is not None:
        record("final_tail_speed_quality", speed_quality)
    if speed_pose_quality is not None:
        record("final_tail_speed_pose_quality", speed_pose_quality)
    if speed_pose_modulation is not None:
        record("final_tail_speed_pose_modulation", speed_pose_modulation)
    if pose_rms is not None:
        for group_name, value in pose_rms.items():
            record(f"final_tail_{group_name}_pose_rms", value)
    if pose_max is not None:
        for group_name, value in pose_max.items():
            record(f"final_tail_{group_name}_max_pose_error", value)
    if pose_mean_score is not None:
        for group_name, value in pose_mean_score.items():
            record(f"final_tail_{group_name}_pose_mean_score", value)
    if pose_worst_score is not None:
        for group_name, value in pose_worst_score.items():
            record(f"final_tail_{group_name}_pose_worst_score", value)
    if pose_group_score is not None:
        for group_name, value in pose_group_score.items():
            record(f"final_tail_{group_name}_pose_group_score", value)
    if speed_rms is not None:
        for group_name, value in speed_rms.items():
            record(f"final_tail_{group_name}_joint_speed_rms", value)
    if speed_max is not None:
        for group_name, value in speed_max.items():
            record(f"final_tail_{group_name}_max_joint_speed", value)
    if speed_mean_score is not None:
        for group_name, value in speed_mean_score.items():
            record(f"final_tail_{group_name}_speed_mean_score", value)
    if speed_worst_score is not None:
        for group_name, value in speed_worst_score.items():
            record(f"final_tail_{group_name}_speed_worst_score", value)
    if speed_group_score is not None:
        for group_name, value in speed_group_score.items():
            record(f"final_tail_{group_name}_speed_group_score", value)


def final_grouped_expert_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    reference_max_joint_speed: float,
    static_window_time_s: float,
    ramp_time_s: float,
    joint_groups: Mapping[str, Sequence[str]],
    group_stds: Mapping[str, float],
    group_weights: Mapping[str, float],
    support_floor: float,
    platform_support_params: Mapping[str, object] | None = None,
    min_total_load_fraction: float = 0.0,
    score_exponent: float | None = None,
    worst_joint_count: int | Mapping[str, int] = 0,
    worst_joint_weight: float | Mapping[str, float] = 0.0,
    group_aggregation: str = "arithmetic",
    worst_group_weight: float = 0.0,
) -> torch.Tensor:
    """Track expert terminal posture by group during the authored static tail.

    Waist and each arm are scored independently so a local abnormal pose can
    no longer hide inside a 29-joint mean.  Legs are also scored explicitly,
    but their configured weight and tolerance can remain looser so physical
    ankle, knee and hip balance corrections are not suppressed.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    pose_details = _grouped_expert_joint_pose_score_details(
        command,
        joint_groups,
        group_stds,
        group_weights,
        score_exponent=score_exponent,
        worst_joint_count=worst_joint_count,
        worst_joint_weight=worst_joint_weight,
        group_aggregation=group_aggregation,
        worst_group_weight=worst_group_weight,
    )
    pose_score = pose_details.aggregate
    soft_support, strict_support = _terminal_expert_support_gate(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
        min_total_load_fraction,
        support_floor,
    )
    static_tail = _expert_static_tail_gate(
        command,
        reference_max_joint_speed,
        static_window_time_s,
        env.step_dt,
        ramp_time_s,
    )
    torso_orientation_error = quat_error_magnitude(
        command.anchor_quat_w, command.robot_anchor_quat_w
    )
    torso_angular_speed = torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=1)
    _record_final_tail_group_metrics(
        command,
        static_tail,
        strict_support=strict_support,
        torso_orientation_error=torso_orientation_error,
        torso_angular_speed=torso_angular_speed,
        pose_quality=pose_score,
        pose_rms=pose_details.rms,
        pose_max=pose_details.maximum,
        pose_mean_score=pose_details.mean_score,
        pose_worst_score=pose_details.worst_score,
        pose_group_score=pose_details.group_score,
    )
    alignment_complete = _terminal_platform_alignment_gate(command).to(dtype=pose_score.dtype)
    return alignment_complete * static_tail * soft_support * pose_score


def final_grouped_actual_joint_velocity_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    reference_max_joint_speed: float,
    static_window_time_s: float,
    ramp_time_s: float,
    joint_groups: Mapping[str, Sequence[str]],
    group_stds: Mapping[str, float],
    group_weights: Mapping[str, float],
    support_floor: float,
    platform_support_params: Mapping[str, object] | None = None,
    min_total_load_fraction: float = 0.0,
    score_exponent: float | None = None,
    worst_joint_count: int | Mapping[str, int] = 0,
    worst_joint_weight: float | Mapping[str, float] = 0.0,
    group_aggregation: str = "arithmetic",
    worst_group_weight: float = 0.0,
    pose_quality_floor: float | None = None,
    pose_quality_group_stds: Mapping[str, float] | None = None,
    pose_quality_group_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    """Settle measured joint velocities, optionally conditioned on expert-pose quality.

    ``pose_quality_floor=None`` is the backward-compatible unmodulated mode.
    An explicit floor in ``[0, 1]`` multiplies the speed score by
    ``floor + (1 - floor) * pose_quality``.  Pose tracking itself remains
    unconditional, so a policy cannot avoid it by continuing to move.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    velocity_details = _grouped_actual_joint_speed_score_details(
        command,
        joint_groups,
        group_stds,
        group_weights,
        score_exponent=score_exponent,
        worst_joint_count=worst_joint_count,
        worst_joint_weight=worst_joint_weight,
        group_aggregation=group_aggregation,
        worst_group_weight=worst_group_weight,
    )
    velocity_score = velocity_details.aggregate
    pose_quality: torch.Tensor | None = None
    if pose_quality_floor is None:
        pose_modulation = torch.ones_like(velocity_score)
    else:
        try:
            floor = float(pose_quality_floor)
        except (TypeError, ValueError) as exc:
            raise TypeError("pose_quality_floor must be a finite scalar in [0, 1] or None.") from exc
        if not math.isfinite(floor) or not 0.0 <= floor <= 1.0:
            raise ValueError(
                f"pose_quality_floor must be finite and in [0, 1], got {pose_quality_floor}."
            )
        if pose_quality_group_stds is None:
            raise ValueError(
                "pose_quality_group_stds is required when pose_quality_floor is enabled."
            )
        pose_details = _grouped_expert_joint_pose_score_details(
            command,
            joint_groups,
            pose_quality_group_stds,
            pose_quality_group_weights if pose_quality_group_weights is not None else group_weights,
            score_exponent=score_exponent,
            worst_joint_count=worst_joint_count,
            worst_joint_weight=worst_joint_weight,
            group_aggregation=group_aggregation,
            worst_group_weight=worst_group_weight,
        )
        pose_quality = pose_details.aggregate
        pose_modulation = floor + (1.0 - floor) * pose_quality
    soft_support, strict_support = _terminal_expert_support_gate(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
        min_total_load_fraction,
        support_floor,
    )
    static_tail = _expert_static_tail_gate(
        command,
        reference_max_joint_speed,
        static_window_time_s,
        env.step_dt,
        ramp_time_s,
    )
    _record_final_tail_group_metrics(
        command,
        static_tail,
        strict_support=strict_support,
        speed_quality=velocity_score,
        speed_pose_quality=pose_quality,
        speed_pose_modulation=pose_modulation,
        speed_rms=velocity_details.rms,
        speed_max=velocity_details.maximum,
        speed_mean_score=velocity_details.mean_score,
        speed_worst_score=velocity_details.worst_score,
        speed_group_score=velocity_details.group_score,
    )
    alignment_complete = _terminal_platform_alignment_gate(command).to(dtype=velocity_score.dtype)
    return alignment_complete * static_tail * soft_support * pose_modulation * velocity_score


def _validate_terminal_root_tensor_pair(
    command: MotionCommand,
    reference_name: str,
    robot_name: str,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a shape-checked expert/robot root tensor pair."""

    reference = getattr(command, reference_name, None)
    robot = getattr(command, robot_name, None)
    if not isinstance(reference, torch.Tensor) or not isinstance(robot, torch.Tensor):
        raise RuntimeError(
            f"Terminal expert root reward requires tensor attributes {reference_name!r} "
            f"and {robot_name!r}."
        )
    expected_shape = (command.time_steps.shape[0], width)
    if reference.shape != expected_shape or robot.shape != expected_shape:
        raise RuntimeError(
            f"Terminal expert root tensors {reference_name!r} and {robot_name!r} must both "
            f"have shape {expected_shape}, got {reference.shape} and {robot.shape}."
        )
    if reference.device != robot.device:
        raise RuntimeError(
            f"Terminal expert root tensors {reference_name!r} and {robot_name!r} must share "
            f"a device, got {reference.device} and {robot.device}."
        )
    if not reference.is_floating_point() or not robot.is_floating_point():
        raise TypeError(
            f"Terminal expert root tensors {reference_name!r} and {robot_name!r} must be floating point."
        )
    if reference.dtype != robot.dtype:
        raise RuntimeError(
            f"Terminal expert root tensors {reference_name!r} and {robot_name!r} must share "
            f"a dtype, got {reference.dtype} and {robot.dtype}."
        )
    return reference, robot


def _validate_terminal_root_score_parameters(std: float, score_exponent: float) -> tuple[float, float]:
    """Validate scale and inverse-power exponent shared by terminal root terms."""

    if isinstance(std, bool):
        raise TypeError("std must be a positive finite scalar.")
    try:
        scale = float(std)
    except (TypeError, ValueError) as exc:
        raise TypeError("std must be a positive finite scalar.") from exc
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"std must be positive and finite, got {std}.")
    exponent = _validate_score_exponent(score_exponent)
    if exponent is None:
        raise ValueError("score_exponent cannot be None for terminal root rewards.")
    return scale, exponent


def final_expert_root_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_max_joint_speed: float,
    static_window_time_s: float,
    std: float,
    ramp_time_s: float | None = None,
    score_exponent: float = 0.5,
) -> torch.Tensor:
    """Track expert root orientation only during the authored stationary tail.

    The historical ``_exp`` suffix follows the module's reward naming
    convention; the implemented score is the configurable heavy-tailed
    ``(1 + (error / std)^2)^(-score_exponent)``.  The gate depends solely on
    reference phase and reference joint speed—never on robot pose, velocity,
    contact, or support—so the policy cannot turn this correction off by
    remaining in a bad state.
    """

    scale, exponent = _validate_terminal_root_score_parameters(std, score_exponent)
    command: MotionCommand = env.command_manager.get_term(command_name)
    reference, robot = _validate_terminal_root_tensor_pair(
        command, "anchor_quat_w", "robot_anchor_quat_w", 4
    )
    error = quat_error_magnitude(reference, robot)
    if error.shape != command.time_steps.shape:
        raise RuntimeError(
            "quat_error_magnitude must return one value per environment, "
            f"got {error.shape} and expected {command.time_steps.shape}."
        )
    score = _inverse_power_score(error, scale, exponent)
    static_tail = _expert_static_tail_gate(
        command,
        reference_max_joint_speed,
        static_window_time_s,
        env.step_dt,
        ramp_time_s,
    )
    _record_final_tail_group_metrics(
        command,
        static_tail,
        torso_orientation_error=error,
        root_orientation_score=score,
    )
    return static_tail * score


def final_expert_root_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_max_joint_speed: float,
    static_window_time_s: float,
    std: float,
    ramp_time_s: float | None = None,
    score_exponent: float = 0.5,
) -> torch.Tensor:
    """Track expert root linear velocity during the reference-static tail."""

    scale, exponent = _validate_terminal_root_score_parameters(std, score_exponent)
    command: MotionCommand = env.command_manager.get_term(command_name)
    reference, robot = _validate_terminal_root_tensor_pair(
        command, "anchor_lin_vel_w", "robot_anchor_lin_vel_w", 3
    )
    error = torch.linalg.vector_norm(reference - robot, dim=1)
    score = _inverse_power_score(error, scale, exponent)
    static_tail = _expert_static_tail_gate(
        command,
        reference_max_joint_speed,
        static_window_time_s,
        env.step_dt,
        ramp_time_s,
    )
    _record_final_tail_group_metrics(
        command,
        static_tail,
        torso_linear_velocity_error=error,
        root_linear_velocity_score=score,
    )
    return static_tail * score


def final_expert_root_angular_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    reference_max_joint_speed: float,
    static_window_time_s: float,
    std: float,
    ramp_time_s: float | None = None,
    score_exponent: float = 0.5,
) -> torch.Tensor:
    """Track expert root angular velocity during the reference-static tail."""

    scale, exponent = _validate_terminal_root_score_parameters(std, score_exponent)
    command: MotionCommand = env.command_manager.get_term(command_name)
    reference, robot = _validate_terminal_root_tensor_pair(
        command, "anchor_ang_vel_w", "robot_anchor_ang_vel_w", 3
    )
    error = torch.linalg.vector_norm(reference - robot, dim=1)
    robot_speed = torch.linalg.vector_norm(robot, dim=1)
    score = _inverse_power_score(error, scale, exponent)
    static_tail = _expert_static_tail_gate(
        command,
        reference_max_joint_speed,
        static_window_time_s,
        env.step_dt,
        ramp_time_s,
    )
    _record_final_tail_group_metrics(
        command,
        static_tail,
        torso_angular_velocity_error=error,
        torso_angular_speed=robot_speed,
        root_angular_velocity_score=score,
    )
    return static_tail * score


def final_expert_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    reference_max_joint_speed: float,
    static_window_time_s: float,
    std: float,
    platform_support_params: Mapping[str, object] | None = None,
    min_total_load_fraction: float = 0.0,
) -> torch.Tensor:
    """Reward a natural full-body expert pose during real final support.

    This term starts only in the verified stationary source tail and requires
    sustained support from both soles plus the configured foot-borne load. It
    therefore cannot alter the dynamic climb, activate on a single planted
    foot, or reward a hand-supported terminal shortcut.  It acts only inside
    the authored stationary source tail, so it reinforces the expert's final
    pose without creating a post-expert standing interval.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    pose_score = _expert_joint_pose_score(command, std)
    per_foot_scores = _platform_foot_contact_scores(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
    )
    two_foot_support = per_foot_scores.amin(dim=1).clamp(min=0.0, max=1.0)
    foot_load_score = _platform_foot_load_score(
        env,
        command,
        platform_cfg,
        base_size,
        platform_support_params,
        min_contact_force=min_contact_force,
        foot_height_std=foot_height_std,
        min_total_load_fraction=min_total_load_fraction,
    )
    static_tail = _expert_static_tail_gate(
        command,
        reference_max_joint_speed,
        static_window_time_s,
        env.step_dt,
    )
    return static_tail * two_foot_support * foot_load_score * pose_score


def platform_foot_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    terminal_window_time_s: float,
    platform_support_params: Mapping[str, object] | None = None,
) -> torch.Tensor:
    """Reward actual two-foot contact with the physical platform top."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    gate = _final_phase_gate(command, terminal_window_time_s, env.step_dt)
    per_foot_scores = _platform_foot_contact_scores(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
    )
    # A mean would let one planted foot hide an unsupported second foot.  The
    # minimum makes this a genuine bilateral-platform reward.
    return gate * per_foot_scores.amin(dim=1)


def platform_foot_surface_alignment(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    platform_support_params: Mapping[str, object],
    min_upward_force: float,
    sole_height_tolerance: float,
    terminal_window_time_s: float,
    yaw_std: float,
) -> torch.Tensor:
    """Reward two terrain-aligned feet while preserving the expert headings.

    This is deliberately independent of expert ankle pitch/roll and contact
    force.  It supplies a geometric correction before impact; the separate
    platform-contact and load terms decide whether the aligned feet are
    physically supporting the robot.
    """

    if not math.isfinite(yaw_std) or yaw_std <= 0.0:
        raise ValueError(f"yaw_std must be positive and finite, got {yaw_std}.")
    command: MotionCommand = env.command_manager.get_term(command_name)
    support = platform_foot_support_state(
        env,
        command.robot,
        command.device,
        platform_cfg,
        base_size,
        platform_support_params,
        min_upward_force=min_upward_force,
        sole_height_tolerance=sole_height_tolerance,
    )
    foot_ids = _named_body_ids(
        command.cfg.body_names,
        support.settings.foot_body_names,
        command.device,
        context="Terminal adaptive surface heading",
    )
    reference_yaw = math_utils.yaw_quat(command.body_quat_relative_w[:, foot_ids])
    actual_yaw = math_utils.yaw_quat(command.robot_body_quat_w[:, foot_ids])
    yaw_error = quat_error_magnitude(reference_yaw, actual_yaw)
    yaw_score = torch.rsqrt(1.0 + torch.square(yaw_error / yaw_std))
    per_foot_score = support.dense_surface_score * yaw_score
    gate = _final_phase_gate(command, terminal_window_time_s, env.step_dt)

    metrics = getattr(command, "metrics", None)
    if metrics is not None:
        side_names = ("left", "right")

        def record(name: str, value: torch.Tensor) -> None:
            metric = metrics.setdefault(name, torch.zeros_like(value))
            if metric.shape != value.shape:
                raise RuntimeError(
                    f"Terminal foot-surface diagnostic {name!r} has shape {metric.shape}, "
                    f"expected {value.shape}."
                )
            metric.copy_(gate * value)

        for foot_index, side_name in enumerate(side_names):
            record(
                f"final_standing_{side_name}_sole_max_height_error",
                support.sole_surface_height_error[:, foot_index],
            )
            record(
                f"final_standing_{side_name}_sole_height_spread",
                support.sole_height_spread[:, foot_index],
            )
            record(
                f"final_standing_{side_name}_dense_surface_score",
                support.dense_surface_score[:, foot_index],
            )
            record(
                f"final_standing_{side_name}_platform_upward_force",
                support.upward_forces[:, foot_index],
            )
            record(
                f"final_standing_{side_name}_strict_active_support",
                support.active_support[:, foot_index].to(dtype=gate.dtype),
            )
    tiny = torch.finfo(per_foot_score.dtype).tiny
    bilateral_score = per_foot_score.shape[1] / torch.sum(
        torch.reciprocal(per_foot_score.clamp_min(tiny)), dim=1
    )
    return gate * bilateral_score


def final_ankle_surface_settling(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    platform_support_params: Mapping[str, object],
    min_upward_force: float,
    sole_height_tolerance: float,
    reference_max_joint_speed: float,
    static_window_time_s: float,
    ramp_time_s: float,
    ankle_joint_names: Sequence[Sequence[str]],
    speed_scale: float,
) -> torch.Tensor:
    """Settle each ankle only in proportion to its terrain-aligned sole quality.

    A motionless toe stand receives only its low dense surface score, so this
    term cannot teach the policy to freeze a bad ankle pose.  Conversely, the
    inverse-square-root velocity score keeps a useful gradient while an ankle
    is still moving.  Strict contact and success remain separate terms.
    """

    if not math.isfinite(speed_scale) or speed_scale <= 0.0:
        raise ValueError(f"speed_scale must be positive and finite, got {speed_scale}.")
    if len(ankle_joint_names) != 2 or any(len(names) == 0 for names in ankle_joint_names):
        raise ValueError("ankle_joint_names must contain two non-empty per-foot joint groups.")
    flattened_names = [str(name) for names in ankle_joint_names for name in names]
    if len(set(flattened_names)) != len(flattened_names):
        raise ValueError("ankle_joint_names must not contain duplicate joints.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    missing = [name for name in flattened_names if name not in command.robot.joint_names]
    if missing:
        raise RuntimeError(f"Terminal ankle settling references unknown robot joints: {missing}.")
    joint_signature = tuple(tuple(str(name) for name in names) for names in ankle_joint_names)
    cached_joint_ids = getattr(command, "_terminal_ankle_surface_joint_ids", None)
    if cached_joint_ids is None or cached_joint_ids[0] != joint_signature:
        resolved_joint_ids = tuple(
            torch.tensor(
                [command.robot.joint_names.index(name) for name in names],
                dtype=torch.long,
                device=command.device,
            )
            for names in joint_signature
        )
        command._terminal_ankle_surface_joint_ids = (joint_signature, resolved_joint_ids)
    else:
        resolved_joint_ids = cached_joint_ids[1]
    support = platform_foot_support_state(
        env,
        command.robot,
        command.device,
        platform_cfg,
        base_size,
        platform_support_params,
        min_upward_force=min_upward_force,
        sole_height_tolerance=sole_height_tolerance,
    )
    per_foot_velocity_scores: list[torch.Tensor] = []
    for joint_ids in resolved_joint_ids:
        joint_scores = torch.rsqrt(
            1.0 + torch.square(torch.abs(command.robot_joint_vel[:, joint_ids]) / speed_scale)
        )
        per_foot_velocity_scores.append(joint_scores.amin(dim=1))
    ankle_velocity_score = torch.stack(per_foot_velocity_scores, dim=1)
    per_foot_score = support.dense_surface_score * ankle_velocity_score
    static_tail = _expert_static_tail_gate(
        command,
        reference_max_joint_speed,
        static_window_time_s,
        env.step_dt,
        ramp_time_s,
    )
    alignment_complete = _terminal_platform_alignment_gate(command).to(dtype=per_foot_score.dtype)
    tiny = torch.finfo(per_foot_score.dtype).tiny
    bilateral_score = per_foot_score.shape[1] / torch.sum(
        torch.reciprocal(per_foot_score.clamp_min(tiny)), dim=1
    )
    return alignment_complete * static_tail * bilateral_score


def final_standing_stability(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    terminal_window_time_s: float,
    root_linear_speed_std: float,
    root_angular_speed_std: float,
    joint_speed_std: float,
    torso_tilt_std: float,
    stability_weights: tuple[float, float, float, float],
    platform_support_params: Mapping[str, object] | None = None,
    min_total_load_fraction: float = 0.0,
    expert_joint_pose_std: float | None = None,
) -> torch.Tensor:
    """Reward a quiet upright stand after the feet contact the platform.

    The platform contact score gates the reward, while the four stability
    scores are combined as a weighted average. Multiplying all four scores
    made this shaping term effectively disappear when one signal (most often
    residual joint speed) was temporarily poor, which gave the policy no
    useful gradient to settle the remaining motion.
    """

    for name, value in (
        ("root_linear_speed_std", root_linear_speed_std),
        ("root_angular_speed_std", root_angular_speed_std),
        ("joint_speed_std", joint_speed_std),
        ("torso_tilt_std", torso_tilt_std),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if len(stability_weights) != 4 or any(weight < 0.0 for weight in stability_weights):
        raise ValueError(
            "stability_weights must contain four non-negative values ordered as "
            "root linear speed, root angular speed, joint speed, torso tilt."
        )
    weight_sum = float(sum(stability_weights))
    if weight_sum <= 0.0:
        raise ValueError("stability_weights must contain at least one positive value.")

    command: MotionCommand = env.command_manager.get_term(command_name)
    gate = _final_phase_gate(command, terminal_window_time_s, env.step_dt)
    per_foot_scores = _platform_foot_contact_scores(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
    )
    contact_score = per_foot_scores.amin(dim=1)
    foot_load_score = _platform_foot_load_score(
        env,
        command,
        platform_cfg,
        base_size,
        platform_support_params,
        min_contact_force=min_contact_force,
        foot_height_std=foot_height_std,
        min_total_load_fraction=min_total_load_fraction,
    )

    root_linear_speed = torch.linalg.vector_norm(command.robot_anchor_lin_vel_w, dim=-1)
    root_angular_speed = torch.linalg.vector_norm(command.robot_anchor_ang_vel_w, dim=-1)
    joint_speed = torch.max(torch.abs(command.robot_joint_vel), dim=1).values
    projected_gravity_b = math_utils.quat_apply_inverse(
        command.robot_anchor_quat_w,
        command.robot.data.GRAVITY_VEC_W,
    )
    torso_tilt = torch.acos((-projected_gravity_b[:, 2]).clamp(min=-1.0, max=1.0))

    def gaussian_score(value: torch.Tensor, std: float) -> torch.Tensor:
        return torch.exp(-0.5 * torch.square(value / std))

    stability_scores = (
        gaussian_score(root_linear_speed, root_linear_speed_std),
        gaussian_score(root_angular_speed, root_angular_speed_std),
        gaussian_score(joint_speed, joint_speed_std)
        * (
            _expert_joint_pose_score(command, expert_joint_pose_std)
            if expert_joint_pose_std is not None
            else torch.ones_like(joint_speed)
        ),
        gaussian_score(torso_tilt, torso_tilt_std),
    )
    stability_score = sum(
        (weight / weight_sum) * score for weight, score in zip(stability_weights, stability_scores, strict=True)
    )
    alignment_complete = _terminal_platform_alignment_gate(command).to(dtype=stability_score.dtype)
    stationary_target = _terminal_stationary_target_gate(command)
    return alignment_complete * stationary_target * gate * contact_score * foot_load_score * stability_score


def final_joint_settling(
    env: ManagerBasedRLEnv,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    foot_body_names: list[str],
    footprint_inset: float,
    foot_height_std: float,
    min_contact_force: float,
    contact_time_scale: float,
    reference_max_joint_speed: float,
    rms_speed_tolerance: float,
    max_speed_tolerance: float,
    rms_speed_scale: float,
    max_speed_scale: float,
    fine_max_speed_scale: float,
    score_weights: tuple[float, float, float],
    platform_support_params: Mapping[str, object] | None = None,
    min_total_load_fraction: float = 0.0,
    expert_joint_pose_std: float | None = None,
) -> torch.Tensor:
    """Reward real joint settling during the expert's stationary tail.

    The reward is enabled only when the immutable reference has already
    stopped and both feet have sustained physical support on the platform.
    It deliberately does not require an upright torso or a particular hand
    state: those requirements would make the transition circular or assume a
    hand-support strategy that has not been established by contact evidence.
    """

    if reference_max_joint_speed < 0.0:
        raise ValueError(
            f"reference_max_joint_speed must be non-negative, got {reference_max_joint_speed}."
        )
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Inspect the immutable source tail, not the temporary q interpolation.
    # The latter correctly has non-zero command velocity during default-pose
    # takeover and is not evidence that the expert clip is still moving.
    if hasattr(command.motion, "joint_vel"):
        source_joint_vel = command.motion.joint_vel[command.time_steps]
    else:
        # Lightweight test doubles commonly expose only the current command
        # velocity; production MotionCommand always follows the branch above.
        source_joint_vel = command.joint_vel
    reference_speed = torch.max(torch.abs(source_joint_vel), dim=1).values
    # The source clips also begin with a short static segment. Requiring the
    # second half of the clip prevents the settling reward from firing at the
    # initial stand before the climb has started.
    motion_starts = command.motion.motion_start_idx[command.motion_ids]
    motion_lengths = command.motion.motion_lengths[command.motion_ids]
    local_time_steps = command.time_steps - motion_starts
    in_terminal_half = 2 * local_time_steps >= motion_lengths - 1
    # During the additional hold ``time_steps`` remains clamped to the final
    # frame, so this gate naturally stays active without creating a new target.
    reference_stopped = (reference_speed <= reference_max_joint_speed) & in_terminal_half
    per_foot_scores = _platform_foot_contact_scores(
        env,
        command,
        platform_cfg,
        contact_sensor_cfg,
        base_size,
        foot_body_names,
        footprint_inset,
        foot_height_std,
        min_contact_force,
        contact_time_scale,
        platform_support_params,
    )
    # The minimum prevents one planted foot from opening the settling reward
    # while the other foot is unsupported or still moving onto the platform.
    two_foot_support = per_foot_scores.amin(dim=1).clamp(min=0.0, max=1.0)
    foot_load_score = _platform_foot_load_score(
        env,
        command,
        platform_cfg,
        base_size,
        platform_support_params,
        min_contact_force=min_contact_force,
        foot_height_std=foot_height_std,
        min_total_load_fraction=min_total_load_fraction,
    )
    settling_score = joint_settling_score(
        command.robot_joint_vel,
        rms_speed_tolerance=rms_speed_tolerance,
        max_speed_tolerance=max_speed_tolerance,
        rms_speed_scale=rms_speed_scale,
        max_speed_scale=max_speed_scale,
        fine_max_speed_scale=fine_max_speed_scale,
        score_weights=score_weights,
    )
    if expert_joint_pose_std is not None:
        # Do not teach the policy to freeze whichever posture it happened to
        # reach.  Full settling credit is available only near the immutable
        # natural expert pose; the dedicated pose reward provides the broad
        # direction for reaching it.
        settling_score *= _expert_joint_pose_score(command, expert_joint_pose_std)
    alignment_complete = _terminal_platform_alignment_gate(command).to(dtype=settling_score.dtype)
    stationary_target = _terminal_stationary_target_gate(command)
    return (
        alignment_complete
        * stationary_target
        * reference_stopped.to(dtype=settling_score.dtype)
        * two_foot_support
        * foot_load_score
        * settling_score
    )


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
