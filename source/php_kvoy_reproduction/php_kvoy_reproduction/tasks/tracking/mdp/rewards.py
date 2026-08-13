from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from php_kvoy_reproduction.tasks.tracking.mdp.commands import MotionCommand
from php_kvoy_reproduction.tasks.tracking.mdp.obstacle import get_climb_box_sizes, points_inside_oriented_box_xy

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


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
    )
    # The minimum prevents one well-supported foot from masking a missing or
    # poorly supported second foot.
    return per_foot_scores.amin(dim=1).clamp(min=0.0, max=1.0)


def _terminal_weighted_body_error_exp(
    error: torch.Tensor,
    body_indexes: list[int],
    terminal_body_indexes: list[int],
    terminal_body_weight: float,
    terminal_gate: torch.Tensor,
    std: float,
) -> torch.Tensor:
    """Fade selected body tracking after the final two-foot contact gate."""

    if not 0.0 <= terminal_body_weight <= 1.0:
        raise ValueError(f"terminal_body_weight must be in [0, 1], got {terminal_body_weight}.")
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")
    terminal_set = set(terminal_body_indexes)
    terminal_mask = torch.tensor(
        [index in terminal_set for index in body_indexes], dtype=error.dtype, device=error.device
    )
    weights = 1.0 - (1.0 - terminal_body_weight) * terminal_gate[:, None] * terminal_mask[None, :]
    # Keep the original body-count normalization. This genuinely weakens the
    # ankle objective instead of renormalizing the remaining body terms.
    return torch.exp(-(error * weights).mean(dim=-1) / std**2)


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
) -> torch.Tensor:
    """Track climb body positions while fading ankle tracking at the end."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    terminal_body_indexes = _get_body_indexes(command, terminal_body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]),
        dim=-1,
    )
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
    )
    return _terminal_weighted_body_error_exp(
        error, body_indexes, terminal_body_indexes, terminal_body_weight, terminal_gate, std
    )


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
) -> torch.Tensor:
    """Track climb body orientations while fading ankle tracking at the end."""

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
    )
    return _terminal_weighted_body_error_exp(
        error, body_indexes, terminal_body_indexes, terminal_body_weight, terminal_gate, std
    )


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
) -> torch.Tensor:
    """Give conservative progress shaping for the physical climb geometry.

    The approach term saturates at the platform's near edge, so it cannot
    reward driving through the box.  The lift term uses the *sampled* platform
    top height and is gated by an actual foot/hand support proximity score and
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
    )
    return gate * per_foot_scores.mean(dim=1)


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
    )
    contact_score = per_foot_scores.mean(dim=1)

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
        gaussian_score(joint_speed, joint_speed_std),
        gaussian_score(torso_tilt, torso_tilt_std),
    )
    stability_score = sum(
        (weight / weight_sum) * score for weight, score in zip(stability_weights, stability_scores, strict=True)
    )
    return gate * contact_score * stability_score


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
