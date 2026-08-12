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
) -> torch.Tensor:
    """Reward a quiet upright stand after the feet contact the platform."""

    for name, value in (
        ("root_linear_speed_std", root_linear_speed_std),
        ("root_angular_speed_std", root_angular_speed_std),
        ("joint_speed_std", joint_speed_std),
        ("torso_tilt_std", torso_tilt_std),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")

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

    stability_score = (
        gaussian_score(root_linear_speed, root_linear_speed_std)
        * gaussian_score(root_angular_speed, root_angular_speed_std)
        * gaussian_score(joint_speed, joint_speed_std)
        * gaussian_score(torso_tilt, torso_tilt_std)
    )
    return gate * contact_score * stability_score


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward
