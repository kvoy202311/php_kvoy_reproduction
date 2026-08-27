"""Route-masked PPO rewards for the unified distillation environment."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import wrap_to_pi

from php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry import (
    filtered_platform_force_score,
    foot_sole_corners_world,
    oriented_box_local_xy,
)
from php_kvoy_reproduction.tasks.tracking.mdp.platform_foot_support import (
    platform_foot_support_state,
)
from php_kvoy_reproduction.distillation.skill_routing import (
    climb_geometry_progress_score,
    down_roll_geometry_progress_score,
    filtered_contact_upward_forces,
    lower_ground_contact_support_score,
)

from .commands import MultiSkillCommand


def _command(env, command_name: str) -> MultiSkillCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, MultiSkillCommand):
        raise TypeError(f"command {command_name!r} must be a MultiSkillCommand")
    return command


def routed_reward(
    env,
    command_name: str,
    skill: str,
    wrapped_func: Callable,
    wrapped_params: Mapping[str, object],
) -> torch.Tensor:
    """Evaluate an existing reward and zero it outside one routed skill."""

    value = wrapped_func(env, **dict(wrapped_params))
    if value.shape != (env.num_envs,):
        raise RuntimeError(
            f"wrapped reward {getattr(wrapped_func, '__name__', wrapped_func)!r} returned {tuple(value.shape)}"
        )
    command = _command(env, command_name)
    route_weight = command.mask_for_skill(skill).to(dtype=value.dtype)
    if skill in ("climb", "down_roll"):
        # Fixed-height motion references are useful only to the extent that
        # the frozen teacher remains in its verified geometry distribution.
        # This weight depends only on sampled geometry, never on the student's
        # tracking error, so the policy cannot turn off a reward by drifting.
        route_weight = route_weight * command.motion_teacher_geometry_confidence
    return value * route_weight


def locomotion_world_velocity_tracking_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if std <= 0.0 or not math.isfinite(std):
        raise ValueError("std must be finite and positive")
    command = _command(env, command_name)
    robot = env.scene[asset_cfg.name]
    error = torch.sum(torch.square(robot.data.root_lin_vel_w[:, :2] - command.world_command), dim=1)
    score = torch.exp(-error / (std * std))
    return score * command.locomotion_mask.to(dtype=score.dtype)


def locomotion_heading_tracking_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if std <= 0.0 or not math.isfinite(std):
        raise ValueError("std must be finite and positive")
    command = _command(env, command_name)
    robot = env.scene[asset_cfg.name]
    moving = torch.linalg.vector_norm(command.world_command, dim=1) > 1.0e-4
    error = wrap_to_pi(command.heading_target - robot.data.heading_w)
    score = torch.exp(-torch.square(error) / (std * std))
    score = torch.where(moving, score, torch.ones_like(score))
    return score * command.locomotion_mask.to(dtype=score.dtype)


def locomotion_upright_l2(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = _command(env, command_name)
    robot = env.scene[asset_cfg.name]
    penalty = torch.sum(torch.square(robot.data.projected_gravity_b[:, :2]), dim=1)
    return penalty * command.locomotion_mask.to(dtype=penalty.dtype)


def _platform_geometry(command: MultiSkillCommand) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sizes = command.platform_sizes
    root = command.robot.data.root_pos_w
    local_xy = oriented_box_local_xy(
        root[:, None, :],
        command.platform_pos_w,
        command.platform_quat_w,
    ).squeeze(1)
    top = command.platform_pos_w[:, 2] + 0.5 * sizes[:, 2]
    return sizes, local_xy, top


def climb_geometry_progress(
    env,
    command_name: str,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    platform_support_params: Mapping[str, object],
    approach_distance: float = 1.2,
    standing_root_clearance: float = 1.0,
    foot_height_tolerance: float = 0.08,
    minimum_upward_force: float = 20.0,
) -> torch.Tensor:
    """Platform-relative climb shaping that remains valid across height/size."""

    if approach_distance <= 0.0 or standing_root_clearance <= 0.0:
        raise ValueError("approach_distance and standing_root_clearance must be positive")
    if foot_height_tolerance <= 0.0 or minimum_upward_force <= 0.0:
        raise ValueError("climb support thresholds are invalid")
    command = _command(env, command_name)
    sizes, root_local_xy, platform_top = _platform_geometry(command)
    ground = env.scene.env_origins[:, 2]

    entry_edge = -0.5 * sizes[:, 0]
    distance_before_entry = (entry_edge - root_local_xy[:, 0]).clamp_min(0.0)
    approach = 1.0 - (distance_before_entry / approach_distance).clamp(0.0, 1.0)
    standing_target = platform_top + standing_root_clearance
    height = (
        (command.robot.data.root_pos_w[:, 2] - (ground + standing_root_clearance))
        / (standing_target - (ground + standing_root_clearance)).clamp_min(1.0e-6)
    ).clamp(0.0, 1.0)

    support_state = platform_foot_support_state(
        env,
        command.robot,
        command.device,
        platform_cfg,
        base_size,
        platform_support_params,
        min_upward_force=minimum_upward_force,
        sole_height_tolerance=foot_height_tolerance,
    )
    force_score = filtered_platform_force_score(
        support_state.platform_forces_w,
        min_upward_force=minimum_upward_force,
    )
    support = (
        support_state.dense_surface_score
        * force_score
        * support_state.active_support.to(dtype=force_score.dtype)
    ).mean(dim=1)

    score = climb_geometry_progress_score(approach, height, support)
    return score * command.climb_mask.to(dtype=score.dtype)


def down_roll_geometry_progress(
    env,
    command_name: str,
    foot_cfg: SceneEntityCfg,
    ground_contact_sensor_names: tuple[str, ...],
    sole_corners_b: tuple[tuple[float, float, float], ...],
    standing_root_clearance: float = 1.0,
    edge_progress_distance: float = 0.8,
    landing_height_tolerance: float = 0.20,
    upright_tolerance: float = 0.35,
    ground_sole_height_tolerance: float = 0.04,
    minimum_ground_upward_force: float = 20.0,
) -> torch.Tensor:
    """Reward crossing the actual far edge and recovering on lower ground."""

    thresholds = (
        standing_root_clearance,
        edge_progress_distance,
        landing_height_tolerance,
        upright_tolerance,
        ground_sole_height_tolerance,
        minimum_ground_upward_force,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in thresholds):
        raise ValueError("down-roll geometry thresholds must be positive")
    command = _command(env, command_name)
    sizes, root_local_xy, platform_top = _platform_geometry(command)
    ground = env.scene.env_origins[:, 2]
    far_edge = 0.5 * sizes[:, 0]
    crossed = ((root_local_xy[:, 0] - far_edge) / edge_progress_distance).clamp(0.0, 1.0)

    upper_standing_root = platform_top + standing_root_clearance
    lower_standing_root = ground + standing_root_clearance
    descent = (
        (upper_standing_root - command.robot.data.root_pos_w[:, 2])
        / (upper_standing_root - lower_standing_root).clamp_min(1.0e-6)
    ).clamp(0.0, 1.0)
    descent = descent * crossed
    landing_height_score = torch.exp(
        -torch.square(
            (command.robot.data.root_pos_w[:, 2] - lower_standing_root)
            / landing_height_tolerance
        )
    )
    upright_error = torch.linalg.vector_norm(
        command.robot.data.projected_gravity_b[:, :2], dim=1
    )
    upright_score = torch.exp(-torch.square(upright_error / upright_tolerance))

    if env.scene[foot_cfg.name] is not command.robot:
        raise ValueError("foot_cfg must refer to the routed robot")
    foot_names = tuple(foot_cfg.body_names or ())
    if not foot_names:
        raise ValueError("foot_cfg must contain at least one explicitly ordered foot")
    if isinstance(foot_cfg.body_ids, slice):
        raise ValueError("ground support requires explicit foot body IDs")
    if isinstance(ground_contact_sensor_names, str):
        raise ValueError("ground_contact_sensor_names must be a per-foot sequence, not a string")
    sensor_names = tuple(ground_contact_sensor_names)
    if (
        len(sensor_names) != len(foot_names)
        or len(set(sensor_names)) != len(sensor_names)
        or any(not isinstance(name, str) or not name for name in sensor_names)
    ):
        raise ValueError(
            "ground_contact_sensor_names must contain one distinct non-empty sensor name per foot"
        )
    foot_body_ids = torch.as_tensor(
        foot_cfg.body_ids,
        device=command.device,
        dtype=torch.long,
    )
    if foot_body_ids.numel() != len(foot_names):
        raise ValueError("ground support requires one explicit robot body ID per foot")

    sole_corners = torch.as_tensor(
        sole_corners_b,
        device=command.device,
        dtype=command.robot.data.body_pos_w.dtype,
    )
    sole_corners_w = foot_sole_corners_world(
        command.robot.data.body_pos_w[:, foot_body_ids],
        command.robot.data.body_quat_w[:, foot_body_ids],
        sole_corners,
    )
    sole_height_errors = torch.abs(
        sole_corners_w[..., 2] - ground[:, None, None]
    ).amin(dim=-1)

    ground_force_matrices: list[torch.Tensor] = []
    for foot_name, sensor_name in zip(foot_names, sensor_names, strict=True):
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        if tuple(contact_sensor.body_names) != (foot_name,):
            raise RuntimeError(
                f"ground-contact sensor {sensor_name!r} must resolve only {foot_name!r}; "
                f"got {tuple(contact_sensor.body_names)}"
            )
        force_matrix_w = contact_sensor.data.force_matrix_w
        if force_matrix_w is None:
            raise RuntimeError(
                f"ground-contact sensor {sensor_name!r} must configure a ground-only filter"
            )
        ground_force_matrices.append(force_matrix_w)
    upward_forces = filtered_contact_upward_forces(
        ground_force_matrices,
        num_envs=env.num_envs,
    )
    ground_support = lower_ground_contact_support_score(
        sole_height_errors,
        upward_forces,
        height_tolerance=ground_sole_height_tolerance,
        minimum_upward_force=minimum_ground_upward_force,
    )

    landing = crossed * landing_height_score * upright_score * ground_support
    score = down_roll_geometry_progress_score(crossed, descent, landing)
    return score * command.down_roll_mask.to(dtype=score.dtype)


__all__ = [
    "locomotion_heading_tracking_exp",
    "locomotion_upright_l2",
    "locomotion_world_velocity_tracking_exp",
    "climb_geometry_progress",
    "down_roll_geometry_progress",
    "routed_reward",
]
