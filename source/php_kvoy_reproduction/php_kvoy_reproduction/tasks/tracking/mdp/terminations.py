from __future__ import annotations

import math
from collections.abc import Sequence
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
from php_kvoy_reproduction.tasks.tracking.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


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
    contact_valid = (foot_contact_forces >= min_foot_contact_force) & (foot_contact_times >= min_foot_contact_time)

    projected_gravity_b = math_utils.quat_apply_inverse(
        command.robot_anchor_quat_w,
        command.robot.data.GRAVITY_VEC_W,
    )
    upright = projected_gravity_b[:, 2] <= -math.cos(max_torso_tilt)
    root_height_valid = (
        torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]) <= max_root_height_error
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
    conditions = {
        "feet_inside": torch.all(inside, dim=1),
        "foot_height_valid": torch.all(height_valid, dim=1),
        "foot_contact_valid": torch.all(contact_valid, dim=1),
        "upright": upright,
        "root_height_valid": root_height_valid,
        "root_linear_speed_valid": root_linear_speed_valid,
        "root_angular_speed_valid": root_angular_speed_valid,
        "joint_speed_valid": joint_speed_valid,
    }
    return conditions, default_joint_pos_rms


class motion_end_success(ManagerTermBase):
    """Classify a clip as successful only after a continuous stable final stand.

    The default articulation pose is deliberately diagnostic-only.  A climb is
    complete when the robot is functionally stable on the platform; requiring
    an unrelated nominal pose would reject valid reproductions of the expert's
    stationary final frame.
    """

    _METRIC_PREFIX = "final_standing_"

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self._required_stable_steps = max(1, math.ceil(float(cfg.params["min_stable_time"]) / env.step_dt - 1.0e-9))

        command: MotionCommand = env.command_manager.get_term(cfg.params["command_name"])
        metric_names = (
            "feet_inside",
            "foot_height_valid",
            "foot_contact_valid",
            "upright",
            "root_height_valid",
            "root_linear_speed_valid",
            "root_angular_speed_valid",
            "joint_speed_valid",
            "default_joint_pos_rms",
            "stable_time",
        )
        for name in metric_names:
            command.metrics.setdefault(self._METRIC_PREFIX + name, torch.zeros(env.num_envs, device=env.device))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._stable_steps[env_ids] = 0

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
        )
        final_frames = command.motion.motion_end_idx[command.motion_ids] - 1
        at_final_frame = command.time_steps >= final_frames
        standing_valid = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        for value in conditions.values():
            standing_valid &= value
        valid_final_stand = at_final_frame & standing_valid
        self._stable_steps = torch.where(valid_final_stand, self._stable_steps + 1, 0)

        for name, value in conditions.items():
            command.metrics[self._METRIC_PREFIX + name].copy_((at_final_frame & value).float())
        command.metrics[self._METRIC_PREFIX + "default_joint_pos_rms"].copy_(
            torch.where(at_final_frame, default_joint_pos_rms, torch.zeros_like(default_joint_pos_rms))
        )
        command.metrics[self._METRIC_PREFIX + "stable_time"].copy_(self._stable_steps.float() * env.step_dt)

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
