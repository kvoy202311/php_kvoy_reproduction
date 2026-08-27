"""Independent routed episode boundaries and safety terminations."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

from .commands import MultiSkillCommand


def _command(env, command_name: str) -> MultiSkillCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, MultiSkillCommand):
        raise TypeError(f"command {command_name!r} must be a MultiSkillCommand")
    return command


def routed_motion_clip_end(env, command_name: str) -> torch.Tensor:
    command = _command(env, command_name)
    direct_motion_end = (
        ~command.composed_episode & command.motion_mask & command.motion_finished
    )
    return direct_motion_end | command.episode_complete


def routed_motion_tracking_failure(env, command_name: str) -> torch.Tensor:
    """Terminate outside a scheduled relaxation of the teacher's scope.

    DAgger labels stop at the original frozen-teacher thresholds.  PPO
    rollouts may continue up to a linearly relaxed boundary, matching PHP's
    intent without allowing arbitrarily corrupted states to run until clip
    end.  The iteration is supplied by the runner and therefore resumes at
    the correct absolute schedule position.
    """

    command = _command(env, command_name)
    motion = command.motion_mask
    scale = command.student_termination_scale
    anchor_z_error = torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2])
    reference_gravity = quat_apply_inverse(
        command.anchor_quat_w,
        command.robot.data.GRAVITY_VEC_W,
    )
    robot_gravity = quat_apply_inverse(
        command.robot_anchor_quat_w,
        command.robot.data.GRAVITY_VEC_W,
    )
    orientation_error = torch.abs(reference_gravity[:, 2] - robot_gravity[:, 2])
    end_effector_ids = torch.tensor(
        [command.cfg.body_names.index(name) for name in command.cfg.teacher_end_effector_names],
        device=command.device,
        dtype=torch.long,
    )
    end_effector_z_error = torch.abs(
        command.body_pos_relative_w[:, end_effector_ids, 2]
        - command.robot_body_pos_w[:, end_effector_ids, 2]
    ).amax(dim=1)
    outside = (
        (anchor_z_error > command.cfg.teacher_anchor_z_threshold * scale)
        | (orientation_error > command.cfg.teacher_orientation_threshold * scale)
        | (end_effector_z_error > command.cfg.teacher_end_effector_z_threshold * scale)
    )
    return motion & command.motion_tracking_termination_enabled & outside


def locomotion_low_root(
    env,
    command_name: str,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    if minimum_height <= 0.0:
        raise ValueError("minimum_height must be positive")
    command = _command(env, command_name)
    robot = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return command.locomotion_mask & (height < minimum_height)


def climb_low_root(
    env,
    command_name: str,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Stop an unrecoverable climb fall without constraining valid down-roll."""

    if minimum_height <= 0.0:
        raise ValueError("minimum_height must be positive")
    command = _command(env, command_name)
    robot = env.scene[asset_cfg.name]
    height = robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return command.climb_mask & (height < minimum_height)


def locomotion_illegal_contact(
    env,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    command = _command(env, command_name)
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contact = torch.any(torch.linalg.vector_norm(forces, dim=-1) > threshold, dim=(1, 2))
    return command.locomotion_mask & contact


def non_finite_robot_state(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    finite = (
        torch.isfinite(robot.data.root_state_w).all(dim=1)
        & torch.isfinite(robot.data.joint_pos).all(dim=1)
        & torch.isfinite(robot.data.joint_vel).all(dim=1)
    )
    return ~finite


__all__ = [
    "climb_low_root",
    "locomotion_illegal_contact",
    "locomotion_low_root",
    "non_finite_robot_state",
    "routed_motion_clip_end",
    "routed_motion_tracking_failure",
]
