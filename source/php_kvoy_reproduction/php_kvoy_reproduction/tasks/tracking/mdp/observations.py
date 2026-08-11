from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from php_kvoy_reproduction.tasks.tracking.mdp.commands import MotionCommand
from php_kvoy_reproduction.tasks.tracking.mdp.obstacle import climb_box_top_height, get_climb_box_sizes

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def box_obstacle_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    offset: float = 0.5,
) -> torch.Tensor:
    """Return a height scan containing the ground and the exact physical box.

    Isaac Lab 4.5's ray caster accepts only one static mesh.  The ELF3 climb
    scene keeps its ground plane and per-environment platform as separate
    collision objects, so the ray caster measures the ground mesh and this
    function overlays the platform using the same per-environment pose and
    dimensions that define its collider. Yaw, translation, and scale are all
    included.

    The returned convention matches :func:`isaaclab.envs.mdp.height_scan`:
    robot body height minus terrain height minus ``offset``.
    """

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    platform: RigidObject = env.scene[asset_cfg.name]
    ray_hits_w = sensor.data.ray_hits_w
    sizes = get_climb_box_sizes(platform, base_size=base_size, device=ray_hits_w.device)
    terrain_height_w, _ = climb_box_top_height(
        ray_hits_w,
        platform.data.root_pos_w,
        platform.data.root_quat_w,
        sizes,
    )
    return sensor.data.pos_w[:, 2].unsqueeze(1) - terrain_height_w - offset
