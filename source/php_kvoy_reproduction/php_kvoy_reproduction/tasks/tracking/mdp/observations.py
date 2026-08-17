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

    factor = getattr(command, "terminal_default_pose_expert_tracking_factor", None)
    if factor is None:
        return pos.view(env.num_envs, -1)
    if factor.shape != (env.num_envs,):
        raise RuntimeError(
            "terminal_default_pose_expert_tracking_factor must have shape "
            f"({env.num_envs},), got {tuple(factor.shape)}."
        )
    return pos.view(env.num_envs, -1) * factor.to(dtype=pos.dtype).unsqueeze(-1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    observation = mat[..., :2].reshape(mat.shape[0], -1)
    factor = getattr(command, "terminal_default_pose_expert_tracking_factor", None)
    if factor is None:
        return observation
    if factor.shape != (env.num_envs,):
        raise RuntimeError(
            "terminal_default_pose_expert_tracking_factor must have shape "
            f"({env.num_envs},), got {tuple(factor.shape)}."
        )
    return observation * factor.to(dtype=observation.dtype).unsqueeze(-1)


def terminal_default_pose_alpha(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Expose terminal q-transition progress so the policy sees the mode switch."""

    command: MotionCommand = env.command_manager.get_term(command_name)
    alpha = getattr(command, "terminal_default_pose_alpha", None)
    if alpha is None:
        return torch.zeros((env.num_envs, 1), dtype=command.joint_pos.dtype, device=command.joint_pos.device)
    if alpha.shape != (env.num_envs,):
        raise RuntimeError(
            f"terminal_default_pose_alpha must have shape ({env.num_envs},), got {tuple(alpha.shape)}."
        )
    return alpha.to(dtype=command.joint_pos.dtype).unsqueeze(-1)


def terminal_default_pose_active(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Expose the one-way terminal target switch separately from its blend.

    ``alpha`` is zero both before latching and on the exact static source-q
    latch frame.  The binary mode bit removes that ambiguity for the policy:
    after it becomes one, source tracking has been disabled and the command
    joint trajectory is the only pose objective.
    """

    command: MotionCommand = env.command_manager.get_term(command_name)
    active = getattr(command, "terminal_default_pose_latched", None)
    if active is None:
        return torch.zeros((env.num_envs, 1), dtype=command.joint_pos.dtype, device=command.joint_pos.device)
    if active.shape != (env.num_envs,):
        raise RuntimeError(
            f"terminal_default_pose_latched must have shape ({env.num_envs},), got {tuple(active.shape)}."
        )
    return active.to(dtype=command.joint_pos.dtype).unsqueeze(-1)


def first_foothold_height_offsets(
    env: ManagerBasedEnv,
    command_name: str,
    foot_body_names: tuple[str, ...] | list[str],
) -> torch.Tensor:
    """Expose the current source Z correction for each configured foot.

    Height randomization changes only the arriving expert foot's body-position
    target.  The exact correction is therefore part of the command state, not
    hidden reward information.  Keeping it in the Actor and Critic makes the
    local foothold adaptation fully observable while preserving the original
    source joint command and all non-foot body targets.
    """

    if not foot_body_names:
        raise ValueError("foot_body_names must contain at least one body name.")
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_names = getattr(command.cfg, "body_names", None)
    if body_names is None:
        raise RuntimeError("First-foothold offset observation requires command.cfg.body_names.")
    missing_body_names = [name for name in foot_body_names if name not in body_names]
    if missing_body_names:
        raise ValueError(
            "First-foothold offset observation requested bodies not tracked by the motion command: "
            f"{missing_body_names}."
        )

    offsets = getattr(command, "first_foothold_height_offsets", None)
    dtype = command.joint_pos.dtype
    device = command.time_steps.device
    if offsets is None:
        return torch.zeros((env.num_envs, len(foot_body_names)), dtype=dtype, device=device)
    if offsets.shape != (env.num_envs, len(body_names)):
        raise RuntimeError(
            "first_foothold_height_offsets must have shape "
            f"({env.num_envs}, {len(body_names)}), got {tuple(offsets.shape)}."
        )
    body_ids = torch.tensor([body_names.index(name) for name in foot_body_names], dtype=torch.long, device=device)
    return offsets[:, body_ids].to(dtype=dtype)


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
