"""Distillation-only domain randomization events."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCamera
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
)


def randomize_camera_extrinsics(
    env,
    env_ids: torch.Tensor,
    sensor_cfg: SceneEntityCfg,
    parent_asset_cfg: SceneEntityCfg,
    translation_range_m: tuple[float, float],
    rotation_range_rad: tuple[float, float],
) -> None:
    """Apply non-cumulative camera-local translation and RPY perturbations."""

    if not (
        len(translation_range_m) == 2
        and translation_range_m[0] <= translation_range_m[1]
        and len(rotation_range_rad) == 2
        and rotation_range_rad[0] <= rotation_range_rad[1]
    ):
        raise ValueError("camera extrinsic ranges must be ordered pairs")
    if max(abs(value) for value in rotation_range_rad) >= math.pi:
        raise ValueError("camera rotation perturbation must stay below pi radians")
    sensor = env.scene.sensors[sensor_cfg.name]
    if not isinstance(sensor, TiledCamera):
        raise TypeError(f"sensor {sensor_cfg.name!r} must be a TiledCamera")
    robot = env.scene[parent_asset_cfg.name]
    if len(parent_asset_cfg.body_ids) != 1:
        raise ValueError("camera parent_asset_cfg must resolve exactly one body")
    body_id = parent_asset_cfg.body_ids[0]
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    count = ids.numel()
    if count == 0:
        return

    nominal_pos = torch.tensor(sensor.cfg.offset.pos, device=env.device).repeat(count, 1)
    nominal_quat = torch.tensor(sensor.cfg.offset.rot, device=env.device).repeat(count, 1)
    nominal_quat = convert_camera_frame_orientation_convention(
        nominal_quat,
        origin=sensor.cfg.offset.convention,
        target="world",
    )
    translation_delta = torch.empty(count, 3, device=env.device).uniform_(*translation_range_m)
    angles = torch.empty(count, 3, device=env.device).uniform_(*rotation_range_rad)
    rotation_delta = quat_from_euler_xyz(angles[:, 0], angles[:, 1], angles[:, 2])
    local_pos = nominal_pos + translation_delta
    local_quat = quat_mul(rotation_delta, nominal_quat)

    parent_pos = robot.data.body_pos_w[ids, body_id]
    parent_quat = robot.data.body_quat_w[ids, body_id]
    world_pos = parent_pos + quat_apply(parent_quat, local_pos)
    world_quat = quat_mul(parent_quat, local_quat)
    sensor.set_world_poses(world_pos, world_quat, env_ids=ids, convention="world")


__all__ = ["randomize_camera_extrinsics"]
