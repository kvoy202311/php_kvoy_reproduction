from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

from .obstacle_geometry import (
    _validate_range,
    climb_box_top_height,
    nominal_environment_mask,
    points_inside_oriented_box_xy,
    sample_climb_box_sizes,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_climb_box_geometry(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    *,
    asset_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    length_range: tuple[float, float],
    width_range: tuple[float, float],
    height_range: tuple[float, float],
    nominal_size_fraction: float = 0.0,
) -> None:
    """Randomize each environment's box scale before PhysX parses the scene.

    The sampled physical dimensions are stored on the rigid object and are
    subsequently shared by pose reset, the analytical height scan, and the
    evaluation checks.  ``nominal_size_fraction`` reserves an exact-size group
    for physically valid random-phase initialization; the remaining group
    carries the configured geometry randomization.  The initialization group
    is exposed as ``_climb_box_random_phase_env_mask``.  The old
    ``_climb_box_nominal_geometry_mask`` name remains a compatibility alias
    for externally configured tasks.  This event must run in ``prestartup``
    mode with ``scene.replicate_physics=False``.
    """

    if env.sim.is_playing():
        raise RuntimeError("Climb-box geometry can only be randomized in prestartup mode.")

    asset: RigidObject = env.scene[asset_cfg.name]
    num_envs = env.scene.num_envs
    if env_ids is None:
        env_ids_cpu = torch.arange(num_envs, dtype=torch.long, device="cpu")
    else:
        env_ids_cpu = torch.as_tensor(env_ids, dtype=torch.long, device="cpu")

    base_size_tensor = torch.tensor(base_size, dtype=torch.float32, device="cpu")
    if base_size_tensor.shape != (3,) or torch.any(base_size_tensor <= 0.0):
        raise ValueError(f"base_size must contain three positive dimensions, got {base_size}.")

    sampled_sizes = sample_climb_box_sizes(
        len(env_ids_cpu),
        length_range=length_range,
        width_range=width_range,
        height_range=height_range,
        device="cpu",
    )
    random_phase_mask = nominal_environment_mask(
        env_ids_cpu,
        num_envs=num_envs,
        nominal_fraction=nominal_size_fraction,
    )
    if nominal_size_fraction > 0.0:
        dimension_ranges = (length_range, width_range, height_range)
        for dimension, (nominal_size, size_range) in enumerate(zip(base_size, dimension_ranges, strict=True)):
            if not size_range[0] <= nominal_size <= size_range[1]:
                raise ValueError(
                    f"base_size[{dimension}]={nominal_size} must lie inside its sampling range {size_range}."
                )
        sampled_sizes[random_phase_mask] = base_size_tensor

    all_sizes = base_size_tensor.repeat(num_envs, 1)
    all_sizes[env_ids_cpu] = sampled_sizes
    asset._climb_box_sizes = all_sizes
    asset._climb_box_scales = all_sizes / base_size_tensor
    all_random_phase_mask = torch.zeros(num_envs, dtype=torch.bool, device="cpu")
    all_random_phase_mask[env_ids_cpu] = random_phase_mask
    asset._climb_box_random_phase_env_mask = all_random_phase_mask
    # Keep the historical attribute as an alias so existing external configs
    # continue to initialize safely.  Current climb code uses the explicit
    # random-phase name above and never interprets this as geometry metadata.
    asset._climb_box_nominal_geometry_mask = all_random_phase_mask

    # Import USD modules lazily so pure geometry helpers remain unit-testable
    # without launching Isaac Sim.
    import omni.usd
    from pxr import Gf, Sdf, UsdGeom, Vt

    stage = omni.usd.get_context().get_stage()
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
    if len(prim_paths) != num_envs:
        raise RuntimeError(
            f"Expected {num_envs} climb-box prims for {asset.cfg.prim_path}, found {len(prim_paths)}."
        )

    scales = sampled_sizes / base_size_tensor
    with Sdf.ChangeBlock():
        for sample_index, env_id in enumerate(env_ids_cpu.tolist()):
            prim_path = prim_paths[env_id]
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)
            scale_spec.default = Gf.Vec3f(*scales[sample_index].tolist())
            if not has_scale_attr:
                order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if order_spec is None:
                    order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])

    # Fail during initialization if USD did not retain exactly the scale that
    # is shared with the analytical height map.
    for env_id in env_ids_cpu.tolist():
        scale_value = stage.GetPrimAtPath(prim_paths[env_id]).GetAttribute("xformOp:scale").Get()
        if scale_value is None:
            raise RuntimeError(f"Climb-box scale attribute was not written for {prim_paths[env_id]}.")
        actual_scale = torch.tensor(tuple(scale_value), dtype=torch.float32)
        expected_scale = asset._climb_box_scales[env_id]
        if not torch.allclose(actual_scale, expected_scale, rtol=0.0, atol=1.0e-6):
            raise RuntimeError(
                f"Climb-box USD scale mismatch for {prim_paths[env_id]}: "
                f"expected {expected_scale.tolist()}, got {actual_scale.tolist()}."
            )


def get_climb_box_sizes(
    asset: RigidObject,
    *,
    base_size: tuple[float, float, float],
    device: str | torch.device,
) -> torch.Tensor:
    """Return the exact per-environment dimensions used by the collider."""

    sizes = getattr(asset, "_climb_box_sizes", None)
    if sizes is None:
        sizes = torch.tensor(base_size, dtype=torch.float32).repeat(asset.num_instances, 1)
    return sizes.to(device=device, dtype=torch.float32)


def reset_climb_box_pose(
    env: ManagerBasedEnv,
    env_ids: Sequence[int] | torch.Tensor,
    *,
    asset_cfg: SceneEntityCfg,
    base_center_xy: tuple[float, float],
    base_size: tuple[float, float, float],
    position_range: dict[str, tuple[float, float]],
    yaw_range: tuple[float, float],
) -> None:
    """Reset box x/y/yaw and keep its lower face exactly on the ground."""

    _validate_range("yaw_range", yaw_range)
    for axis in ("x", "y"):
        _validate_range(f"position_range[{axis!r}]", position_range.get(axis, (0.0, 0.0)))

    asset: RigidObject = env.scene[asset_cfg.name]
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device)
    sizes = get_climb_box_sizes(asset, base_size=base_size, device=asset.device)[env_ids]

    xy_ranges = torch.tensor(
        (position_range.get("x", (0.0, 0.0)), position_range.get("y", (0.0, 0.0))),
        dtype=torch.float32,
        device=asset.device,
    )
    xy_delta = math_utils.sample_uniform(
        xy_ranges[:, 0], xy_ranges[:, 1], (len(env_ids), 2), device=asset.device
    )
    yaw = math_utils.sample_uniform(*yaw_range, (len(env_ids),), device=asset.device)

    positions = env.scene.env_origins[env_ids].clone()
    positions[:, :2] += torch.tensor(base_center_xy, dtype=torch.float32, device=asset.device) + xy_delta
    positions[:, 2] += 0.5 * sizes[:, 2]
    zeros = torch.zeros_like(yaw)
    orientations = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    asset.write_root_pose_to_sim(torch.cat((positions, orientations), dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=asset.device), env_ids=env_ids)


def climb_box_geometry_errors(
    env: ManagerBasedEnv,
    *,
    asset_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
) -> dict[str, torch.Tensor]:
    """Return per-environment invariants used by the evaluation gate.

    A valid platform has its bottom face on the environment ground and carries
    the exact size-to-USD-scale relationship recorded before PhysX startup.
    """

    asset: RigidObject = env.scene[asset_cfg.name]
    sizes = get_climb_box_sizes(asset, base_size=base_size, device=asset.device)
    bottom_height_error = torch.abs(
        asset.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2] - 0.5 * sizes[:, 2]
    )

    scales = getattr(asset, "_climb_box_scales", None)
    if scales is None:
        scales = torch.ones_like(sizes, device="cpu")
    scales = scales.to(device=asset.device, dtype=torch.float32)
    base = torch.tensor(base_size, dtype=torch.float32, device=asset.device)
    size_scale_error = torch.max(torch.abs(sizes - scales * base), dim=1).values
    return {
        "bottom_height_error": bottom_height_error,
        "size_scale_error": size_scale_error,
    }
