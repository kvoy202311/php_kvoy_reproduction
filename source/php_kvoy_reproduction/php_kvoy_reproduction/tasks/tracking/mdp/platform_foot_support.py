"""Shared, platform-filtered physical foot-support helpers for the climb task.

The generic robot contact sensor reports the aggregate force from every
counterpart.  That is useful for diagnostics, but it cannot tell whether an
ankle is actually supported by the climb platform.  The climb scene therefore
owns one filtered contact sensor per ankle.  This module combines those
filtered forces with real sole geometry and deliberately contains no reward
state, so rewards, success criteria, and the terminal command mode all use
the same physical definition.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NamedTuple

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from php_kvoy_reproduction.tasks.tracking.mdp.obstacle import get_climb_box_sizes
from php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry import (
    filtered_platform_force_score,
    foot_sole_corners_world,
    foothold_safety_score,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PlatformFootSupportSettings(NamedTuple):
    """Validated immutable geometry and sensor settings for the two feet."""

    foot_body_names: tuple[str, ...]
    platform_contact_sensor_names: tuple[str, ...]
    sole_corners_b: tuple[tuple[float, float, float], ...]
    approach_side: float
    max_heel_overhang: float
    min_forefoot_inside: float
    far_edge_margin: float
    lateral_margin: float


class PlatformFootSupportState(NamedTuple):
    """Current, real platform support facts for each configured foot."""

    settings: PlatformFootSupportSettings
    platform: RigidObject
    sizes: torch.Tensor
    sole_corners_w: torch.Tensor
    sole_geometry_score: torch.Tensor
    sole_geometry_valid: torch.Tensor
    sole_plane_height_error: torch.Tensor
    platform_forces_w: torch.Tensor
    upward_forces: torch.Tensor
    active_support: torch.Tensor


def platform_foot_support_settings(params: Mapping[str, object]) -> PlatformFootSupportSettings:
    """Parse the shared two-foot platform-support configuration.

    This intentionally has no contact-time field.  Continuous support is
    stateful and is owned by :class:`MotionCommand`, where it can be advanced
    exactly once per policy step regardless of how many reward terms inspect
    it.
    """

    required_keys = (
        "foot_body_names",
        "platform_contact_sensor_names",
        "sole_corners_b",
        "approach_side",
        "max_heel_overhang",
        "min_forefoot_inside",
        "far_edge_margin",
        "lateral_margin",
    )
    missing = [key for key in required_keys if key not in params]
    if missing:
        raise ValueError(f"platform_foot_support_params is missing required keys: {missing}.")

    raw_foot_names = params["foot_body_names"]
    raw_sensor_names = params["platform_contact_sensor_names"]
    raw_sole_corners = params["sole_corners_b"]
    if not isinstance(raw_foot_names, (list, tuple)) or len(raw_foot_names) != 2:
        raise ValueError("platform_foot_support_params['foot_body_names'] must contain exactly two feet.")
    if not isinstance(raw_sensor_names, (list, tuple)) or len(raw_sensor_names) != len(raw_foot_names):
        raise ValueError(
            "platform_foot_support_params['platform_contact_sensor_names'] must match foot_body_names one-to-one."
        )
    if not isinstance(raw_sole_corners, (list, tuple)) or len(raw_sole_corners) < 4:
        raise ValueError("platform_foot_support_params['sole_corners_b'] must contain at least four sole samples.")

    foot_body_names = tuple(str(name) for name in raw_foot_names)
    platform_contact_sensor_names = tuple(str(name) for name in raw_sensor_names)
    if len(set(foot_body_names)) != len(foot_body_names):
        raise ValueError("platform_foot_support_params['foot_body_names'] must not contain duplicates.")
    if len(set(platform_contact_sensor_names)) != len(platform_contact_sensor_names):
        raise ValueError("platform_foot_support_params['platform_contact_sensor_names'] must not contain duplicates.")

    sole_corners: list[tuple[float, float, float]] = []
    for corner in raw_sole_corners:
        if not isinstance(corner, (list, tuple)) or len(corner) != 3:
            raise ValueError("Every platform-foot sole corner must contain exactly three coordinates.")
        sole_corners.append((float(corner[0]), float(corner[1]), float(corner[2])))

    approach_side = float(params["approach_side"])
    if approach_side not in (-1.0, 1.0):
        raise ValueError(f"approach_side must be -1 or 1, got {approach_side}.")
    for name in ("max_heel_overhang", "min_forefoot_inside", "far_edge_margin", "lateral_margin"):
        if float(params[name]) < 0.0:
            raise ValueError(f"platform-foot support setting {name} must be non-negative, got {params[name]}.")

    return PlatformFootSupportSettings(
        foot_body_names=foot_body_names,
        platform_contact_sensor_names=platform_contact_sensor_names,
        sole_corners_b=tuple(sole_corners),
        approach_side=approach_side,
        max_heel_overhang=float(params["max_heel_overhang"]),
        min_forefoot_inside=float(params["min_forefoot_inside"]),
        far_edge_margin=float(params["far_edge_margin"]),
        lateral_margin=float(params["lateral_margin"]),
    )


def platform_foot_support_state(
    env: ManagerBasedRLEnv,
    robot: Articulation,
    device: str | torch.device,
    platform_cfg: SceneEntityCfg,
    base_size: tuple[float, float, float],
    params: Mapping[str, object],
    *,
    min_upward_force: float,
    sole_height_tolerance: float,
) -> PlatformFootSupportState:
    """Read strict per-foot support from the filtered platform sensors.

    A valid support requires all of the following for that foot:

    * its real sole is inside the safe platform footprint (including the
      requested 5 cm maximum heel overhang);
    * its lowest sole point is close to the physical platform top;
    * the force from *ClimbPlatform* has a sufficient positive world-z
      component.

    No generic ankle force, ground contact, wrist contact, side strike, or
    visual-only proximity can satisfy this state.
    """

    if min_upward_force <= 0.0:
        raise ValueError(f"min_upward_force must be positive, got {min_upward_force}.")
    if sole_height_tolerance <= 0.0:
        raise ValueError(f"sole_height_tolerance must be positive, got {sole_height_tolerance}.")

    settings = platform_foot_support_settings(params)
    platform: RigidObject = env.scene[platform_cfg.name]
    sizes = get_climb_box_sizes(platform, base_size=base_size, device=platform.device)
    if sizes.shape != (env.num_envs, 3):
        raise RuntimeError(
            "Platform size query returned the wrong shape: "
            f"expected {(env.num_envs, 3)}, got {tuple(sizes.shape)}."
        )

    foot_body_ids = torch.tensor(
        [robot.body_names.index(name) for name in settings.foot_body_names],
        dtype=torch.long,
        device=device,
    )
    sole_corners_b = torch.tensor(
        settings.sole_corners_b,
        dtype=robot.data.body_pos_w.dtype,
        device=device,
    )
    sole_corners_w = foot_sole_corners_world(
        robot.data.body_pos_w[:, foot_body_ids],
        robot.data.body_quat_w[:, foot_body_ids],
        sole_corners_b,
    )
    sole_geometry_score, sole_geometry_valid = foothold_safety_score(
        sole_corners_w,
        platform.data.root_pos_w,
        platform.data.root_quat_w,
        sizes,
        approach_side=settings.approach_side,
        max_heel_overhang=settings.max_heel_overhang,
        min_forefoot_inside=settings.min_forefoot_inside,
        far_edge_margin=settings.far_edge_margin,
        lateral_margin=settings.lateral_margin,
    )

    # ``foothold_safety_score`` permits the user-approved 5 cm rear overhang.
    # The terminal task additionally needs a genuine top-surface proximity;
    # using the lowest sole point is robust to a still-pitched landing foot and
    # never mistakes the ankle origin for contact geometry.
    platform_top = platform.data.root_pos_w[:, 2] + 0.5 * sizes[:, 2]
    sole_plane_height_error = sole_corners_w[..., 2].amin(dim=-1) - platform_top[:, None]

    platform_forces: list[torch.Tensor] = []
    for sensor_name in settings.platform_contact_sensor_names:
        sensor: ContactSensor = env.scene.sensors[sensor_name]
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError(
                f"Platform-foot sensor '{sensor_name}' must configure filter_prim_paths_expr for ClimbPlatform."
            )
        if force_matrix.ndim != 4 or force_matrix.shape[0] != env.num_envs or force_matrix.shape[1] != 1:
            raise RuntimeError(
                f"Platform-foot sensor '{sensor_name}' must contain exactly one ankle body; "
                f"got force matrix shape {tuple(force_matrix.shape)}."
            )
        if force_matrix.shape[2] == 0:
            raise RuntimeError(f"Platform-foot sensor '{sensor_name}' resolved no ClimbPlatform filter bodies.")
        platform_forces.append(force_matrix[:, 0].sum(dim=1))
    platform_forces_w = torch.stack(platform_forces, dim=1)
    upward_forces = platform_forces_w[..., 2].clamp_min(0.0)
    active_support = (
        sole_geometry_valid
        & (sole_plane_height_error.abs() <= sole_height_tolerance)
        & (upward_forces >= min_upward_force)
    )

    return PlatformFootSupportState(
        settings=settings,
        platform=platform,
        sizes=sizes,
        sole_corners_w=sole_corners_w,
        sole_geometry_score=sole_geometry_score,
        sole_geometry_valid=sole_geometry_valid,
        sole_plane_height_error=sole_plane_height_error,
        platform_forces_w=platform_forces_w,
        upward_forces=upward_forces,
        active_support=active_support,
    )


def platform_foot_support_score(
    state: PlatformFootSupportState,
    filtered_contact_time: torch.Tensor,
    *,
    min_upward_force: float,
    contact_time_scale: float,
    sole_height_tolerance: float,
) -> torch.Tensor:
    """Return a smooth [0, 1] support score from strict platform facts."""

    expected_shape = state.upward_forces.shape
    if filtered_contact_time.shape != expected_shape:
        raise ValueError(
            "filtered_contact_time must match the configured feet, "
            f"got {filtered_contact_time.shape} and {expected_shape}."
        )
    if contact_time_scale <= 0.0:
        raise ValueError(f"contact_time_scale must be positive, got {contact_time_scale}.")
    if sole_height_tolerance <= 0.0:
        raise ValueError(f"sole_height_tolerance must be positive, got {sole_height_tolerance}.")

    force_score = filtered_platform_force_score(state.platform_forces_w, min_upward_force=min_upward_force)
    time_score = (filtered_contact_time / contact_time_scale).clamp(min=0.0, max=1.0)
    height_score = torch.exp(-0.5 * torch.square(state.sole_plane_height_error / sole_height_tolerance))
    return state.sole_geometry_score * height_score * force_score * time_score


def platform_foot_load_score(
    state: PlatformFootSupportState,
    robot: Articulation,
    *,
    min_total_load_fraction: float,
    gravity_magnitude: float = 9.81,
) -> torch.Tensor:
    """Score how much of the robot's weight is carried by platform feet.

    ``min_total_load_fraction=0`` keeps the helper neutral.  A positive value
    prevents a hand from carrying the primary load while both feet merely
    touch the platform.  The denominator uses the actual USD mass parsed by
    Isaac Lab, rather than a hard-coded robot mass.
    """

    if not 0.0 <= min_total_load_fraction <= 1.0:
        raise ValueError(
            "min_total_load_fraction must lie in [0, 1], "
            f"got {min_total_load_fraction}."
        )
    if gravity_magnitude <= 0.0:
        raise ValueError(f"gravity_magnitude must be positive, got {gravity_magnitude}.")
    if min_total_load_fraction == 0.0:
        return torch.ones(
            state.upward_forces.shape[0],
            dtype=state.upward_forces.dtype,
            device=state.upward_forces.device,
        )

    default_mass = getattr(robot.data, "default_mass", None)
    if default_mass is None:
        raise RuntimeError("Platform foot-load gating requires Articulation.data.default_mass.")
    if default_mass.ndim == 1:
        total_mass = default_mass.sum().expand(state.upward_forces.shape[0])
    elif default_mass.ndim == 2 and default_mass.shape[0] == state.upward_forces.shape[0]:
        total_mass = default_mass.sum(dim=1)
    else:
        raise RuntimeError(
            "Platform foot-load gating expected default_mass with shape [bodies] or [num_envs, bodies], "
            f"got {tuple(default_mass.shape)}."
        )
    # ``min_total_load_fraction`` and ``gravity_magnitude`` were validated as
    # strictly positive above, so this native-device mass check is equivalent
    # to checking ``minimum_load`` below.  In the common Isaac Lab case where
    # default masses live on CPU, doing it before the transfer avoids a CUDA
    # scalar synchronization in a helper evaluated by several terms per step.
    if torch.any(total_mass <= 0.0):
        raise RuntimeError("Platform foot-load gating received a non-positive robot weight.")

    # Isaac Lab can keep articulation default masses on CPU while contact
    # forces are produced on the simulation device.  Move both dtype and
    # device explicitly before combining them; ``Tensor.to(dtype=...)`` alone
    # deliberately preserves the CPU device and would fail on the first
    # training step.
    total_weight = total_mass.to(
        device=state.upward_forces.device,
        dtype=state.upward_forces.dtype,
    ) * gravity_magnitude
    minimum_load = total_weight * min_total_load_fraction
    return (state.upward_forces.sum(dim=1) / minimum_load).clamp(min=0.0, max=1.0)


def platform_foot_load_valid(
    state: PlatformFootSupportState,
    robot: Articulation,
    *,
    min_total_load_fraction: float,
    gravity_magnitude: float = 9.81,
) -> torch.Tensor:
    """Return whether the two platform feet carry the requested load share."""

    if min_total_load_fraction == 0.0:
        return torch.ones(state.upward_forces.shape[0], dtype=torch.bool, device=state.upward_forces.device)
    return platform_foot_load_score(
        state,
        robot,
        min_total_load_fraction=min_total_load_fraction,
        gravity_magnitude=gravity_magnitude,
    ) >= 1.0
