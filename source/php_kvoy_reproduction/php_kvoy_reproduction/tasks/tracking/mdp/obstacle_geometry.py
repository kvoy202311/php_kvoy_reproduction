"""Pure PyTorch geometry used by ELF3 climb obstacles and offline tests."""

from __future__ import annotations

import math

import torch


def _validate_range(name: str, value_range: tuple[float, float], *, positive: bool = False) -> None:
    if len(value_range) != 2 or value_range[0] > value_range[1]:
        raise ValueError(f"{name} must be an ordered (minimum, maximum) pair, got {value_range}.")
    if positive and value_range[0] <= 0.0:
        raise ValueError(f"{name} must remain positive, got {value_range}.")


def sample_climb_box_sizes(
    count: int,
    *,
    length_range: tuple[float, float],
    width_range: tuple[float, float],
    height_range: tuple[float, float],
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Sample physical box dimensions as ``[length, width, height]``."""

    if count < 0:
        raise ValueError(f"count must be non-negative, got {count}.")
    for name, value_range in (
        ("length_range", length_range),
        ("width_range", width_range),
        ("height_range", height_range),
    ):
        _validate_range(name, value_range, positive=True)
    ranges = torch.tensor((length_range, width_range, height_range), dtype=torch.float32, device=device)
    return ranges[:, 0] + torch.rand((count, 3), device=device) * (ranges[:, 1] - ranges[:, 0])


def nominal_environment_mask(
    env_ids: torch.Tensor,
    *,
    num_envs: int,
    nominal_fraction: float,
) -> torch.Tensor:
    """Return a deterministic, exact-size nominal-environment partition."""

    if env_ids.ndim != 1 or env_ids.dtype != torch.long:
        raise TypeError("env_ids must be a one-dimensional torch.long tensor.")
    if num_envs <= 0:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")
    if not 0.0 <= nominal_fraction <= 1.0:
        raise ValueError(f"nominal_fraction must be in [0, 1], got {nominal_fraction}.")
    if torch.any(env_ids < 0) or torch.any(env_ids >= num_envs):
        raise IndexError(f"env_ids contains an index outside [0, {num_envs - 1}].")

    nominal_count = int(num_envs * nominal_fraction + 0.5)
    return env_ids < nominal_count


def oriented_box_local_xy(
    points_w: torch.Tensor,
    centers_w: torch.Tensor,
    orientations_w: torch.Tensor,
) -> torch.Tensor:
    """Express batched world points in an oriented box's local ``x/y`` plane.

    ``points_w`` may contain any number of point dimensions after the leading
    environment dimension, e.g. ``[N, P, 3]`` or ``[N, F, C, 3]``.  The box
    is yaw-only in the climb task, but extracting yaw from the full wxyz
    quaternion keeps this helper valid when the platform pose is randomized.
    """

    if points_w.ndim < 3 or points_w.shape[-1] != 3:
        raise ValueError(f"points_w must have shape [N, ..., 3], got {points_w.shape}.")
    if centers_w.shape != (points_w.shape[0], 3):
        raise ValueError(f"centers_w must have shape {(points_w.shape[0], 3)}, got {centers_w.shape}.")
    if orientations_w.shape != (points_w.shape[0], 4):
        raise ValueError(
            f"orientations_w must have shape {(points_w.shape[0], 4)}, got {orientations_w.shape}."
        )

    # Extract yaw from a wxyz quaternion, then apply the inverse planar
    # rotation. Roll and pitch do not affect this kinematic platform.
    w, x, y, z = orientations_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    broadcast_shape = (points_w.shape[0],) + (1,) * (points_w.ndim - 2)
    cos_yaw = torch.cos(yaw).view(broadcast_shape)
    sin_yaw = torch.sin(yaw).view(broadcast_shape)
    center_xy = centers_w[:, None, :2]
    while center_xy.ndim < points_w.ndim:
        center_xy = center_xy.unsqueeze(1)
    delta = points_w[..., :2] - center_xy
    local_x = cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1]
    local_y = -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1]
    return torch.stack((local_x, local_y), dim=-1)


def points_inside_oriented_box_xy(
    points_w: torch.Tensor,
    centers_w: torch.Tensor,
    orientations_w: torch.Tensor,
    sizes: torch.Tensor,
    *,
    margin: float = 0.0,
) -> torch.Tensor:
    """Test batched world points against yaw-oriented rectangular footprints."""

    if points_w.ndim != 3 or points_w.shape[-1] != 3:
        raise ValueError(f"points_w must have shape [N, P, 3], got {points_w.shape}.")
    if centers_w.shape != (points_w.shape[0], 3):
        raise ValueError(f"centers_w must have shape {(points_w.shape[0], 3)}, got {centers_w.shape}.")
    if orientations_w.shape != (points_w.shape[0], 4):
        raise ValueError(
            f"orientations_w must have shape {(points_w.shape[0], 4)}, got {orientations_w.shape}."
        )
    if sizes.shape != (points_w.shape[0], 3):
        raise ValueError(f"sizes must have shape {(points_w.shape[0], 3)}, got {sizes.shape}.")
    if margin < 0.0:
        raise ValueError(f"margin must be non-negative, got {margin}.")

    local_xy = oriented_box_local_xy(points_w, centers_w, orientations_w)

    half_size = 0.5 * sizes[:, None, :2] + margin
    return (local_xy[..., 0].abs() <= half_size[..., 0]) & (local_xy[..., 1].abs() <= half_size[..., 1])


def foot_sole_corners_world(
    foot_positions_w: torch.Tensor,
    foot_orientations_w: torch.Tensor,
    sole_corners_b: torch.Tensor,
) -> torch.Tensor:
    """Transform one set of sole sample points from each foot-link frame.

    The ankle link origin is not a safe proxy for the physical toe or heel.
    This helper transforms the actual sole corners, preserving pitch and roll
    so a valid forefoot-first contact is not mistaken for a flat-foot target.
    """

    if foot_positions_w.ndim != 3 or foot_positions_w.shape[-1] != 3:
        raise ValueError(f"foot_positions_w must have shape [N, F, 3], got {foot_positions_w.shape}.")
    if foot_orientations_w.shape != foot_positions_w.shape[:-1] + (4,):
        raise ValueError(
            "foot_orientations_w must have shape "
            f"{foot_positions_w.shape[:-1] + (4,)}, got {foot_orientations_w.shape}."
        )
    if sole_corners_b.ndim != 2 or sole_corners_b.shape[-1] != 3 or sole_corners_b.shape[0] == 0:
        raise ValueError(f"sole_corners_b must have shape [C, 3] with C > 0, got {sole_corners_b.shape}.")

    local_corners = sole_corners_b.to(device=foot_positions_w.device, dtype=foot_positions_w.dtype)
    corner_count = local_corners.shape[0]
    vectors = local_corners.view(1, 1, corner_count, 3)
    quat_vector = foot_orientations_w[..., 1:].unsqueeze(-2).expand(-1, -1, corner_count, -1)
    quat_scalar = foot_orientations_w[..., :1].unsqueeze(-2)

    # Quaternion-vector rotation for wxyz quaternions.  This avoids importing
    # Isaac Lab math utilities into the pure geometry module used by tests.
    twice_cross = 2.0 * torch.cross(quat_vector, vectors.expand_as(quat_vector), dim=-1)
    rotated = vectors + quat_scalar * twice_cross + torch.cross(quat_vector, twice_cross, dim=-1)
    return foot_positions_w.unsqueeze(-2) + rotated


def terminal_sole_support_plane_z(
    foot_positions_w: torch.Tensor,
    foot_orientations_w: torch.Tensor,
    sole_corners_b: torch.Tensor,
) -> torch.Tensor:
    """Return the lowest physical sole point for every terminal reference.

    The source clips may finish with pitched or rolled feet, so an ankle-link
    origin (or a nominal box height) is not a reliable terminal support plane.
    Taking the minimum across both configured feet and all their sole samples
    gives one clip-specific plane that can be aligned to a runtime platform.
    """

    sole_corners_w = foot_sole_corners_world(foot_positions_w, foot_orientations_w, sole_corners_b)
    return sole_corners_w[..., 2].amin(dim=(1, 2))


def terminal_platform_z_alignment(
    source_support_z: torch.Tensor,
    platform_top_z: torch.Tensor,
    final_hold_count: torch.Tensor,
    *,
    ramp_steps: int,
    step_dt: float,
    terminal_clearance: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return smooth terminal-reference z position/velocity corrections.

    The correction is zero until a clip has reached its extra final-frame
    hold.  It then smoothly moves the *whole* immutable body reference so its
    source sole support plane reaches the actual sampled platform top.  The
    matching z velocity makes the temporary translation kinematically
    explicit to velocity-tracking rewards.  No joint-space reference is
    altered.
    """

    if source_support_z.ndim != 1:
        raise ValueError(f"source_support_z must have shape [N], got {source_support_z.shape}.")
    if platform_top_z.shape != source_support_z.shape:
        raise ValueError(
            "platform_top_z must match source_support_z, "
            f"got {platform_top_z.shape} and {source_support_z.shape}."
        )
    if final_hold_count.shape != source_support_z.shape:
        raise ValueError(
            "final_hold_count must match source_support_z, "
            f"got {final_hold_count.shape} and {source_support_z.shape}."
        )
    if platform_top_z.device != source_support_z.device or final_hold_count.device != source_support_z.device:
        raise ValueError("terminal alignment tensors must all share one device.")
    if ramp_steps <= 0:
        raise ValueError(f"ramp_steps must be positive, got {ramp_steps}.")
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}.")
    if not math.isfinite(terminal_clearance) or terminal_clearance < 0.0:
        raise ValueError(
            "terminal_clearance must be a finite non-negative value, "
            f"got {terminal_clearance}."
        )

    progress = (final_hold_count.to(dtype=source_support_z.dtype) / float(ramp_steps)).clamp(0.0, 1.0)
    blend = progress * progress * (3.0 - 2.0 * progress)
    # This is the derivative of the cubic blend.  It is exactly zero at both
    # endpoints, including every post-ramp final-frame-hold step.
    blend_rate = 6.0 * progress * (1.0 - progress) / (float(ramp_steps) * step_dt)
    target_delta_z = platform_top_z.to(dtype=source_support_z.dtype) + terminal_clearance - source_support_z
    return blend * target_delta_z, blend_rate * target_delta_z, final_hold_count >= ramp_steps


def _smoothstep_between(value: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Return a cubic smooth step from zero at ``start`` to one at ``end``."""

    normalized = ((value - start) / (end - start)).clamp(min=0.0, max=1.0)
    return normalized * normalized * (3.0 - 2.0 * normalized)


def _foothold_boundary_values(
    sole_corners_w: torch.Tensor,
    centers_w: torch.Tensor,
    orientations_w: torch.Tensor,
    sizes: torch.Tensor,
    *,
    approach_side: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the four physical sole-boundary quantities used by foothold terms."""

    local_xy = oriented_box_local_xy(sole_corners_w, centers_w, orientations_w)
    # ``progress`` increases from the physical approach edge toward the far
    # edge for either configured approach side.
    progress = -approach_side * local_xy[..., 0]
    half_length = 0.5 * sizes[:, 0].view(-1, 1)
    near_edge = -half_length
    far_edge = half_length
    rear_progress = progress.amin(dim=-1)
    front_progress = progress.amax(dim=-1)
    heel_overhang = (near_edge - rear_progress).clamp_min(0.0)
    forefoot_inside = front_progress - near_edge
    far_clearance = far_edge - front_progress
    lateral_clearance = 0.5 * sizes[:, 1].view(-1, 1) - local_xy[..., 1].abs().amax(dim=-1)
    return heel_overhang, forefoot_inside, far_clearance, lateral_clearance


def foothold_safety_score(
    sole_corners_w: torch.Tensor,
    centers_w: torch.Tensor,
    orientations_w: torch.Tensor,
    sizes: torch.Tensor,
    *,
    approach_side: float,
    max_heel_overhang: float,
    min_forefoot_inside: float,
    far_edge_margin: float,
    lateral_margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score safe sole placement on a yaw-oriented, per-environment platform.

    The returned score has shape ``[N, F]``.  The accompanying boolean mask
    is deliberately strict: a foot receives *no positive support reward* if
    its rear-most sole point exceeds ``max_heel_overhang``, its forefoot does
    not get onto the platform, or any sole point violates the far/lateral safe
    boundaries.  The score remains smooth inside that valid set.
    """

    if sole_corners_w.ndim != 4 or sole_corners_w.shape[-1] != 3 or sole_corners_w.shape[2] == 0:
        raise ValueError(f"sole_corners_w must have shape [N, F, C, 3], got {sole_corners_w.shape}.")
    if sizes.shape != (sole_corners_w.shape[0], 3):
        raise ValueError(f"sizes must have shape {(sole_corners_w.shape[0], 3)}, got {sizes.shape}.")
    if approach_side not in (-1.0, 1.0):
        raise ValueError(f"approach_side must be -1.0 or 1.0, got {approach_side}.")
    for name, value in (
        ("max_heel_overhang", max_heel_overhang),
        ("min_forefoot_inside", min_forefoot_inside),
        ("far_edge_margin", far_edge_margin),
        ("lateral_margin", lateral_margin),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    if max_heel_overhang <= 0.0:
        raise ValueError("max_heel_overhang must be positive.")

    heel_overhang, forefoot_inside, far_clearance, lateral_clearance = _foothold_boundary_values(
        sole_corners_w,
        centers_w,
        orientations_w,
        sizes,
        approach_side=approach_side,
    )

    # The small epsilon is only numerical tolerance at the user-selected
    # 5 cm boundary; it does not create an additional rewarded overhang band.
    boundary_epsilon = torch.finfo(sole_corners_w.dtype).eps * 16.0
    valid = (
        (heel_overhang <= max_heel_overhang + boundary_epsilon)
        & (forefoot_inside >= min_forefoot_inside - boundary_epsilon)
        & (far_clearance >= far_edge_margin - boundary_epsilon)
        & (lateral_clearance >= lateral_margin - boundary_epsilon)
    )

    # Within the hard safe region, gently prefer more platform overlap without
    # requiring a flat foot or forcing it toward the platform centre.
    heel_score = torch.exp(-0.5 * torch.square(heel_overhang / max_heel_overhang))
    forefoot_scale = max(min_forefoot_inside, 1.0e-6)
    far_scale = max(far_edge_margin, 1.0e-6)
    lateral_scale = max(lateral_margin, 1.0e-6)
    forefoot_score = 1.0 - torch.exp(-forefoot_inside.clamp_min(0.0) / forefoot_scale)
    far_score = 1.0 - torch.exp(-far_clearance.clamp_min(0.0) / far_scale)
    lateral_score = 1.0 - torch.exp(-lateral_clearance.clamp_min(0.0) / lateral_scale)
    score = heel_score * forefoot_score * far_score * lateral_score
    return score * valid.to(dtype=score.dtype), valid


def foothold_safety_violation(
    sole_corners_w: torch.Tensor,
    centers_w: torch.Tensor,
    orientations_w: torch.Tensor,
    sizes: torch.Tensor,
    *,
    approach_side: float,
    max_heel_overhang: float,
    min_forefoot_inside: float,
    far_edge_margin: float,
    lateral_margin: float,
) -> torch.Tensor:
    """Return a bounded, continuous penalty signal for unsafe sole placement.

    ``foothold_safety_score`` deliberately gives no positive support credit
    outside the valid footprint.  This companion signal supplies the missing
    learning direction: after a foot makes top-surface contact, crossing the
    5 cm heel limit (or another configured boundary) is actively worse than
    returning to the safe region.  The result is zero on every exact safe
    boundary and approaches one as a violation grows.
    """

    if sole_corners_w.ndim != 4 or sole_corners_w.shape[-1] != 3 or sole_corners_w.shape[2] == 0:
        raise ValueError(f"sole_corners_w must have shape [N, F, C, 3], got {sole_corners_w.shape}.")
    if sizes.shape != (sole_corners_w.shape[0], 3):
        raise ValueError(f"sizes must have shape {(sole_corners_w.shape[0], 3)}, got {sizes.shape}.")
    if approach_side not in (-1.0, 1.0):
        raise ValueError(f"approach_side must be -1.0 or 1.0, got {approach_side}.")
    for name, value in (
        ("max_heel_overhang", max_heel_overhang),
        ("min_forefoot_inside", min_forefoot_inside),
        ("far_edge_margin", far_edge_margin),
        ("lateral_margin", lateral_margin),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    if max_heel_overhang <= 0.0:
        raise ValueError("max_heel_overhang must be positive.")

    heel_overhang, forefoot_inside, far_clearance, lateral_clearance = _foothold_boundary_values(
        sole_corners_w,
        centers_w,
        orientations_w,
        sizes,
        approach_side=approach_side,
    )

    # A quarter of the configured margin makes a 1--2 cm post-limit error
    # materially visible to PPO, while preserving a continuous signal at the
    # exact user-approved 5 cm heel boundary.
    def _violation(excess: torch.Tensor, margin: float) -> torch.Tensor:
        scale = max(0.25 * margin, 1.0e-6)
        return 1.0 - torch.exp(-excess.clamp_min(0.0) / scale)

    heel = _violation(heel_overhang - max_heel_overhang, max_heel_overhang)
    forefoot = _violation(min_forefoot_inside - forefoot_inside, min_forefoot_inside)
    far = _violation(far_edge_margin - far_clearance, far_edge_margin)
    lateral = _violation(lateral_margin - lateral_clearance, lateral_margin)
    return torch.maximum(torch.maximum(heel, forefoot), torch.maximum(far, lateral))


def sole_top_height_score(
    sole_corners_w: torch.Tensor,
    centers_w: torch.Tensor,
    sizes: torch.Tensor,
    *,
    height_std: float,
) -> torch.Tensor:
    """Score whether any physical sole sample is close to the current top."""

    if sole_corners_w.ndim != 4 or sole_corners_w.shape[-1] != 3:
        raise ValueError(f"sole_corners_w must have shape [N, F, C, 3], got {sole_corners_w.shape}.")
    if centers_w.shape != (sole_corners_w.shape[0], 3):
        raise ValueError(f"centers_w must have shape {(sole_corners_w.shape[0], 3)}, got {centers_w.shape}.")
    if sizes.shape != (sole_corners_w.shape[0], 3):
        raise ValueError(f"sizes must have shape {(sole_corners_w.shape[0], 3)}, got {sizes.shape}.")
    if height_std <= 0.0:
        raise ValueError(f"height_std must be positive, got {height_std}.")

    platform_top = centers_w[:, 2].view(-1, 1, 1) + 0.5 * sizes[:, 2].view(-1, 1, 1)
    closest_height_error = torch.abs(sole_corners_w[..., 2] - platform_top).amin(dim=-1)
    return torch.exp(-0.5 * torch.square(closest_height_error / height_std))


def sole_surface_alignment_score(
    sole_corners_w: torch.Tensor,
    centers_w: torch.Tensor,
    sizes: torch.Tensor,
    *,
    height_std: float,
    height_tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score whether the complete sampled sole is aligned with the box top.

    ``sole_top_height_score`` intentionally accepts a forefoot-first landing by
    looking only at the closest sole sample.  That is useful for detecting the
    initial impact, but it must not classify a persistent toe stand as settled
    support.  This stricter helper uses the largest absolute height error over
    every configured sole sample.  It therefore remains independent of the
    expert ankle pitch/roll and adapts directly to the physical platform top.

    Returns the smooth alignment score, a hard readiness mask, and the maximum
    absolute sole-to-surface error for diagnostics.
    """

    if sole_corners_w.ndim != 4 or sole_corners_w.shape[-1] != 3 or sole_corners_w.shape[2] == 0:
        raise ValueError(f"sole_corners_w must have shape [N, F, C, 3], got {sole_corners_w.shape}.")
    if centers_w.shape != (sole_corners_w.shape[0], 3):
        raise ValueError(f"centers_w must have shape {(sole_corners_w.shape[0], 3)}, got {centers_w.shape}.")
    if sizes.shape != (sole_corners_w.shape[0], 3):
        raise ValueError(f"sizes must have shape {(sole_corners_w.shape[0], 3)}, got {sizes.shape}.")
    if not math.isfinite(height_std) or height_std <= 0.0:
        raise ValueError(f"height_std must be positive and finite, got {height_std}.")
    if not math.isfinite(height_tolerance) or height_tolerance <= 0.0:
        raise ValueError(f"height_tolerance must be positive and finite, got {height_tolerance}.")

    platform_top = centers_w[:, 2].view(-1, 1, 1) + 0.5 * sizes[:, 2].view(-1, 1, 1)
    maximum_height_error = torch.abs(sole_corners_w[..., 2] - platform_top).amax(dim=-1)
    score = torch.exp(-0.5 * torch.square(maximum_height_error / height_std))
    valid = maximum_height_error <= height_tolerance
    return score, valid, maximum_height_error


def sole_surface_shaping_score(
    sole_corners_w: torch.Tensor,
    centers_w: torch.Tensor,
    sizes: torch.Tensor,
    *,
    tilt_scale: float,
    height_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return non-saturating sole-to-top shaping and its two physical errors.

    Strict support intentionally checks the *largest* absolute sample error in
    :func:`sole_surface_alignment_score`.  That binary fact must not also be
    used as the only learning signal: a badly pitched incoming sole can be
    tens of centimetres from strict support, where a narrow Gaussian is
    numerically indistinguishable from zero.  This helper instead combines
    inverse-square-root scores for sole tilt and closest top approach.  It
    remains informative far from the target but never changes the strict
    support classification.

    Returns the dense score, sole height spread, and closest absolute height
    error, each with shape ``[N, F]``.
    """

    if sole_corners_w.ndim != 4 or sole_corners_w.shape[-1] != 3 or sole_corners_w.shape[2] == 0:
        raise ValueError(f"sole_corners_w must have shape [N, F, C, 3], got {sole_corners_w.shape}.")
    if centers_w.shape != (sole_corners_w.shape[0], 3):
        raise ValueError(f"centers_w must have shape {(sole_corners_w.shape[0], 3)}, got {centers_w.shape}.")
    if sizes.shape != (sole_corners_w.shape[0], 3):
        raise ValueError(f"sizes must have shape {(sole_corners_w.shape[0], 3)}, got {sizes.shape}.")
    for name, value in (("tilt_scale", tilt_scale), ("height_scale", height_scale)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}.")

    sole_heights = sole_corners_w[..., 2]
    platform_top = centers_w[:, 2].view(-1, 1, 1) + 0.5 * sizes[:, 2].view(-1, 1, 1)
    height_spread = sole_heights.amax(dim=-1) - sole_heights.amin(dim=-1)
    closest_height_error = torch.abs(sole_heights - platform_top).amin(dim=-1)
    tilt_score = torch.rsqrt(1.0 + torch.square(height_spread / tilt_scale))
    height_score = torch.rsqrt(1.0 + torch.square(closest_height_error / height_scale))
    return tilt_score * height_score, height_spread, closest_height_error


def foothold_precontact_score(
    sole_corners_w: torch.Tensor,
    centers_w: torch.Tensor,
    orientations_w: torch.Tensor,
    sizes: torch.Tensor,
    *,
    approach_side: float,
    approach_distance: float,
    height_std: float,
) -> torch.Tensor:
    """Smoothly recognize a foot that is arriving at the platform top.

    This intentionally does not require the final safe-footprint mask.  It is
    used to fade conflicting expert tracking *before* contact, allowing the
    policy to correct an otherwise over-long step instead of only reacting
    after it has already struck an edge.
    """

    if approach_side not in (-1.0, 1.0):
        raise ValueError(f"approach_side must be -1.0 or 1.0, got {approach_side}.")
    if approach_distance <= 0.0:
        raise ValueError(f"approach_distance must be positive, got {approach_distance}.")

    local_xy = oriented_box_local_xy(sole_corners_w, centers_w, orientations_w)
    progress = -approach_side * local_xy[..., 0]
    front_progress = progress.amax(dim=-1)
    near_edge = -0.5 * sizes[:, 0].view(-1, 1)
    arrival_score = _smoothstep_between(front_progress, near_edge - approach_distance, near_edge)
    return arrival_score * sole_top_height_score(sole_corners_w, centers_w, sizes, height_std=height_std)


def filtered_platform_contact_score(
    platform_forces_w: torch.Tensor,
    contact_times: torch.Tensor,
    *,
    min_upward_force: float,
    contact_time_scale: float,
) -> torch.Tensor:
    """Return smooth platform-contact quality from filtered world-frame forces."""

    if platform_forces_w.ndim != 3 or platform_forces_w.shape[-1] != 3:
        raise ValueError(
            f"platform_forces_w must have shape [N, F, 3], got {platform_forces_w.shape}."
        )
    if contact_times.shape != platform_forces_w.shape[:-1]:
        raise ValueError(
            f"contact_times must have shape {platform_forces_w.shape[:-1]}, got {contact_times.shape}."
        )
    if min_upward_force <= 0.0 or contact_time_scale <= 0.0:
        raise ValueError("min_upward_force and contact_time_scale must be positive.")

    force_score = filtered_platform_force_score(platform_forces_w, min_upward_force=min_upward_force)
    time_score = (contact_times / contact_time_scale).clamp(min=0.0, max=1.0)
    return force_score * time_score


def filtered_platform_force_score(
    platform_forces_w: torch.Tensor,
    *,
    min_upward_force: float,
) -> torch.Tensor:
    """Score only upward force from a platform-filtered contact matrix."""

    if platform_forces_w.ndim != 3 or platform_forces_w.shape[-1] != 3:
        raise ValueError(
            f"platform_forces_w must have shape [N, F, 3], got {platform_forces_w.shape}."
        )
    if min_upward_force <= 0.0:
        raise ValueError(f"min_upward_force must be positive, got {min_upward_force}.")
    upward_force = platform_forces_w[..., 2].clamp_min(0.0)
    return 1.0 - torch.exp(-upward_force / min_upward_force)


def advance_filtered_platform_contact_time(
    previous_contact_time: torch.Tensor,
    active_platform_support: torch.Tensor,
    *,
    step_dt: float,
) -> torch.Tensor:
    """Accumulate time spent in *filtered* platform support only.

    Isaac Lab's ``current_contact_time`` tracks all contacts of a body, not
    only the counterparts selected by a contact-force filter.  This pure
    helper is reset by :class:`MotionCommand` and lets the first-foothold term
    require continuous contact with the actual ClimbPlatform instead.
    """

    if previous_contact_time.ndim != 2:
        raise ValueError(
            "previous_contact_time must have shape [N, F], "
            f"got {previous_contact_time.shape}."
        )
    if active_platform_support.shape != previous_contact_time.shape:
        raise ValueError(
            "active_platform_support must match previous_contact_time shape, "
            f"got {active_platform_support.shape} and {previous_contact_time.shape}."
        )
    if step_dt <= 0.0:
        raise ValueError(f"step_dt must be positive, got {step_dt}.")
    return torch.where(
        active_platform_support.to(dtype=torch.bool),
        previous_contact_time + step_dt,
        torch.zeros_like(previous_contact_time),
    )


def first_foothold_reference_gate(
    reference_foot_positions_w: torch.Tensor,
    centers_w: torch.Tensor,
    orientations_w: torch.Tensor,
    sizes: torch.Tensor,
    phase: torch.Tensor,
    *,
    approach_side: float,
    reference_activation_distance: float,
    reference_activation_inside: float,
    reference_release_distance: float,
    reference_release_inside: float,
    phase_start: float,
    phase_ramp: float,
    phase_end: float,
    phase_fade: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gate the first-foot transfer and identify the reference-leading foot.

    The spatial component adapts to each environment's sampled platform
    length.  The broad phase window protects against unrelated initial/final
    poses in a motion clip; it does not alter the reference trajectory.
    """

    if reference_foot_positions_w.ndim != 3 or reference_foot_positions_w.shape[-1] != 3:
        raise ValueError(
            "reference_foot_positions_w must have shape [N, F, 3], "
            f"got {reference_foot_positions_w.shape}."
        )
    if reference_foot_positions_w.shape[1] < 2:
        raise ValueError("first_foothold_reference_gate requires at least two feet.")
    if phase.shape != (reference_foot_positions_w.shape[0],):
        raise ValueError(f"phase must have shape {(reference_foot_positions_w.shape[0],)}, got {phase.shape}.")
    if approach_side not in (-1.0, 1.0):
        raise ValueError(f"approach_side must be -1.0 or 1.0, got {approach_side}.")
    for name, value in (
        ("reference_activation_distance", reference_activation_distance),
        ("reference_activation_inside", reference_activation_inside),
        ("reference_release_distance", reference_release_distance),
        ("reference_release_inside", reference_release_inside),
        ("phase_ramp", phase_ramp),
        ("phase_fade", phase_fade),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if not 0.0 <= phase_start < phase_end <= 1.0 or phase_start + phase_ramp > 1.0 or phase_end + phase_fade > 1.0:
        raise ValueError("first-foothold phase boundaries must remain ordered and inside [0, 1].")

    local_xy = oriented_box_local_xy(reference_foot_positions_w, centers_w, orientations_w)
    progress = -approach_side * local_xy[..., 0]
    lead_progress, lead_indices = progress.max(dim=1)
    trailing_progress = progress.min(dim=1).values
    near_edge = -0.5 * sizes[:, 0]
    lead_arrival = _smoothstep_between(
        lead_progress,
        near_edge - reference_activation_distance,
        near_edge + reference_activation_inside,
    )
    trailing_arrival = _smoothstep_between(
        trailing_progress,
        near_edge - reference_release_distance,
        near_edge + reference_release_inside,
    )
    phase_in = _smoothstep_between(
        phase,
        torch.full_like(phase, phase_start),
        torch.full_like(phase, phase_start + phase_ramp),
    )
    phase_out = 1.0 - _smoothstep_between(
        phase,
        torch.full_like(phase, phase_end),
        torch.full_like(phase, phase_end + phase_fade),
    )
    gate = phase_in * phase_out * lead_arrival * (1.0 - trailing_arrival)
    lead_mask = torch.nn.functional.one_hot(lead_indices, num_classes=reference_foot_positions_w.shape[1]).to(
        dtype=reference_foot_positions_w.dtype
    )
    return gate, lead_mask


def climb_box_top_height(
    ground_hits_w: torch.Tensor,
    centers_w: torch.Tensor,
    orientations_w: torch.Tensor,
    sizes: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Overlay an oriented box top on ground-ray hits."""

    on_box = points_inside_oriented_box_xy(ground_hits_w, centers_w, orientations_w, sizes)
    top = centers_w[:, None, 2] + 0.5 * sizes[:, None, 2]
    height = torch.where(on_box, torch.maximum(ground_hits_w[..., 2], top), ground_hits_w[..., 2])
    return height, on_box
