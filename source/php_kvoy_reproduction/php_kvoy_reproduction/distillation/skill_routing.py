"""Dependency-light route identifiers and balanced episode allocation."""

from __future__ import annotations

import math

import torch


LOCOMOTION_SKILL_ID = 0
CLIMB_SKILL_ID = 1
DOWN_ROLL_SKILL_ID = 2
NUM_SKILLS = 3


def planar_command_speed_valid(
    commands: torch.Tensor,
    *,
    maximum_speed: float,
) -> torch.Tensor:
    """Return which finite planar commands stay inside the deployment speed contract."""

    if commands.ndim != 2 or commands.shape[1] != 2 or not commands.is_floating_point():
        raise ValueError("commands must be a floating-point [N, 2] tensor")
    if not torch.isfinite(commands).all():
        raise ValueError("commands must contain only finite values")
    if not math.isfinite(maximum_speed) or maximum_speed <= 0.0:
        raise ValueError("maximum_speed must be finite and positive")
    return torch.linalg.vector_norm(commands, dim=1) <= maximum_speed + 1.0e-6


def approach_transition_status(
    longitudinal_error: torch.Tensor,
    lateral_error: torch.Tensor,
    elapsed_time: torch.Tensor,
    *,
    switch_distance: float,
    lateral_tolerance: float,
    maximum_overshoot: float,
    timeout: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return robust approach-ready and failed masks in platform coordinates.

    A signed longitudinal condition remains true after crossing the target
    line, unlike Euclidean distance to one point.  The bounded overshoot and
    timeout turn a missed lateral corridor into an explicit failed rollout
    instead of leaving the environment in locomotion forever.
    """

    tensors = (longitudinal_error, lateral_error, elapsed_time)
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("approach transition inputs must be one-dimensional")
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("approach transition inputs must share one shape")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("approach transition inputs must be finite floating-point tensors")
    limits = (switch_distance, lateral_tolerance, maximum_overshoot, timeout)
    if any(not math.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("approach transition limits must be finite and positive")

    ready = (
        (longitudinal_error <= switch_distance)
        & (longitudinal_error >= -maximum_overshoot)
        & (torch.abs(lateral_error) <= lateral_tolerance)
    )
    failed = (~ready) & (
        (longitudinal_error < -maximum_overshoot) | (elapsed_time >= timeout)
    )
    return ready, failed


def motion_boundary_alignment_ready(
    joint_position_rms: torch.Tensor,
    joint_speed_rms: torch.Tensor,
    gravity_xy_norm: torch.Tensor,
    *,
    maximum_joint_position_rms: float,
    maximum_joint_speed_rms: float,
    maximum_gravity_xy_norm: float,
) -> torch.Tensor:
    """Gate a teacher switch on overlapping default-like boundary states."""

    tensors = (joint_position_rms, joint_speed_rms, gravity_xy_norm)
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("boundary-alignment inputs must be one-dimensional")
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("boundary-alignment inputs must share one shape")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("boundary-alignment inputs must be finite floating-point tensors")
    if any(torch.any(value < 0.0) for value in tensors):
        raise ValueError("boundary-alignment errors must be non-negative")
    limits = (
        maximum_joint_position_rms,
        maximum_joint_speed_rms,
        maximum_gravity_xy_norm,
    )
    if any(not math.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("boundary-alignment limits must be finite and positive")
    return (
        (joint_position_rms <= maximum_joint_position_rms)
        & (joint_speed_rms <= maximum_joint_speed_rms)
        & (gravity_xy_norm <= maximum_gravity_xy_norm)
    )


def climb_settle_geometry_ready(
    longitudinal_error: torch.Tensor,
    lateral_error: torch.Tensor,
    heading_error: torch.Tensor,
    *,
    switch_distance: float,
    maximum_overshoot: float,
    lateral_tolerance: float,
    maximum_heading_error: float,
) -> torch.Tensor:
    """Revalidate the climb entrance immediately before teacher activation."""

    tensors = (longitudinal_error, lateral_error, heading_error)
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("climb-settle geometry inputs must be one-dimensional")
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("climb-settle geometry inputs must share one shape")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("climb-settle geometry inputs must be finite floating-point tensors")
    limits = (switch_distance, maximum_overshoot, lateral_tolerance, maximum_heading_error)
    if any(not math.isfinite(value) or value <= 0.0 for value in limits):
        raise ValueError("climb-settle geometry limits must be finite and positive")
    return (
        (longitudinal_error <= switch_distance)
        & (longitudinal_error >= -maximum_overshoot)
        & (torch.abs(lateral_error) <= lateral_tolerance)
        & (torch.abs(heading_error) <= maximum_heading_error)
    )


def down_roll_settle_geometry_ready(
    forward_edge_distance: torch.Tensor,
    lateral_offset: torch.Tensor,
    half_width: torch.Tensor,
    heading_error: torch.Tensor,
    *,
    minimum_edge_distance: float,
    maximum_edge_distance: float,
    lateral_margin: float,
    maximum_heading_error: float,
) -> torch.Tensor:
    """Revalidate the platform edge without requiring motion during settle."""

    tensors = (forward_edge_distance, lateral_offset, half_width, heading_error)
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("down-roll-settle geometry inputs must be one-dimensional")
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("down-roll-settle geometry inputs must share one shape")
    if any(not value.is_floating_point() or not torch.isfinite(value).all() for value in tensors):
        raise ValueError("down-roll-settle geometry inputs must be finite floating-point tensors")
    if not all(
        math.isfinite(value)
        for value in (
            minimum_edge_distance,
            maximum_edge_distance,
            lateral_margin,
            maximum_heading_error,
        )
    ):
        raise ValueError("down-roll-settle geometry limits must be finite")
    if minimum_edge_distance > maximum_edge_distance:
        raise ValueError("edge-distance bounds must be ordered")
    if lateral_margin < 0.0 or maximum_heading_error <= 0.0:
        raise ValueError("lateral margin must be non-negative and heading limit positive")

    lateral_limit = (half_width - lateral_margin).clamp_min(0.0)
    return (
        (forward_edge_distance >= minimum_edge_distance)
        & (forward_edge_distance <= maximum_edge_distance)
        & (torch.abs(lateral_offset) <= lateral_limit)
        & (torch.abs(heading_error) <= maximum_heading_error)
    )


def platform_height_teacher_confidence(
    heights: torch.Tensor,
    *,
    nominal_height: float,
    full_tolerance: float,
    zero_tolerance: float,
) -> torch.Tensor:
    """Smooth confidence for a teacher trained at one platform height.

    Confidence stays exactly one inside ``full_tolerance`` and reaches zero
    at ``zero_tolerance``.  The cubic transition has zero slope at both ends,
    avoiding a discontinuous imitation gradient between neighboring geometry
    environments.
    """

    if heights.ndim != 1 or not heights.is_floating_point():
        raise ValueError("heights must be a one-dimensional floating-point tensor")
    if not torch.isfinite(heights).all():
        raise ValueError("heights must be finite")
    if not math.isfinite(nominal_height) or nominal_height <= 0.0:
        raise ValueError("nominal_height must be positive")
    if (
        not math.isfinite(full_tolerance)
        or not math.isfinite(zero_tolerance)
        or full_tolerance < 0.0
        or zero_tolerance <= full_tolerance
    ):
        raise ValueError("height tolerances must satisfy 0 <= full < zero")
    normalized = (
        (torch.abs(heights - nominal_height) - full_tolerance)
        / (zero_tolerance - full_tolerance)
    ).clamp(0.0, 1.0)
    smoothstep = normalized.square() * (3.0 - 2.0 * normalized)
    return 1.0 - smoothstep


def platform_reference_center_offsets(
    lengths: torch.Tensor,
    *,
    nominal_length: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offsets from the physical center for edge-aligned motion references.

    Platform length grows away from the fixed climb entry edge.  The climb
    reference therefore moves half the length delta backward from the new
    physical center, while the down-roll reference moves half forward to keep
    the source exit edge aligned with the sampled far edge.
    """

    if lengths.ndim != 1 or not lengths.is_floating_point():
        raise ValueError("lengths must be a one-dimensional floating-point tensor")
    if not torch.isfinite(lengths).all() or torch.any(lengths <= 0.0):
        raise ValueError("lengths must contain finite positive values")
    if nominal_length <= 0.0:
        raise ValueError("nominal_length must be positive")
    half_delta = 0.5 * (lengths - nominal_length)
    return -half_delta, half_delta


def down_roll_transition_ready(
    forward_edge_distance: torch.Tensor,
    lateral_offset: torch.Tensor,
    half_width: torch.Tensor,
    command_forward_speed: torch.Tensor,
    command_heading_error: torch.Tensor,
    body_heading_error: torch.Tensor,
    gravity_xy_norm: torch.Tensor,
    *,
    minimum_edge_distance: float,
    maximum_edge_distance: float,
    lateral_margin: float,
    minimum_forward_speed: float,
    maximum_heading_error: float,
    maximum_gravity_xy_norm: float,
) -> torch.Tensor:
    """Geometry/state guard for switching from top locomotion to down-roll."""

    tensors = (
        forward_edge_distance,
        lateral_offset,
        half_width,
        command_forward_speed,
        command_heading_error,
        body_heading_error,
        gravity_xy_norm,
    )
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("down-roll transition inputs must be one-dimensional")
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("down-roll transition inputs must share one shape")
    if any(not torch.isfinite(value).all() for value in tensors):
        raise ValueError("down-roll transition inputs must be finite")
    if minimum_edge_distance > maximum_edge_distance:
        raise ValueError("edge-distance bounds must be ordered")
    if lateral_margin < 0.0 or minimum_forward_speed < 0.0:
        raise ValueError("lateral_margin and minimum_forward_speed must be non-negative")
    if maximum_heading_error <= 0.0 or maximum_gravity_xy_norm <= 0.0:
        raise ValueError("heading and gravity limits must be positive")

    lateral_limit = (half_width - lateral_margin).clamp_min(0.0)
    return (
        (forward_edge_distance >= minimum_edge_distance)
        & (forward_edge_distance <= maximum_edge_distance)
        & (torch.abs(lateral_offset) <= lateral_limit)
        & (command_forward_speed >= minimum_forward_speed)
        & (torch.abs(command_heading_error) <= maximum_heading_error)
        & (torch.abs(body_heading_error) <= maximum_heading_error)
        & (gravity_xy_norm <= maximum_gravity_xy_norm)
    )


def climb_geometry_progress_score(
    approach: torch.Tensor,
    height: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """Blend climb progress while making real foot support dominant.

    Approach and root elevation preserve a dense learning signal, but neither
    can exceed 40% of the objective without platform-filtered foot support.
    """

    return 0.20 * approach + 0.20 * height + 0.60 * height * support


def down_roll_geometry_progress_score(
    crossed: torch.Tensor,
    descent: torch.Tensor,
    landing: torch.Tensor,
) -> torch.Tensor:
    """Blend down-roll progress while making upright landing dominant.

    Merely crossing and falling can earn at most 40%; the remaining 60%
    requires a recovered landing at the lower standing height.
    """

    return 0.20 * crossed + 0.20 * descent + 0.60 * landing


def lower_ground_contact_support_score(
    sole_height_errors: torch.Tensor,
    upward_forces: torch.Tensor,
    *,
    height_tolerance: float,
    minimum_upward_force: float,
) -> torch.Tensor:
    """Return per-environment lower-ground support from real foot contact.

    A foot contributes only when an actual sole sample is near the lower
    ground plane and its measured upward contact force reaches the configured
    threshold.  Taking the best foot permits a physically valid one-foot
    landing without allowing a force-free ballistic state to score.
    """

    if (
        sole_height_errors.ndim != 2
        or sole_height_errors.shape[1] == 0
        or sole_height_errors.shape != upward_forces.shape
    ):
        raise ValueError("ground-support inputs must share a non-empty [N, feet] shape")
    if not sole_height_errors.is_floating_point() or not upward_forces.is_floating_point():
        raise ValueError("ground-support inputs must be floating-point tensors")
    if sole_height_errors.device != upward_forces.device:
        raise ValueError("ground-support inputs must share one device")
    if (
        not torch.isfinite(sole_height_errors).all()
        or not torch.isfinite(upward_forces).all()
        or torch.any(sole_height_errors < 0.0)
        or torch.any(upward_forces < 0.0)
    ):
        raise ValueError("ground-support errors and upward forces must be finite and non-negative")
    if not math.isfinite(height_tolerance) or height_tolerance <= 0.0:
        raise ValueError("height_tolerance must be finite and positive")
    if not math.isfinite(minimum_upward_force) or minimum_upward_force <= 0.0:
        raise ValueError("minimum_upward_force must be finite and positive")

    height_score = torch.exp(-torch.square(sole_height_errors / height_tolerance))
    forces = upward_forces.to(dtype=sole_height_errors.dtype)
    force_score = 1.0 - torch.exp(-forces / minimum_upward_force)
    active_contact = forces >= minimum_upward_force
    return (height_score * force_score * active_contact.to(dtype=height_score.dtype)).amax(dim=1)


def filtered_contact_upward_forces(
    force_matrices_w: tuple[torch.Tensor, ...] | list[torch.Tensor],
    *,
    num_envs: int,
) -> torch.Tensor:
    """Extract one independently filtered world-z contact force per foot.

    Each input must come from a one-body contact sensor with exactly one
    filtered counterpart.  Requiring the full ``[N, 1, 1, 3]`` contract here
    prevents an aggregate ``net_forces_w`` tensor from silently being used as
    a ground-only contact measurement.
    """

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    if not isinstance(force_matrices_w, (tuple, list)) or not force_matrices_w:
        raise ValueError("force_matrices_w must contain at least one per-foot matrix")

    expected_shape = (num_envs, 1, 1, 3)
    reference_device: torch.device | None = None
    reference_dtype: torch.dtype | None = None
    upward_forces: list[torch.Tensor] = []
    for matrix in force_matrices_w:
        if not isinstance(matrix, torch.Tensor) or tuple(matrix.shape) != expected_shape:
            raise ValueError(
                "each filtered contact matrix must have shape "
                f"{expected_shape}, got {getattr(matrix, 'shape', None)}"
            )
        if not matrix.is_floating_point():
            raise ValueError("filtered contact matrices must be floating-point tensors")
        if reference_device is None:
            reference_device = matrix.device
            reference_dtype = matrix.dtype
        elif matrix.device != reference_device or matrix.dtype != reference_dtype:
            raise ValueError("filtered contact matrices must share one device and dtype")
        upward_forces.append(matrix[:, 0, 0, 2].clamp_min(0.0))
    return torch.stack(upward_forces, dim=1)


def balanced_skill_ids(
    count: int,
    *,
    cursor: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, int]:
    """Allocate a shuffled contiguous segment of an exactly balanced stream."""

    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("count must be a non-negative integer")
    if isinstance(cursor, bool) or not isinstance(cursor, int) or not 0 <= cursor < NUM_SKILLS:
        raise ValueError(f"cursor must lie in [0, {NUM_SKILLS})")
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=device), cursor
    routes = (torch.arange(count, device=device, dtype=torch.long) + cursor) % NUM_SKILLS
    permutation = torch.randperm(count, device=device, generator=generator)
    return routes[permutation], (cursor + count) % NUM_SKILLS


__all__ = [
    "CLIMB_SKILL_ID",
    "DOWN_ROLL_SKILL_ID",
    "LOCOMOTION_SKILL_ID",
    "NUM_SKILLS",
    "approach_transition_status",
    "balanced_skill_ids",
    "climb_settle_geometry_ready",
    "climb_geometry_progress_score",
    "down_roll_geometry_progress_score",
    "down_roll_transition_ready",
    "down_roll_settle_geometry_ready",
    "filtered_contact_upward_forces",
    "lower_ground_contact_support_score",
    "motion_boundary_alignment_ready",
    "planar_command_speed_valid",
    "platform_height_teacher_confidence",
    "platform_reference_center_offsets",
]
