"""Pure PyTorch geometry used by ELF3 climb obstacles and offline tests."""

from __future__ import annotations

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

    # Extract yaw from a wxyz quaternion, then apply the inverse planar
    # rotation. Roll and pitch do not affect this kinematic platform.
    w, x, y, z = orientations_w.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw = torch.cos(yaw)[:, None]
    sin_yaw = torch.sin(yaw)[:, None]
    delta = points_w[..., :2] - centers_w[:, None, :2]
    local_x = cos_yaw * delta[..., 0] + sin_yaw * delta[..., 1]
    local_y = -sin_yaw * delta[..., 0] + cos_yaw * delta[..., 1]

    half_size = 0.5 * sizes[:, None, :2] + margin
    return (local_x.abs() <= half_size[..., 0]) & (local_y.abs() <= half_size[..., 1])


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
