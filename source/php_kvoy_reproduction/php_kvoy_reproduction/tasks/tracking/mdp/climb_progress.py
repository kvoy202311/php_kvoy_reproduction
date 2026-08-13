"""Pure state update for bounded, non-repeatable climb progress."""

from __future__ import annotations

import torch


def bounded_episode_progress_increment(
    potential: torch.Tensor,
    initialized: torch.Tensor,
    episode_max: torch.Tensor,
    max_delta_per_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return newly reached progress and the updated episode maximum.

    The first sample establishes the reset baseline. Afterwards only progress
    above the previous episode maximum is eligible for reward. The per-step
    cap deliberately discards any excess instead of deferring it: revisiting
    the same height or contact state can therefore never recover or farm
    previously clipped reward.
    """

    if max_delta_per_step <= 0.0:
        raise ValueError(f"max_delta_per_step must be positive, got {max_delta_per_step}.")
    if potential.shape != initialized.shape or potential.shape != episode_max.shape:
        raise ValueError(
            "potential, initialized, and episode_max must have identical shapes, got "
            f"{potential.shape}, {initialized.shape}, and {episode_max.shape}."
        )
    if initialized.dtype != torch.bool:
        raise TypeError(f"initialized must be a boolean tensor, got {initialized.dtype}.")

    previous_max = torch.where(initialized, episode_max, potential)
    updated_max = torch.maximum(previous_max, potential)
    increment = (updated_max - previous_max).clamp(min=0.0, max=max_delta_per_step)
    return increment, updated_max
