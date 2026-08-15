"""Pure joint-speed scoring helpers for the ELF3 final standing phase."""

from __future__ import annotations

import torch


def joint_speed_statistics(joint_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-environment RMS and maximum absolute joint speeds."""

    if joint_velocity.ndim != 2 or joint_velocity.shape[1] == 0:
        raise ValueError(
            "joint_velocity must have shape [num_envs, num_joints] with at least one joint, "
            f"got {joint_velocity.shape}."
        )
    absolute_speed = torch.abs(joint_velocity)
    rms_speed = torch.sqrt(torch.mean(torch.square(joint_velocity), dim=1))
    max_speed = torch.max(absolute_speed, dim=1).values
    return rms_speed, max_speed


def joint_settling_score(
    joint_velocity: torch.Tensor,
    *,
    rms_speed_tolerance: float,
    max_speed_tolerance: float,
    rms_speed_scale: float,
    max_speed_scale: float,
    fine_max_speed_scale: float,
    score_weights: tuple[float, float, float],
) -> torch.Tensor:
    """Return a smooth settling score in ``[0, 1]``.

    Scoring the *excess over the success tolerances* makes every final state
    within the acceptance band high-value, while still giving a smooth and
    substantial gradient to states that exceed it.  Combining RMS and maximum
    speed prevents the average from hiding one fast joint without making the
    full objective depend only on one joint.
    """

    for name, value in (
        ("rms_speed_tolerance", rms_speed_tolerance),
        ("max_speed_tolerance", max_speed_tolerance),
    ):
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative, got {value}.")
    for name, value in (
        ("rms_speed_scale", rms_speed_scale),
        ("max_speed_scale", max_speed_scale),
        ("fine_max_speed_scale", fine_max_speed_scale),
    ):
        if value <= 0.0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if len(score_weights) != 3 or any(weight < 0.0 for weight in score_weights):
        raise ValueError("score_weights must contain three non-negative values.")
    weight_sum = float(sum(score_weights))
    if weight_sum <= 0.0:
        raise ValueError("score_weights must contain at least one positive value.")

    rms_speed, max_speed = joint_speed_statistics(joint_velocity)
    rms_excess = (rms_speed - rms_speed_tolerance).clamp_min(0.0)
    max_excess = (max_speed - max_speed_tolerance).clamp_min(0.0)
    rms_broad_score = torch.reciprocal(1.0 + torch.square(rms_excess / rms_speed_scale))
    max_broad_score = torch.reciprocal(1.0 + torch.square(max_excess / max_speed_scale))
    max_fine_score = torch.exp(-0.5 * torch.square(max_excess / fine_max_speed_scale))
    scores = (rms_broad_score, max_broad_score, max_fine_score)
    return sum(
        (weight / weight_sum) * score for weight, score in zip(score_weights, scores, strict=True)
    )
