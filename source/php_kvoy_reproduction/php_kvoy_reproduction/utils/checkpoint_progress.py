from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral
from typing import Any


RUNNER_PROGRESS_CHECKPOINT_KEY = "whole_body_tracking_runner_progress_v1"


def _nonnegative_iteration(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}.")
    return value


def build_runner_progress_state(last_completed_iteration: int) -> dict[str, int]:
    """Build versioned progress metadata for a checkpoint saved after an update."""

    last_completed_iteration = _nonnegative_iteration("last_completed_iteration", last_completed_iteration)
    return {
        "version": 1,
        "last_completed_iteration": last_completed_iteration,
        "next_iteration": last_completed_iteration + 1,
    }


def resolve_resume_iteration(checkpoint_infos: object, loaded_iteration: int) -> int:
    """Return the first not-yet-completed iteration represented by a checkpoint.

    RSL-RL stores the index of the update that has just completed. Legacy whole
    body tracking checkpoints contain no explicit progress metadata, so their
    correct resume index is also ``loaded_iteration + 1``.
    """

    loaded_iteration = _nonnegative_iteration("loaded_iteration", loaded_iteration)
    if not isinstance(checkpoint_infos, Mapping) or RUNNER_PROGRESS_CHECKPOINT_KEY not in checkpoint_infos:
        return loaded_iteration + 1

    progress = checkpoint_infos[RUNNER_PROGRESS_CHECKPOINT_KEY]
    if not isinstance(progress, Mapping):
        raise TypeError("Runner progress checkpoint state must be a mapping.")
    if progress.get("version") != 1:
        raise ValueError(f"Unsupported runner progress checkpoint version: {progress.get('version')!r}.")

    last_completed = _nonnegative_iteration(
        "progress.last_completed_iteration", progress.get("last_completed_iteration")
    )
    next_iteration = _nonnegative_iteration("progress.next_iteration", progress.get("next_iteration"))
    if last_completed != loaded_iteration:
        raise ValueError(
            "Runner progress metadata disagrees with RSL-RL's checkpoint iteration: "
            f"progress={last_completed}, rsl_rl={loaded_iteration}."
        )
    if next_iteration != last_completed + 1:
        raise ValueError(
            "Runner progress metadata must resume immediately after the completed iteration: "
            f"last_completed={last_completed}, next={next_iteration}."
        )
    return next_iteration
