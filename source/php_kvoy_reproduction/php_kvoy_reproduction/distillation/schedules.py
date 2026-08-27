"""Loss schedules used by PHP-style joint DAgger and PPO training."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DistillationWeights:
    """Weights applied to the two optimization objectives at one iteration."""

    dagger: float
    ppo: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.dagger) and math.isfinite(self.ppo)):
            raise ValueError("Distillation weights must be finite.")
        if self.dagger < 0.0 or self.ppo < 0.0:
            raise ValueError("Distillation weights must be non-negative.")
        if abs(self.dagger + self.ppo - 1.0) > 1.0e-9:
            raise ValueError("DAgger and PPO weights must sum to one.")


@dataclass(frozen=True)
class PhpLossSchedule:
    """Linear PHP curriculum with a persistent DAgger floor.

    For the paper's 20k-iteration setup, ``curriculum_iterations=10_000``
    corresponds to the first half of training.  The iteration is supplied by
    the runner rather than stored here, so checkpoint resume cannot silently
    restart the curriculum.
    """

    curriculum_iterations: int = 10_000
    minimum_dagger_weight: float = 0.1
    adaptive_lr_minimum_ppo_weight: float = 0.1

    def __post_init__(self) -> None:
        if self.curriculum_iterations <= 0:
            raise ValueError("curriculum_iterations must be positive.")
        if not 0.0 <= self.minimum_dagger_weight <= 1.0:
            raise ValueError("minimum_dagger_weight must lie in [0, 1].")
        if not 0.0 <= self.adaptive_lr_minimum_ppo_weight < 1.0:
            raise ValueError("adaptive_lr_minimum_ppo_weight must lie in [0, 1).")

    def at(self, iteration: int) -> DistillationWeights:
        """Return objective weights for a zero-based learning iteration."""

        if iteration < 0:
            raise ValueError(f"iteration must be non-negative, got {iteration}.")
        dagger = max(
            self.minimum_dagger_weight,
            1.0 - float(iteration) / float(self.curriculum_iterations),
        )
        return DistillationWeights(dagger=dagger, ppo=1.0 - dagger)

    def adaptive_lr_enabled(self, iteration: int) -> bool:
        """Whether KL-based learning-rate adaptation is active."""

        return self.at(iteration).ppo > self.adaptive_lr_minimum_ppo_weight
