"""Deployment-compatible hard option routing for the visual Student.

The neural selector proposes a skill every control step.  This controller is
the deliberately small amount of temporal state required to turn those noisy
frame-wise proposals into safe whole-body options: locomotion may enter one
motion skill after confirmation, while climb and down-roll remain committed
until their learned completion signal is confirmed (or a conservative safety
timeout is reached).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch


@dataclass(frozen=True)
class OptionSelection:
    """One controller decision for a vectorized environment step."""

    active_skill_ids: torch.Tensor
    predicted_skill_ids: torch.Tensor
    teacher_forcing_mask: torch.Tensor
    switched: torch.Tensor


def _probability(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite probability")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive_steps(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


class OptionStateController:
    """Convert selector logits into sticky, hard-routed skill IDs.

    Teacher forcing is sampled once per environment episode.  It is never
    sampled per step, which would create artificial mid-motion route changes.
    Autonomous environments always reset into locomotion, matching deployment.
    """

    def __init__(
        self,
        num_envs: int,
        skill_names: Sequence[str],
        *,
        activation_probability: float = 0.60,
        release_probability: float = 0.55,
        activation_confirmation_steps: int = 3,
        release_confirmation_steps: int = 5,
        post_release_cooldown_steps: int = 25,
        minimum_skill_duration_steps: Mapping[str, int] | None = None,
        maximum_skill_duration_steps: Mapping[str, int] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        names = tuple(skill_names)
        if len(names) < 2 or len(set(names)) != len(names) or names[0] != "locomotion":
            raise ValueError("skill_names must be unique and begin with 'locomotion'")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("skill_names must contain non-empty strings")

        self.num_envs = num_envs
        self.skill_names = names
        self.num_skills = len(names)
        self.device = torch.device(device)
        self.activation_probability = _probability(
            activation_probability, "activation_probability"
        )
        self.release_probability = _probability(
            release_probability, "release_probability"
        )
        self.activation_confirmation_steps = _positive_steps(
            activation_confirmation_steps, "activation_confirmation_steps"
        )
        self.release_confirmation_steps = _positive_steps(
            release_confirmation_steps, "release_confirmation_steps"
        )
        self.post_release_cooldown_steps = _positive_steps(
            post_release_cooldown_steps, "post_release_cooldown_steps"
        )

        default_minimum = {name: 100 for name in names[1:]}
        default_maximum = {name: 400 for name in names[1:]}
        minimum = default_minimum if minimum_skill_duration_steps is None else dict(minimum_skill_duration_steps)
        maximum = default_maximum if maximum_skill_duration_steps is None else dict(maximum_skill_duration_steps)
        expected_motion_names = set(names[1:])
        if set(minimum) != expected_motion_names or set(maximum) != expected_motion_names:
            raise ValueError(
                "minimum/maximum skill durations must contain every non-locomotion skill exactly once"
            )
        minimum_by_id = [0]
        maximum_by_id = [2**31 - 1]
        for name in names[1:]:
            minimum_steps = _positive_steps(minimum[name], f"minimum duration for {name!r}")
            maximum_steps = _positive_steps(maximum[name], f"maximum duration for {name!r}")
            if maximum_steps <= minimum_steps:
                raise ValueError(f"maximum duration for {name!r} must exceed its minimum")
            minimum_by_id.append(minimum_steps)
            maximum_by_id.append(maximum_steps)
        self.minimum_duration = torch.tensor(minimum_by_id, device=self.device, dtype=torch.long)
        self.maximum_duration = torch.tensor(maximum_by_id, device=self.device, dtype=torch.long)

        self.active_skill_ids = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.candidate_skill_ids = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.candidate_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.elapsed_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.cooldown_steps = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.teacher_forcing_mask = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self._initialized = False

    @staticmethod
    def _column_ids(
        values: torch.Tensor,
        *,
        name: str,
        num_envs: int,
        num_skills: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(values, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        result = values.reshape(-1).to(device=device)
        if result.shape != (num_envs,):
            raise ValueError(f"{name} must contain one value per environment")
        if result.dtype == torch.bool or result.is_floating_point() or result.is_complex():
            raise TypeError(f"{name} must use an integer dtype")
        result = result.to(dtype=torch.long)
        if torch.any((result < 0) | (result >= num_skills)):
            raise ValueError(f"{name} contains an out-of-range skill ID")
        return result

    def reset(
        self,
        dones: torch.Tensor | None = None,
        *,
        teacher_forcing_probability: float = 0.0,
    ) -> None:
        """Reset completed episodes and sample one forcing decision per episode."""

        probability = _probability(
            teacher_forcing_probability, "teacher_forcing_probability"
        )
        if dones is None:
            reset_mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            if not isinstance(dones, torch.Tensor):
                raise TypeError("dones must be a torch.Tensor or None")
            reset_mask = dones.reshape(-1).to(device=self.device, dtype=torch.bool)
            if reset_mask.shape != (self.num_envs,):
                raise ValueError("dones must contain one value per environment")
        if not torch.any(reset_mask):
            return
        self.active_skill_ids[reset_mask] = 0
        self.candidate_skill_ids[reset_mask] = 0
        self.candidate_steps[reset_mask] = 0
        self.elapsed_steps[reset_mask] = 0
        self.cooldown_steps[reset_mask] = 0
        if probability <= 0.0:
            self.teacher_forcing_mask[reset_mask] = False
        elif probability >= 1.0:
            self.teacher_forcing_mask[reset_mask] = True
        else:
            self.teacher_forcing_mask[reset_mask] = (
                torch.rand(int(reset_mask.sum().item()), device=self.device) < probability
            )
        self._initialized = True

    def _autonomous_selection(
        self,
        predicted_ids: torch.Tensor,
        probabilities: torch.Tensor,
        autonomous: torch.Tensor,
    ) -> torch.Tensor:
        previous = self.active_skill_ids.clone()
        cooling = autonomous & (self.active_skill_ids == 0) & (self.cooldown_steps > 0)
        self.cooldown_steps[cooling] -= 1
        locomoting = autonomous & (self.active_skill_ids == 0) & (self.cooldown_steps == 0)
        proposed_motion = locomoting & (predicted_ids != 0)
        confident_motion = proposed_motion & (
            probabilities.gather(1, predicted_ids[:, None]).squeeze(1)
            >= self.activation_probability
        )
        same_candidate = predicted_ids == self.candidate_skill_ids
        continue_candidate = confident_motion & same_candidate
        new_candidate = confident_motion & ~same_candidate
        self.candidate_steps[continue_candidate] += 1
        self.candidate_skill_ids[new_candidate] = predicted_ids[new_candidate]
        self.candidate_steps[new_candidate] = 1
        clear_locomotion_candidate = locomoting & ~confident_motion
        self.candidate_skill_ids[clear_locomotion_candidate] = 0
        self.candidate_steps[clear_locomotion_candidate] = 0
        activate = locomoting & (
            self.candidate_steps >= self.activation_confirmation_steps
        )
        self.active_skill_ids[activate] = self.candidate_skill_ids[activate]
        self.elapsed_steps[activate] = 0
        self.candidate_skill_ids[activate] = 0
        self.candidate_steps[activate] = 0

        active_motion = autonomous & (self.active_skill_ids != 0)
        self.elapsed_steps[active_motion] += 1
        active_ids = self.active_skill_ids.clamp(0, self.num_skills - 1)
        minimum_reached = self.elapsed_steps >= self.minimum_duration[active_ids]
        maximum_reached = self.elapsed_steps >= self.maximum_duration[active_ids]
        release_proposed = active_motion & minimum_reached & (predicted_ids == 0) & (
            probabilities[:, 0] >= self.release_probability
        )
        self.candidate_steps[release_proposed] += 1
        self.candidate_steps[active_motion & ~release_proposed] = 0
        release = active_motion & (
            (self.candidate_steps >= self.release_confirmation_steps) | maximum_reached
        )
        self.active_skill_ids[release] = 0
        self.candidate_skill_ids[release] = 0
        self.candidate_steps[release] = 0
        self.elapsed_steps[release] = 0
        self.cooldown_steps[release] = self.post_release_cooldown_steps
        return self.active_skill_ids != previous

    def select(
        self,
        selector_logits: torch.Tensor,
        *,
        oracle_skill_ids: torch.Tensor | None = None,
        teacher_forcing_probability: float = 0.0,
    ) -> OptionSelection:
        """Return hard execution routes without blending action heads."""

        if not isinstance(selector_logits, torch.Tensor) or selector_logits.shape != (
            self.num_envs,
            self.num_skills,
        ):
            raise ValueError(
                f"selector_logits must have shape [{self.num_envs}, {self.num_skills}]"
            )
        if selector_logits.device != self.device or not selector_logits.is_floating_point():
            raise ValueError("selector_logits must be floating point on the controller device")
        if not torch.isfinite(selector_logits).all():
            raise ValueError("selector_logits contains NaN or infinity")
        if not self._initialized:
            self.reset(teacher_forcing_probability=teacher_forcing_probability)

        probabilities = torch.softmax(selector_logits, dim=-1)
        predicted_ids = torch.argmax(probabilities, dim=-1)
        autonomous = ~self.teacher_forcing_mask
        switched = self._autonomous_selection(predicted_ids, probabilities, autonomous)

        if torch.any(self.teacher_forcing_mask):
            if oracle_skill_ids is None:
                raise ValueError("oracle_skill_ids are required for teacher-forced environments")
            oracle = self._column_ids(
                oracle_skill_ids,
                name="oracle_skill_ids",
                num_envs=self.num_envs,
                num_skills=self.num_skills,
                device=self.device,
            )
            forced = self.teacher_forcing_mask
            switched[forced] = self.active_skill_ids[forced] != oracle[forced]
            self.active_skill_ids[forced] = oracle[forced]
            self.elapsed_steps[forced] = 0
            self.candidate_skill_ids[forced] = 0
            self.candidate_steps[forced] = 0
        return OptionSelection(
            active_skill_ids=self.active_skill_ids[:, None].clone(),
            predicted_skill_ids=predicted_ids[:, None],
            teacher_forcing_mask=self.teacher_forcing_mask[:, None].clone(),
            switched=switched[:, None],
        )


def linear_teacher_forcing_probability(
    iteration: int,
    *,
    start: float,
    end: float,
    curriculum_iterations: int,
) -> float:
    """Linearly interpolate a stage-local route-teacher-forcing schedule."""

    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
        raise ValueError("iteration must be a non-negative integer")
    start_probability = _probability(start, "start")
    end_probability = _probability(end, "end")
    if isinstance(curriculum_iterations, bool) or not isinstance(curriculum_iterations, int):
        raise TypeError("curriculum_iterations must be an integer")
    if curriculum_iterations <= 0:
        raise ValueError("curriculum_iterations must be positive")
    progress = min(1.0, float(iteration) / float(curriculum_iterations))
    return start_probability + progress * (end_probability - start_probability)


__all__ = [
    "OptionSelection",
    "OptionStateController",
    "linear_teacher_forcing_probability",
]
