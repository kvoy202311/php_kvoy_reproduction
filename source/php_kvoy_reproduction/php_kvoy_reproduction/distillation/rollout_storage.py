"""Rollout storage for joint PPO and online DAgger optimization.

The upstream RSL-RL 2.3.x distillation storage intentionally omits the value,
return, advantage, and log-probability tensors needed by PPO.  This module
keeps one aligned tensor for every transition instead of maintaining separate
RL and imitation buffers, which makes it impossible for a shuffle to separate
a student observation from its teacher label or route.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Iterator, Sequence

import torch


@dataclass
class HybridTransition:
    """One vectorized environment transition before it is copied to storage."""

    observations: torch.Tensor | None = None
    privileged_observations: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    rewards: torch.Tensor | None = None
    dones: torch.Tensor | None = None
    values: torch.Tensor | None = None
    actions_log_prob: torch.Tensor | None = None
    action_mean: torch.Tensor | None = None
    action_sigma: torch.Tensor | None = None
    teacher_actions: torch.Tensor | None = None
    dagger_mask: torch.Tensor | None = None
    skill_ids: torch.Tensor | None = None

    def clear(self) -> None:
        """Release references to environment tensors after insertion."""

        for field in fields(self):
            setattr(self, field.name, None)


@dataclass(frozen=True)
class HybridBatch:
    """A named, aligned minibatch consumed by :class:`DAggerPPO`."""

    observations: torch.Tensor
    privileged_observations: torch.Tensor
    actions: torch.Tensor
    target_values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    old_actions_log_prob: torch.Tensor
    old_action_mean: torch.Tensor
    old_action_sigma: torch.Tensor
    teacher_actions: torch.Tensor
    dagger_mask: torch.Tensor
    skill_ids: torch.Tensor


def _shape_tuple(shape: Sequence[int], name: str) -> tuple[int, ...]:
    result = tuple(int(value) for value in shape)
    if not result or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain positive dimensions, got {result}.")
    return result


class HybridRolloutStorage:
    """Flat feed-forward rollout storage with teacher labels.

    Recurrent policies are deliberately outside the first implementation.  A
    fixed eight-frame proprioceptive history is already part of each actor
    observation, so random flat minibatches preserve the complete temporal
    input expected by the feed-forward student.
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: Sequence[int],
        critic_obs_shape: Sequence[int],
        actions_shape: Sequence[int],
        *,
        num_skills: int,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if num_transitions_per_env <= 0:
            raise ValueError("num_transitions_per_env must be positive.")
        if num_skills <= 0:
            raise ValueError("num_skills must be positive.")

        self.num_envs = int(num_envs)
        self.num_transitions_per_env = int(num_transitions_per_env)
        self.num_skills = int(num_skills)
        self.actor_obs_shape = _shape_tuple(actor_obs_shape, "actor_obs_shape")
        self.critic_obs_shape = _shape_tuple(critic_obs_shape, "critic_obs_shape")
        self.actions_shape = _shape_tuple(actions_shape, "actions_shape")
        self.device = torch.device(device)
        self.step = 0

        leading = (self.num_transitions_per_env, self.num_envs)
        self.observations = torch.zeros(*leading, *self.actor_obs_shape, device=self.device)
        self.privileged_observations = torch.zeros(*leading, *self.critic_obs_shape, device=self.device)
        self.actions = torch.zeros(*leading, *self.actions_shape, device=self.device)
        self.rewards = torch.zeros(*leading, 1, device=self.device)
        self.dones = torch.zeros(*leading, 1, dtype=torch.bool, device=self.device)
        self.values = torch.zeros(*leading, 1, device=self.device)
        self.actions_log_prob = torch.zeros(*leading, 1, device=self.device)
        self.mu = torch.zeros(*leading, *self.actions_shape, device=self.device)
        self.sigma = torch.zeros(*leading, *self.actions_shape, device=self.device)
        self.returns = torch.zeros(*leading, 1, device=self.device)
        self.advantages = torch.zeros(*leading, 1, device=self.device)

        self.teacher_actions = torch.zeros(*leading, *self.actions_shape, device=self.device)
        # Continuous confidence in [0, 1].  Zero excludes an OOD teacher
        # label; intermediate values retain a verified but weaker prior.
        self.dagger_mask = torch.zeros(*leading, 1, device=self.device)
        self.skill_ids = torch.zeros(*leading, 1, dtype=torch.long, device=self.device)

    @property
    def is_full(self) -> bool:
        return self.step == self.num_transitions_per_env

    @staticmethod
    def _require(name: str, value: torch.Tensor | None) -> torch.Tensor:
        if value is None:
            raise ValueError(f"Transition field {name!r} was not populated.")
        return value

    @staticmethod
    def _copy_exact(destination: torch.Tensor, source: torch.Tensor, name: str) -> None:
        if source.shape != destination.shape:
            raise ValueError(
                f"Transition field {name!r} has shape {tuple(source.shape)}; "
                f"expected {tuple(destination.shape)}."
            )
        if source.dtype.is_floating_point and not torch.isfinite(source).all():
            raise ValueError(f"Transition field {name!r} contains NaN or infinity.")
        destination.copy_(source.to(device=destination.device, dtype=destination.dtype))

    def add_transitions(self, transition: HybridTransition) -> None:
        """Atomically copy one complete vectorized step into storage."""

        if self.is_full:
            raise OverflowError("Rollout buffer is full; call clear() after update().")

        index = self.step
        tensor_fields = (
            "observations",
            "privileged_observations",
            "actions",
            "values",
            "actions_log_prob",
            "action_mean",
            "action_sigma",
            "teacher_actions",
        )
        destinations = (
            self.observations[index],
            self.privileged_observations[index],
            self.actions[index],
            self.values[index],
            self.actions_log_prob[index],
            self.mu[index],
            self.sigma[index],
            self.teacher_actions[index],
        )
        for name, destination in zip(tensor_fields, destinations, strict=True):
            self._copy_exact(destination, self._require(name, getattr(transition, name)), name)

        rewards = self._require("rewards", transition.rewards).reshape(-1, 1)
        dones = self._require("dones", transition.dones).reshape(-1, 1)
        dagger_mask = self._require("dagger_mask", transition.dagger_mask).reshape(-1, 1)
        skill_ids = self._require("skill_ids", transition.skill_ids).reshape(-1, 1)
        self._copy_exact(self.rewards[index], rewards, "rewards")
        self._copy_exact(self.dones[index], dones, "dones")
        self._copy_exact(self.dagger_mask[index], dagger_mask, "dagger_mask")
        self._copy_exact(self.skill_ids[index], skill_ids, "skill_ids")

        if torch.any((self.dagger_mask[index] < 0.0) | (self.dagger_mask[index] > 1.0)):
            raise ValueError("dagger_mask confidence weights must lie in [0, 1].")

        if ((self.skill_ids[index] < 0) | (self.skill_ids[index] >= self.num_skills)).any():
            invalid = self.skill_ids[index][
                (self.skill_ids[index] < 0) | (self.skill_ids[index] >= self.num_skills)
            ]
            raise ValueError(
                f"skill_ids must lie in [0, {self.num_skills}), got "
                f"{invalid.unique().tolist()}."
            )
        if (self.sigma[index] <= 0.0).any():
            raise ValueError("action_sigma must be strictly positive.")
        self.step += 1

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        *,
        normalize_advantage: bool = True,
    ) -> None:
        """Compute GAE returns after a complete rollout."""

        if not self.is_full:
            raise RuntimeError(
                "Cannot compute returns before the rollout is full: "
                f"{self.step}/{self.num_transitions_per_env} steps."
            )
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must lie in [0, 1].")
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lam must lie in [0, 1].")
        expected = (self.num_envs, 1)
        if last_values.shape != expected:
            raise ValueError(f"last_values must have shape {expected}, got {tuple(last_values.shape)}.")
        if not torch.isfinite(last_values).all():
            raise ValueError("last_values contains NaN or infinity.")

        advantage = torch.zeros_like(last_values)
        for index in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if index == self.num_transitions_per_env - 1 else self.values[index + 1]
            not_terminal = (~self.dones[index]).to(self.rewards.dtype)
            delta = self.rewards[index] + not_terminal * gamma * next_values - self.values[index]
            advantage = delta + not_terminal * gamma * lam * advantage
            self.returns[index] = advantage + self.values[index]

        self.advantages.copy_(self.returns - self.values)
        if normalize_advantage:
            mean = self.advantages.mean()
            std = self.advantages.std(unbiased=False)
            self.advantages.sub_(mean).div_(std + 1.0e-8)

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int,
    ) -> Iterator[HybridBatch]:
        """Yield shuffled aligned minibatches without silently dropping data."""

        if not self.is_full:
            raise RuntimeError("Cannot create minibatches from an incomplete rollout.")
        if num_mini_batches <= 0 or num_epochs <= 0:
            raise ValueError("num_mini_batches and num_epochs must be positive.")
        batch_size = self.num_envs * self.num_transitions_per_env
        if batch_size % num_mini_batches != 0:
            raise ValueError(
                f"Rollout batch {batch_size} is not divisible by {num_mini_batches} minibatches; "
                "refusing to drop samples."
            )
        mini_batch_size = batch_size // num_mini_batches

        flattened = {
            "observations": self.observations.flatten(0, 1),
            "privileged_observations": self.privileged_observations.flatten(0, 1),
            "actions": self.actions.flatten(0, 1),
            "target_values": self.values.flatten(0, 1),
            "advantages": self.advantages.flatten(0, 1),
            "returns": self.returns.flatten(0, 1),
            "old_actions_log_prob": self.actions_log_prob.flatten(0, 1),
            "old_action_mean": self.mu.flatten(0, 1),
            "old_action_sigma": self.sigma.flatten(0, 1),
            "teacher_actions": self.teacher_actions.flatten(0, 1),
            "dagger_mask": self.dagger_mask.flatten(0, 1),
            "skill_ids": self.skill_ids.flatten(0, 1),
        }

        for _ in range(num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                batch_indices = indices[start : start + mini_batch_size]
                yield HybridBatch(**{name: values[batch_indices] for name, values in flattened.items()})

    def clear(self) -> None:
        """Mark storage empty without reallocating its tensors."""

        self.step = 0
