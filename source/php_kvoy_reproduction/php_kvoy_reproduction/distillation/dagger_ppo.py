"""Joint online DAgger and PPO optimization used by the PHP student."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal

import torch
from torch import nn

from .rollout_storage import HybridBatch, HybridRolloutStorage, HybridTransition
from .schedules import PhpLossSchedule


DAggerReduction = Literal["mean_per_dof", "sum_per_sample"]


def _as_column(values: torch.Tensor, name: str, batch_size: int) -> torch.Tensor:
    result = values.reshape(-1, 1)
    if result.shape[0] != batch_size:
        raise ValueError(
            f"{name} must contain one value per environment ({batch_size}), "
            f"got shape {tuple(values.shape)}."
        )
    return result


def _skill_id_column(
    values: torch.Tensor,
    name: str,
    batch_size: int,
    num_skills: int,
) -> torch.Tensor:
    result = _as_column(values, name, batch_size)
    if result.dtype == torch.bool or result.is_floating_point() or result.is_complex():
        raise TypeError(f"{name} must use an integer dtype")
    result = result.to(dtype=torch.long)
    if torch.any((result < 0) | (result >= num_skills)):
        raise ValueError(f"{name} must lie in [0, {num_skills})")
    return result


def masked_dagger_losses(
    student_mean: torch.Tensor,
    teacher_actions: torch.Tensor,
    dagger_weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return weighted mean-per-DoF, sum-per-sample, and effective weight.

    The arithmetic mask formulation deliberately keeps a differentiable zero
    connected to ``student_mean`` when no labels are valid.  Consequently an
    all-zero mask produces neither NaN nor a detached loss that breaks
    ``backward()``.
    """

    if student_mean.ndim != 2:
        raise ValueError(f"student_mean must be [batch, actions], got {tuple(student_mean.shape)}.")
    if teacher_actions.shape != student_mean.shape:
        raise ValueError(
            f"teacher_actions must match student_mean {tuple(student_mean.shape)}, "
            f"got {tuple(teacher_actions.shape)}."
        )
    if dagger_weights.shape != (student_mean.shape[0], 1):
        raise ValueError(
            "dagger_weights must be "
            f"[{student_mean.shape[0]}, 1], got {tuple(dagger_weights.shape)}."
        )
    if not torch.isfinite(student_mean).all() or not torch.isfinite(dagger_weights).all():
        raise ValueError("Student actions or DAgger weights contain NaN or infinity.")
    if torch.any((dagger_weights < 0.0) | (dagger_weights > 1.0)):
        raise ValueError("dagger_weights must lie in [0, 1].")

    weights = dagger_weights.to(dtype=student_mean.dtype)
    effective_weight = weights.sum()
    positive = weights[:, 0] > 0.0
    valid_count = torch.count_nonzero(positive).to(dtype=student_mean.dtype)
    if not torch.any(positive):
        connected_zero = student_mean.sum() * 0.0
        return connected_zero, connected_zero, effective_weight
    valid_student = student_mean[positive]
    valid_teacher = teacher_actions[positive]
    valid_weights = weights[positive]
    if not torch.isfinite(valid_teacher).all():
        raise ValueError("A positive-confidence teacher label contains NaN or infinity.")
    # Subset before subtraction: 0 * inf and 0 * huge values are not safe
    # masks and can contaminate gradients even though their confidence is zero.
    squared_sum = ((valid_student - valid_teacher).square() * valid_weights).sum()
    # Normalize by the number of labels, not by their summed confidence.
    # This preserves the historical binary-mask mean while ensuring that a
    # confidence of 0.1 contributes one tenth of a fully trusted label.
    safe_count = valid_count.clamp_min(1.0)
    mean_per_dof = squared_sum / (safe_count * student_mean.shape[1])
    sum_per_sample = squared_sum / safe_count
    return mean_per_dof, sum_per_sample, effective_weight


def skill_balanced_mean(
    sample_values: torch.Tensor,
    skill_ids: torch.Tensor,
    num_skills: int,
    validity_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Average per-skill means, optionally with continuous sample weights."""

    values = sample_values.reshape(sample_values.shape[0], -1).mean(dim=1)
    route = skill_ids.reshape(-1)
    if values.shape[0] != route.shape[0]:
        raise ValueError("sample_values and skill_ids must have the same batch size")
    if isinstance(num_skills, bool) or not isinstance(num_skills, int) or num_skills <= 0:
        raise ValueError("num_skills must be a positive integer")
    if torch.any((route < 0) | (route >= num_skills)):
        raise ValueError("skill_ids contains an out-of-range route")
    if validity_mask is None:
        weights = torch.ones_like(values)
    else:
        weights = validity_mask.reshape(-1).to(device=values.device, dtype=values.dtype)
        if weights.shape != route.shape:
            raise ValueError("validity_mask must contain one value per sample")
        if not torch.isfinite(weights).all() or torch.any((weights < 0.0) | (weights > 1.0)):
            raise ValueError("validity_mask weights must be finite and lie in [0, 1]")
    group_means = []
    for skill_id in range(num_skills):
        selected = route == skill_id
        positive = selected & (weights > 0.0)
        valid_count = torch.count_nonzero(positive)
        if valid_count > 0:
            # Keep every represented skill equally balanced, but retain the
            # absolute confidence scale within that skill.  Dividing by the
            # summed confidence would cancel a uniform confidence reduction.
            group_means.append(
                (values[positive] * weights[positive]).sum()
                / valid_count.to(dtype=values.dtype)
            )
    if not group_means:
        return values.sum() * 0.0
    return torch.stack(group_means).mean()


def skill_balanced_dagger_losses(
    student_mean: torch.Tensor,
    teacher_actions: torch.Tensor,
    dagger_weights: torch.Tensor,
    skill_ids: torch.Tensor,
    num_skills: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DAgger errors with equal skill weight and weighted samples per skill."""

    _, _, effective_weight = masked_dagger_losses(
        student_mean, teacher_actions, dagger_weights
    )
    route = skill_ids.reshape(-1)
    weights = dagger_weights.reshape(-1).to(
        device=student_mean.device, dtype=student_mean.dtype
    )
    if route.shape != weights.shape or route.shape[0] != student_mean.shape[0]:
        raise ValueError("skill_ids and dagger_weights must contain one value per sample")
    if route.dtype == torch.bool or route.is_floating_point() or route.is_complex():
        raise TypeError("skill_ids must use an integer dtype")
    if torch.any((route < 0) | (route >= num_skills)):
        raise ValueError("skill_ids contains an out-of-range route")
    group_mean_losses: list[torch.Tensor] = []
    group_sum_losses: list[torch.Tensor] = []
    for skill_id in range(num_skills):
        positive = (route == skill_id) & (weights > 0.0)
        count = torch.count_nonzero(positive)
        if count == 0:
            continue
        difference = student_mean[positive] - teacher_actions[positive]
        squared = difference.square()
        sample_weights = weights[positive]
        denominator = count.to(dtype=student_mean.dtype)
        group_mean_losses.append(
            (squared.mean(dim=1) * sample_weights).sum() / denominator
        )
        group_sum_losses.append(
            (squared.sum(dim=1) * sample_weights).sum() / denominator
        )
    if not group_mean_losses:
        connected_zero = student_mean.sum() * 0.0
        return connected_zero, connected_zero, effective_weight
    return (
        torch.stack(group_mean_losses).mean(),
        torch.stack(group_sum_losses).mean(),
        effective_weight,
    )


class DAggerPPO:
    """RSL-RL 2.3-style algorithm with a PHP loss curriculum.

    Teacher labels are generated on the states visited by the student.  The
    sampled student action returned by :meth:`act` is the only action that the
    runner is allowed to send to the environment.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        num_learning_epochs: int = 2,
        num_mini_batches: int = 96,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.001,
        learning_rate: float = 3.0e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float | None = 0.01,
        dagger_base_coef: float = 10.0,
        selector_loss_coef: float = 1.0,
        dagger_reduction: DAggerReduction = "sum_per_sample",
        curriculum_iterations: int = 10_000,
        minimum_dagger_weight: float = 0.1,
        adaptive_lr_minimum_ppo_weight: float = 0.1,
        balance_skill_losses: bool = True,
        maximum_action_magnitude: float = 1_000.0,
        normalize_advantage_per_mini_batch: bool = False,
        skill_names: Sequence[str] = ("locomotion", "climb", "down_roll"),
        device: str | torch.device = "cpu",
    ) -> None:
        if getattr(policy, "is_recurrent", False):
            raise ValueError("DAggerPPO currently supports only feed-forward policies.")
        if num_learning_epochs <= 0 or num_mini_batches <= 0:
            raise ValueError("num_learning_epochs and num_mini_batches must be positive.")
        if clip_param <= 0.0:
            raise ValueError("clip_param must be positive.")
        if not 0.0 <= gamma <= 1.0 or not 0.0 <= lam <= 1.0:
            raise ValueError("gamma and lam must lie in [0, 1].")
        if value_loss_coef < 0.0 or entropy_coef < 0.0:
            raise ValueError("Loss coefficients must be non-negative.")
        if learning_rate <= 0.0 or max_grad_norm <= 0.0:
            raise ValueError("learning_rate and max_grad_norm must be positive.")
        if schedule not in ("fixed", "adaptive"):
            raise ValueError("schedule must be 'fixed' or 'adaptive'.")
        if desired_kl is not None and desired_kl <= 0.0:
            raise ValueError("desired_kl must be positive when provided.")
        if dagger_base_coef < 0.0:
            raise ValueError("dagger_base_coef must be non-negative.")
        if not math.isfinite(selector_loss_coef) or selector_loss_coef < 0.0:
            raise ValueError("selector_loss_coef must be finite and non-negative.")
        if dagger_reduction not in ("mean_per_dof", "sum_per_sample"):
            raise ValueError("dagger_reduction must be 'mean_per_dof' or 'sum_per_sample'.")
        if not skill_names or len(set(skill_names)) != len(skill_names):
            raise ValueError("skill_names must be non-empty and unique.")
        if not isinstance(balance_skill_losses, bool):
            raise TypeError("balance_skill_losses must be a boolean.")
        if not math.isfinite(maximum_action_magnitude) or maximum_action_magnitude <= 0.0:
            raise ValueError("maximum_action_magnitude must be finite and positive.")

        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.storage: HybridRolloutStorage | None = None
        self.transition = HybridTransition()

        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.clip_param = float(clip_param)
        self.gamma = float(gamma)
        self.lam = float(lam)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.dagger_base_coef = float(dagger_base_coef)
        self.selector_loss_coef = float(selector_loss_coef)
        self.dagger_reduction = dagger_reduction
        self.balance_skill_losses = balance_skill_losses
        self.maximum_action_magnitude = float(maximum_action_magnitude)
        self.normalize_advantage_per_mini_batch = bool(normalize_advantage_per_mini_batch)
        self.skill_names = tuple(str(name) for name in skill_names)
        self.loss_schedule = PhpLossSchedule(
            curriculum_iterations=curriculum_iterations,
            minimum_dagger_weight=minimum_dagger_weight,
            adaptive_lr_minimum_ppo_weight=adaptive_lr_minimum_ppo_weight,
        )

    def init_storage(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        actor_obs_shape: Sequence[int],
        critic_obs_shape: Sequence[int],
        actions_shape: Sequence[int],
    ) -> None:
        self.storage = HybridRolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            num_skills=len(self.skill_names),
            device=self.device,
        )

    def _require_storage(self) -> HybridRolloutStorage:
        if self.storage is None:
            raise RuntimeError("init_storage() must be called before collecting rollouts.")
        return self.storage

    def act(
        self,
        actor_obs: torch.Tensor,
        critic_obs: torch.Tensor,
        teacher_actions: torch.Tensor,
        dagger_mask: torch.Tensor,
        skill_ids: torch.Tensor,
        actor_skill_ids: torch.Tensor,
        teacher_forcing_mask: torch.Tensor,
        actor_outputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Sample and record a student action for one vectorized step."""

        storage = self._require_storage()
        batch_size = storage.num_envs
        if actor_obs.shape != (batch_size, *storage.actor_obs_shape):
            raise ValueError(
                f"actor_obs shape must be {(batch_size, *storage.actor_obs_shape)}, "
                f"got {tuple(actor_obs.shape)}."
            )
        if critic_obs.shape != (batch_size, *storage.critic_obs_shape):
            raise ValueError(
                f"critic_obs shape must be {(batch_size, *storage.critic_obs_shape)}, "
                f"got {tuple(critic_obs.shape)}."
            )
        if teacher_actions.shape != (batch_size, *storage.actions_shape):
            raise ValueError(
                f"teacher_actions shape must be {(batch_size, *storage.actions_shape)}, "
                f"got {tuple(teacher_actions.shape)}."
            )

        oracle_routes = _skill_id_column(
            skill_ids, "skill_ids", batch_size, len(self.skill_names)
        )
        actor_routes = _skill_id_column(
            actor_skill_ids,
            "actor_skill_ids",
            batch_size,
            len(self.skill_names),
        )
        forcing = _as_column(
            teacher_forcing_mask, "teacher_forcing_mask", batch_size
        ).to(dtype=torch.bool)
        if actor_outputs is None:
            actions = self.policy.act(actor_obs, skill_ids=actor_routes)
        else:
            if not isinstance(actor_outputs, tuple) or len(actor_outputs) != 2:
                raise TypeError("actor_outputs must be an (all_action_means, selector_logits) tuple")
            self.policy.update_distribution_from_outputs(
                actor_outputs[0], actor_outputs[1], skill_ids=actor_routes
            )
            actions = self.policy.distribution.sample()
            if not torch.isfinite(actions).all():
                raise FloatingPointError("Sampled actions contain NaN or infinity.")
        if actions.shape != teacher_actions.shape:
            raise ValueError(
                f"Policy produced actions {tuple(actions.shape)} but teacher labels are "
                f"{tuple(teacher_actions.shape)}."
            )
        values = self.policy.evaluate(critic_obs)
        expected_values = (batch_size, 1)
        if values.shape != expected_values:
            raise ValueError(f"Policy critic must return {expected_values}, got {tuple(values.shape)}.")
        dagger_column = _as_column(dagger_mask, "dagger_mask", batch_size)
        positive_labels = dagger_column[:, 0] > 0.0
        if torch.any(positive_labels):
            valid_teacher = teacher_actions[positive_labels]
            if not torch.isfinite(valid_teacher).all():
                raise ValueError("Positive-confidence teacher actions contain NaN or infinity.")
            if valid_teacher.abs().max() > self.maximum_action_magnitude:
                raise FloatingPointError("Teacher action magnitude exceeds the configured safety bound.")
        if actions.abs().max() > self.maximum_action_magnitude:
            raise FloatingPointError("Student action magnitude exceeds the configured safety bound.")

        self.transition.actions = actions.detach()
        self.transition.values = values.detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(actions).detach().reshape(-1, 1)
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = actor_obs.detach()
        self.transition.privileged_observations = critic_obs.detach()
        self.transition.teacher_actions = teacher_actions.detach()
        self.transition.dagger_mask = dagger_column.detach()
        self.transition.skill_ids = oracle_routes.detach()
        self.transition.actor_skill_ids = actor_routes.detach()
        self.transition.teacher_forcing_mask = forcing.detach()
        return self.transition.actions

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict,
    ) -> None:
        """Finish and insert the transition after stepping the environment."""

        storage = self._require_storage()
        batch_size = storage.num_envs
        reward_column = _as_column(rewards, "rewards", batch_size).clone()
        done_column = _as_column(dones, "dones", batch_size)

        if "time_outs" in infos:
            time_outs = _as_column(infos["time_outs"].to(self.device), "time_outs", batch_size)
            values = HybridRolloutStorage._require("values", self.transition.values)
            reward_column.add_(self.gamma * values * time_outs.to(dtype=values.dtype))

        self.transition.rewards = reward_column
        self.transition.dones = done_column
        storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(done_column)

    def compute_returns(self, last_critic_obs: torch.Tensor) -> None:
        storage = self._require_storage()
        with torch.no_grad():
            last_values = self.policy.evaluate(last_critic_obs).detach()
        storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def _adapt_learning_rate(self, batch: HybridBatch) -> float:
        """Apply the upstream RSL-RL adaptive-KL rule and return mean KL."""

        sigma = self.policy.action_std
        mu = self.policy.action_mean
        with torch.no_grad():
            kl = torch.sum(
                torch.log(sigma / batch.old_action_sigma + 1.0e-5)
                + (
                    batch.old_action_sigma.square()
                    + (batch.old_action_mean - mu).square()
                )
                / (2.0 * sigma.square())
                - 0.5,
                dim=-1,
            )
            kl_mean = (
                skill_balanced_mean(kl, batch.actor_skill_ids, len(self.skill_names))
                if self.balance_skill_losses
                else kl.mean()
            )
            if not torch.isfinite(kl_mean):
                raise FloatingPointError("Adaptive-KL computation produced NaN or infinity.")
            if kl_mean > self.desired_kl * 2.0:  # type: ignore[operator]
                self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
            elif 0.0 < kl_mean < self.desired_kl / 2.0:  # type: ignore[operator]
                self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
            for parameter_group in self.optimizer.param_groups:
                parameter_group["lr"] = self.learning_rate
        return float(kl_mean.item())

    def update(self, iteration: int) -> dict[str, float]:
        """Optimize one rollout using the schedule at the absolute iteration."""

        storage = self._require_storage()
        if iteration < 0:
            raise ValueError("iteration must be non-negative.")
        weights = self.loss_schedule.at(iteration)
        adaptive_lr = (
            self.schedule == "adaptive"
            and self.desired_kl is not None
            and self.loss_schedule.adaptive_lr_enabled(iteration)
        )

        metric_sums: dict[str, float] = {
            "value_function": 0.0,
            "surrogate": 0.0,
            "entropy": 0.0,
            "ppo_total": 0.0,
            "dagger": 0.0,
            "dagger_mse_mean_per_dof": 0.0,
            "dagger_mse_sum_per_sample": 0.0,
            "selector": 0.0,
            "selector_accuracy": 0.0,
            "route_agreement": 0.0,
            "teacher_forcing_fraction": 0.0,
            "total": 0.0,
            "kl": 0.0,
        }
        skill_error_sums = [0.0] * len(self.skill_names)
        skill_valid_counts = [0.0] * len(self.skill_names)
        skill_weight_sums = [0.0] * len(self.skill_names)
        selector_correct_counts = [0.0] * len(self.skill_names)
        selector_sample_counts = [0.0] * len(self.skill_names)
        action_maxima = {
            "student_action_max_abs": 0.0,
            "teacher_action_max_abs": 0.0,
        }
        update_count = 0
        kl_update_count = 0

        generator = storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            advantages = batch.advantages
            if self.normalize_advantage_per_mini_batch:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

            # Rebuild exactly the hard-routed distribution that generated the
            # rollout.  Oracle routes are used only by selector/DAgger losses.
            self.policy.update_distribution(
                batch.observations, skill_ids=batch.actor_skill_ids
            )
            actions_log_prob = self.policy.get_actions_log_prob(batch.actions).reshape(-1, 1)
            values = self.policy.evaluate(batch.privileged_observations)
            oracle_route = batch.skill_ids.squeeze(-1)
            actor_route = batch.actor_skill_ids.squeeze(-1)
            if self.balance_skill_losses:
                def reduce_samples(sample_values: torch.Tensor) -> torch.Tensor:
                    return skill_balanced_mean(
                        sample_values,
                        actor_route,
                        len(self.skill_names),
                    )
            else:
                def reduce_samples(sample_values: torch.Tensor) -> torch.Tensor:
                    return sample_values.mean()
            entropy = reduce_samples(self.policy.entropy)

            if adaptive_lr:
                metric_sums["kl"] += self._adapt_learning_rate(batch)
                kl_update_count += 1

            ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = reduce_samples(torch.maximum(surrogate, surrogate_clipped))

            if self.use_clipped_value_loss:
                value_clipped = batch.target_values + (values - batch.target_values).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = reduce_samples(
                    torch.maximum(
                        (values - batch.returns).square(),
                        (value_clipped - batch.returns).square(),
                    )
                )
            else:
                value_loss = reduce_samples((batch.returns - values).square())

            ppo_loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            oracle_action_mean = self.policy.action_mean_for_skill(batch.skill_ids)
            if oracle_action_mean.abs().max() > self.maximum_action_magnitude:
                raise FloatingPointError(
                    "Student oracle-head action magnitude exceeds the configured safety bound."
                )
            selector_per_sample = nn.functional.cross_entropy(
                self.policy.selector_logits,
                oracle_route,
                reduction="none",
            )
            selector_loss = (
                skill_balanced_mean(
                    selector_per_sample,
                    oracle_route,
                    len(self.skill_names),
                )
                if self.balance_skill_losses
                else selector_per_sample.mean()
            )
            if self.balance_skill_losses:
                dagger_mean, dagger_sum, _ = skill_balanced_dagger_losses(
                    oracle_action_mean,
                    batch.teacher_actions,
                    batch.dagger_mask,
                    batch.skill_ids,
                    len(self.skill_names),
                )
            else:
                dagger_mean, dagger_sum, _ = masked_dagger_losses(
                    oracle_action_mean,
                    batch.teacher_actions,
                    batch.dagger_mask,
                )
            dagger_loss = dagger_mean if self.dagger_reduction == "mean_per_dof" else dagger_sum
            total_loss = (
                weights.ppo * ppo_loss
                + self.dagger_base_coef * weights.dagger * dagger_loss
                + self.selector_loss_coef * selector_loss
            )
            if not torch.isfinite(total_loss):
                raise FloatingPointError("DAgger-PPO loss produced NaN or infinity.")

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            metric_sums["value_function"] += float(value_loss.detach().item())
            metric_sums["surrogate"] += float(surrogate_loss.detach().item())
            metric_sums["entropy"] += float(entropy.detach().item())
            metric_sums["ppo_total"] += float(ppo_loss.detach().item())
            metric_sums["dagger"] += float(dagger_loss.detach().item())
            metric_sums["dagger_mse_mean_per_dof"] += float(dagger_mean.detach().item())
            metric_sums["dagger_mse_sum_per_sample"] += float(dagger_sum.detach().item())
            metric_sums["selector"] += float(selector_loss.detach().item())
            predictions = torch.argmax(self.policy.selector_logits.detach(), dim=-1)
            metric_sums["selector_accuracy"] += float(
                (predictions == oracle_route).float().mean().item()
            )
            metric_sums["route_agreement"] += float(
                (actor_route == oracle_route).float().mean().item()
            )
            metric_sums["teacher_forcing_fraction"] += float(
                batch.teacher_forcing_mask.float().mean().item()
            )
            action_maxima["student_action_max_abs"] = max(
                action_maxima["student_action_max_abs"],
                float(self.policy.action_mean.detach().abs().max().item()),
            )
            valid_teacher_mask = batch.dagger_mask[:, 0] > 0.0
            teacher_max = (
                float(batch.teacher_actions[valid_teacher_mask].abs().max().item())
                if torch.any(valid_teacher_mask)
                else 0.0
            )
            action_maxima["teacher_action_max_abs"] = max(
                action_maxima["teacher_action_max_abs"], teacher_max
            )
            metric_sums["total"] += float(total_loss.detach().item())

            sample_weights = batch.dagger_mask.squeeze(-1)
            valid = sample_weights > 0.0
            for skill_id in range(len(self.skill_names)):
                selector_selected = oracle_route == skill_id
                selector_count = int(selector_selected.sum().item())
                if selector_count:
                    selector_correct_counts[skill_id] += float(
                        (predictions[selector_selected] == skill_id).sum().item()
                    )
                    selector_sample_counts[skill_id] += selector_count
                selected = valid & (oracle_route == skill_id)
                count = int(selected.sum().item())
                if count:
                    weights_for_skill = sample_weights[selected]
                    weight_sum = float(weights_for_skill.sum().item())
                    squared_per_sample = (
                        oracle_action_mean.detach()[selected]
                        - batch.teacher_actions[selected]
                    ).square().mean(dim=-1)
                    skill_error_sums[skill_id] += float(
                        (squared_per_sample * weights_for_skill).sum().item()
                    )
                    skill_valid_counts[skill_id] += count
                    skill_weight_sums[skill_id] += weight_sum
            update_count += 1

        if update_count != self.num_learning_epochs * self.num_mini_batches:
            raise RuntimeError(
                f"Expected {self.num_learning_epochs * self.num_mini_batches} optimizer updates, "
                f"got {update_count}."
            )

        storage.clear()
        metrics = {name: total / update_count for name, total in metric_sums.items()}
        metrics.update(action_maxima)
        if kl_update_count == 0:
            metrics["kl"] = 0.0
        else:
            metrics["kl"] = metric_sums["kl"] / kl_update_count
        metrics["dagger_weight"] = weights.dagger
        metrics["ppo_weight"] = weights.ppo
        metrics["learning_rate"] = self.learning_rate
        for skill_id, skill_name in enumerate(self.skill_names):
            count = skill_valid_counts[skill_id]
            metrics[f"dagger_valid_count/{skill_name}"] = count / self.num_learning_epochs
            effective_weight = skill_weight_sums[skill_id]
            metrics[f"dagger_effective_weight/{skill_name}"] = (
                effective_weight / self.num_learning_epochs
            )
            metrics[f"dagger_mse/{skill_name}"] = (
                skill_error_sums[skill_id] / effective_weight
                if effective_weight > 0.0
                else 0.0
            )
            selector_count = selector_sample_counts[skill_id]
            metrics[f"selector_accuracy/{skill_name}"] = (
                selector_correct_counts[skill_id] / selector_count
                if selector_count > 0.0
                else 0.0
            )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise FloatingPointError("Non-finite value found in DAgger-PPO metrics.")
        return metrics
