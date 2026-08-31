from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn
from torch.distributions import Normal

from php_kvoy_reproduction.distillation.dagger_ppo import (
    DAggerPPO,
    masked_dagger_losses,
    skill_balanced_dagger_losses,
    skill_balanced_mean,
)
from php_kvoy_reproduction.distillation.observation import (
    BlockwiseObservationNormalizer,
    VisionObservationLayout,
)
from php_kvoy_reproduction.distillation.rollout_storage import HybridRolloutStorage, HybridTransition
from php_kvoy_reproduction.distillation.schedules import PhpLossSchedule


def test_php_schedule_boundaries_and_absolute_resume_iteration():
    schedule = PhpLossSchedule(curriculum_iterations=10_000, minimum_dagger_weight=0.1)
    assert schedule.at(0).dagger == pytest.approx(1.0)
    assert schedule.at(0).ppo == pytest.approx(0.0)
    assert schedule.at(1_000).ppo == pytest.approx(0.1)
    assert not schedule.adaptive_lr_enabled(1_000)
    assert schedule.adaptive_lr_enabled(1_001)
    assert schedule.at(9_000).dagger == pytest.approx(0.1)
    assert schedule.at(100_000).dagger == pytest.approx(0.1)
    # The schedule is a pure function of the restored absolute iteration.
    assert schedule.at(7_321) == PhpLossSchedule().at(7_321)


def test_blockwise_normalizer_leaves_command_and_depth_bit_exact():
    layout = VisionObservationLayout(
        proprio_frame_dim=2,
        proprio_history_length=2,
        command_dim=2,
        depth_height=15,
        depth_width=15,
    )
    observations = torch.randn(5, layout.actor_obs_dim)
    command_before = observations[:, layout.command_slice].clone()
    depth_before = observations[:, layout.depth_slice].clone()
    normalizer = BlockwiseObservationNormalizer(layout)
    transformed = normalizer.transform(observations, update=True)
    assert torch.equal(transformed[:, layout.command_slice], command_before)
    assert torch.equal(transformed[:, layout.depth_slice], depth_before)
    assert normalizer.proprio.count.item() == 5


def test_observation_layout_rejects_wrong_dimension_and_order_contract():
    layout = VisionObservationLayout()
    with pytest.raises(ValueError, match="dimension mismatch"):
        layout.validate(torch.zeros(2, layout.actor_obs_dim - 1))
    assert layout.proprio_slice.stop == 744
    assert layout.command_slice == slice(744, 746)
    assert layout.depth_slice == slice(746, 5792)


def _transition(step: int, num_envs: int = 4) -> HybridTransition:
    ids = torch.arange(num_envs, dtype=torch.float32).unsqueeze(1) + 10.0 * step
    actions = ids.repeat(1, 2)
    return HybridTransition(
        observations=torch.cat((ids, ids + 100.0, ids + 200.0), dim=1),
        privileged_observations=torch.cat((ids, ids + 300.0), dim=1),
        actions=actions,
        rewards=torch.ones(num_envs),
        dones=torch.zeros(num_envs, dtype=torch.bool),
        values=ids,
        actions_log_prob=ids,
        action_mean=actions + 1.0,
        action_sigma=torch.ones_like(actions),
        teacher_actions=actions + 1_000.0,
        dagger_mask=torch.ones(num_envs, 1, dtype=torch.bool),
        skill_ids=(ids.long() % 3),
        actor_skill_ids=((ids.long() + 1) % 3),
        teacher_forcing_mask=torch.zeros(num_envs, 1, dtype=torch.bool),
    )


def test_hybrid_storage_shuffle_keeps_labels_routes_and_actions_aligned():
    storage = HybridRolloutStorage(4, 2, [3], [2], [2], num_skills=3)
    storage.add_transitions(_transition(0))
    storage.add_transitions(_transition(1))
    storage.compute_returns(torch.zeros(4, 1), gamma=0.99, lam=0.95)
    seen = 0
    for batch in storage.mini_batch_generator(num_mini_batches=4, num_epochs=1):
        identity = batch.observations[:, 0]
        assert torch.equal(batch.actions[:, 0], identity)
        assert torch.equal(batch.teacher_actions[:, 0], identity + 1_000.0)
        assert torch.equal(batch.skill_ids[:, 0], identity.long() % 3)
        assert torch.equal(batch.actor_skill_ids[:, 0], (identity.long() + 1) % 3)
        seen += batch.observations.shape[0]
    assert seen == 8


def test_storage_refuses_to_drop_remainder_samples():
    storage = HybridRolloutStorage(4, 2, [3], [2], [2], num_skills=3)
    storage.add_transitions(_transition(0))
    storage.add_transitions(_transition(1))
    storage.compute_returns(torch.zeros(4, 1), gamma=0.99, lam=0.95)
    with pytest.raises(ValueError, match="not divisible"):
        list(storage.mini_batch_generator(num_mini_batches=3, num_epochs=1))


def test_all_zero_dagger_mask_is_finite_and_has_zero_gradient():
    student = torch.randn(7, 29, requires_grad=True)
    teacher = torch.randn(7, 29)
    mask = torch.zeros(7, 1, dtype=torch.bool)
    mean_loss, sum_loss, count = masked_dagger_losses(student, teacher, mask)
    assert mean_loss.item() == 0.0
    assert sum_loss.item() == 0.0
    assert count.item() == 0.0
    mean_loss.backward()
    assert student.grad is not None
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_skill_balanced_reduction_is_independent_of_route_duration() -> None:
    values = torch.tensor([1.0] * 8 + [3.0] + [5.0])
    routes = torch.tensor([0] * 8 + [1] + [2])
    assert values.mean().item() != pytest.approx(3.0)
    assert skill_balanced_mean(values, routes, 3).item() == pytest.approx(3.0)


def test_skill_balanced_dagger_uses_equal_valid_skill_weight() -> None:
    routes = torch.tensor([[0], [0], [0], [1], [2]])
    student = torch.zeros(5, 2, requires_grad=True)
    teacher = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]
    )
    mean_loss, sum_loss, count = skill_balanced_dagger_losses(
        student,
        teacher,
        torch.ones(5, 1, dtype=torch.bool),
        routes,
        3,
    )
    assert mean_loss.item() == pytest.approx((1.0 + 4.0 + 9.0) / 3.0)
    assert sum_loss.item() == pytest.approx(2.0 * (1.0 + 4.0 + 9.0) / 3.0)
    assert count.item() == 5.0
    mean_loss.backward()
    assert torch.isfinite(student.grad).all()


def test_continuous_dagger_confidence_is_weighted_within_each_skill() -> None:
    routes = torch.tensor([[0], [0], [1]])
    student = torch.zeros(3, 2, requires_grad=True)
    teacher = torch.tensor([[1.0, 1.0], [3.0, 3.0], [2.0, 2.0]])
    weights = torch.tensor([[1.0], [0.25], [1.0]])
    mean_loss, sum_loss, effective_weight = skill_balanced_dagger_losses(
        student,
        teacher,
        weights,
        routes,
        2,
    )
    skill_zero = (1.0 * 1.0 + 9.0 * 0.25) / 2.0
    expected_mean = (skill_zero + 4.0) / 2.0
    assert mean_loss.item() == pytest.approx(expected_mean)
    assert sum_loss.item() == pytest.approx(2.0 * expected_mean)
    assert effective_weight.item() == pytest.approx(2.25)


def test_uniform_continuous_confidence_reduces_dagger_strength() -> None:
    routes = torch.tensor([[0], [0], [1], [1], [2], [2]])
    student = torch.zeros(6, 29)
    teacher = torch.ones(6, 29)
    full_loss = skill_balanced_dagger_losses(
        student,
        teacher,
        torch.ones(6, 1),
        routes,
        3,
    )[1]
    weak_loss = skill_balanced_dagger_losses(
        student,
        teacher,
        torch.full((6, 1), 0.1),
        routes,
        3,
    )[1]
    assert weak_loss.item() == pytest.approx(0.1 * full_loss.item())


def test_zero_confidence_labels_remain_excluded_from_binary_mask_mean() -> None:
    student = torch.zeros(2, 2)
    teacher = torch.tensor([[1.0, 1.0], [10.0, 10.0]])
    mean_loss, sum_loss, effective_weight = masked_dagger_losses(
        student,
        teacher,
        torch.tensor([[1.0], [0.0]]),
    )
    assert mean_loss.item() == pytest.approx(1.0)
    assert sum_loss.item() == pytest.approx(2.0)
    assert effective_weight.item() == pytest.approx(1.0)


def test_php_sum_reduction_is_exactly_action_dim_times_per_dof_mean() -> None:
    student = torch.randn(7, 29)
    teacher = torch.randn(7, 29)
    weights = torch.rand(7, 1)
    mean_loss, sum_loss, _ = masked_dagger_losses(student, teacher, weights)
    assert sum_loss.item() == pytest.approx(29.0 * mean_loss.item(), rel=1.0e-6)


class _TinyPolicy(nn.Module):
    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(3, 3 * 29)
        self.selector = nn.Linear(3, 3)
        self.critic = nn.Linear(2, 1)
        self.log_std = nn.Parameter(torch.full((29,), -2.0))
        self.distribution = None
        self._all_action_means = None
        self._selector_logits = None

    def reset(self, dones=None):
        pass

    def update_distribution(self, observations, skill_ids=None):
        self._all_action_means = self.actor(observations).reshape(-1, 3, 29)
        self._selector_logits = self.selector(observations)
        if skill_ids is None:
            skill_ids = self._selector_logits.argmax(dim=-1)
        route = skill_ids.reshape(-1)
        mean = self._all_action_means[
            torch.arange(observations.shape[0]), route
        ]
        self.distribution = Normal(mean, self.log_std.exp().expand_as(mean))

    def act(self, observations, **kwargs):
        self.update_distribution(observations, kwargs.get("skill_ids"))
        return self.distribution.sample()

    @property
    def selector_logits(self):
        return self._selector_logits

    def action_mean_for_skill(self, skill_ids):
        route = skill_ids.reshape(-1)
        return self._all_action_means[
            torch.arange(route.shape[0]), route
        ]

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(-1)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(-1)

    def evaluate(self, observations, **kwargs):
        return self.critic(observations)


def test_algorithm_uses_absolute_iteration_and_reports_each_skill():
    torch.manual_seed(1)
    algorithm = DAggerPPO(
        _TinyPolicy(),
        num_learning_epochs=1,
        num_mini_batches=1,
        schedule="fixed",
        learning_rate=1.0e-4,
    )
    algorithm.init_storage(4, 2, [3], [2], [29])
    for step in range(2):
        actor_obs = torch.randn(4, 3)
        critic_obs = torch.randn(4, 2)
        teacher = torch.randn(4, 29)
        actions = algorithm.act(
            actor_obs,
            critic_obs,
            teacher,
            torch.ones(4, 1, dtype=torch.bool),
            torch.tensor([[0], [1], [2], [0]]),
            torch.tensor([[0], [1], [2], [0]]),
            torch.ones(4, 1, dtype=torch.bool),
        )
        assert actions.shape == (4, 29)
        algorithm.process_env_step(torch.ones(4), torch.zeros(4), {})
    algorithm.compute_returns(torch.randn(4, 2))
    metrics = algorithm.update(iteration=5_000)
    assert metrics["dagger_weight"] == pytest.approx(0.5)
    assert metrics["ppo_weight"] == pytest.approx(0.5)
    assert metrics["dagger_valid_count/locomotion"] == 4
    assert metrics["dagger_valid_count/climb"] == 2
    assert metrics["dagger_valid_count/down_roll"] == 2
    assert "selector_accuracy" in metrics
