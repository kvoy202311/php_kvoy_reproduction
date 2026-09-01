from __future__ import annotations

import pytest
import torch

from php_kvoy_reproduction.distillation.motion_boundary import (
    motion_reference_advance_mask,
)
from php_kvoy_reproduction.distillation.option_controller import (
    OptionStateController,
    linear_teacher_forcing_probability,
)


def _controller(num_envs: int = 1) -> OptionStateController:
    return OptionStateController(
        num_envs,
        ("locomotion", "climb", "down_roll"),
        activation_probability=0.6,
        release_probability=0.55,
        activation_confirmation_steps=2,
        release_confirmation_steps=2,
        post_release_cooldown_steps=2,
        minimum_skill_duration_steps={"climb": 2, "down_roll": 2},
        maximum_skill_duration_steps={"climb": 6, "down_roll": 6},
    )


def test_motion_activation_requires_confident_consecutive_frames() -> None:
    controller = _controller()
    climb = torch.tensor([[0.0, 3.0, 0.0]])
    first = controller.select(climb)
    assert first.active_skill_ids.item() == 0
    second = controller.select(climb)
    assert second.active_skill_ids.item() == 1
    assert second.switched.item()


def test_motion_clock_waits_for_the_confirmed_action_head() -> None:
    controller = OptionStateController(
        1,
        ("locomotion", "climb", "down_roll"),
        activation_probability=0.6,
        release_probability=0.55,
        activation_confirmation_steps=3,
        release_confirmation_steps=2,
        post_release_cooldown_steps=2,
        minimum_skill_duration_steps={"climb": 2, "down_roll": 2},
        maximum_skill_duration_steps={"climb": 6, "down_roll": 6},
    )
    climb = torch.tensor([[0.0, 3.0, 0.0]])
    advance_decisions = []
    for _ in range(3):
        selection = controller.select(climb)
        advance = motion_reference_advance_mask(
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([1], dtype=torch.long),
            selection.active_skill_ids[:, 0],
            torch.tensor([False]),
        )
        advance_decisions.append(bool(advance.item()))

    assert advance_decisions == [False, False, True]


def test_committed_motion_cannot_switch_directly_to_another_motion() -> None:
    controller = _controller()
    climb = torch.tensor([[0.0, 3.0, 0.0]])
    controller.select(climb)
    controller.select(climb)
    down_roll = torch.tensor([[0.0, 0.0, 3.0]])
    for _ in range(3):
        selection = controller.select(down_roll)
        assert selection.active_skill_ids.item() == 1


def test_motion_release_requires_minimum_duration_and_confirmation() -> None:
    controller = _controller()
    climb = torch.tensor([[0.0, 3.0, 0.0]])
    locomotion = torch.tensor([[3.0, 0.0, 0.0]])
    controller.select(climb)
    controller.select(climb)
    first_release_vote = controller.select(locomotion)
    assert first_release_vote.active_skill_ids.item() == 1
    released = controller.select(locomotion)
    assert released.active_skill_ids.item() == 0


def test_safety_timeout_releases_into_a_retrigger_cooldown() -> None:
    controller = _controller()
    climb = torch.tensor([[0.0, 3.0, 0.0]])
    controller.select(climb)
    controller.select(climb)
    for _ in range(5):
        selection = controller.select(climb)
    assert selection.active_skill_ids.item() == 0
    # A still-high climb score cannot immediately re-enter the option.
    assert controller.select(climb).active_skill_ids.item() == 0


def test_teacher_forcing_is_sampled_per_episode_not_per_step() -> None:
    controller = _controller(num_envs=2)
    logits = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    oracle = torch.tensor([[1], [2]])
    forced = controller.select(
        logits,
        oracle_skill_ids=oracle,
        teacher_forcing_probability=1.0,
    )
    torch.testing.assert_close(forced.active_skill_ids, oracle)
    still_forced = controller.select(
        logits,
        oracle_skill_ids=oracle,
        teacher_forcing_probability=0.0,
    )
    assert still_forced.teacher_forcing_mask.all()
    controller.reset(
        torch.tensor([True, False]), teacher_forcing_probability=0.0
    )
    assert not controller.teacher_forcing_mask[0]
    assert controller.teacher_forcing_mask[1]


def test_linear_teacher_forcing_schedule_has_exact_boundaries() -> None:
    assert linear_teacher_forcing_probability(
        0, start=1.0, end=0.25, curriculum_iterations=100
    ) == pytest.approx(1.0)
    assert linear_teacher_forcing_probability(
        50, start=1.0, end=0.25, curriculum_iterations=100
    ) == pytest.approx(0.625)
    assert linear_teacher_forcing_probability(
        1000, start=1.0, end=0.25, curriculum_iterations=100
    ) == pytest.approx(0.25)
