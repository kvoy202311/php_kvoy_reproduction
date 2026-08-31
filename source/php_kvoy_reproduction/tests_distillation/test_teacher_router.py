from __future__ import annotations

from typing import Mapping

import pytest
import torch
from torch import nn

from php_kvoy_reproduction.distillation.action_transform import CanonicalActionTransform
from php_kvoy_reproduction.distillation.teacher_router import TeacherRouter


class RecordingTeacher(nn.Module):
    def __init__(self, observation_dim: int, value: float, fingerprint: str) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.action_dim = 29
        self.value = value
        self._fingerprint = fingerprint
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.received: list[torch.Tensor] = []

    def fingerprint(self) -> str:
        return self._fingerprint

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        self.received.append(observations.detach().clone())
        return torch.full(
            (observations.shape[0], 29),
            self.value,
            dtype=observations.dtype,
            device=observations.device,
        )


def make_router() -> tuple[TeacherRouter, dict[str, RecordingTeacher]]:
    teachers = {
        "locomotion": RecordingTeacher(3, 1.0, "loc-fingerprint"),
        "climb": RecordingTeacher(4, 2.0, "climb-fingerprint"),
        "down_roll": RecordingTeacher(5, 3.0, "down-fingerprint"),
    }
    router = TeacherRouter(teachers, {"climb": 1, "down_roll": 2, "locomotion": 0})
    return router, teachers


def observations(batch_size: int = 6) -> dict[str, torch.Tensor]:
    rows = torch.arange(batch_size, dtype=torch.float32).unsqueeze(1)
    return {
        "locomotion": rows.repeat(1, 3),
        "climb": rows.repeat(1, 4),
        "down_roll": rows.repeat(1, 5),
    }


@pytest.mark.parametrize("column", [False, True])
def test_subset_inference_and_scatter_preserve_environment_order(column: bool) -> None:
    router, teachers = make_router()
    routes = torch.tensor([2, 0, 1, 2, 1, 0])
    routed = routes[:, None] if column else routes
    result = router.act(observations(), routed)
    expected = torch.tensor([3.0, 1.0, 2.0, 3.0, 2.0, 1.0])
    torch.testing.assert_close(result.actions[:, 0], expected)
    assert result.actions.shape == (6, 29)
    assert result.valid_mask.shape == (6, 1)
    assert result.valid_mask.eq(1.0).all()
    torch.testing.assert_close(result.skill_ids[:, 0], routes)
    torch.testing.assert_close(teachers["locomotion"].received[0][:, 0], torch.tensor([1.0, 5.0]))
    torch.testing.assert_close(teachers["climb"].received[0][:, 0], torch.tensor([2.0, 4.0]))
    torch.testing.assert_close(teachers["down_roll"].received[0][:, 0], torch.tensor([0.0, 3.0]))


def test_global_and_per_skill_validity_scatter() -> None:
    router, _ = make_router()
    routes = torch.tensor([2, 0, 1, 2, 1, 0])
    global_mask = torch.tensor([True, False, True, False, True, True])
    result = router.act(observations(), routes, validity_mask=global_mask)
    torch.testing.assert_close(result.valid_mask[:, 0], global_mask.float())

    per_skill = {
        "locomotion": torch.tensor([False, False, False, False, False, False]),
        "climb": torch.tensor([False, False, False, False, True, False]),
        "down_roll": torch.tensor([True, False, False, False, False, False]),
    }
    result = router.act(observations(), routes, validity_by_skill=per_skill)
    torch.testing.assert_close(
        result.valid_mask[:, 0], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    )

    confidence = torch.tensor([1.0, 0.0, 0.75, 0.25, 0.5, 1.0])
    result = router.act(observations(), routes, validity_mask=confidence)
    torch.testing.assert_close(result.valid_mask[:, 0], confidence)


def test_teachers_are_frozen_and_stay_in_eval_mode() -> None:
    router, teachers = make_router()
    assert all(not parameter.requires_grad for parameter in router.parameters())
    assert not router.training
    router.train(True)
    assert not router.training
    assert all(not teacher.training for teacher in teachers.values())
    assert router.fingerprints() == {
        "locomotion": "loc-fingerprint",
        "climb": "climb-fingerprint",
        "down_roll": "down-fingerprint",
    }


@pytest.mark.parametrize(
    "routes",
    [
        torch.tensor([-1, 0]),
        torch.tensor([0, 3]),
        torch.tensor([0.0, 1.5]),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([[0, 1]]),
    ],
)
def test_invalid_routes_are_rejected(routes: torch.Tensor) -> None:
    router, _ = make_router()
    with pytest.raises((TypeError, ValueError)):
        router.act(observations(batch_size=2), routes)


def test_missing_observations_wrong_dimensions_and_nan_are_rejected() -> None:
    router, _ = make_router()
    routes = torch.tensor([0, 1, 2])
    missing = observations(3)
    del missing["climb"]
    with pytest.raises(ValueError, match="missing routed skill"):
        router.act(missing, routes)

    wrong = observations(3)
    wrong["climb"] = torch.zeros(3, 3)
    with pytest.raises(ValueError, match="expected 4"):
        router.act(wrong, routes)

    invalid = observations(3)
    invalid["down_roll"][2, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        router.act(invalid, routes)


def test_zero_confidence_rows_never_enter_the_teacher() -> None:
    router, teachers = make_router()
    obs = observations(3)
    obs["down_roll"][2, 0] = float("nan")
    result = router.act(
        obs,
        torch.tensor([0, 1, 2]),
        validity_mask=torch.tensor([1.0, 1.0, 0.0]),
    )
    assert result.actions[2].eq(0.0).all()
    assert result.valid_mask[2].item() == 0.0
    assert teachers["down_roll"].received == []


def test_invalid_validity_contract_is_rejected() -> None:
    router, _ = make_router()
    routes = torch.tensor([0, 1, 2])
    with pytest.raises(TypeError, match="bool or floating"):
        router.act(observations(3), routes, validity_mask=torch.ones(3, dtype=torch.long))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        router.act(observations(3), routes, validity_mask=torch.tensor([1.0, 1.1, 0.0]))
    with pytest.raises(ValueError, match="only one"):
        router.act(
            observations(3),
            routes,
            validity_mask=torch.ones(3, dtype=torch.bool),
            validity_by_skill={"locomotion": torch.ones(3, dtype=torch.bool)},
        )


def test_skill_ids_are_explicit_unique_and_contiguous() -> None:
    teachers = {
        "a": RecordingTeacher(2, 1.0, "a"),
        "b": RecordingTeacher(2, 2.0, "b"),
    }
    with pytest.raises(ValueError, match="duplicate"):
        TeacherRouter(teachers, {"a": 0, "b": 0})
    with pytest.raises(ValueError, match="contiguous"):
        TeacherRouter(teachers, {"a": 0, "b": 2})
    with pytest.raises(ValueError, match="same skill names"):
        TeacherRouter(teachers, {"a": 0, "c": 1})


def test_explicit_action_transforms_produce_canonical_labels() -> None:
    teachers = {
        "a": RecordingTeacher(2, 1.0, "a"),
        "b": RecordingTeacher(2, 2.0, "b"),
    }

    def transform(teacher_scale: float) -> CanonicalActionTransform:
        return CanonicalActionTransform(
            teacher_default_q=[0.0] * 29,
            teacher_action_scale=teacher_scale,
            student_default_q=[0.0] * 29,
            student_action_scale=0.5,
            hard_lower_limits=[-10.0] * 29,
            hard_upper_limits=[10.0] * 29,
        )

    router = TeacherRouter(
        teachers,
        {"a": 0, "b": 1},
        action_transforms={"a": transform(1.0), "b": transform(0.5)},
    )
    obs = {"a": torch.zeros(2, 2), "b": torch.zeros(2, 2)}
    result = router.act(obs, torch.tensor([0, 1]))
    # Both physical targets are 1 radian and therefore both canonical labels are 2.
    torch.testing.assert_close(result.actions, torch.full((2, 29), 2.0))


def test_observations_for_unused_skills_are_not_required() -> None:
    router, teachers = make_router()
    obs = {"climb": torch.zeros(2, 4)}
    result = router.act(obs, torch.tensor([1, 1]))
    assert result.actions.eq(2.0).all()
    assert teachers["locomotion"].received == []
    assert teachers["down_roll"].received == []
