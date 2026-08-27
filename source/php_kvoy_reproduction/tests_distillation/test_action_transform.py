from __future__ import annotations

import pytest
import torch

from php_kvoy_reproduction.distillation.action_transform import (
    CanonicalActionTransform,
    transform_teacher_actions,
)


def vectors() -> dict[str, torch.Tensor]:
    return {
        "zero": torch.zeros(29),
        "one": torch.ones(29),
        "lower": torch.full((29,), -10.0),
        "upper": torch.full((29,), 10.0),
    }


def test_identity_conversion_and_audit_targets() -> None:
    values = vectors()
    actions = torch.linspace(-1.0, 1.0, 29)
    result = transform_teacher_actions(
        actions,
        teacher_default_q=values["zero"],
        teacher_action_scale=1.0,
        student_default_q=values["zero"],
        student_action_scale=1.0,
        hard_lower_limits=values["lower"],
        hard_upper_limits=values["upper"],
    )
    torch.testing.assert_close(result.actions, actions)
    torch.testing.assert_close(result.q_target, actions)
    torch.testing.assert_close(result.q_target_clamped, actions)
    assert not result.clamped_mask.any()


def test_different_teacher_and_student_conventions() -> None:
    values = vectors()
    transform = CanonicalActionTransform(
        teacher_default_q=values["one"],
        teacher_action_scale=0.5,
        student_default_q=values["zero"],
        student_action_scale=0.25,
        hard_lower_limits=values["lower"],
        hard_upper_limits=values["upper"],
    )
    result = transform(torch.full((2, 29), 2.0))
    torch.testing.assert_close(result.q_target, torch.full((2, 29), 2.0))
    torch.testing.assert_close(result.actions, torch.full((2, 29), 8.0))


def test_hard_limits_clamp_absolute_targets_before_student_conversion() -> None:
    transform = CanonicalActionTransform(
        teacher_default_q=torch.zeros(29),
        teacher_action_scale=2.0,
        student_default_q=torch.zeros(29),
        student_action_scale=0.5,
        hard_lower_limits=torch.full((29,), -1.0),
        hard_upper_limits=torch.full((29,), 1.0),
    )
    result = transform(torch.tensor([[-2.0] * 10 + [0.25] * 9 + [2.0] * 10]))
    assert result.clamped_mask.sum().item() == 20
    torch.testing.assert_close(result.q_target_clamped[0, :10], torch.full((10,), -1.0))
    torch.testing.assert_close(result.actions[0, -10:], torch.full((10,), 2.0))


def test_arbitrary_batch_prefix_and_dtype_are_preserved() -> None:
    transform = CanonicalActionTransform(
        teacher_default_q=torch.zeros(29),
        teacher_action_scale=torch.ones(29),
        student_default_q=torch.zeros(29),
        student_action_scale=torch.ones(29),
        hard_lower_limits=torch.full((29,), -5.0),
        hard_upper_limits=torch.full((29,), 5.0),
    )
    actions = torch.zeros((2, 3, 29), dtype=torch.float64)
    result = transform(actions)
    assert result.actions.shape == (2, 3, 29)
    assert result.actions.dtype == torch.float64


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("teacher_default_q", [0.0] * 28),
        ("teacher_action_scale", 0.0),
        ("teacher_action_scale", -0.1),
        ("student_action_scale", [1.0] * 28 + [-0.1]),
        ("student_action_scale", [1.0] * 28 + [float("nan")]),
        ("hard_lower_limits", [0.0] * 28),
    ],
)
def test_invalid_control_vectors_are_rejected(keyword: str, value: object) -> None:
    arguments: dict[str, object] = {
        "teacher_default_q": [0.0] * 29,
        "teacher_action_scale": 1.0,
        "student_default_q": [0.0] * 29,
        "student_action_scale": 1.0,
        "hard_lower_limits": [-1.0] * 29,
        "hard_upper_limits": [1.0] * 29,
    }
    arguments[keyword] = value
    with pytest.raises(ValueError):
        CanonicalActionTransform(**arguments)


def test_reversed_hard_limits_are_rejected() -> None:
    with pytest.raises(ValueError, match="less than or equal"):
        CanonicalActionTransform(
            teacher_default_q=[0.0] * 29,
            teacher_action_scale=1.0,
            student_default_q=[0.0] * 29,
            student_action_scale=1.0,
            hard_lower_limits=[1.0] + [-1.0] * 28,
            hard_upper_limits=[0.0] + [1.0] * 28,
        )


def test_input_shape_dtype_and_finiteness_are_strict() -> None:
    transform = CanonicalActionTransform(
        teacher_default_q=[0.0] * 29,
        teacher_action_scale=1.0,
        student_default_q=[0.0] * 29,
        student_action_scale=1.0,
        hard_lower_limits=[-1.0] * 29,
        hard_upper_limits=[1.0] * 29,
    )
    with pytest.raises(ValueError, match="end in 29"):
        transform(torch.zeros(2, 28))
    with pytest.raises(TypeError, match="floating-point"):
        transform(torch.zeros(2, 29, dtype=torch.int64))
    invalid = torch.zeros(2, 29)
    invalid[0, 0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        transform(invalid)


def test_joint_orders_must_be_both_present_and_identical() -> None:
    base = dict(
        teacher_default_q=[0.0] * 29,
        teacher_action_scale=1.0,
        student_default_q=[0.0] * 29,
        student_action_scale=1.0,
        hard_lower_limits=[-1.0] * 29,
        hard_upper_limits=[1.0] * 29,
    )
    names = [f"joint_{index}" for index in range(29)]
    with pytest.raises(ValueError, match="provided together"):
        CanonicalActionTransform(**base, teacher_joint_order=names)
    reversed_names = list(reversed(names))
    with pytest.raises(ValueError, match="orders differ"):
        CanonicalActionTransform(
            **base, teacher_joint_order=names, student_joint_order=reversed_names
        )
