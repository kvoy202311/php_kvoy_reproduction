"""Canonical action conversion for heterogeneous teacher policies.

The policies used by the distillation pipeline predict normalized joint-position
commands.  A raw command is only meaningful together with the default pose and
action scale used during that policy's training.  This module converts through
the physical (absolute joint-position) representation before producing a label
in the student's action convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


NUM_CANONICAL_JOINTS = 29


def _as_joint_vector(
    value: float | Sequence[float] | torch.Tensor,
    *,
    name: str,
    allow_scalar: bool,
) -> torch.Tensor:
    """Return a finite float64 vector in canonical joint order."""

    try:
        tensor = torch.as_tensor(value, dtype=torch.float64)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if tensor.ndim == 0 and allow_scalar:
        tensor = tensor.repeat(NUM_CANONICAL_JOINTS)
    elif tensor.ndim != 1 or tensor.shape[0] != NUM_CANONICAL_JOINTS:
        expected = "a scalar or a 29-element vector" if allow_scalar else "a 29-element vector"
        raise ValueError(f"{name} must be {expected}; got shape {tuple(tensor.shape)}")

    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains a non-finite value")
    return tensor.contiguous()


def _validate_joint_order(joint_order: Sequence[str], *, name: str) -> tuple[str, ...]:
    order = tuple(joint_order)
    if len(order) != NUM_CANONICAL_JOINTS:
        raise ValueError(f"{name} must contain exactly {NUM_CANONICAL_JOINTS} names")
    if any(not isinstance(joint, str) or not joint for joint in order):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(order)) != len(order):
        raise ValueError(f"{name} contains duplicate joint names")
    return order


@dataclass(frozen=True)
class ActionTransformResult:
    """Auditable result of a teacher-to-student action conversion.

    Attributes:
        actions: Labels expressed in the student's normalized action convention.
        q_target: Unclamped absolute joint target produced by the teacher.
        q_target_clamped: Absolute target after applying URDF hard joint limits.
        clamped_mask: Element-wise indication that a hard limit changed a target.
    """

    actions: torch.Tensor
    q_target: torch.Tensor
    q_target_clamped: torch.Tensor
    clamped_mask: torch.Tensor


class CanonicalActionTransform:
    """Convert raw teacher actions into a student's normalized convention."""

    def __init__(
        self,
        *,
        teacher_default_q: Sequence[float] | torch.Tensor,
        teacher_action_scale: float | Sequence[float] | torch.Tensor,
        student_default_q: Sequence[float] | torch.Tensor,
        student_action_scale: float | Sequence[float] | torch.Tensor,
        hard_lower_limits: Sequence[float] | torch.Tensor,
        hard_upper_limits: Sequence[float] | torch.Tensor,
        teacher_joint_order: Sequence[str] | None = None,
        student_joint_order: Sequence[str] | None = None,
    ) -> None:
        self.teacher_default_q = _as_joint_vector(
            teacher_default_q, name="teacher_default_q", allow_scalar=False
        )
        self.teacher_action_scale = _as_joint_vector(
            teacher_action_scale, name="teacher_action_scale", allow_scalar=True
        )
        self.student_default_q = _as_joint_vector(
            student_default_q, name="student_default_q", allow_scalar=False
        )
        self.student_action_scale = _as_joint_vector(
            student_action_scale, name="student_action_scale", allow_scalar=True
        )
        self.hard_lower_limits = _as_joint_vector(
            hard_lower_limits, name="hard_lower_limits", allow_scalar=False
        )
        self.hard_upper_limits = _as_joint_vector(
            hard_upper_limits, name="hard_upper_limits", allow_scalar=False
        )

        if bool((self.teacher_action_scale <= 0).any()):
            raise ValueError("teacher_action_scale must contain only positive values")
        if bool((self.student_action_scale <= 0).any()):
            raise ValueError("student_action_scale must contain only positive values")
        if bool((self.hard_lower_limits > self.hard_upper_limits).any()):
            raise ValueError("hard_lower_limits must be less than or equal to hard_upper_limits")

        if (teacher_joint_order is None) != (student_joint_order is None):
            raise ValueError("teacher_joint_order and student_joint_order must be provided together")
        if teacher_joint_order is not None and student_joint_order is not None:
            teacher_order = _validate_joint_order(teacher_joint_order, name="teacher_joint_order")
            student_order = _validate_joint_order(student_joint_order, name="student_joint_order")
            if teacher_order != student_order:
                raise ValueError("teacher and student joint orders differ; implicit reordering is forbidden")
            self.joint_order: tuple[str, ...] | None = teacher_order
        else:
            self.joint_order = None

    def __call__(self, teacher_actions: torch.Tensor) -> ActionTransformResult:
        return self.transform(teacher_actions)

    def transform(self, teacher_actions: torch.Tensor) -> ActionTransformResult:
        """Transform a tensor whose final dimension is the canonical 29 joints."""

        if not isinstance(teacher_actions, torch.Tensor):
            raise TypeError("teacher_actions must be a torch.Tensor")
        if not teacher_actions.is_floating_point():
            raise TypeError("teacher_actions must have a floating-point dtype")
        if teacher_actions.ndim < 1 or teacher_actions.shape[-1] != NUM_CANONICAL_JOINTS:
            raise ValueError(
                f"teacher_actions must end in {NUM_CANONICAL_JOINTS}; got shape {tuple(teacher_actions.shape)}"
            )
        if not bool(torch.isfinite(teacher_actions).all()):
            raise ValueError("teacher_actions contains a non-finite value")

        device = teacher_actions.device
        dtype = teacher_actions.dtype

        def materialize(vector: torch.Tensor) -> torch.Tensor:
            return vector.to(device=device, dtype=dtype)

        teacher_default_q = materialize(self.teacher_default_q)
        teacher_scale = materialize(self.teacher_action_scale)
        student_default_q = materialize(self.student_default_q)
        student_scale = materialize(self.student_action_scale)
        lower = materialize(self.hard_lower_limits)
        upper = materialize(self.hard_upper_limits)

        q_target = teacher_default_q + teacher_scale * teacher_actions
        q_target_clamped = torch.maximum(torch.minimum(q_target, upper), lower)
        student_actions = (q_target_clamped - student_default_q) / student_scale

        if not bool(torch.isfinite(student_actions).all()):
            # Constructor validation makes this unreachable for finite inputs, but retaining
            # the check makes the invariant explicit if a future dtype overflows.
            raise ValueError("transformed student actions contain a non-finite value")

        return ActionTransformResult(
            actions=student_actions,
            q_target=q_target,
            q_target_clamped=q_target_clamped,
            clamped_mask=q_target_clamped.ne(q_target),
        )


def transform_teacher_actions(
    teacher_actions: torch.Tensor,
    *,
    teacher_default_q: Sequence[float] | torch.Tensor,
    teacher_action_scale: float | Sequence[float] | torch.Tensor,
    student_default_q: Sequence[float] | torch.Tensor,
    student_action_scale: float | Sequence[float] | torch.Tensor,
    hard_lower_limits: Sequence[float] | torch.Tensor,
    hard_upper_limits: Sequence[float] | torch.Tensor,
) -> ActionTransformResult:
    """Functional convenience wrapper around :class:`CanonicalActionTransform`."""

    transform = CanonicalActionTransform(
        teacher_default_q=teacher_default_q,
        teacher_action_scale=teacher_action_scale,
        student_default_q=student_default_q,
        student_action_scale=student_action_scale,
        hard_lower_limits=hard_lower_limits,
        hard_upper_limits=hard_upper_limits,
    )
    return transform(teacher_actions)
