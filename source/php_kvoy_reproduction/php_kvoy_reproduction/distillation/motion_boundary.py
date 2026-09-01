"""Deployment-observable execution boundaries for motion-teacher clips."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

_POSE_FIELDS = ("joint_pos", "body_pos_w")
_VELOCITY_FIELDS = ("joint_vel", "body_lin_vel_w", "body_ang_vel_w")
_REQUIRED_FIELDS = _POSE_FIELDS + ("body_quat_w",) + _VELOCITY_FIELDS


def _validate_motion_arrays(
    arrays: Mapping[str, torch.Tensor],
    motion_start_idx: torch.Tensor,
    motion_end_idx: torch.Tensor,
) -> int:
    missing = [name for name in _REQUIRED_FIELDS if name not in arrays]
    if missing:
        raise KeyError(f"motion arrays are missing required fields: {missing}")
    if motion_start_idx.ndim != 1 or motion_end_idx.shape != motion_start_idx.shape:
        raise ValueError("motion_start_idx and motion_end_idx must be one-dimensional tensors of equal shape")
    if motion_start_idx.dtype != torch.long or motion_end_idx.dtype != torch.long:
        raise TypeError("motion_start_idx and motion_end_idx must use torch.long indices")
    if motion_start_idx.numel() == 0:
        raise ValueError("motion boundaries must contain at least one clip")
    if motion_start_idx.device != motion_end_idx.device:
        raise ValueError("motion boundary tensors must share one device")

    frame_count: int | None = None
    reference_device = motion_start_idx.device
    for name in _REQUIRED_FIELDS:
        value = arrays[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"motion field {name!r} must be a torch.Tensor")
        if not value.is_floating_point() or value.ndim < 2:
            raise TypeError(f"motion field {name!r} must be a floating-point tensor with a frame dimension")
        if value.device != reference_device:
            raise ValueError(f"motion field {name!r} and boundary tensors must share one device")
        if not torch.isfinite(value).all():
            raise ValueError(f"motion field {name!r} contains NaN or infinity")
        if frame_count is None:
            frame_count = int(value.shape[0])
        elif value.shape[0] != frame_count:
            raise ValueError("all motion fields must contain the same number of frames")

    assert frame_count is not None
    if torch.any(motion_start_idx < 0) or torch.any(motion_end_idx > frame_count):
        raise IndexError("motion boundaries fall outside the concatenated frame arrays")
    if torch.any(motion_end_idx - motion_start_idx < 2):
        raise ValueError("each motion clip must contain at least two frames")
    if motion_start_idx.numel() > 1 and not torch.equal(motion_start_idx[1:], motion_end_idx[:-1]):
        raise ValueError("motion clips must form one contiguous concatenated dataset")
    return frame_count


def detect_motion_execution_starts(
    arrays: Mapping[str, torch.Tensor],
    motion_start_idx: torch.Tensor,
    motion_end_idx: torch.Tensor,
    *,
    pose_tolerance: float = 1.0e-6,
    velocity_tolerance: float = 1.0e-6,
) -> torch.Tensor:
    """Return the first physically active reference frame of every clip.

    Motion teachers observe a time-indexed reference, while the deployment
    Student observes only vision and physical history.  A converter-appended
    static prefix therefore creates an unobservable timer: identical Student
    observations receive both hold and motion-start labels.  This function
    removes only that exact prefix.  It does not trim slow, non-zero motion.

    Pose activity is measured relative to the raw first frame.  Velocity
    activity is measured against zero.  The first frame exceeding either
    tolerance becomes the execution start.  A clip that is already active at
    its raw first frame keeps that frame as its execution start.
    """

    _validate_motion_arrays(arrays, motion_start_idx, motion_end_idx)
    for name, value in (("pose_tolerance", pose_tolerance), ("velocity_tolerance", velocity_tolerance)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value)
        ):
            raise ValueError(f"{name} must be a positive finite number")

    execution_starts = torch.empty_like(motion_start_idx)
    for motion_id, (start_tensor, end_tensor) in enumerate(zip(motion_start_idx, motion_end_idx, strict=True)):
        start = int(start_tensor.item())
        end = int(end_tensor.item())
        active = torch.zeros(end - start, device=motion_start_idx.device, dtype=torch.bool)
        for name in _POSE_FIELDS:
            values = arrays[name][start:end].reshape(end - start, -1)
            initial = values[0]
            active |= torch.amax(torch.abs(values - initial), dim=1) > float(pose_tolerance)
        quaternions = arrays["body_quat_w"][start:end]
        initial_quaternions = quaternions[0]
        alignment = torch.abs(torch.sum(quaternions * initial_quaternions, dim=-1))
        normalization = torch.linalg.vector_norm(quaternions, dim=-1) * torch.linalg.vector_norm(
            initial_quaternions, dim=-1
        )
        if torch.any(normalization <= 0.0):
            raise ValueError(f"motion clip {motion_id} contains a zero-norm body quaternion")
        orientation_error = 1.0 - (alignment / normalization).clamp(max=1.0)
        active |= torch.amax(orientation_error, dim=1) > float(pose_tolerance)
        for name in _VELOCITY_FIELDS:
            values = arrays[name][start:end].reshape(end - start, -1)
            active |= torch.amax(torch.abs(values), dim=1) > float(velocity_tolerance)

        candidates = torch.where(active)[0]
        if candidates.numel() == 0:
            raise ValueError(f"motion clip {motion_id} contains no physical activity")
        execution_starts[motion_id] = start + candidates[0]

    return execution_starts


def motion_reference_advance_mask(
    motion_mask: torch.Tensor,
    motion_finished: torch.Tensor,
    reference_skill_ids: torch.Tensor,
    active_student_skill_ids: torch.Tensor,
    reset_this_step: torch.Tensor,
) -> torch.Tensor:
    """Return motions whose reference may advance after the applied action.

    The reference clock belongs to the action head that actually controlled
    the preceding physics interval.  It must therefore pause while the Option
    controller is confirming a newly requested motion, and it must not consume
    one frame in Isaac Lab's post-termination reset/update pass.
    """

    if not isinstance(motion_mask, torch.Tensor):
        raise TypeError("motion_mask must be a torch.Tensor")
    tensors = {
        "motion_mask": motion_mask,
        "motion_finished": motion_finished,
        "reference_skill_ids": reference_skill_ids,
        "active_student_skill_ids": active_student_skill_ids,
        "reset_this_step": reset_this_step,
    }
    if motion_mask.ndim != 1:
        raise ValueError("motion advance inputs must be one-dimensional")
    shape = motion_mask.shape
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != shape:
            raise ValueError("motion advance inputs must share one-dimensional shape")
        if value.device != motion_mask.device:
            raise ValueError("motion advance inputs must share one device")
    for name in ("motion_mask", "motion_finished", "reset_this_step"):
        if tensors[name].dtype != torch.bool:
            raise TypeError(f"{name} must use torch.bool")
    for name in ("reference_skill_ids", "active_student_skill_ids"):
        value = tensors[name]
        if value.dtype == torch.bool or value.is_floating_point() or value.is_complex():
            raise TypeError(f"{name} must use an integer dtype")

    return (
        motion_mask
        & ~motion_finished
        & ~reset_this_step
        & (reference_skill_ids == active_student_skill_ids)
    )


__all__ = ["detect_motion_execution_starts", "motion_reference_advance_mask"]
