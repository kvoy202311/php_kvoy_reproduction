"""Runtime verification of the student's raw-action convention."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Any, Sequence

import torch

from .action_transform import NUM_CANONICAL_JOINTS


def _constant_vector(value: Any, *, width: int, name: str) -> torch.Tensor:
    """Convert a scalar/vector/batched vector and require identical environment rows."""

    if isinstance(value, bool):
        raise TypeError(f"runtime action {name} must be numeric")
    if isinstance(value, (int, float)):
        scalar = float(value)
        if not math.isfinite(scalar):
            raise ValueError(f"runtime action {name} contains NaN or infinity")
        return torch.full((width,), scalar, dtype=torch.float64)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"runtime action {name} must be a number or tensor")
    tensor = value.detach().to(device="cpu", dtype=torch.float64)
    if tensor.ndim == 0:
        return _constant_vector(tensor.item(), width=width, name=name)
    if tensor.shape[-1] != width:
        raise ValueError(f"runtime action {name} must end in {width}, got {tuple(tensor.shape)}")
    rows = tensor.reshape(-1, width)
    if rows.shape[0] == 0 or not bool(torch.isfinite(rows).all()):
        raise ValueError(f"runtime action {name} is empty or contains NaN or infinity")
    reference = rows[0]
    if not torch.equal(rows, reference.expand_as(rows)):
        raise ValueError(f"runtime action {name} differs between environments")
    return reference


def _constant_limits(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"runtime action {name} must be a tensor")
    tensor = value.detach().to(device="cpu", dtype=torch.float64)
    expected_tail = (NUM_CANONICAL_JOINTS, 2)
    if tensor.ndim < 2 or tuple(tensor.shape[-2:]) != expected_tail:
        raise ValueError(f"runtime action {name} must end in {expected_tail}, got {tuple(tensor.shape)}")
    rows = tensor.reshape(-1, *expected_tail)
    if rows.shape[0] == 0 or not bool(torch.isfinite(rows).all()):
        raise ValueError(f"runtime action {name} is empty or contains NaN or infinity")
    reference = rows[0]
    if not torch.equal(rows, reference.expand_as(rows)):
        raise ValueError(f"runtime action {name} differs between environments")
    return reference


def _assert_close(
    actual: torch.Tensor,
    expected: Sequence[float],
    *,
    name: str,
    atol: float,
) -> None:
    expected_tensor = torch.tensor(tuple(expected), dtype=torch.float64)
    if actual.shape != expected_tensor.shape:
        raise ValueError(
            f"runtime action {name} has shape {tuple(actual.shape)}, expected {tuple(expected_tensor.shape)}"
        )
    if not torch.allclose(actual, expected_tensor, rtol=0.0, atol=atol):
        max_error = torch.max(torch.abs(actual - expected_tensor)).item()
        raise ValueError(
            f"runtime action {name} differs from the frozen-teacher contract "
            f"(maximum absolute error {max_error:.3g})"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_action_contract(teacher_router: Any, action_term: Any, *, atol: float = 1.0e-6) -> None:
    """Require the live Isaac action term to match the frozen teachers exactly.

    The current router emits raw policy actions without a teacher-to-student
    transform.  Therefore the live position term must preserve the same joint
    order and affine mapping, including its final target clipping bounds.
    """

    if not math.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative")
    teachers = tuple(getattr(teacher_router, "teachers", ()))
    if not teachers:
        raise ValueError("teacher router contains no teachers")
    manifests = tuple(getattr(teacher, "manifest", None) for teacher in teachers)
    if any(manifest is None for manifest in manifests):
        raise ValueError("every frozen teacher must expose its verified manifest")
    transforms = tuple(getattr(teacher_router, "_action_transforms", ()))
    if len(transforms) != len(teachers) or any(transform is not None for transform in transforms):
        raise ValueError(
            "runtime action validation currently requires teachers to share the student's raw-action contract"
        )
    reference = manifests[0]

    if int(getattr(action_term, "action_dim", -1)) != NUM_CANONICAL_JOINTS:
        raise ValueError(f"runtime action term must control exactly {NUM_CANONICAL_JOINTS} joints")
    cfg = getattr(action_term, "cfg", None)
    if cfg is None or getattr(cfg, "use_default_offset", None) is not True:
        raise ValueError("runtime action term must use the articulation default position as its offset")
    joint_names = tuple(getattr(action_term, "_joint_names", ()))
    if joint_names != tuple(reference.joint_order):
        raise ValueError("runtime action joint order differs from the frozen-teacher contract")

    scale = _constant_vector(
        getattr(action_term, "_scale", None),
        width=NUM_CANONICAL_JOINTS,
        name="scale",
    )
    offset = _constant_vector(
        getattr(action_term, "_offset", None),
        width=NUM_CANONICAL_JOINTS,
        name="offset",
    )
    limits = _constant_limits(getattr(action_term, "_clip", None), name="target limits")
    _assert_close(scale, reference.action_scale, name="scale", atol=atol)
    _assert_close(offset, reference.default_q, name="offset", atol=atol)
    _assert_close(limits[:, 0], reference.hard_lower_limits, name="lower target limits", atol=atol)
    _assert_close(limits[:, 1], reference.hard_upper_limits, name="upper target limits", atol=atol)

    asset = getattr(action_term, "_asset", None)
    if asset is None:
        raise ValueError("runtime action term does not expose its articulation asset")
    asset_joint_names = tuple(getattr(asset, "joint_names", ()))
    if len(asset_joint_names) < NUM_CANONICAL_JOINTS:
        raise ValueError("runtime articulation exposes an incomplete joint order")
    try:
        joint_indices = [asset_joint_names.index(name) for name in reference.joint_order]
    except ValueError as exc:
        raise ValueError("runtime articulation is missing a teacher-contract joint") from exc
    effort = getattr(getattr(asset, "data", None), "joint_effort_limits", None)
    if not isinstance(effort, torch.Tensor):
        raise ValueError("runtime articulation exposes no joint effort limits")
    effort = _constant_vector(
        effort[:, joint_indices],
        width=NUM_CANONICAL_JOINTS,
        name="effort limits",
    )
    _assert_close(effort, reference.effort_limits, name="effort limits", atol=atol)

    spawn = getattr(getattr(asset, "cfg", None), "spawn", None)
    live_usd = getattr(spawn, "usd_path", None)
    if not isinstance(live_usd, str) or not live_usd:
        raise ValueError("runtime articulation does not expose a local USD path")
    live_usd_path = Path(live_usd).expanduser().resolve()
    if not live_usd_path.is_file():
        raise FileNotFoundError(f"runtime articulation USD does not exist: {live_usd_path}")
    live_usd_sha256 = _sha256_file(live_usd_path)
    if live_usd_sha256 != reference.usd_sha256:
        raise ValueError(
            "runtime articulation USD differs from the frozen-teacher asset contract "
            f"(live={live_usd_sha256}, expected={reference.usd_sha256})"
        )


__all__ = ["validate_runtime_action_contract"]
