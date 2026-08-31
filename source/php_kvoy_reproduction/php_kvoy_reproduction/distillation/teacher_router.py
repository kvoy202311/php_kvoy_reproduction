"""Stable subset routing for multiple frozen teacher actors."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from .action_transform import ActionTransformResult, CanonicalActionTransform, NUM_CANONICAL_JOINTS


@dataclass(frozen=True)
class TeacherBatch:
    """Teacher labels scattered back into the original environment order."""

    actions: torch.Tensor
    valid_mask: torch.Tensor
    skill_ids: torch.Tensor

    def __post_init__(self) -> None:
        if self.actions.ndim != 2 or self.actions.shape[1] != NUM_CANONICAL_JOINTS:
            raise ValueError(f"actions must have shape [N, {NUM_CANONICAL_JOINTS}]")
        if not self.actions.is_floating_point() or not torch.isfinite(self.actions).all():
            raise ValueError("actions must be a finite floating-point tensor")
        expected_column = (self.actions.shape[0], 1)
        if self.valid_mask.shape != expected_column or not self.valid_mask.is_floating_point():
            raise ValueError("valid_mask must be a floating-point confidence tensor with shape [N, 1]")
        if not torch.isfinite(self.valid_mask).all() or torch.any(
            (self.valid_mask < 0.0) | (self.valid_mask > 1.0)
        ):
            raise ValueError("valid_mask confidence must be finite and lie in [0, 1]")
        if self.skill_ids.shape != expected_column or self.skill_ids.dtype != torch.int64:
            raise ValueError("skill_ids must be an int64 tensor with shape [N, 1]")
        if not (self.actions.device == self.valid_mask.device == self.skill_ids.device):
            raise ValueError("TeacherBatch tensors must share a device")


def _normalize_skill_ids(skill_ids: torch.Tensor) -> torch.Tensor:
    if not isinstance(skill_ids, torch.Tensor):
        raise TypeError("skill_ids must be a torch.Tensor")
    if skill_ids.ndim == 2 and skill_ids.shape[1] == 1:
        skill_ids = skill_ids[:, 0]
    elif skill_ids.ndim != 1:
        raise ValueError(f"skill_ids must have shape [N] or [N, 1]; got {tuple(skill_ids.shape)}")
    if skill_ids.numel() == 0:
        raise ValueError("skill_ids must not be empty")
    if skill_ids.dtype == torch.bool:
        raise TypeError("skill_ids must contain integer route labels, not booleans")
    if skill_ids.is_floating_point():
        if not bool(torch.isfinite(skill_ids).all()):
            raise ValueError("skill_ids contains a non-finite value")
        rounded = torch.round(skill_ids)
        if not bool(skill_ids.eq(rounded).all()):
            raise ValueError("skill_ids contains a non-integer floating-point value")
        return rounded.to(dtype=torch.int64)
    if skill_ids.is_complex():
        raise TypeError("skill_ids must contain real integer route labels")
    return skill_ids.to(dtype=torch.int64)


def _normalize_validity(mask: torch.Tensor, *, batch_size: int, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if mask.ndim == 2 and mask.shape[1] == 1:
        mask = mask[:, 0]
    elif mask.ndim != 1:
        raise ValueError(f"{name} must have shape [N] or [N, 1]")
    if mask.shape[0] != batch_size:
        raise ValueError(f"{name} has batch size {mask.shape[0]}, expected {batch_size}")
    if mask.dtype == torch.bool:
        mask = mask.to(dtype=torch.float32)
    elif not mask.is_floating_point():
        raise TypeError(f"{name} must have bool or floating-point dtype")
    if not torch.isfinite(mask).all() or torch.any((mask < 0.0) | (mask > 1.0)):
        raise ValueError(f"{name} confidence must be finite and lie in [0, 1]")
    return mask


class TeacherRouter(nn.Module):
    """Run each teacher on only its routed environments and scatter labels back.

    ``skill_to_id`` is explicit and must use contiguous IDs beginning at zero;
    dictionary insertion order is never used as route semantics.  If teacher
    action conventions differ, one transform per skill is mandatory.
    """

    def __init__(
        self,
        teachers: Mapping[str, nn.Module],
        skill_to_id: Mapping[str, int],
        *,
        action_transforms: Mapping[str, CanonicalActionTransform] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(teachers, Mapping) or not teachers:
            raise ValueError("teachers must be a non-empty mapping")
        if not isinstance(skill_to_id, Mapping):
            raise TypeError("skill_to_id must be a mapping")
        if set(teachers) != set(skill_to_id):
            raise ValueError("teachers and skill_to_id must contain exactly the same skill names")
        if any(not isinstance(name, str) or not name for name in teachers):
            raise ValueError("teacher skill names must be non-empty strings")

        ids: list[int] = []
        for name, route_id in skill_to_id.items():
            if isinstance(route_id, bool) or not isinstance(route_id, int):
                raise ValueError(f"route ID for {name!r} must be an integer")
            ids.append(route_id)
        if len(set(ids)) != len(ids):
            raise ValueError("skill_to_id contains duplicate route IDs")
        if sorted(ids) != list(range(len(ids))):
            raise ValueError("route IDs must be contiguous and begin at zero")

        ordered_names = tuple(name for name, _ in sorted(skill_to_id.items(), key=lambda item: item[1]))
        ordered_teachers: list[nn.Module] = []
        for name in ordered_names:
            teacher = teachers[name]
            if not isinstance(teacher, nn.Module):
                raise TypeError(f"teacher {name!r} must be a torch.nn.Module")
            action_dim = getattr(teacher, "action_dim", NUM_CANONICAL_JOINTS)
            if action_dim != NUM_CANONICAL_JOINTS:
                raise ValueError(f"teacher {name!r} action_dim must be {NUM_CANONICAL_JOINTS}")
            manifest = getattr(teacher, "manifest", None)
            if manifest is not None and getattr(manifest, "skill_name", name) != name:
                raise ValueError(f"teacher manifest skill {manifest.skill_name!r} does not match registration {name!r}")
            teacher.requires_grad_(False)
            teacher.eval()
            ordered_teachers.append(teacher)

        if action_transforms is not None:
            if set(action_transforms) != set(teachers):
                raise ValueError("action_transforms must contain exactly one transform for every teacher")
            for name, transform in action_transforms.items():
                if not isinstance(transform, CanonicalActionTransform):
                    raise TypeError(f"action transform for {name!r} must be a CanonicalActionTransform")
            ordered_transforms: tuple[CanonicalActionTransform | None, ...] = tuple(
                action_transforms[name] for name in ordered_names
            )
        else:
            _validate_common_raw_action_contract([teachers[name] for name in ordered_names])
            ordered_transforms = (None,) * len(ordered_names)

        self._skill_names = ordered_names
        self._skill_to_id = {name: int(skill_to_id[name]) for name in ordered_names}
        self.teachers = nn.ModuleList(ordered_teachers)
        self._action_transforms = ordered_transforms
        self.requires_grad_(False)
        super().train(False)

    @property
    def skill_to_id(self) -> dict[str, int]:
        return dict(self._skill_to_id)

    @property
    def id_to_skill(self) -> tuple[str, ...]:
        return self._skill_names

    @property
    def skill_names(self) -> tuple[str, ...]:
        """Return skill names in their canonical contiguous route-ID order."""

        return self._skill_names

    def train(self, mode: bool = True) -> "TeacherRouter":  # noqa: ARG002
        super().train(False)
        for teacher in self.teachers:
            teacher.train(False)
        return self

    def fingerprints(self) -> dict[str, str]:
        """Return stable per-skill teacher fingerprints for run provenance."""

        result: dict[str, str] = {}
        for name, teacher in zip(self._skill_names, self.teachers, strict=True):
            fingerprint_method = getattr(teacher, "fingerprint", None)
            if callable(fingerprint_method):
                fingerprint = fingerprint_method()
                if not isinstance(fingerprint, str) or not fingerprint:
                    raise ValueError(f"teacher {name!r} returned an invalid fingerprint")
                result[name] = fingerprint
            else:
                result[name] = _module_fingerprint(teacher)
        return result

    @torch.no_grad()
    def act(
        self,
        teacher_observations: Mapping[str, torch.Tensor],
        skill_ids: torch.Tensor,
        validity_mask: torch.Tensor | Mapping[str, torch.Tensor] | None = None,
        *,
        validity_by_skill: Mapping[str, torch.Tensor] | None = None,
    ) -> TeacherBatch:
        if not isinstance(teacher_observations, Mapping):
            raise TypeError("teacher_observations must be a mapping")
        route_ids = _normalize_skill_ids(skill_ids)
        batch_size = route_ids.shape[0]
        used_ids = set(int(value) for value in torch.unique(route_ids).cpu().tolist())
        valid_ids = set(range(len(self._skill_names)))
        invalid_ids = used_ids - valid_ids
        if invalid_ids:
            raise ValueError(f"skill_ids contains unregistered route IDs: {sorted(invalid_ids)}")

        if validity_by_skill is not None:
            if validity_mask is not None:
                raise ValueError("pass only one of validity_mask and validity_by_skill")
            validity_mask = validity_by_skill
        global_validity: torch.Tensor | None = None
        per_skill_validity: Mapping[str, torch.Tensor] | None = None
        if isinstance(validity_mask, Mapping):
            per_skill_validity = validity_mask
        elif validity_mask is not None:
            global_validity = _normalize_validity(validity_mask, batch_size=batch_size, name="validity_mask")

        pieces: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        output_device: torch.device | None = None
        output_dtype: torch.dtype | None = None
        for route_id, (name, teacher, transform) in enumerate(
            zip(self._skill_names, self.teachers, self._action_transforms, strict=True)
        ):
            route_indices = torch.nonzero(route_ids == route_id, as_tuple=False).squeeze(-1)
            if route_indices.numel() == 0:
                continue
            if name not in teacher_observations:
                raise ValueError(f"teacher_observations is missing routed skill {name!r}")
            observations = teacher_observations[name]
            if not isinstance(observations, torch.Tensor):
                raise TypeError(f"teacher observations for {name!r} must be a torch.Tensor")
            if not observations.is_floating_point():
                raise TypeError(f"teacher observations for {name!r} must have a floating-point dtype")
            if observations.ndim != 2 or observations.shape[0] != batch_size:
                raise ValueError(
                    f"teacher observations for {name!r} must have shape [N, D] with N={batch_size}; "
                    f"got {tuple(observations.shape)}"
                )
            expected_dim = getattr(teacher, "observation_dim", None)
            if expected_dim is not None and observations.shape[1] != expected_dim:
                raise ValueError(
                    f"teacher observations for {name!r} have dimension {observations.shape[1]}, "
                    f"expected {expected_dim}"
                )
            observation_indices = route_indices.to(device=observations.device)
            if global_validity is not None:
                subset_validity = global_validity.index_select(
                    0, route_indices.to(device=global_validity.device)
                ).to(device=observations.device, dtype=observations.dtype)
            elif per_skill_validity is not None:
                if name not in per_skill_validity:
                    raise ValueError(f"per-skill validity is missing routed skill {name!r}")
                skill_validity = _normalize_validity(
                    per_skill_validity[name], batch_size=batch_size, name=f"validity for {name!r}"
                )
                subset_validity = skill_validity.index_select(
                    0, route_indices.to(device=skill_validity.device)
                ).to(device=observations.device, dtype=observations.dtype)
            else:
                subset_validity = torch.ones(
                    route_indices.numel(), dtype=observations.dtype, device=observations.device
                )
            positive = subset_validity > 0.0
            active_observation_indices = observation_indices[positive]
            active_observations = observations.index_select(0, active_observation_indices)
            if active_observations.numel() > 0 and not bool(
                torch.isfinite(active_observations).all()
            ):
                raise ValueError(
                    f"positive-validity teacher observations for {name!r} contain a non-finite value"
                )

            # Zero-confidence rows never enter a frozen teacher.  This is
            # essential for randomized geometry where a teacher observation
            # can be outside its verified numerical scope.
            if torch.any(positive):
                raw_actions = teacher(active_observations)
                if not isinstance(raw_actions, torch.Tensor):
                    raise TypeError(f"teacher {name!r} must return a torch.Tensor")
                expected_shape = (int(positive.sum().item()), NUM_CANONICAL_JOINTS)
                if raw_actions.shape != expected_shape:
                    raise ValueError(
                        f"teacher {name!r} returned shape {tuple(raw_actions.shape)}, "
                        f"expected {expected_shape}"
                    )
                if not raw_actions.is_floating_point() or not bool(
                    torch.isfinite(raw_actions).all()
                ):
                    raise ValueError(f"teacher {name!r} returned invalid actions")
                if transform is None:
                    active_actions = raw_actions
                else:
                    transformed = transform(raw_actions)
                    if not isinstance(transformed, ActionTransformResult):
                        raise TypeError(
                            f"action transform for {name!r} returned an invalid result"
                        )
                    active_actions = transformed.actions
                canonical_actions = torch.zeros(
                    route_indices.numel(),
                    NUM_CANONICAL_JOINTS,
                    device=active_actions.device,
                    dtype=active_actions.dtype,
                )
                canonical_actions[positive.to(device=active_actions.device)] = active_actions
                subset_validity = subset_validity.to(
                    device=active_actions.device, dtype=active_actions.dtype
                )
            else:
                canonical_actions = torch.zeros(
                    route_indices.numel(),
                    NUM_CANONICAL_JOINTS,
                    device=observations.device,
                    dtype=observations.dtype,
                )

            if output_device is None:
                output_device = canonical_actions.device
                output_dtype = canonical_actions.dtype
            elif canonical_actions.device != output_device or canonical_actions.dtype != output_dtype:
                raise ValueError("all teacher actions must share a device and dtype")
            pieces.append((route_indices, canonical_actions, subset_validity))

        if output_device is None or output_dtype is None:
            raise RuntimeError("no teacher produced actions")
        actions = torch.empty(
            (batch_size, NUM_CANONICAL_JOINTS), device=output_device, dtype=output_dtype
        )
        valid = torch.empty((batch_size,), device=output_device, dtype=output_dtype)
        assigned = torch.zeros((batch_size,), device=output_device, dtype=torch.int64)
        for route_indices, subset_actions, subset_validity in pieces:
            scatter_indices = route_indices.to(device=output_device)
            actions.index_copy_(0, scatter_indices, subset_actions)
            valid.index_copy_(0, scatter_indices, subset_validity)
            assigned.index_add_(0, scatter_indices, torch.ones_like(scatter_indices, dtype=torch.int64))
        if not bool(assigned.eq(1).all()):
            raise RuntimeError("each environment must receive exactly one teacher label")

        routed_ids = route_ids.to(device=output_device).unsqueeze(-1)
        return TeacherBatch(actions=actions, valid_mask=valid.unsqueeze(-1), skill_ids=routed_ids)

    forward = act


def _validate_common_raw_action_contract(teachers: list[nn.Module]) -> None:
    manifests = [getattr(teacher, "manifest", None) for teacher in teachers]
    if any(manifest is None for manifest in manifests):
        return
    reference = manifests[0]
    fields = (
        "joint_order",
        "default_q",
        "action_scale",
        "hard_lower_limits",
        "hard_upper_limits",
        "effort_limits",
        "usd_sha256",
    )
    for manifest in manifests[1:]:
        mismatched = [field for field in fields if getattr(manifest, field) != getattr(reference, field)]
        if mismatched:
            raise ValueError(
                "teacher raw action conventions differ; provide explicit action_transforms "
                f"(mismatched fields: {mismatched})"
            )


def _module_fingerprint(module: nn.Module) -> str:
    digest = hashlib.sha256()
    qualified_name = f"{type(module).__module__}.{type(module).__qualname__}"
    digest.update(qualified_name.encode("utf-8"))
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        raw_bytes = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw_bytes)
    return digest.hexdigest()
