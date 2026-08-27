"""Frozen, dependency-light teacher actor loading.

This module intentionally depends only on PyTorch and the strict manifest
schema.  In particular, it never imports Isaac Lab, RSL-RL, TensorDict, or the
Python packages that happened to exist when an expert was trained.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from .teacher_manifest import NormalizerSpec, TeacherManifest


_SAFE_WRAPPER_METADATA_KEYS = frozenset({"format_version", "provenance"})
_ACTOR_ARTIFACT_FORMAT_VERSION = "actor_state_v1"


def _activation(name: str) -> nn.Module:
    activations: dict[str, type[nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    try:
        return activations[name]()
    except KeyError as exc:
        raise ValueError(f"unsupported actor activation: {name!r}") from exc


def build_actor_mlp(manifest: TeacherManifest) -> nn.Sequential:
    """Build the exact feed-forward actor topology declared by a manifest."""

    dimensions = (manifest.actor_input_dim, *manifest.hidden_dims, manifest.action_dim)
    modules: list[nn.Module] = []
    for index, (input_dim, output_dim) in enumerate(zip(dimensions[:-1], dimensions[1:], strict=True)):
        modules.append(nn.Linear(input_dim, output_dim))
        if index < len(dimensions) - 2:
            modules.append(_activation(manifest.activation))
    return nn.Sequential(*modules)


def _validate_safe_value(value: Any, *, path: str = "artifact") -> None:
    """Reject non-tensor/non-primitive objects even after a weights-only load."""

    if isinstance(value, torch.Tensor) or value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} contains a non-string mapping key")
            _validate_safe_value(item, path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_safe_value(item, path=f"{path}[{index}]")
        return
    raise ValueError(f"{path} contains unsupported object type {type(value).__name__}")


def _load_weights_only(path: Path, *, map_location: torch.device | str) -> Any:
    try:
        artifact = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:  # pragma: no cover - defensive guard for unsupported PyTorch versions
        raise RuntimeError("teacher artifacts require a PyTorch version with weights_only=True") from exc
    _validate_safe_value(artifact)
    return artifact


def _extract_tensor_mapping(
    artifact: Any,
    *,
    state_key: str | None,
    other_allowed_state_keys: set[str] | None = None,
    context: str,
) -> Mapping[str, torch.Tensor]:
    if not isinstance(artifact, Mapping):
        raise ValueError(f"{context} artifact root must be a mapping")

    if "format_version" in artifact and artifact["format_version"] != _ACTOR_ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"{context} artifact format_version must be {_ACTOR_ARTIFACT_FORMAT_VERSION!r}"
        )

    if state_key is None:
        state = artifact
    else:
        allowed = set(_SAFE_WRAPPER_METADATA_KEYS)
        allowed.add(state_key)
        allowed.update(other_allowed_state_keys or set())
        unknown = set(artifact) - allowed
        if unknown:
            raise ValueError(f"{context} artifact contains unknown wrapper keys: {sorted(unknown)}")
        if state_key not in artifact:
            raise ValueError(f"{context} artifact is missing wrapper key {state_key!r}")
        state = artifact[state_key]

    if not isinstance(state, Mapping):
        raise ValueError(f"{context} state must be a tensor mapping")
    result: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} state contains a non-string key")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{context} state value {key!r} is not a tensor")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{context} state tensor {key!r} contains a non-finite value")
        result[key] = tensor
    return result


def _strip_explicit_prefix(
    state: Mapping[str, torch.Tensor], *, prefix: str, context: str
) -> dict[str, torch.Tensor]:
    if not prefix:
        return dict(state)
    wrong_prefix = [key for key in state if not key.startswith(prefix)]
    if wrong_prefix:
        raise ValueError(
            f"{context} state contains keys outside the declared prefix {prefix!r}: {sorted(wrong_prefix)}"
        )
    stripped = {key[len(prefix) :]: tensor for key, tensor in state.items()}
    if any(not key for key in stripped):
        raise ValueError(f"{context} state prefix {prefix!r} produces an empty key")
    if len(stripped) != len(state):
        raise ValueError(f"{context} state prefix {prefix!r} produces duplicate keys")
    return stripped


def _validate_state_dict_exact(module: nn.Module, state: Mapping[str, torch.Tensor], *, context: str) -> None:
    expected = module.state_dict()
    expected_keys = set(expected)
    actual_keys = set(state)
    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={sorted(missing)}")
        if extra:
            details.append(f"extra={sorted(extra)}")
        raise ValueError(f"{context} state keys do not match exactly ({', '.join(details)})")
    for key, expected_tensor in expected.items():
        actual_tensor = state[key]
        if actual_tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"{context} tensor {key!r} has shape {tuple(actual_tensor.shape)}, "
                f"expected {tuple(expected_tensor.shape)}"
            )


class FrozenEmpiricalNormalizer(nn.Module):
    """Immutable ``(x - mean) / (std + epsilon)`` normalization."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, *, epsilon: float) -> None:
        super().__init__()
        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise TypeError("normalizer mean and std must be tensors")
        if mean.ndim != 1 or std.ndim != 1 or mean.shape != std.shape:
            raise ValueError("normalizer mean and std must be equal-length one-dimensional tensors")
        if not mean.is_floating_point() or not std.is_floating_point():
            raise TypeError("normalizer mean and std must have floating-point dtypes")
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(std).all()):
            raise ValueError("normalizer mean/std contains a non-finite value")
        if bool((std < 0).any()):
            raise ValueError("normalizer std must not be negative")
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("normalizer epsilon must be finite and positive")
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clone())
        self.epsilon = float(epsilon)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenEmpiricalNormalizer":  # noqa: ARG002
        super().train(False)
        return self

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations - self.mean) / (self.std + self.epsilon)


class IdentityNormalizer(nn.Module):
    """Stateless identity module that remains in evaluation mode."""

    def __init__(self) -> None:
        super().__init__()
        super().train(False)

    def train(self, mode: bool = True) -> "IdentityNormalizer":  # noqa: ARG002
        super().train(False)
        return self

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


def _normalizer_tensors(
    state: Mapping[str, Any], *, spec: NormalizerSpec
) -> tuple[torch.Tensor, torch.Tensor]:
    """Parse one of the explicitly supported converted-normalizer formats."""

    keys = set(state)
    if keys in ({"mean", "std"}, {"mean", "std", "count"}):
        mean = state["mean"]
        std = state["std"]
        if "count" in state:
            count = state["count"]
            if isinstance(count, torch.Tensor):
                if count.numel() != 1 or not bool(torch.isfinite(count).all()) or float(count.item()) <= 0:
                    raise ValueError("normalizer count must be a finite positive scalar")
            elif isinstance(count, (int, float)) and not isinstance(count, bool):
                if not math.isfinite(float(count)) or float(count) <= 0:
                    raise ValueError("normalizer count must be a finite positive scalar")
            else:
                raise ValueError("normalizer count must be a finite positive scalar")
    elif keys == {"kind", "mean", "std", "epsilon"}:
        if state["kind"] != spec.kind:
            raise ValueError("normalizer artifact kind does not match manifest")
        if not isinstance(state["epsilon"], (int, float)) or isinstance(state["epsilon"], bool):
            raise ValueError("normalizer artifact epsilon must be numeric")
        if not math.isclose(float(state["epsilon"]), float(spec.epsilon), rel_tol=0.0, abs_tol=0.0):
            raise ValueError("normalizer artifact epsilon does not match manifest")
        mean = state["mean"]
        std = state["std"]
    elif keys == {"_mean", "_std", "count"}:
        # This exact form is the RSL-RL 2.3.3 empirical-normalizer state.  It is
        # accepted only as an explicit, bounded compatibility format.
        mean = state["_mean"]
        std = state["_std"]
        count = state["count"]
        if not isinstance(count, torch.Tensor) or count.numel() != 1 or float(count.item()) <= 0:
            raise ValueError("normalizer count must be a positive scalar tensor")
        if isinstance(mean, torch.Tensor) and mean.ndim == 2 and mean.shape[0] == 1:
            mean = mean.squeeze(0)
        if isinstance(std, torch.Tensor) and std.ndim == 2 and std.shape[0] == 1:
            std = std.squeeze(0)
    else:
        raise ValueError(f"unsupported normalizer state keys: {sorted(keys)}")

    if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
        raise ValueError("normalizer mean/std must be tensors")
    if spec.dimension is None:
        raise ValueError("empirical normalizer manifest is missing its dimension")
    if mean.shape != (spec.dimension,) or std.shape != (spec.dimension,):
        raise ValueError(
            f"normalizer mean/std must have shape ({spec.dimension},); got {tuple(mean.shape)} and {tuple(std.shape)}"
        )
    return mean, std


class TeacherPolicy(nn.Module):
    """Deterministic and permanently frozen MLP teacher actor."""

    def __init__(
        self,
        manifest: TeacherManifest,
        actor: nn.Sequential,
        normalizer: nn.Module,
    ) -> None:
        super().__init__()
        if not isinstance(manifest, TeacherManifest):
            raise TypeError("manifest must be a TeacherManifest")
        self.manifest = manifest
        self.actor = actor
        self.normalizer = normalizer
        self.requires_grad_(False)
        super().train(False)

    @classmethod
    def from_manifest(
        cls,
        manifest: TeacherManifest,
        *,
        base_directory: str | Path | None = None,
        device: torch.device | str = "cpu",
    ) -> "TeacherPolicy":
        """Hash verify and strictly load a converted teacher artifact."""

        checkpoint_path = manifest.verify_checkpoint(base_directory=base_directory)
        target_device = torch.device(device)
        artifact_cache: dict[Path, Any] = {}

        def load(path: Path) -> Any:
            resolved = path.resolve(strict=True)
            if resolved not in artifact_cache:
                artifact_cache[resolved] = _load_weights_only(resolved, map_location=target_device)
            return artifact_cache[resolved]

        checkpoint_artifact = load(checkpoint_path)
        other_state_keys: set[str] = set()
        if (
            manifest.normalizer.kind != "none"
            and manifest.normalizer.artifact_path == manifest.checkpoint_path
            and manifest.normalizer.state_key is not None
        ):
            other_state_keys.add(manifest.normalizer.state_key)
        actor_state = _extract_tensor_mapping(
            checkpoint_artifact,
            state_key=manifest.state_dict_key,
            other_allowed_state_keys=other_state_keys,
            context="actor",
        )
        actor_state = _strip_explicit_prefix(actor_state, prefix=manifest.state_prefix, context="actor")

        actor = build_actor_mlp(manifest).to(device=target_device)
        _validate_state_dict_exact(actor, actor_state, context="actor")
        actor.load_state_dict(actor_state, strict=True)

        if manifest.normalizer.kind == "none":
            normalizer: nn.Module = IdentityNormalizer()
        else:
            normalizer_path = manifest.verify_normalizer(base_directory=base_directory)
            if normalizer_path is None or manifest.normalizer.state_key is None:
                raise RuntimeError("validated non-identity normalizer is missing its load contract")
            normalizer_artifact = load(normalizer_path)
            normalizer_state = _extract_normalizer_mapping(
                normalizer_artifact,
                state_key=manifest.normalizer.state_key,
                actor_state_key=(
                    manifest.state_dict_key if normalizer_path == checkpoint_path else None
                ),
            )
            mean, std = _normalizer_tensors(normalizer_state, spec=manifest.normalizer)
            if manifest.normalizer.epsilon is None:
                raise RuntimeError("validated empirical normalizer is missing epsilon")
            normalizer = FrozenEmpiricalNormalizer(
                mean.to(device=target_device),
                std.to(device=target_device),
                epsilon=manifest.normalizer.epsilon,
            )

        policy = cls(manifest=manifest, actor=actor, normalizer=normalizer)
        policy.requires_grad_(False)
        policy.eval()
        return policy

    @property
    def observation_dim(self) -> int:
        return self.manifest.actor_input_dim

    @property
    def action_dim(self) -> int:
        return self.manifest.action_dim

    @property
    def skill_name(self) -> str:
        return self.manifest.skill_name

    def fingerprint(self) -> str:
        return self.manifest.fingerprint()

    def train(self, mode: bool = True) -> "TeacherPolicy":  # noqa: ARG002
        # A parent student's ``train()`` call must never reactivate teacher-side
        # dropout/statistics if the actor architecture grows in the future.
        super().train(False)
        self.actor.train(False)
        self.normalizer.train(False)
        return self

    @torch.no_grad()
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if not isinstance(observations, torch.Tensor):
            raise TypeError("teacher observations must be a torch.Tensor")
        if not observations.is_floating_point():
            raise TypeError("teacher observations must have a floating-point dtype")
        if observations.ndim < 1 or observations.shape[-1] != self.observation_dim:
            raise ValueError(
                f"teacher observations must end in {self.observation_dim}; got shape {tuple(observations.shape)}"
            )
        if not bool(torch.isfinite(observations).all()):
            raise ValueError("teacher observations contain a non-finite value")

        parameter = next(self.actor.parameters())
        if observations.device != parameter.device:
            raise ValueError(
                f"teacher observations are on {observations.device}, but policy is on {parameter.device}"
            )
        if observations.dtype != parameter.dtype:
            raise ValueError(
                f"teacher observations use {observations.dtype}, but policy uses {parameter.dtype}"
            )

        normalized = self.normalizer(observations)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError("normalized teacher observations contain a non-finite value")
        actions = self.actor(normalized)
        expected_shape = (*observations.shape[:-1], self.action_dim)
        if actions.shape != expected_shape:
            raise RuntimeError(f"teacher actor returned shape {tuple(actions.shape)}, expected {expected_shape}")
        if not bool(torch.isfinite(actions).all()):
            raise ValueError("teacher actor returned a non-finite action")
        return actions

    act = forward


def _extract_normalizer_mapping(
    artifact: Any,
    *,
    state_key: str,
    actor_state_key: str | None,
) -> Mapping[str, Any]:
    if not isinstance(artifact, Mapping):
        raise ValueError("normalizer artifact root must be a mapping")
    allowed = set(_SAFE_WRAPPER_METADATA_KEYS)
    allowed.add(state_key)
    if actor_state_key is not None:
        allowed.add(actor_state_key)
    unknown = set(artifact) - allowed
    if unknown:
        raise ValueError(f"normalizer artifact contains unknown wrapper keys: {sorted(unknown)}")
    if state_key not in artifact:
        raise ValueError(f"normalizer artifact is missing wrapper key {state_key!r}")
    state = artifact[state_key]
    if not isinstance(state, Mapping):
        raise ValueError("normalizer state must be a mapping")
    return state
