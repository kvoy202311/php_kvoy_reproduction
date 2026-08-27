"""One-time conversion of expert checkpoints into safe actor-only artifacts.

Legacy training checkpoints are provenance inputs, not runtime dependencies.
The distillation runtime consumes the small ``actor_state_v1`` format emitted
here and can therefore use ``torch.load(weights_only=True)`` exclusively.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch

from .teacher_manifest import NUM_CANONICAL_JOINTS, sha256_file, verify_file_sha256
from .teacher_policy import _load_weights_only


ARTIFACT_FORMAT_VERSION = "actor_state_v1"
_TRACKING_CHECKPOINT_KEYS = frozenset(
    {
        "model_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
        "obs_norm_state_dict",
        "privileged_obs_norm_state_dict",
    }
)


@dataclass(frozen=True)
class ArtifactBuildResult:
    path: Path
    sha256: str
    source_sha256: str


def _positive_dimensions(
    actor_input_dim: int, hidden_dims: Sequence[int], action_dim: int
) -> tuple[int, ...]:
    values = (actor_input_dim, *tuple(hidden_dims), action_dim)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise ValueError("actor dimensions must be positive integers")
    if action_dim != NUM_CANONICAL_JOINTS:
        raise ValueError(f"action_dim must be exactly {NUM_CANONICAL_JOINTS}")
    if not hidden_dims:
        raise ValueError("hidden_dims must not be empty")
    return values


def _expected_actor_shapes(dimensions: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {}
    for layer_index, (input_dim, output_dim) in enumerate(
        zip(dimensions[:-1], dimensions[1:], strict=True)
    ):
        sequential_index = 2 * layer_index
        expected[f"{sequential_index}.weight"] = (output_dim, input_dim)
        expected[f"{sequential_index}.bias"] = (output_dim,)
    return expected


def _canonical_actor_state(
    source_state: Mapping[str, Any],
    *,
    state_prefix: str,
    dimensions: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    if not isinstance(source_state, Mapping):
        raise ValueError("source actor state must be a mapping")
    if not isinstance(state_prefix, str):
        raise TypeError("state_prefix must be a string")
    selected: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if not isinstance(key, str):
            raise ValueError("source actor state contains a non-string key")
        if state_prefix and not key.startswith(state_prefix):
            continue
        canonical_key = key[len(state_prefix) :] if state_prefix else key
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"source actor value {key!r} is not a tensor")
        if canonical_key in selected:
            raise ValueError(f"source prefix produces duplicate actor key {canonical_key!r}")
        selected[canonical_key] = value

    expected = _expected_actor_shapes(dimensions)
    missing = set(expected) - set(selected)
    extra = set(selected) - set(expected)
    if missing or extra:
        raise ValueError(
            "source actor keys do not match expected MLP "
            f"(missing={sorted(missing)}, extra={sorted(extra)})"
        )

    output: dict[str, torch.Tensor] = {}
    for key, shape in expected.items():
        tensor = selected[key]
        if tensor.shape != shape:
            raise ValueError(f"source actor tensor {key!r} has shape {tuple(tensor.shape)}, expected {shape}")
        if not tensor.is_floating_point():
            raise ValueError(f"source actor tensor {key!r} must be floating point")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"source actor tensor {key!r} contains a non-finite value")
        output[key] = tensor.detach().cpu().contiguous().clone()
    return output


def _converted_tracking_normalizer(state: Any, *, actor_input_dim: int) -> dict[str, Any]:
    if not isinstance(state, Mapping):
        raise ValueError("tracking observation normalizer must be a mapping")
    required = {"_mean", "_var", "_std", "count"}
    if set(state) != required:
        raise ValueError(
            f"tracking observation normalizer keys must be exactly {sorted(required)}; got {sorted(state)}"
        )
    mean = state["_mean"]
    variance = state["_var"]
    std = state["_std"]
    count = state["count"]
    if not all(isinstance(value, torch.Tensor) for value in (mean, variance, std, count)):
        raise ValueError("tracking normalizer mean/variance/std/count must all be tensors")
    if mean.shape == (1, actor_input_dim):
        mean = mean.squeeze(0)
    if variance.shape == (1, actor_input_dim):
        variance = variance.squeeze(0)
    if std.shape == (1, actor_input_dim):
        std = std.squeeze(0)
    if mean.shape != (actor_input_dim,) or variance.shape != (actor_input_dim,) or std.shape != (
        actor_input_dim,
    ):
        raise ValueError(f"tracking normalizer vectors must flatten to shape ({actor_input_dim},)")
    if count.numel() != 1 or not bool(torch.isfinite(count).all()) or float(count.item()) <= 0:
        raise ValueError("tracking normalizer count must be a finite positive scalar")
    for name, tensor in (("mean", mean), ("variance", variance), ("std", std)):
        if not tensor.is_floating_point() or not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"tracking normalizer {name} must be a finite floating-point tensor")
    if bool((variance < 0).any()) or bool((std < 0).any()):
        raise ValueError("tracking normalizer variance/std must not be negative")
    if not torch.allclose(std.square(), variance, rtol=1.0e-4, atol=1.0e-6):
        raise ValueError("tracking normalizer _std is inconsistent with _var")
    return {
        "kind": "empirical_std_plus_eps",
        "mean": mean.detach().cpu().contiguous().clone(),
        "std": std.detach().cpu().contiguous().clone(),
        "epsilon": 0.01,
    }


def _provenance(
    *,
    source_path: Path,
    source_sha256: str,
    skill_name: str,
    source_format: str,
) -> dict[str, Any]:
    return {
        "source_filename": source_path.name,
        "source_sha256": source_sha256,
        "skill_name": skill_name,
        "source_format": source_format,
    }


def build_tracking_teacher_artifact(
    source_checkpoint: str | Path,
    destination: str | Path,
    *,
    expected_source_sha256: str,
    skill_name: Literal["climb", "down_roll"],
    actor_input_dim: int = 347,
    hidden_dims: Sequence[int] = (512, 256, 128),
    action_dim: int = NUM_CANONICAL_JOINTS,
    overwrite: bool = False,
) -> ArtifactBuildResult:
    """Convert a climb/down-roll RSL-RL checkpoint using weights-only loading."""

    if skill_name not in {"climb", "down_roll"}:
        raise ValueError("tracking artifact skill_name must be 'climb' or 'down_roll'")
    dimensions = _positive_dimensions(actor_input_dim, hidden_dims, action_dim)
    source_path = verify_file_sha256(source_checkpoint, expected_source_sha256)
    # Legacy RSL-RL checkpoints contain Adam state dictionaries whose keys are
    # integer parameter IDs.  They are valid outputs of PyTorch's restricted
    # ``weights_only`` loader but intentionally fail the stricter JSON-like
    # runtime-artifact validator.  Conversion never consumes optimizer state:
    # validate the exact top-level wrapper below, then validate only the actor
    # and observation-normalizer tensor mappings that are copied out.
    try:
        source = torch.load(source_path, map_location="cpu", weights_only=True)
    except TypeError as exc:  # pragma: no cover - supported PyTorch is >= 2.4
        raise RuntimeError("PyTorch with weights_only=True support is required") from exc
    if not isinstance(source, Mapping):
        raise ValueError("tracking checkpoint root must be a mapping")
    if set(source) != _TRACKING_CHECKPOINT_KEYS:
        raise ValueError(
            "tracking checkpoint wrapper does not match the audited RSL-RL format "
            f"(missing={sorted(_TRACKING_CHECKPOINT_KEYS - set(source))}, "
            f"unknown={sorted(set(source) - _TRACKING_CHECKPOINT_KEYS)})"
        )
    actor = _canonical_actor_state(
        source["model_state_dict"], state_prefix="actor.", dimensions=dimensions
    )
    normalizer = _converted_tracking_normalizer(source["obs_norm_state_dict"], actor_input_dim=actor_input_dim)
    artifact = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "actor_state_dict": actor,
        "actor_normalizer": normalizer,
        "provenance": _provenance(
            source_path=source_path,
            source_sha256=expected_source_sha256.lower(),
            skill_name=skill_name,
            source_format="rsl_rl_2p3_tracking_checkpoint",
        ),
    }
    return _atomic_write_artifact(artifact, destination, overwrite=overwrite, source_sha256=expected_source_sha256)


def build_locomotion_teacher_artifact(
    source: str | Path,
    destination: str | Path,
    *,
    expected_source_sha256: str,
    source_format: Literal["safe_tensor", "trusted_torchscript"],
    actor_input_dim: int = 1207,
    hidden_dims: Sequence[int] = (512, 256, 128),
    action_dim: int = NUM_CANONICAL_JOINTS,
    state_dict_key: str | None = "actor_state_dict",
    state_prefix: str = "",
    trusted_source: bool = False,
    overwrite: bool = False,
) -> ArtifactBuildResult:
    """Convert a locomotion export without ever unpickling its legacy checkpoint.

    ``safe_tensor`` uses ``torch.load(weights_only=True)`` and is the preferred
    route for a trusted external state-dict conversion.  ``trusted_torchscript``
    accepts an exported ``policy.pt`` only after the caller explicitly marks the
    hash-pinned source as trusted.  There is intentionally no legacy-pickle mode.
    """

    dimensions = _positive_dimensions(actor_input_dim, hidden_dims, action_dim)
    source_path = verify_file_sha256(source, expected_source_sha256)
    if source_format == "safe_tensor":
        loaded = _load_weights_only(source_path, map_location="cpu")
        if not isinstance(loaded, Mapping):
            raise ValueError("safe locomotion export root must be a mapping")
        if state_dict_key is None:
            state = loaded
        else:
            allowed = {state_dict_key, "format_version", "provenance"}
            unknown = set(loaded) - allowed
            if unknown:
                raise ValueError(f"safe locomotion export contains unknown wrapper keys: {sorted(unknown)}")
            if state_dict_key not in loaded:
                raise ValueError(f"safe locomotion export is missing {state_dict_key!r}")
            state = loaded[state_dict_key]
        source_format_name = "weights_only_actor_state"
    elif source_format == "trusted_torchscript":
        if not trusted_source:
            raise PermissionError(
                "TorchScript loading may execute serialized model code; set trusted_source=True only "
                "for a hash-audited export"
            )
        try:
            scripted = torch.jit.load(str(source_path), map_location="cpu")
        except (RuntimeError, ValueError) as exc:
            raise ValueError("trusted locomotion source is not a valid TorchScript policy export") from exc
        scripted.eval()
        state = scripted.state_dict()
        source_format_name = "trusted_torchscript_policy"
    else:  # pragma: no cover - Literal protects typed callers; retained for runtime inputs
        raise ValueError(f"unsupported locomotion source_format: {source_format!r}")

    actor = _canonical_actor_state(state, state_prefix=state_prefix, dimensions=dimensions)
    artifact = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "actor_state_dict": actor,
        "provenance": _provenance(
            source_path=source_path,
            source_sha256=expected_source_sha256.lower(),
            skill_name="locomotion",
            source_format=source_format_name,
        ),
    }
    return _atomic_write_artifact(artifact, destination, overwrite=overwrite, source_sha256=expected_source_sha256)


def _atomic_write_artifact(
    artifact: Mapping[str, Any],
    destination: str | Path,
    *,
    overwrite: bool,
    source_sha256: str,
) -> ArtifactBuildResult:
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or not all(
        character in "0123456789abcdefABCDEF" for character in source_sha256
    ):
        # The source is normally verified before this helper is reached; keep
        # this invariant local so a future internal caller cannot write first
        # and fail provenance validation afterwards.
        raise ValueError("source_sha256 must be a 64-character hex digest")
    destination_path = Path(destination)
    parent = destination_path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise NotADirectoryError(f"artifact destination parent is not a directory: {parent}")
    destination_path = parent / destination_path.name
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"artifact destination already exists: {destination_path}")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            torch.save(dict(artifact), stream)
            stream.flush()
            os.fsync(stream.fileno())
        # Re-check immediately before the atomic replace to reduce accidental
        # clobbering; overwrite=False is the default one-time conversion mode.
        if destination_path.exists() and not overwrite:
            raise FileExistsError(f"artifact destination already exists: {destination_path}")
        os.replace(temporary_path, destination_path)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    output_sha256 = sha256_file(destination_path)
    return ArtifactBuildResult(
        path=destination_path.resolve(strict=True),
        sha256=output_sha256,
        source_sha256=source_sha256.lower(),
    )
