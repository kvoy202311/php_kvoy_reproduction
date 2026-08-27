"""Strict, portable teacher-policy manifests.

The manifest is deliberately more explicit than an ordinary experiment config:
it is the contract between a frozen teacher and a future distillation run.  No
field is inferred from checkpoint contents, and every file used by a teacher is
hash verified before loading.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 1
NUM_CANONICAL_JOINTS = 29
_SHA256_LENGTH = 64
_SUPPORTED_ACTIVATIONS = frozenset({"elu", "relu", "tanh", "silu"})
_SUPPORTED_NORMALIZERS = frozenset({"none", "empirical_std_plus_eps"})


def _strict_keys(
    data: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> None:
    if not isinstance(data, Mapping):
        raise TypeError(f"{context} must be a mapping")
    optional = optional or set()
    keys = set(data)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {sorted(unknown)}")


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _float_vector(value: Any, *, name: str, allow_scalar: bool = False) -> tuple[float, ...]:
    if allow_scalar and isinstance(value, (int, float)) and not isinstance(value, bool):
        scalar = _finite_float(value, name=name)
        return (scalar,) * NUM_CANONICAL_JOINTS
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        expected = "a scalar or 29-element sequence" if allow_scalar else "a 29-element sequence"
        raise ValueError(f"{name} must be {expected}")
    if len(value) != NUM_CANONICAL_JOINTS:
        raise ValueError(f"{name} must contain exactly {NUM_CANONICAL_JOINTS} values")
    return tuple(_finite_float(item, name=f"{name}[{index}]") for index, item in enumerate(value))


def _sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        raise ValueError(f"{name} must be a 64-character SHA256 hex digest")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"{name} must be a 64-character SHA256 hex digest")
    return normalized


def _relative_artifact_path(value: Any, *, name: str) -> str:
    """Validate a manifest-owned relative path without resolving the file yet."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty relative path")
    if "\\" in value:
        raise ValueError(f"{name} must use POSIX separators")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} must be a normalized relative path without traversal")
    return path.as_posix()


def _json_safe(value: Any, *, name: str) -> Any:
    """Return a detached JSON-safe value while rejecting NaN and exotic objects."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{name} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{name} contains a non-string mapping key")
            result[key] = _json_safe(item, name=f"{name}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, name=f"{name}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{name} contains a non-JSON value of type {type(value).__name__}")


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA256 digest without loading the complete file into memory."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"artifact is not a regular file: {file_path}")
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: str | Path, expected_sha256: str) -> Path:
    """Verify a file and return its resolved path, or raise on any mismatch."""

    expected = _sha256(expected_sha256, name="expected_sha256")
    file_path = Path(path).resolve(strict=True)
    actual = sha256_file(file_path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch for {file_path}: expected {expected}, got {actual}")
    return file_path


def resolve_manifest_path(base_directory: str | Path, relative_path: str) -> Path:
    """Resolve an artifact path and reject both lexical and symlink traversal."""

    safe_path = _relative_artifact_path(relative_path, name="relative_path")
    base = Path(base_directory).resolve(strict=True)
    if not base.is_dir():
        raise NotADirectoryError(f"manifest base is not a directory: {base}")
    candidate = (base / safe_path).resolve(strict=True)
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes manifest directory: {relative_path}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"artifact is not a regular file: {candidate}")
    return candidate


@dataclass(frozen=True)
class ObservationTerm:
    """One ordered segment of an actor observation vector."""

    name: str
    dimension: int
    history_length: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("observation term name must be a non-empty string")
        object.__setattr__(self, "dimension", _positive_int(self.dimension, name=f"{self.name}.dimension"))
        object.__setattr__(
            self, "history_length", _positive_int(self.history_length, name=f"{self.name}.history_length")
        )

    @property
    def flattened_dimension(self) -> int:
        return self.dimension * self.history_length

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservationTerm":
        _strict_keys(
            data,
            required={"name", "dimension", "history_length"},
            context="observation term",
        )
        return cls(name=data["name"], dimension=data["dimension"], history_length=data["history_length"])

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dimension": self.dimension, "history_length": self.history_length}


@dataclass(frozen=True)
class ObservationSchema:
    """Ordered actor-observation contract, including history and clipping semantics."""

    terms: tuple[ObservationTerm, ...]
    history_order: str
    previous_action_semantics: str
    clip_low: float | None
    clip_high: float | None

    def __post_init__(self) -> None:
        terms = tuple(self.terms)
        if not terms or any(not isinstance(term, ObservationTerm) for term in terms):
            raise ValueError("observation_schema.terms must contain ObservationTerm values")
        names = [term.name for term in terms]
        if len(set(names)) != len(names):
            raise ValueError("observation_schema contains duplicate term names")
        object.__setattr__(self, "terms", terms)

        if self.history_order not in {"none", "oldest_to_newest", "newest_to_oldest"}:
            raise ValueError("observation_schema.history_order is unsupported")
        has_history = any(term.history_length > 1 for term in terms)
        if has_history == (self.history_order == "none"):
            raise ValueError("observation_schema.history_order is inconsistent with term history lengths")
        if not isinstance(self.previous_action_semantics, str) or not self.previous_action_semantics:
            raise ValueError("observation_schema.previous_action_semantics must be a non-empty string")

        if (self.clip_low is None) != (self.clip_high is None):
            raise ValueError("observation clipping bounds must either both be set or both be null")
        if self.clip_low is not None and self.clip_high is not None:
            low = _finite_float(self.clip_low, name="observation_schema.clip_low")
            high = _finite_float(self.clip_high, name="observation_schema.clip_high")
            if low > high:
                raise ValueError("observation_schema.clip_low must not exceed clip_high")
            object.__setattr__(self, "clip_low", low)
            object.__setattr__(self, "clip_high", high)

    @property
    def dimension(self) -> int:
        return sum(term.flattened_dimension for term in self.terms)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ObservationSchema":
        _strict_keys(
            data,
            required={"terms", "history_order", "previous_action_semantics", "clip_low", "clip_high"},
            context="observation_schema",
        )
        if not isinstance(data["terms"], list):
            raise ValueError("observation_schema.terms must be a list")
        return cls(
            terms=tuple(ObservationTerm.from_dict(term) for term in data["terms"]),
            history_order=data["history_order"],
            previous_action_semantics=data["previous_action_semantics"],
            clip_low=data["clip_low"],
            clip_high=data["clip_high"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": [term.to_dict() for term in self.terms],
            "history_order": self.history_order,
            "previous_action_semantics": self.previous_action_semantics,
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
        }


@dataclass(frozen=True)
class NormalizerSpec:
    """Frozen actor-normalization state contract."""

    kind: str
    dimension: int | None
    artifact_path: str | None
    artifact_sha256: str | None
    state_key: str | None
    epsilon: float | None

    def __post_init__(self) -> None:
        if self.kind not in _SUPPORTED_NORMALIZERS:
            raise ValueError(f"unsupported normalizer kind: {self.kind!r}")
        if self.kind == "none":
            if any(
                value is not None
                for value in (self.dimension, self.artifact_path, self.artifact_sha256, self.state_key, self.epsilon)
            ):
                raise ValueError("normalizer kind 'none' must not declare state fields")
            return

        dimension = _positive_int(self.dimension, name="normalizer.dimension")
        path = _relative_artifact_path(self.artifact_path, name="normalizer.artifact_path")
        digest = _sha256(self.artifact_sha256, name="normalizer.artifact_sha256")
        if not isinstance(self.state_key, str) or not self.state_key:
            raise ValueError("normalizer.state_key must be a non-empty string")
        epsilon = _finite_float(self.epsilon, name="normalizer.epsilon")
        if epsilon <= 0:
            raise ValueError("normalizer.epsilon must be positive")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "artifact_path", path)
        object.__setattr__(self, "artifact_sha256", digest)
        object.__setattr__(self, "epsilon", epsilon)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizerSpec":
        _strict_keys(
            data,
            required={"kind", "dimension", "artifact_path", "artifact_sha256", "state_key", "epsilon"},
            context="normalizer",
        )
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "dimension": self.dimension,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "state_key": self.state_key,
            "epsilon": self.epsilon,
        }


@dataclass(frozen=True)
class TeacherManifest:
    """Validated manifest for one immutable teacher policy artifact."""

    schema_version: int
    skill_name: str
    checkpoint_path: str
    checkpoint_sha256: str
    state_dict_key: str | None
    state_prefix: str
    actor_input_dim: int
    action_dim: int
    hidden_dims: tuple[int, ...]
    activation: str
    joint_order: tuple[str, ...]
    default_q: tuple[float, ...]
    action_scale: tuple[float, ...] | float
    observation_schema: ObservationSchema
    normalizer: NormalizerSpec
    urdf_path: str
    urdf_sha256: str
    usd_path: str
    usd_sha256: str
    control_dt: float
    hard_lower_limits: tuple[float, ...]
    hard_upper_limits: tuple[float, ...]
    effort_limits: tuple[float, ...]
    artifact_metadata: Mapping[str, Any] = field(default_factory=dict)
    _source_directory: Path | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema_version {self.schema_version!r}; expected {MANIFEST_SCHEMA_VERSION}"
            )
        if not isinstance(self.skill_name, str) or not self.skill_name:
            raise ValueError("skill_name must be a non-empty string")

        object.__setattr__(
            self, "checkpoint_path", _relative_artifact_path(self.checkpoint_path, name="checkpoint_path")
        )
        object.__setattr__(self, "checkpoint_sha256", _sha256(self.checkpoint_sha256, name="checkpoint_sha256"))
        if self.state_dict_key is not None and (
            not isinstance(self.state_dict_key, str) or not self.state_dict_key
        ):
            raise ValueError("state_dict_key must be null or a non-empty string")
        if not isinstance(self.state_prefix, str):
            raise ValueError("state_prefix must be a string")

        object.__setattr__(self, "actor_input_dim", _positive_int(self.actor_input_dim, name="actor_input_dim"))
        action_dim = _positive_int(self.action_dim, name="action_dim")
        if action_dim != NUM_CANONICAL_JOINTS:
            raise ValueError(f"action_dim must be exactly {NUM_CANONICAL_JOINTS}")
        object.__setattr__(self, "action_dim", action_dim)

        hidden_dims = tuple(_positive_int(dim, name="hidden_dims item") for dim in self.hidden_dims)
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")
        object.__setattr__(self, "hidden_dims", hidden_dims)
        if not isinstance(self.activation, str) or self.activation.lower() not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(f"unsupported activation: {self.activation!r}")
        object.__setattr__(self, "activation", self.activation.lower())

        joint_order = tuple(self.joint_order)
        if len(joint_order) != NUM_CANONICAL_JOINTS:
            raise ValueError(f"joint_order must contain exactly {NUM_CANONICAL_JOINTS} names")
        if any(not isinstance(name, str) or not name for name in joint_order):
            raise ValueError("joint_order must contain non-empty strings")
        if len(set(joint_order)) != len(joint_order):
            raise ValueError("joint_order contains duplicate names")
        object.__setattr__(self, "joint_order", joint_order)

        default_q = _float_vector(self.default_q, name="default_q")
        scale = _float_vector(self.action_scale, name="action_scale", allow_scalar=True)
        if any(value <= 0 for value in scale):
            raise ValueError("action_scale must contain only positive values")
        lower = _float_vector(self.hard_lower_limits, name="hard_lower_limits")
        upper = _float_vector(self.hard_upper_limits, name="hard_upper_limits")
        effort = _float_vector(self.effort_limits, name="effort_limits")
        if any(low > high for low, high in zip(lower, upper, strict=True)):
            raise ValueError("hard_lower_limits must be less than or equal to hard_upper_limits")
        if any(value <= 0 for value in effort):
            raise ValueError("effort_limits must contain only positive URDF values")
        object.__setattr__(self, "default_q", default_q)
        object.__setattr__(self, "action_scale", scale)
        object.__setattr__(self, "hard_lower_limits", lower)
        object.__setattr__(self, "hard_upper_limits", upper)
        object.__setattr__(self, "effort_limits", effort)

        if not isinstance(self.observation_schema, ObservationSchema):
            raise TypeError("observation_schema must be an ObservationSchema")
        if self.observation_schema.dimension != self.actor_input_dim:
            raise ValueError(
                "observation schema dimension does not match actor_input_dim: "
                f"{self.observation_schema.dimension} != {self.actor_input_dim}"
            )
        if not isinstance(self.normalizer, NormalizerSpec):
            raise TypeError("normalizer must be a NormalizerSpec")
        if self.normalizer.kind != "none" and self.normalizer.dimension != self.actor_input_dim:
            raise ValueError("normalizer dimension must equal actor_input_dim")

        object.__setattr__(self, "urdf_path", _relative_artifact_path(self.urdf_path, name="urdf_path"))
        object.__setattr__(self, "urdf_sha256", _sha256(self.urdf_sha256, name="urdf_sha256"))
        object.__setattr__(self, "usd_path", _relative_artifact_path(self.usd_path, name="usd_path"))
        object.__setattr__(self, "usd_sha256", _sha256(self.usd_sha256, name="usd_sha256"))
        control_dt = _finite_float(self.control_dt, name="control_dt")
        if control_dt <= 0:
            raise ValueError("control_dt must be positive")
        object.__setattr__(self, "control_dt", control_dt)
        object.__setattr__(self, "artifact_metadata", _json_safe(self.artifact_metadata, name="artifact_metadata"))

        if self._source_directory is not None:
            source_directory = Path(self._source_directory).resolve(strict=True)
            if not source_directory.is_dir():
                raise NotADirectoryError(f"manifest source directory is not a directory: {source_directory}")
            object.__setattr__(self, "_source_directory", source_directory)

    @property
    def source_directory(self) -> Path | None:
        return self._source_directory

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any],
        *,
        source_directory: str | Path | None = None,
    ) -> "TeacherManifest":
        required = {
            "schema_version",
            "skill_name",
            "checkpoint_path",
            "checkpoint_sha256",
            "state_dict_key",
            "state_prefix",
            "actor_input_dim",
            "action_dim",
            "hidden_dims",
            "activation",
            "joint_order",
            "default_q",
            "action_scale",
            "observation_schema",
            "normalizer",
            "urdf_path",
            "urdf_sha256",
            "usd_path",
            "usd_sha256",
            "control_dt",
            "hard_lower_limits",
            "hard_upper_limits",
            "effort_limits",
        }
        _strict_keys(data, required=required, optional={"artifact_metadata"}, context="teacher manifest")
        hidden_dims = data["hidden_dims"]
        joint_order = data["joint_order"]
        if not isinstance(hidden_dims, list):
            raise ValueError("hidden_dims must be a list")
        if not isinstance(joint_order, list):
            raise ValueError("joint_order must be a list")
        return cls(
            schema_version=data["schema_version"],
            skill_name=data["skill_name"],
            checkpoint_path=data["checkpoint_path"],
            checkpoint_sha256=data["checkpoint_sha256"],
            state_dict_key=data["state_dict_key"],
            state_prefix=data["state_prefix"],
            actor_input_dim=data["actor_input_dim"],
            action_dim=data["action_dim"],
            hidden_dims=tuple(hidden_dims),
            activation=data["activation"],
            joint_order=tuple(joint_order),
            default_q=tuple(data["default_q"]) if isinstance(data["default_q"], list) else data["default_q"],
            action_scale=(
                tuple(data["action_scale"])
                if isinstance(data["action_scale"], list)
                else data["action_scale"]
            ),
            observation_schema=ObservationSchema.from_dict(data["observation_schema"]),
            normalizer=NormalizerSpec.from_dict(data["normalizer"]),
            urdf_path=data["urdf_path"],
            urdf_sha256=data["urdf_sha256"],
            usd_path=data["usd_path"],
            usd_sha256=data["usd_sha256"],
            control_dt=data["control_dt"],
            hard_lower_limits=(
                tuple(data["hard_lower_limits"])
                if isinstance(data["hard_lower_limits"], list)
                else data["hard_lower_limits"]
            ),
            hard_upper_limits=(
                tuple(data["hard_upper_limits"])
                if isinstance(data["hard_upper_limits"], list)
                else data["hard_upper_limits"]
            ),
            effort_limits=(
                tuple(data["effort_limits"])
                if isinstance(data["effort_limits"], list)
                else data["effort_limits"]
            ),
            artifact_metadata=data.get("artifact_metadata", {}),
            _source_directory=Path(source_directory) if source_directory is not None else None,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TeacherManifest":
        manifest_path = Path(path).resolve(strict=True)
        if not manifest_path.is_file():
            raise FileNotFoundError(f"manifest is not a regular file: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, Mapping):
            raise ValueError("teacher manifest JSON root must be an object")
        return cls.from_dict(data, source_directory=manifest_path.parent)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "skill_name": self.skill_name,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "state_dict_key": self.state_dict_key,
            "state_prefix": self.state_prefix,
            "actor_input_dim": self.actor_input_dim,
            "action_dim": self.action_dim,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "joint_order": list(self.joint_order),
            "default_q": list(self.default_q),
            "action_scale": list(self.action_scale),
            "observation_schema": self.observation_schema.to_dict(),
            "normalizer": self.normalizer.to_dict(),
            "urdf_path": self.urdf_path,
            "urdf_sha256": self.urdf_sha256,
            "usd_path": self.usd_path,
            "usd_sha256": self.usd_sha256,
            "control_dt": self.control_dt,
            "hard_lower_limits": list(self.hard_lower_limits),
            "hard_upper_limits": list(self.hard_upper_limits),
            "effort_limits": list(self.effort_limits),
            "artifact_metadata": _json_safe(self.artifact_metadata, name="artifact_metadata"),
        }

    def save(self, path: str | Path) -> Path:
        manifest_path = Path(path)
        if not manifest_path.parent.is_dir():
            raise FileNotFoundError(f"manifest parent directory does not exist: {manifest_path.parent}")
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        return manifest_path.resolve(strict=True)

    def resolve_path(self, relative_path: str) -> Path:
        if self._source_directory is None:
            raise ValueError("manifest has no source directory; load it from a file or pass a base directory")
        return resolve_manifest_path(self._source_directory, relative_path)

    def verify_checkpoint(self, *, base_directory: str | Path | None = None) -> Path:
        path = self._resolve_for_verification(self.checkpoint_path, base_directory)
        return verify_file_sha256(path, self.checkpoint_sha256)

    def verify_assets(self, *, base_directory: str | Path | None = None) -> dict[str, Path]:
        urdf = verify_file_sha256(self._resolve_for_verification(self.urdf_path, base_directory), self.urdf_sha256)
        usd = verify_file_sha256(self._resolve_for_verification(self.usd_path, base_directory), self.usd_sha256)
        return {"urdf": urdf, "usd": usd}

    def verify_normalizer(self, *, base_directory: str | Path | None = None) -> Path | None:
        if self.normalizer.kind == "none":
            return None
        if self.normalizer.artifact_path is None or self.normalizer.artifact_sha256 is None:
            raise RuntimeError("validated non-identity normalizer is missing its artifact contract")
        path = self._resolve_for_verification(self.normalizer.artifact_path, base_directory)
        return verify_file_sha256(path, self.normalizer.artifact_sha256)

    def verify_all_files(self, *, base_directory: str | Path | None = None) -> dict[str, Path]:
        verified = {"checkpoint": self.verify_checkpoint(base_directory=base_directory)}
        verified.update(self.verify_assets(base_directory=base_directory))
        normalizer = self.verify_normalizer(base_directory=base_directory)
        if normalizer is not None:
            verified["normalizer"] = normalizer
        return verified

    def fingerprint(self) -> str:
        """Return a stable digest of the complete serialized contract."""

        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _resolve_for_verification(
        self, relative_path: str, base_directory: str | Path | None
    ) -> Path:
        base = Path(base_directory) if base_directory is not None else self._source_directory
        if base is None:
            raise ValueError("a base_directory is required for a manifest created from a dictionary")
        return resolve_manifest_path(base, relative_path)
