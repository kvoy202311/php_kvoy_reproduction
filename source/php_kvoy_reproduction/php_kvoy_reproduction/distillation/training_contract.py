"""Stable fingerprints for semantic distillation training inputs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


_IGNORED_ENVIRONMENT_KEYS = {
    "climb_motion_dir",
    "climb_motion_file",
    "device",
    "down_roll_motion_dir",
    "down_roll_motion_file",
    "num_envs",
    "seed",
    "viewer",
}


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if callable(value):
        return f"{value.__module__}:{value.__qualname__}"
    return str(value)


def contract_fingerprint(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(_canonical(contract), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def motion_directory_contract(directory: str | Path) -> list[dict[str, Any]]:
    root = Path(directory).expanduser().resolve()
    files = sorted(root.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"motion directory contains no direct .npz files: {root}")
    return [
        {"name": path.name, "size": path.stat().st_size, "sha256": _sha256(path)}
        for path in files
    ]


def _strip_operational_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_operational_fields(item)
            for key, item in value.items()
            if str(key) not in _IGNORED_ENVIRONMENT_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_operational_fields(item) for item in value]
    return value


def environment_training_contract(
    env_cfg: Any,
    *,
    task: str,
    climb_motion_dir: str | Path,
    down_roll_motion_dir: str | Path,
) -> dict[str, Any]:
    """Capture environment semantics while allowing device/env-count changes."""

    to_dict = getattr(env_cfg, "to_dict", None)
    if not callable(to_dict):
        raise TypeError("env_cfg must expose to_dict()")
    return {
        "task": task,
        "environment": _strip_operational_fields(to_dict()),
        "climb_motion_files": motion_directory_contract(climb_motion_dir),
        "down_roll_motion_files": motion_directory_contract(down_roll_motion_dir),
    }


__all__ = [
    "contract_fingerprint",
    "environment_training_contract",
    "motion_directory_contract",
]
