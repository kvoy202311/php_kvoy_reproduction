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

# These fields deliberately change between the atomic, transition and full
# curricula.  They affect augmentation, not the meaning, order or units of a
# policy input, so they must not prevent an intentional stage warm-start.
_POLICY_AUGMENTATION_KEYS = {
    "delay_range_s",
    "image_offset_range",
    "noise_enabled",
    "pixel_noise_std",
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


def _strip_policy_augmentation_fields(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _strip_policy_augmentation_fields(item)
            for key, item in value.items()
            if str(key) not in _POLICY_AUGMENTATION_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_strip_policy_augmentation_fields(item) for item in value]
    return value


def _config_dict(value: Any, *, name: str) -> dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(f"{name} must expose to_dict()")
    result = to_dict()
    if not isinstance(result, Mapping):
        raise TypeError(f"{name}.to_dict() must return a mapping")
    return dict(result)


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


def student_policy_input_contract(env_cfg: Any, *, task: str) -> dict[str, Any]:
    """Capture deployment-critical Actor input and action semantics.

    This deliberately excludes curriculum-only image corruption and camera
    extrinsic randomization.  It includes the nominal camera transform and
    intrinsics, depth tensor geometry/range, ordered policy-observation terms,
    control period and action mapping.  Consequently atomic -> transition ->
    full warm-starts remain valid, while a torso/head camera, FOV, crop, term
    order or actuator-contract mismatch is rejected before a checkpoint can be
    used.
    """

    if not isinstance(task, str) or not task:
        raise ValueError("task must be a non-empty string")
    try:
        scene = env_cfg.scene
        camera = scene.depth_camera
        policy_observations = env_cfg.observations.policy
        actions = env_cfg.actions
        sim_dt = float(env_cfg.sim.dt)
        decimation = int(env_cfg.decimation)
    except AttributeError as exc:
        raise TypeError(
            "env_cfg must expose scene.depth_camera, observations.policy, actions, sim.dt and decimation"
        ) from exc
    if sim_dt <= 0.0 or decimation <= 0:
        raise ValueError("environment simulation dt and decimation must be positive")

    return {
        "task": task,
        # Bump this string whenever the meaning of an unchanged-shape Actor
        # input changes.  The command channel carries the latest live deployment
        # request; committed climb/down-roll control ignores it until the
        # post-motion release gate, without hiding it from the Actor.
        "actor_command_semantics": (
            "bounded_live_requested_world_velocity_body_frame_motion_lock_v4"
        ),
        "motion_execution_semantics": (
            "auto_static_prefix_trim_stationary_boundary_reset_safe_student_head_clock_v2"
        ),
        "control_dt": sim_dt * decimation,
        "depth_camera": _config_dict(camera, name="env_cfg.scene.depth_camera"),
        "policy_observations": _strip_policy_augmentation_fields(
            _config_dict(policy_observations, name="env_cfg.observations.policy")
        ),
        "actions": _config_dict(actions, name="env_cfg.actions"),
    }


__all__ = [
    "contract_fingerprint",
    "environment_training_contract",
    "motion_directory_contract",
    "student_policy_input_contract",
]
