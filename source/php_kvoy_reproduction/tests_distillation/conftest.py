from __future__ import annotations

import copy
from pathlib import Path
import sys
import types
from typing import Any

import pytest


# The extension package's top-level __init__ registers Isaac Lab environments.
# These unit tests intentionally exercise only the dependency-light distillation
# modules, so install namespace shells rather than importing that side-effectful
# package initializer.
_SOURCE_ROOT = Path(__file__).parents[1] / "php_kvoy_reproduction"
if "php_kvoy_reproduction" not in sys.modules:
    package = types.ModuleType("php_kvoy_reproduction")
    package.__path__ = [str(_SOURCE_ROOT)]
    sys.modules["php_kvoy_reproduction"] = package
if "php_kvoy_reproduction.distillation" not in sys.modules:
    distillation_package = types.ModuleType("php_kvoy_reproduction.distillation")
    distillation_package.__path__ = [str(_SOURCE_ROOT / "distillation")]
    sys.modules["php_kvoy_reproduction.distillation"] = distillation_package
for namespace, path in (
    ("php_kvoy_reproduction.tasks", _SOURCE_ROOT / "tasks"),
    ("php_kvoy_reproduction.tasks.distillation", _SOURCE_ROOT / "tasks" / "distillation"),
    (
        "php_kvoy_reproduction.tasks.distillation.mdp",
        _SOURCE_ROOT / "tasks" / "distillation" / "mdp",
    ),
):
    if namespace not in sys.modules:
        module = types.ModuleType(namespace)
        module.__path__ = [str(path)]
        sys.modules[namespace] = module

from php_kvoy_reproduction.distillation.teacher_manifest import sha256_file


def manifest_dict(
    *,
    actor_input_dim: int = 4,
    hidden_dims: list[int] | None = None,
    checkpoint_path: str = "teacher.pt",
    checkpoint_sha256: str = "0" * 64,
    normalizer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    joints = [f"joint_{index}" for index in range(29)]
    return {
        "schema_version": 1,
        "skill_name": "climb",
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": checkpoint_sha256,
        "state_dict_key": "actor_state_dict",
        "state_prefix": "",
        "actor_input_dim": actor_input_dim,
        "action_dim": 29,
        "hidden_dims": hidden_dims or [5],
        "activation": "elu",
        "joint_order": joints,
        "default_q": [0.0] * 29,
        "action_scale": 0.25,
        "observation_schema": {
            "terms": [{"name": "actor_observation", "dimension": actor_input_dim, "history_length": 1}],
            "history_order": "none",
            "previous_action_semantics": "raw_policy_action",
            "clip_low": None,
            "clip_high": None,
        },
        "normalizer": normalizer
        or {
            "kind": "none",
            "dimension": None,
            "artifact_path": None,
            "artifact_sha256": None,
            "state_key": None,
            "epsilon": None,
        },
        "urdf_path": "robot.urdf",
        "urdf_sha256": "1" * 64,
        "usd_path": "robot.usd",
        "usd_sha256": "2" * 64,
        "control_dt": 0.02,
        "hard_lower_limits": [-2.0] * 29,
        "hard_upper_limits": [2.0] * 29,
        "effort_limits": [100.0] * 29,
        "artifact_metadata": {"source": "unit-test"},
    }


@pytest.fixture
def valid_manifest_dict() -> dict[str, Any]:
    return copy.deepcopy(manifest_dict())


def write_asset_files(directory: Path) -> dict[str, str]:
    paths = {
        "urdf": directory / "robot.urdf",
        "usd": directory / "robot.usd",
    }
    paths["urdf"].write_bytes(b"urdf-test")
    paths["usd"].write_bytes(b"usd-test")
    return {name: sha256_file(path) for name, path in paths.items()}
