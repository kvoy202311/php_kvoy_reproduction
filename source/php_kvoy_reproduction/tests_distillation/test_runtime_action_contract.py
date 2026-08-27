from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch

from php_kvoy_reproduction.distillation.action_contract import validate_runtime_action_contract


def _fixture(tmp_path) -> tuple[SimpleNamespace, SimpleNamespace]:
    joint_order = tuple(f"joint_{index}" for index in range(29))
    default_q = tuple(0.01 * index for index in range(29))
    action_scale = tuple(0.1 + 0.01 * index for index in range(29))
    lower = tuple(-2.0 - 0.01 * index for index in range(29))
    upper = tuple(2.0 + 0.01 * index for index in range(29))
    effort = tuple(100.0 + index for index in range(29))
    usd_path = tmp_path / "elf3.usd"
    usd_path.write_bytes(b"runtime-usd")
    usd_sha256 = hashlib.sha256(b"runtime-usd").hexdigest()
    manifest = SimpleNamespace(
        joint_order=joint_order,
        default_q=default_q,
        action_scale=action_scale,
        hard_lower_limits=lower,
        hard_upper_limits=upper,
        effort_limits=effort,
        usd_sha256=usd_sha256,
    )
    teachers = [SimpleNamespace(manifest=manifest) for _ in range(3)]
    router = SimpleNamespace(teachers=teachers, _action_transforms=(None, None, None))
    limits = torch.tensor(list(zip(lower, upper, strict=True)), dtype=torch.float32)
    term = SimpleNamespace(
        action_dim=29,
        cfg=SimpleNamespace(use_default_offset=True),
        _joint_names=list(joint_order),
        _scale=torch.tensor(action_scale).repeat(4, 1),
        _offset=torch.tensor(default_q).repeat(4, 1),
        _clip=limits.repeat(4, 1, 1),
        _asset=SimpleNamespace(
            joint_names=list(joint_order),
            data=SimpleNamespace(
                joint_effort_limits=torch.tensor(effort).repeat(4, 1),
            ),
            cfg=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(usd_path))),
        ),
    )
    return router, term


def test_matching_runtime_action_contract_is_accepted(tmp_path) -> None:
    router, term = _fixture(tmp_path)
    validate_runtime_action_contract(router, term)


def test_runtime_action_contract_rejects_joint_order_or_scale_drift(tmp_path) -> None:
    router, term = _fixture(tmp_path)
    term._joint_names[0], term._joint_names[1] = term._joint_names[1], term._joint_names[0]
    with pytest.raises(ValueError, match="joint order"):
        validate_runtime_action_contract(router, term)

    router, term = _fixture(tmp_path)
    term._scale[:, 3] += 0.02
    with pytest.raises(ValueError, match="scale differs"):
        validate_runtime_action_contract(router, term)


def test_runtime_action_contract_rejects_environment_dependent_mapping(tmp_path) -> None:
    router, term = _fixture(tmp_path)
    term._offset[2, 5] += 0.01
    with pytest.raises(ValueError, match="differs between environments"):
        validate_runtime_action_contract(router, term)


def test_runtime_action_contract_rejects_unexpressed_action_transform(tmp_path) -> None:
    router, term = _fixture(tmp_path)
    router._action_transforms = (None, object(), None)
    with pytest.raises(ValueError, match="share the student's raw-action contract"):
        validate_runtime_action_contract(router, term)


def test_runtime_action_contract_rejects_effort_or_live_usd_drift(tmp_path) -> None:
    router, term = _fixture(tmp_path)
    term._asset.data.joint_effort_limits[:, 4] += 1.0
    with pytest.raises(ValueError, match="effort limits differs"):
        validate_runtime_action_contract(router, term)

    router, term = _fixture(tmp_path)
    with open(term._asset.cfg.spawn.usd_path, "ab") as stream:
        stream.write(b"-changed")
    with pytest.raises(ValueError, match="USD differs"):
        validate_runtime_action_contract(router, term)
