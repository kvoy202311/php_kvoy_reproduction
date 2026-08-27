from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from php_kvoy_reproduction.distillation.teacher_manifest import (
    NormalizerSpec,
    TeacherManifest,
    resolve_manifest_path,
    sha256_file,
    verify_file_sha256,
)

from conftest import write_asset_files


def test_valid_manifest_round_trip(valid_manifest_dict: dict, tmp_path: Path) -> None:
    manifest = TeacherManifest.from_dict(valid_manifest_dict)
    assert manifest.action_scale == (0.25,) * 29
    assert manifest.observation_schema.dimension == 4
    path = tmp_path / "teacher.json"
    manifest.save(path)
    loaded = TeacherManifest.load(path)
    assert loaded == manifest
    assert loaded.to_dict() == manifest.to_dict()
    assert loaded.source_directory == tmp_path.resolve()
    assert len(loaded.fingerprint()) == 64


@pytest.mark.parametrize("field", ["skill_name", "joint_order", "normalizer", "control_dt"])
def test_missing_root_fields_are_rejected(valid_manifest_dict: dict, field: str) -> None:
    del valid_manifest_dict[field]
    with pytest.raises(ValueError, match="missing fields"):
        TeacherManifest.from_dict(valid_manifest_dict)


def test_unknown_root_and_nested_fields_are_rejected(valid_manifest_dict: dict) -> None:
    root = copy.deepcopy(valid_manifest_dict)
    root["mystery"] = 3
    with pytest.raises(ValueError, match="unknown fields"):
        TeacherManifest.from_dict(root)

    nested = copy.deepcopy(valid_manifest_dict)
    nested["normalizer"]["clip"] = 10.0
    with pytest.raises(ValueError, match="unknown fields"):
        TeacherManifest.from_dict(nested)


@pytest.mark.parametrize("digest", ["abc", "g" * 64, "0" * 63])
def test_invalid_hashes_are_rejected(valid_manifest_dict: dict, digest: str) -> None:
    valid_manifest_dict["checkpoint_sha256"] = digest
    with pytest.raises(ValueError, match="SHA256"):
        TeacherManifest.from_dict(valid_manifest_dict)


def test_exactly_29_unique_joints_and_actions_are_required(valid_manifest_dict: dict) -> None:
    wrong_dim = copy.deepcopy(valid_manifest_dict)
    wrong_dim["action_dim"] = 28
    with pytest.raises(ValueError, match="exactly 29"):
        TeacherManifest.from_dict(wrong_dim)

    duplicate = copy.deepcopy(valid_manifest_dict)
    duplicate["joint_order"][-1] = duplicate["joint_order"][0]
    with pytest.raises(ValueError, match="duplicate"):
        TeacherManifest.from_dict(duplicate)


def test_action_scale_must_be_strictly_positive(valid_manifest_dict: dict) -> None:
    non_positive = copy.deepcopy(valid_manifest_dict)
    non_positive["action_scale"] = [0.25] * 28 + [-0.25]
    with pytest.raises(ValueError, match="positive"):
        TeacherManifest.from_dict(non_positive)


def test_observation_dimensions_and_history_contract_are_checked(valid_manifest_dict: dict) -> None:
    wrong_sum = copy.deepcopy(valid_manifest_dict)
    wrong_sum["observation_schema"]["terms"][0]["dimension"] = 3
    with pytest.raises(ValueError, match="does not match"):
        TeacherManifest.from_dict(wrong_sum)

    wrong_history = copy.deepcopy(valid_manifest_dict)
    wrong_history["observation_schema"]["terms"][0]["history_length"] = 2
    with pytest.raises(ValueError, match="history_order"):
        TeacherManifest.from_dict(wrong_history)


def test_none_and_empirical_normalizer_schemas(valid_manifest_dict: dict) -> None:
    none_manifest = TeacherManifest.from_dict(valid_manifest_dict)
    assert none_manifest.normalizer.kind == "none"

    empirical = copy.deepcopy(valid_manifest_dict)
    empirical["normalizer"] = {
        "kind": "empirical_std_plus_eps",
        "dimension": 4,
        "artifact_path": "teacher.pt",
        "artifact_sha256": "0" * 64,
        "state_key": "actor_normalizer",
        "epsilon": 0.01,
    }
    manifest = TeacherManifest.from_dict(empirical)
    assert manifest.normalizer.epsilon == 0.01

    empirical["normalizer"]["dimension"] = 3
    with pytest.raises(ValueError, match="actor_input_dim"):
        TeacherManifest.from_dict(empirical)


def test_none_normalizer_cannot_hide_state() -> None:
    with pytest.raises(ValueError, match="must not declare"):
        NormalizerSpec(
            kind="none",
            dimension=4,
            artifact_path=None,
            artifact_sha256=None,
            state_key=None,
            epsilon=None,
        )


@pytest.mark.parametrize("path", ["../teacher.pt", "/tmp/teacher.pt", "artifacts/../teacher.pt"])
def test_manifest_artifact_paths_cannot_escape(valid_manifest_dict: dict, path: str) -> None:
    valid_manifest_dict["checkpoint_path"] = path
    with pytest.raises(ValueError, match="relative path|traversal"):
        TeacherManifest.from_dict(valid_manifest_dict)


def test_symlink_escape_is_rejected(tmp_path: Path) -> None:
    base = tmp_path / "base"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    target = outside / "secret.pt"
    target.write_bytes(b"secret")
    (base / "link.pt").symlink_to(target)
    with pytest.raises(ValueError, match="escapes"):
        resolve_manifest_path(base, "link.pt")


def test_streamed_file_hash_success_and_failure(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"0123456789" * 1000)
    digest = sha256_file(path, chunk_size=7)
    assert verify_file_sha256(path, digest) == path.resolve()
    with pytest.raises(ValueError, match="mismatch"):
        verify_file_sha256(path, "0" * 64)


def test_explicit_checkpoint_asset_and_normalizer_verification(
    valid_manifest_dict: dict, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"teacher")
    hashes = write_asset_files(tmp_path)
    valid_manifest_dict["checkpoint_sha256"] = sha256_file(checkpoint)
    valid_manifest_dict["urdf_sha256"] = hashes["urdf"]
    valid_manifest_dict["usd_sha256"] = hashes["usd"]
    valid_manifest_dict["normalizer"] = {
        "kind": "empirical_std_plus_eps",
        "dimension": 4,
        "artifact_path": "teacher.pt",
        "artifact_sha256": sha256_file(checkpoint),
        "state_key": "actor_normalizer",
        "epsilon": 0.01,
    }
    manifest = TeacherManifest.from_dict(valid_manifest_dict, source_directory=tmp_path)
    verified = manifest.verify_all_files()
    assert set(verified) == {"checkpoint", "urdf", "usd", "normalizer"}

    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="mismatch"):
        manifest.verify_checkpoint()


def test_json_loader_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="root must be an object"):
        TeacherManifest.load(path)
