from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from php_kvoy_reproduction.distillation.teacher_artifact import (
    ARTIFACT_FORMAT_VERSION,
    build_locomotion_teacher_artifact,
    build_tracking_teacher_artifact,
)
from php_kvoy_reproduction.distillation.teacher_manifest import sha256_file


class _UnsafeLegacyObject:
    """Pickleable test payload that is intentionally unsafe for weights-only load."""

    pass


def actor_state(input_dim: int = 4, hidden_dim: int = 5) -> dict[str, torch.Tensor]:
    actor = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(), nn.Linear(hidden_dim, 29))
    return {name: tensor.detach().clone() for name, tensor in actor.state_dict().items()}


def tracking_checkpoint(*, inconsistent_std: bool = False) -> dict[str, object]:
    state = {f"actor.{name}": tensor for name, tensor in actor_state().items()}
    state["std"] = torch.ones(29)
    state["critic.0.weight"] = torch.zeros(1, 1)
    std = torch.full((1, 4), 2.0)
    variance = std.square()
    if inconsistent_std:
        variance[0, 0] = 9.0
    return {
        "model_state_dict": state,
        "optimizer_state_dict": {},
        "iter": 10,
        "infos": {},
        "obs_norm_state_dict": {
            "_mean": torch.arange(4, dtype=torch.float32).unsqueeze(0),
            "_var": variance,
            "_std": std,
            "count": torch.tensor(100.0),
        },
        "privileged_obs_norm_state_dict": {},
    }


def test_tracking_conversion_extracts_only_actor_and_frozen_normalizer(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "teacher.pt"
    torch.save(tracking_checkpoint(), source)
    source_hash = sha256_file(source)
    result = build_tracking_teacher_artifact(
        source,
        destination,
        expected_source_sha256=source_hash,
        skill_name="climb",
        actor_input_dim=4,
        hidden_dims=(5,),
    )
    assert result.path == destination.resolve()
    assert result.sha256 == sha256_file(destination)
    artifact = torch.load(destination, map_location="cpu", weights_only=True)
    assert artifact["format_version"] == ARTIFACT_FORMAT_VERSION
    assert set(artifact["actor_state_dict"]) == {"0.weight", "0.bias", "2.weight", "2.bias"}
    assert artifact["actor_normalizer"]["epsilon"] == 0.01
    torch.testing.assert_close(artifact["actor_normalizer"]["std"], torch.full((4,), 2.0))
    assert artifact["provenance"]["source_sha256"] == source_hash


def test_tracking_conversion_verifies_hash_and_normalizer_consistency(tmp_path: Path) -> None:
    source = tmp_path / "source.pt"
    torch.save(tracking_checkpoint(), source)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        build_tracking_teacher_artifact(
            source,
            tmp_path / "teacher.pt",
            expected_source_sha256="0" * 64,
            skill_name="climb",
            actor_input_dim=4,
            hidden_dims=(5,),
        )

    torch.save(tracking_checkpoint(inconsistent_std=True), source)
    with pytest.raises(ValueError, match="inconsistent"):
        build_tracking_teacher_artifact(
            source,
            tmp_path / "teacher.pt",
            expected_source_sha256=sha256_file(source),
            skill_name="down_roll",
            actor_input_dim=4,
            hidden_dims=(5,),
        )


def test_safe_locomotion_conversion_and_no_overwrite_default(tmp_path: Path) -> None:
    source = tmp_path / "exported_state.pt"
    destination = tmp_path / "locomotion.pt"
    torch.save({"actor_state_dict": actor_state()}, source)
    source_hash = sha256_file(source)
    build_locomotion_teacher_artifact(
        source,
        destination,
        expected_source_sha256=source_hash,
        source_format="safe_tensor",
        actor_input_dim=4,
        hidden_dims=(5,),
    )
    artifact = torch.load(destination, map_location="cpu", weights_only=True)
    assert artifact["format_version"] == ARTIFACT_FORMAT_VERSION
    assert "actor_normalizer" not in artifact
    with pytest.raises(FileExistsError):
        build_locomotion_teacher_artifact(
            source,
            destination,
            expected_source_sha256=source_hash,
            source_format="safe_tensor",
            actor_input_dim=4,
            hidden_dims=(5,),
        )


def test_locomotion_has_no_unsafe_legacy_pickle_fallback(tmp_path: Path) -> None:
    source = tmp_path / "legacy.pt"
    torch.save({"actor_state_dict": _UnsafeLegacyObject()}, source)
    with pytest.raises(Exception):
        build_locomotion_teacher_artifact(
            source,
            tmp_path / "output.pt",
            expected_source_sha256=sha256_file(source),
            source_format="safe_tensor",
            actor_input_dim=4,
            hidden_dims=(5,),
        )
    assert not (tmp_path / "output.pt").exists()


def test_torchscript_requires_explicit_trust_before_loading(tmp_path: Path) -> None:
    source = tmp_path / "policy.pt"
    scripted = torch.jit.script(nn.Sequential(nn.Linear(4, 5), nn.ELU(), nn.Linear(5, 29)))
    torch.jit.save(scripted, source)
    with pytest.raises(PermissionError, match="trusted_source=True"):
        build_locomotion_teacher_artifact(
            source,
            tmp_path / "output.pt",
            expected_source_sha256=sha256_file(source),
            source_format="trusted_torchscript",
            actor_input_dim=4,
            hidden_dims=(5,),
            trusted_source=False,
        )


def test_atomic_conversion_cleans_temporary_file_on_validation_failure(tmp_path: Path) -> None:
    source = tmp_path / "bad.pt"
    bad_state = actor_state()
    bad_state["extra"] = torch.zeros(1)
    torch.save({"actor_state_dict": bad_state}, source)
    with pytest.raises(ValueError, match="actor keys"):
        build_locomotion_teacher_artifact(
            source,
            tmp_path / "output.pt",
            expected_source_sha256=sha256_file(source),
            source_format="safe_tensor",
            actor_input_dim=4,
            hidden_dims=(5,),
        )
    assert list(tmp_path.glob(".output.pt.*.tmp")) == []
