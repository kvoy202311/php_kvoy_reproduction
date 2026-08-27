from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from php_kvoy_reproduction.distillation.teacher_manifest import TeacherManifest, sha256_file
from php_kvoy_reproduction.distillation.teacher_policy import TeacherPolicy, build_actor_mlp

from conftest import manifest_dict, write_asset_files


def make_artifact_and_manifest(
    tmp_path: Path,
    *,
    mutate_state=None,
    empirical: bool = False,
) -> tuple[TeacherManifest, dict[str, torch.Tensor]]:
    initial = TeacherManifest.from_dict(manifest_dict(actor_input_dim=4, hidden_dims=[5]))
    actor = build_actor_mlp(initial)
    with torch.no_grad():
        for index, parameter in enumerate(actor.parameters()):
            parameter.fill_(0.01 * (index + 1))
    state: dict[str, torch.Tensor] = {name: tensor.clone() for name, tensor in actor.state_dict().items()}
    if mutate_state is not None:
        mutate_state(state)

    artifact: dict[str, object] = {"format_version": "actor_state_v1", "actor_state_dict": state}
    normalizer = None
    if empirical:
        artifact["actor_normalizer"] = {
            "kind": "empirical_std_plus_eps",
            "mean": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "std": torch.tensor([1.0, 2.0, 4.0, 8.0]),
            "epsilon": 0.01,
        }
    path = tmp_path / "teacher.pt"
    torch.save(artifact, path)
    asset_hashes = write_asset_files(tmp_path)
    if empirical:
        normalizer = {
            "kind": "empirical_std_plus_eps",
            "dimension": 4,
            "artifact_path": "teacher.pt",
            "artifact_sha256": sha256_file(path),
            "state_key": "actor_normalizer",
            "epsilon": 0.01,
        }
    data = manifest_dict(
        actor_input_dim=4,
        hidden_dims=[5],
        checkpoint_sha256=sha256_file(path),
        normalizer=normalizer,
    )
    data["urdf_sha256"] = asset_hashes["urdf"]
    data["usd_sha256"] = asset_hashes["usd"]
    return TeacherManifest.from_dict(data, source_directory=tmp_path), state


def test_strict_actor_load_is_deterministic_and_permanently_frozen(tmp_path: Path) -> None:
    manifest, _ = make_artifact_and_manifest(tmp_path)
    policy = TeacherPolicy.from_manifest(manifest)
    observations = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0
    first = policy.act(observations)
    second = policy(observations)
    torch.testing.assert_close(first, second)
    assert first.shape == (3, 29)
    assert not policy.training
    assert all(not parameter.requires_grad for parameter in policy.parameters())

    policy.train(True)
    assert not policy.training
    assert not policy.actor.training
    assert not policy.normalizer.training


@pytest.mark.parametrize("kind", ["missing", "extra", "shape"])
def test_missing_extra_and_wrong_shape_state_are_rejected(tmp_path: Path, kind: str) -> None:
    def mutate(state: dict[str, torch.Tensor]) -> None:
        if kind == "missing":
            state.pop("0.bias")
        elif kind == "extra":
            state["extra"] = torch.zeros(1)
        else:
            state["0.weight"] = torch.zeros((1, 1))

    manifest, _ = make_artifact_and_manifest(tmp_path, mutate_state=mutate)
    with pytest.raises(ValueError, match="state keys|has shape"):
        TeacherPolicy.from_manifest(manifest)


def test_non_tensor_actor_state_is_rejected(tmp_path: Path) -> None:
    manifest, state = make_artifact_and_manifest(tmp_path)
    unsafe_state: dict[str, object] = dict(state)
    unsafe_state["0.bias"] = [0.0] * 5
    path = tmp_path / "teacher.pt"
    torch.save({"actor_state_dict": unsafe_state}, path)
    data = manifest.to_dict()
    data["checkpoint_sha256"] = sha256_file(path)
    strict_manifest = TeacherManifest.from_dict(data, source_directory=tmp_path)
    with pytest.raises(ValueError, match="not a tensor"):
        TeacherPolicy.from_manifest(strict_manifest)


def test_unknown_wrapper_content_is_rejected(tmp_path: Path) -> None:
    manifest, state = make_artifact_and_manifest(tmp_path)
    path = tmp_path / "teacher.pt"
    torch.save({"actor_state_dict": state, "optimizer_state_dict": {}}, path)
    data = manifest.to_dict()
    data["checkpoint_sha256"] = sha256_file(path)
    strict_manifest = TeacherManifest.from_dict(data, source_directory=tmp_path)
    with pytest.raises(ValueError, match="unknown wrapper keys"):
        TeacherPolicy.from_manifest(strict_manifest)


def test_explicit_prefix_is_required_and_strict(tmp_path: Path) -> None:
    manifest, state = make_artifact_and_manifest(tmp_path)
    prefixed = {f"actor.{name}": value for name, value in state.items()}
    path = tmp_path / "teacher.pt"
    torch.save({"actor_state_dict": prefixed}, path)
    data = manifest.to_dict()
    data["checkpoint_sha256"] = sha256_file(path)
    data["state_prefix"] = "actor."
    prefixed_manifest = TeacherManifest.from_dict(data, source_directory=tmp_path)
    policy = TeacherPolicy.from_manifest(prefixed_manifest)
    assert policy(torch.zeros(1, 4)).shape == (1, 29)

    mixed = copy.deepcopy(prefixed)
    mixed["other.weight"] = torch.zeros(1)
    torch.save({"actor_state_dict": mixed}, path)
    data["checkpoint_sha256"] = sha256_file(path)
    mixed_manifest = TeacherManifest.from_dict(data, source_directory=tmp_path)
    with pytest.raises(ValueError, match="outside the declared prefix"):
        TeacherPolicy.from_manifest(mixed_manifest)


def test_empirical_normalizer_uses_std_plus_point_zero_one_and_never_updates(tmp_path: Path) -> None:
    manifest, _ = make_artifact_and_manifest(tmp_path, empirical=True)
    policy = TeacherPolicy.from_manifest(manifest)
    normalizer = policy.normalizer
    observations = torch.tensor([[2.0, 4.0, 7.0, 12.0]])
    before_mean = normalizer.mean.clone()
    before_std = normalizer.std.clone()
    normalized = normalizer(observations)
    expected = (observations - torch.tensor([1.0, 2.0, 3.0, 4.0])) / (
        torch.tensor([1.0, 2.0, 4.0, 8.0]) + 0.01
    )
    torch.testing.assert_close(normalized, expected)
    policy(observations)
    torch.testing.assert_close(normalizer.mean, before_mean)
    torch.testing.assert_close(normalizer.std, before_std)


def test_observation_dimension_dtype_and_finiteness_are_checked(tmp_path: Path) -> None:
    manifest, _ = make_artifact_and_manifest(tmp_path)
    policy = TeacherPolicy.from_manifest(manifest)
    with pytest.raises(ValueError, match="end in 4"):
        policy(torch.zeros(2, 3))
    with pytest.raises(TypeError, match="floating-point"):
        policy(torch.zeros(2, 4, dtype=torch.int64))
    with pytest.raises(ValueError, match="non-finite"):
        policy(torch.full((2, 4), float("nan")))
    with pytest.raises(ValueError, match="policy uses"):
        policy(torch.zeros(2, 4, dtype=torch.float64))


def test_non_finite_checkpoint_tensor_is_rejected(tmp_path: Path) -> None:
    def mutate(state: dict[str, torch.Tensor]) -> None:
        state["0.weight"][0, 0] = float("nan")

    manifest, _ = make_artifact_and_manifest(tmp_path, mutate_state=mutate)
    with pytest.raises(ValueError, match="non-finite"):
        TeacherPolicy.from_manifest(manifest)


def test_checkpoint_hash_is_verified_before_loading(tmp_path: Path) -> None:
    manifest, _ = make_artifact_and_manifest(tmp_path)
    (tmp_path / "teacher.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        TeacherPolicy.from_manifest(manifest)
