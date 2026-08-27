from __future__ import annotations

from php_kvoy_reproduction.distillation.training_contract import (
    contract_fingerprint,
    environment_training_contract,
    motion_directory_contract,
)


class _Cfg:
    def __init__(self, num_envs: int, reward_weight: float) -> None:
        self.num_envs = num_envs
        self.reward_weight = reward_weight

    def to_dict(self):
        return {
            "scene": {"num_envs": self.num_envs, "robot": {"device": "cuda:0"}},
            "seed": 42,
            "viewer": {"eye": [1.0, 2.0, 3.0]},
            "reward": {"weight": self.reward_weight},
            "commands": {"climb_motion_dir": "/machine/specific/path"},
        }


def _motions(tmp_path):
    climb = tmp_path / "climb"
    down = tmp_path / "down"
    climb.mkdir()
    down.mkdir()
    (climb / "a.npz").write_bytes(b"climb")
    (down / "b.npz").write_bytes(b"down")
    return climb, down


def test_environment_contract_ignores_operational_fields_but_tracks_semantics(tmp_path) -> None:
    climb, down = _motions(tmp_path)
    first = environment_training_contract(
        _Cfg(2048, 1.0),
        task="Distillation-Test-v0",
        climb_motion_dir=climb,
        down_roll_motion_dir=down,
    )
    resized = environment_training_contract(
        _Cfg(64, 1.0),
        task="Distillation-Test-v0",
        climb_motion_dir=climb,
        down_roll_motion_dir=down,
    )
    changed = environment_training_contract(
        _Cfg(64, 2.0),
        task="Distillation-Test-v0",
        climb_motion_dir=climb,
        down_roll_motion_dir=down,
    )
    assert contract_fingerprint(first) == contract_fingerprint(resized)
    assert contract_fingerprint(first) != contract_fingerprint(changed)


def test_motion_contract_tracks_content_not_absolute_directory(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "clip.npz").write_bytes(b"same-motion")
    (second / "clip.npz").write_bytes(b"same-motion")
    assert motion_directory_contract(first) == motion_directory_contract(second)
    (second / "clip.npz").write_bytes(b"changed-motion")
    assert motion_directory_contract(first) != motion_directory_contract(second)
