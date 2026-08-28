from __future__ import annotations

from types import SimpleNamespace

from php_kvoy_reproduction.distillation.training_contract import (
    contract_fingerprint,
    environment_training_contract,
    motion_directory_contract,
    student_policy_input_contract,
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


class _DictCfg:
    def __init__(self, value):
        self.value = value

    def to_dict(self):
        return self.value


def _policy_cfg(*, parent: str = "head", noise_enabled: bool = True):
    return SimpleNamespace(
        scene=SimpleNamespace(
            depth_camera=_DictCfg(
                {
                    "prim_path": f"/Robot/{parent}/DepthCamera",
                    "width": 87,
                    "height": 58,
                    "vfov": 58.0,
                }
            )
        ),
        observations=SimpleNamespace(
            policy=_DictCfg(
                {
                    "proprio_history": {"history_length": 8},
                    "depth": {
                        "near_clip": 0.15,
                        "far_clip": 2.0,
                        "noise_enabled": noise_enabled,
                        "pixel_noise_std": 0.03 if noise_enabled else 0.0,
                        "delay_range_s": (0.06, 0.08) if noise_enabled else (0.0, 0.0),
                        "image_offset_range": (-0.03, 0.03) if noise_enabled else (0.0, 0.0),
                    },
                }
            )
        ),
        actions=_DictCfg({"joint_pos": {"scale": 0.25, "joint_order": ["a", "b"]}}),
        sim=SimpleNamespace(dt=0.005),
        decimation=4,
    )


def test_policy_input_contract_ignores_curriculum_augmentation_only() -> None:
    noisy = student_policy_input_contract(_policy_cfg(noise_enabled=True), task="Task-v0")
    clean = student_policy_input_contract(_policy_cfg(noise_enabled=False), task="Task-v0")
    different_camera = student_policy_input_contract(
        _policy_cfg(parent="torso", noise_enabled=False),
        task="Task-v0",
    )
    assert contract_fingerprint(noisy) == contract_fingerprint(clean)
    assert contract_fingerprint(noisy) != contract_fingerprint(different_camera)
    assert noisy["actor_command_semantics"] == (
        "bounded_live_requested_world_velocity_body_frame_motion_lock_v4"
    )
