from __future__ import annotations

import pytest
import torch

from php_kvoy_reproduction.distillation.runner import DistillationRunner, _cpu_cuda_rng_states
from php_kvoy_reproduction.distillation.teacher_router import TeacherBatch


class _FakeEnvironment:
    num_envs = 3
    num_actions = 29
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.step_count = 0
        self.stepped_actions: list[torch.Tensor] = []

    def _pack(self):
        actor = torch.full((self.num_envs, 231), 0.01 * self.step_count)
        routes = torch.arange(self.num_envs, dtype=torch.float32).unsqueeze(1)
        groups = {
            "critic": torch.full((self.num_envs, 5), 0.02 * self.step_count),
            "skill_id": routes,
            "teacher_valid": torch.ones(self.num_envs, 1),
            "locomotion_teacher": torch.zeros(self.num_envs, 3),
            "motion_teacher": torch.zeros(self.num_envs, 4),
        }
        return actor, {"observations": groups}

    def get_observations(self):
        return self._pack()

    def step(self, actions: torch.Tensor):
        assert actions.shape == (self.num_envs, self.num_actions)
        # The teacher below returns exactly 5.0.  A regression that advances
        # physics with labels instead of the sampled student action fails here.
        assert not torch.equal(actions, torch.full_like(actions, 5.0))
        self.stepped_actions.append(actions.detach().clone())
        self.step_count += 1
        actor, extras = self._pack()
        rewards = torch.ones(self.num_envs)
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        return actor, rewards, dones, extras


class _FakeTeacherRouter:
    skill_names = ("locomotion", "climb", "down_roll")

    def __init__(self, fingerprints: dict[str, str] | None = None) -> None:
        self._fingerprints = fingerprints or {
            "locomotion": "locomotion-test-teacher",
            "climb": "climb-test-teacher",
            "down_roll": "down-roll-test-teacher",
        }

    def eval(self):
        return self

    def fingerprints(self):
        return dict(self._fingerprints)

    def act(self, teacher_observations, skill_ids, validity_mask=None):
        assert teacher_observations["locomotion"].shape == (3, 3)
        assert teacher_observations["climb"].shape == (3, 4)
        assert teacher_observations["down_roll"].shape == (3, 4)
        return TeacherBatch(
            actions=torch.full((3, 29), 5.0),
            valid_mask=validity_mask.clone(),
            skill_ids=skill_ids.clone(),
        )


def _train_cfg() -> dict:
    return {
        "device": "cpu",
        "num_steps_per_env": 2,
        "empirical_normalization": False,
        "policy": {
            "class_name": "VisionActorCritic",
            "init_noise_std": 0.01,
            "noise_std_type": "log",
            "actor_hidden_dims": [16],
            "critic_hidden_dims": [8],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "DAggerPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "learning_rate": 1.0e-4,
            "schedule": "fixed",
            "desired_kl": None,
            "skill_names": ("locomotion", "climb", "down_roll"),
        },
        "save_interval": 10,
        "log_interval": 1,
        "logger": "tensorboard",
        "observation_layout": {
            "proprio_frame_dim": 2,
            "proprio_history_length": 2,
            "command_dim": 2,
            "depth_height": 15,
            "depth_width": 15,
        },
    }


def test_runner_advances_fake_environment_only_with_student_actions() -> None:
    torch.manual_seed(3)
    env = _FakeEnvironment()
    runner = DistillationRunner(env, _train_cfg(), _FakeTeacherRouter(), device="cpu")
    runner.learn(1)

    assert len(env.stepped_actions) == 2
    assert runner.current_learning_iteration == 1
    assert runner.tot_timesteps == 6
    assert all(torch.isfinite(actions).all() for actions in env.stepped_actions)
    assert all(not torch.equal(actions, torch.full_like(actions, 5.0)) for actions in env.stepped_actions)


def test_checkpoint_restores_absolute_iteration_and_adapted_learning_rate(tmp_path) -> None:
    train_cfg = _train_cfg()
    train_cfg["algorithm"]["schedule"] = "adaptive"
    train_cfg["algorithm"]["desired_kl"] = 0.01
    source = DistillationRunner(
        _FakeEnvironment(),
        train_cfg,
        _FakeTeacherRouter(),
        device="cpu",
    )
    source.current_learning_iteration = 4_321
    source.tot_timesteps = 123_456
    source.alg.learning_rate = 7.5e-5
    for parameter_group in source.alg.optimizer.param_groups:
        parameter_group["lr"] = source.alg.learning_rate

    checkpoint = tmp_path / "model_4320.pt"
    source.save(checkpoint)

    resumed = DistillationRunner(
        _FakeEnvironment(),
        train_cfg,
        _FakeTeacherRouter(),
        device="cpu",
    )
    resumed.load(checkpoint)

    assert resumed.current_learning_iteration == 4_321
    assert resumed.tot_timesteps == 123_456
    assert resumed.alg.learning_rate == 7.5e-5
    assert {group["lr"] for group in resumed.alg.optimizer.param_groups} == {7.5e-5}


def test_checkpoint_rejects_changed_teacher_fingerprints(tmp_path) -> None:
    source = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
    )
    checkpoint = tmp_path / "model_0.pt"
    source.save(checkpoint)

    changed_fingerprints = source.teacher_router.fingerprints()
    changed_fingerprints["climb"] = "different-climb-teacher"
    resumed = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(changed_fingerprints),
        device="cpu",
    )

    with pytest.raises(ValueError, match="Frozen teachers differ"):
        resumed.load(checkpoint)


def test_checkpoint_rejects_changed_environment_contract_on_resume(tmp_path) -> None:
    source = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        environment_contract={"motion_sha256": "a", "termination_scale": 2.0},
    )
    checkpoint = tmp_path / "model_0.pt"
    source.save(checkpoint)
    resumed = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        environment_contract={"motion_sha256": "b", "termination_scale": 2.0},
    )
    with pytest.raises(ValueError, match="semantics differ"):
        resumed.load(checkpoint)


def test_warm_start_allows_intentional_environment_contract_change(tmp_path) -> None:
    source = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        environment_contract={"version": 1},
    )
    checkpoint = tmp_path / "model_0.pt"
    source.save(checkpoint)
    warm_started = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        environment_contract={"version": 2},
    )
    warm_started.load(checkpoint, load_optimizer=False)


def test_checkpoint_warm_start_loads_student_but_resets_training_state(tmp_path) -> None:
    source = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
    )
    with torch.no_grad():
        next(source.policy.parameters()).fill_(0.125)
    source.current_learning_iteration = 321
    source.tot_timesteps = 9_876
    source.alg.learning_rate = 7.5e-5
    for parameter_group in source.alg.optimizer.param_groups:
        parameter_group["lr"] = source.alg.learning_rate
    checkpoint = tmp_path / "model_320.pt"
    source.save(checkpoint)

    warm_started = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
    )
    warm_started.load(checkpoint, load_optimizer=False)

    for expected, actual in zip(source.policy.parameters(), warm_started.policy.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)
    assert warm_started.current_learning_iteration == 0
    assert warm_started.tot_timesteps == 0
    assert warm_started.tot_time == 0.0
    assert warm_started.alg.learning_rate == 1.0e-4
    assert {group["lr"] for group in warm_started.alg.optimizer.param_groups} == {1.0e-4}


def test_serialized_cuda_rng_states_are_restored_as_cpu_byte_tensors() -> None:
    serialized = [
        torch.arange(8, dtype=torch.uint8),
        torch.arange(8, dtype=torch.uint8) + 1,
    ]
    restored = _cpu_cuda_rng_states(serialized)
    assert all(state.device.type == "cpu" and state.dtype == torch.uint8 for state in restored)
    assert all(state.ndim == 1 for state in restored)
    with pytest.raises(TypeError, match="one-dimensional"):
        _cpu_cuda_rng_states([torch.zeros(2, 2, dtype=torch.uint8)])
    with pytest.raises(TypeError, match="non-empty"):
        _cpu_cuda_rng_states([])


@pytest.mark.parametrize("invalid_learning_rate", [0.0, -1.0e-4, float("nan"), float("inf")])
def test_checkpoint_rejects_invalid_optimizer_learning_rate(
    tmp_path,
    invalid_learning_rate: float,
) -> None:
    source = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
    )
    checkpoint = tmp_path / "model_0.pt"
    source.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["optimizer_state_dict"]["param_groups"][0]["lr"] = invalid_learning_rate
    torch.save(payload, checkpoint)

    resumed = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
    )
    with pytest.raises(ValueError, match="finite and positive"):
        resumed.load(checkpoint)
