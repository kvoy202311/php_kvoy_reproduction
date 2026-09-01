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


class _FakeIterationCommand:
    def __init__(self) -> None:
        self.iterations: list[int] = []
        self.active_skill_updates: list[torch.Tensor] = []

    def set_training_iteration(self, iteration: int) -> None:
        self.iterations.append(iteration)

    def set_student_active_skill_ids(self, skill_ids: torch.Tensor) -> None:
        values = skill_ids.reshape(-1).detach().clone()
        assert values.shape == (3,)
        self.active_skill_updates.append(values)


class _FakeCommandManager:
    def __init__(self, command: _FakeIterationCommand) -> None:
        self.command = command

    def get_term(self, name: str):
        assert name == "multi_skill"
        return self.command


class _FakeEnvironmentWithIteration(_FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.iteration_command = _FakeIterationCommand()
        self.command_manager = _FakeCommandManager(self.iteration_command)
        self.unwrapped = self

    def step(self, actions: torch.Tensor):
        assert len(self.iteration_command.active_skill_updates) == self.step_count + 1
        return super().step(actions)


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
        "option_control": {
            "activation_confirmation_steps": 1,
            "release_confirmation_steps": 1,
            "minimum_skill_duration_steps": {"climb": 1, "down_roll": 1},
            "maximum_skill_duration_steps": {"climb": 3, "down_roll": 3},
            "teacher_forcing_start": 1.0,
            "teacher_forcing_end": 1.0,
            "teacher_forcing_iterations": 1,
        },
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


def test_runner_publishes_applied_student_head_before_every_physics_step() -> None:
    torch.manual_seed(3)
    env = _FakeEnvironmentWithIteration()
    cfg = _train_cfg()
    cfg["environment_iteration_command"] = "multi_skill"
    runner = DistillationRunner(env, cfg, _FakeTeacherRouter(), device="cpu")

    runner.learn(1)

    assert len(env.iteration_command.active_skill_updates) == 2
    assert env.iteration_command.active_skill_updates[0].tolist() == [0, 1, 2]
    assert env.iteration_command.active_skill_updates[1].tolist() == [0, 1, 2]


def test_inference_uses_autonomous_selector_and_hard_action_head() -> None:
    runner = DistillationRunner(
        _FakeEnvironment(), _train_cfg(), _FakeTeacherRouter(), device="cpu"
    )
    with torch.no_grad():
        runner.policy.actor.selector.weight.zero_()
        runner.policy.actor.selector.bias.copy_(torch.tensor([0.0, 10.0, 0.0]))
        for skill_id, head in enumerate(runner.policy.actor.action_heads):
            head.weight.zero_()
            head.bias.fill_(float(skill_id + 1))
    inference = runner.get_inference_policy(device="cpu")
    raw_observations, _ = runner.env.get_observations()
    actions = inference(raw_observations)
    assert actions.eq(2.0).all()
    assert inference.active_skill_ids.eq(1).all()
    inference.reset(torch.tensor([True, False, False]))
    assert inference.active_skill_ids[:, 0].tolist() == [0, 1, 1]


def test_inference_fixed_skill_matches_atomic_teacher_forced_route_from_first_step() -> None:
    runner = DistillationRunner(
        _FakeEnvironment(), _train_cfg(), _FakeTeacherRouter(), device="cpu"
    )
    with torch.no_grad():
        # Deliberately make the autonomous selector disagree with the fixed
        # climb route.  Atomic-aligned inference must still execute climb from
        # the first step, including immediately after a partial episode reset.
        runner.policy.actor.selector.weight.zero_()
        runner.policy.actor.selector.bias.copy_(torch.tensor([10.0, 0.0, 0.0]))
        for skill_id, head in enumerate(runner.policy.actor.action_heads):
            head.weight.zero_()
            head.bias.fill_(float(skill_id + 1))

    inference = runner.get_inference_policy(device="cpu", fixed_skill_id=1)
    raw_observations, _ = runner.env.get_observations()
    assert inference(raw_observations).eq(2.0).all()
    assert inference.active_skill_ids.eq(1).all()

    inference.reset(torch.tensor([True, False, False]))
    assert inference(raw_observations).eq(2.0).all()
    assert inference.active_skill_ids.eq(1).all()


def test_inference_publishes_the_same_hard_route_that_selects_actions() -> None:
    env = _FakeEnvironmentWithIteration()
    cfg = _train_cfg()
    cfg["environment_iteration_command"] = "multi_skill"
    runner = DistillationRunner(env, cfg, _FakeTeacherRouter(), device="cpu")
    with torch.no_grad():
        runner.policy.actor.selector.weight.zero_()
        runner.policy.actor.selector.bias.copy_(torch.tensor([0.0, 10.0, 0.0]))
    inference = runner.get_inference_policy(device="cpu")
    raw_observations, _ = env.get_observations()

    inference(raw_observations)

    assert env.iteration_command.active_skill_updates[-1].tolist() == [1, 1, 1]


@pytest.mark.parametrize("fixed_skill_id", [True, -1, 3])
def test_inference_rejects_invalid_fixed_skill_id(fixed_skill_id) -> None:
    runner = DistillationRunner(
        _FakeEnvironment(), _train_cfg(), _FakeTeacherRouter(), device="cpu"
    )
    expected_error = TypeError if isinstance(fixed_skill_id, bool) else ValueError
    with pytest.raises(expected_error):
        runner.get_inference_policy(device="cpu", fixed_skill_id=fixed_skill_id)


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


def test_checkpoint_stage_allows_same_resume_and_only_adjacent_warm_start(tmp_path) -> None:
    atomic = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        training_stage="atomic",
    )
    atomic_checkpoint = tmp_path / "atomic.pt"
    atomic.save(atomic_checkpoint)

    atomic_resume = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        training_stage="atomic",
    )
    atomic_resume.load(atomic_checkpoint)
    assert atomic_resume.loaded_training_stage == "atomic"

    transition = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        training_stage="transition",
    )
    transition.load(atomic_checkpoint, load_optimizer=False)
    transition_checkpoint = tmp_path / "transition.pt"
    transition.save(transition_checkpoint)

    full = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        training_stage="full",
    )
    full.load(transition_checkpoint, load_optimizer=False)
    with pytest.raises(ValueError, match="immediately preceding"):
        full.load(atomic_checkpoint, load_optimizer=False)


def test_checkpoint_stage_rejects_cross_stage_resume_and_unverifiable_source(tmp_path) -> None:
    atomic = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        training_stage="atomic",
    )
    atomic_checkpoint = tmp_path / "atomic.pt"
    atomic.save(atomic_checkpoint)

    transition = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        training_stage="transition",
    )
    with pytest.raises(ValueError, match="same curriculum stage"):
        transition.load(atomic_checkpoint)

    unspecified = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
    )
    unspecified_checkpoint = tmp_path / "unspecified.pt"
    unspecified.save(unspecified_checkpoint)
    with pytest.raises(ValueError, match="does not record"):
        transition.load(unspecified_checkpoint, load_optimizer=False)


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


@pytest.mark.parametrize(
    "load_kwargs",
    ({}, {"load_optimizer": False}, {"load_optimizer": False, "restore_training_state": True}),
)
def test_checkpoint_always_rejects_changed_policy_input_contract(tmp_path, load_kwargs) -> None:
    source = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        policy_input_contract={"camera_parent": "head", "vfov": 58.0},
    )
    checkpoint = tmp_path / "model_0.pt"
    source.save(checkpoint)
    target = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        policy_input_contract={"camera_parent": "torso", "vfov": 58.0},
    )
    with pytest.raises(ValueError, match="Student camera"):
        target.load(checkpoint, **load_kwargs)


def test_checkpoint_rejects_changed_deployment_option_semantics(tmp_path) -> None:
    source = DistillationRunner(
        _FakeEnvironment(), _train_cfg(), _FakeTeacherRouter(), device="cpu"
    )
    checkpoint = tmp_path / "model_0.pt"
    source.save(checkpoint)
    changed_cfg = _train_cfg()
    changed_cfg["option_control"]["activation_confirmation_steps"] = 2
    target = DistillationRunner(
        _FakeEnvironment(), changed_cfg, _FakeTeacherRouter(), device="cpu"
    )
    with pytest.raises(ValueError, match="Option-controller semantics"):
        target.load(checkpoint, load_optimizer=False)


@pytest.mark.parametrize("remove_fingerprint_key", [False, True])
def test_checkpoint_without_policy_input_contract_is_rejected(
    tmp_path,
    remove_fingerprint_key: bool,
) -> None:
    source = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        policy_input_contract={"camera_parent": "head"},
    )
    checkpoint = tmp_path / "legacy.pt"
    source.save(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if remove_fingerprint_key:
        payload.pop("policy_input_contract_fingerprint")
    else:
        payload["policy_input_contract_fingerprint"] = None
    torch.save(payload, checkpoint)

    target = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
        policy_input_contract={"camera_parent": "head"},
    )
    with pytest.raises(ValueError, match="does not contain a Student policy-input contract"):
        target.load(checkpoint, load_optimizer=False)


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


def test_checkpoint_inference_restores_iteration_without_optimizer(tmp_path) -> None:
    source = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
    )
    source.current_learning_iteration = 10_001
    source.tot_timesteps = 123_456
    checkpoint = tmp_path / "model_10000.pt"
    source.save(checkpoint)

    inference_cfg = _train_cfg()
    inference_cfg["environment_iteration_command"] = "multi_skill"
    inference_env = _FakeEnvironmentWithIteration()
    inference = DistillationRunner(
        inference_env,
        inference_cfg,
        _FakeTeacherRouter(),
        device="cpu",
    )
    inference.load(
        checkpoint,
        load_optimizer=False,
        restore_training_state=True,
    )

    assert inference.current_learning_iteration == 10_001
    assert inference.tot_timesteps == 123_456
    assert inference.alg.learning_rate == 1.0e-4
    assert inference_env.iteration_command.iterations == [10_000]


def test_checkpoint_rejects_optimizer_without_training_state(tmp_path) -> None:
    source = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
    )
    checkpoint = tmp_path / "model_0.pt"
    source.save(checkpoint)
    target = DistillationRunner(
        _FakeEnvironment(),
        _train_cfg(),
        _FakeTeacherRouter(),
        device="cpu",
    )
    with pytest.raises(ValueError, match="optimizer requires"):
        target.load(checkpoint, load_optimizer=True, restore_training_state=False)


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
