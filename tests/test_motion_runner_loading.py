from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rsl_rl.runners.on_policy_runner import OnPolicyRunner


_REPO_ROOT = Path(__file__).parents[1]
_SOURCE_ROOT = _REPO_ROOT / "source/php_kvoy_reproduction"
_RUNNER_PATH = _SOURCE_ROOT / "php_kvoy_reproduction/utils/my_on_policy_runner.py"
_PROGRESS_PATH = _SOURCE_ROOT / "php_kvoy_reproduction/utils/checkpoint_progress.py"


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_runner_module():
    """Load the runner without importing the Isaac Sim application stack."""

    package_names = ("php_kvoy_reproduction", "php_kvoy_reproduction.utils")
    saved_modules = {name: sys.modules.get(name) for name in package_names}
    saved_isaaclab_rl = sys.modules.get("isaaclab_rl")
    saved_isaaclab_rsl_rl = sys.modules.get("isaaclab_rl.rsl_rl")
    saved_wandb = sys.modules.get("wandb")
    saved_exporter = sys.modules.get("php_kvoy_reproduction.utils.exporter")
    saved_progress = sys.modules.get("php_kvoy_reproduction.utils.checkpoint_progress")
    try:
        for name in package_names:
            package = types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package

        isaaclab_rl = types.ModuleType("isaaclab_rl")
        isaaclab_rsl_rl = types.ModuleType("isaaclab_rl.rsl_rl")
        isaaclab_rsl_rl.export_policy_as_onnx = lambda *args, **kwargs: None
        isaaclab_rl.rsl_rl = isaaclab_rsl_rl
        sys.modules["isaaclab_rl"] = isaaclab_rl
        sys.modules["isaaclab_rl.rsl_rl"] = isaaclab_rsl_rl
        sys.modules["wandb"] = types.ModuleType("wandb")

        exporter = types.ModuleType("php_kvoy_reproduction.utils.exporter")
        exporter.attach_onnx_metadata = lambda *args, **kwargs: None
        exporter.export_motion_policy_as_onnx = lambda *args, **kwargs: None
        sys.modules["php_kvoy_reproduction.utils.exporter"] = exporter
        _load_module_from_path("php_kvoy_reproduction.utils.checkpoint_progress", _PROGRESS_PATH)
        return _load_module_from_path("motion_runner_under_test", _RUNNER_PATH)
    finally:
        restore = {
            **saved_modules,
            "isaaclab_rl": saved_isaaclab_rl,
            "isaaclab_rl.rsl_rl": saved_isaaclab_rsl_rl,
            "wandb": saved_wandb,
            "php_kvoy_reproduction.utils.exporter": saved_exporter,
            "php_kvoy_reproduction.utils.checkpoint_progress": saved_progress,
        }
        for name, module in restore.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


runner_module = _load_runner_module()


class _StateRecorder:
    def __init__(self):
        self.loaded = []

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        self.loaded.append(state)


class _PolicyStateRecorder(_StateRecorder):
    def load_state_dict(self, state):
        super().load_state_dict(state)
        return True


class MotionRunnerLoadingTest(unittest.TestCase):
    def test_current_rsl_rl_load_without_optimizer_loads_model_and_normalizers(self):
        upstream_runner = OnPolicyRunner.__new__(OnPolicyRunner)
        policy = _PolicyStateRecorder()
        optimizer = _StateRecorder()
        obs_normalizer = _StateRecorder()
        privileged_obs_normalizer = _StateRecorder()
        upstream_runner.alg = SimpleNamespace(policy=policy, rnd=False, optimizer=optimizer)
        upstream_runner.empirical_normalization = True
        upstream_runner.obs_normalizer = obs_normalizer
        upstream_runner.privileged_obs_normalizer = privileged_obs_normalizer
        upstream_runner.current_learning_iteration = 0
        checkpoint = {
            "model_state_dict": {"actor_and_critic": "weights"},
            "obs_norm_state_dict": {"actor": "normalizer"},
            "privileged_obs_norm_state_dict": {"critic": "normalizer"},
            "optimizer_state_dict": {"optimizer": "old"},
            "iter": 37,
            "infos": {},
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "model.pt"
            torch.save(checkpoint, checkpoint_path)
            OnPolicyRunner.load(upstream_runner, str(checkpoint_path), load_optimizer=False)

        self.assertEqual(policy.loaded, [{"actor_and_critic": "weights"}])
        self.assertEqual(obs_normalizer.loaded, [{"actor": "normalizer"}])
        self.assertEqual(privileged_obs_normalizer.loaded, [{"critic": "normalizer"}])
        self.assertEqual(optimizer.loaded, [])
        # The custom MotionOnPolicyRunner explicitly resets this value to zero
        # after the upstream call when it performs a warm start.
        self.assertEqual(upstream_runner.current_learning_iteration, 37)

    def setUp(self):
        self.sampler = _StateRecorder()
        self.curriculum = _StateRecorder()
        term_cfg = SimpleNamespace(func=self.curriculum)
        curriculum_manager = SimpleNamespace(_term_names=["platform_pose"], _term_cfgs=[term_cfg])
        command_manager = SimpleNamespace(get_term=lambda _name: SimpleNamespace(motion_sampler=self.sampler))
        unwrapped = SimpleNamespace(
            curriculum_manager=curriculum_manager,
            command_manager=command_manager,
        )
        self.runner = runner_module.MotionOnPolicyRunner.__new__(runner_module.MotionOnPolicyRunner)
        self.runner.env = SimpleNamespace(unwrapped=unwrapped)
        self.runner.current_learning_iteration = 0

    def _upstream_load(self, iteration: int, infos: dict):
        def load(runner, path, load_optimizer=True):
            runner.current_learning_iteration = iteration
            return infos

        return load

    def test_policy_only_resets_iteration_and_skips_training_control_state(self):
        infos = {
            runner_module._MOTION_SAMPLER_CHECKPOINT_KEY: {"sampler": "old"},
            runner_module._CURRICULUM_CHECKPOINT_KEY: {"platform_pose": {"stage": 3}},
        }
        with patch.object(OnPolicyRunner, "load", new=self._upstream_load(123, infos)):
            self.runner.load_policy_only("model.pt")

        self.assertEqual(self.runner.current_learning_iteration, 0)
        self.assertEqual(self.sampler.loaded, [])
        self.assertEqual(self.curriculum.loaded, [])

    def test_full_resume_still_restores_iteration_sampler_and_curriculum(self):
        infos = {
            runner_module.RUNNER_PROGRESS_CHECKPOINT_KEY: {
                "version": 1,
                "last_completed_iteration": 14,
                "next_iteration": 15,
            },
            runner_module._MOTION_SAMPLER_CHECKPOINT_KEY: {"sampler": "saved"},
            runner_module._CURRICULUM_CHECKPOINT_KEY: {"platform_pose": {"stage": 2}},
        }
        with patch.object(OnPolicyRunner, "load", new=self._upstream_load(14, infos)):
            self.runner.load("model.pt", load_optimizer=True)

        self.assertEqual(self.runner.current_learning_iteration, 15)
        self.assertEqual(self.sampler.loaded, [{"sampler": "saved"}])
        self.assertEqual(self.curriculum.loaded, [{"stage": 2}])


if __name__ == "__main__":
    unittest.main()
