from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


fake_isaaclab = types.ModuleType("isaaclab")
fake_managers = types.ModuleType("isaaclab.managers")
fake_managers.CurriculumTermCfg = object
fake_managers.ManagerTermBase = _ManagerTermBase
saved_modules = {name: sys.modules.get(name) for name in ("isaaclab", "isaaclab.managers")}
sys.modules["isaaclab"] = fake_isaaclab
sys.modules["isaaclab.managers"] = fake_managers
try:
    module_path = (
        Path(__file__).parents[1]
        / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/curriculums.py"
    )
    spec = importlib.util.spec_from_file_location("php_kvoy_reproduction_climb_curriculum", module_path)
    curriculums = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(curriculums)
finally:
    for module_name, saved_module in saved_modules.items():
        if saved_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = saved_module


class _EventManager:
    def __init__(self):
        self.cfg = SimpleNamespace(
            params={
                "position_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                "yaw_range": (-0.8, 0.8),
            }
        )

    def get_term_cfg(self, name):
        assert name == "platform_pose"
        return self.cfg

    def set_term_cfg(self, name, cfg):
        assert name == "platform_pose"
        self.cfg = cfg


class _TerminationManager:
    def __init__(self, num_envs):
        self.success = torch.zeros(num_envs, dtype=torch.bool)

    def get_term(self, name):
        assert name == "motion_end_success"
        return self.success


class ClimbBoxPoseCurriculumTest(unittest.TestCase):
    def setUp(self):
        self.params = {
            "event_term_name": "platform_pose",
            "success_term_name": "motion_end_success",
            "full_position_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            "full_yaw_range": (-0.8, 0.8),
            "stage_scales": (0.0, 0.5, 1.0),
            "advance_success_rate": 0.8,
            "regress_success_rate": 0.5,
            "min_evaluated_episodes": 4,
        }
        self.env = SimpleNamespace(
            num_envs=4,
            device="cpu",
            common_step_counter=0,
            event_manager=_EventManager(),
            termination_manager=_TerminationManager(4),
        )
        self.cfg = SimpleNamespace(params=self.params)
        self.term = curriculums.climb_box_pose_curriculum(self.cfg, self.env)

    def _compute(self):
        return self.term(self.env, torch.arange(4), **self.params)

    def test_initial_reset_is_not_counted_and_uses_fixed_pose(self):
        state = self._compute()

        self.assertEqual(state["window_episodes"], 0.0)
        self.assertEqual(self.env.event_manager.cfg.params["position_range"]["x"], (-0.0, 0.0))
        self.assertEqual(self.env.event_manager.cfg.params["yaw_range"], (-0.0, 0.0))

    def test_success_advances_and_failure_regresses_complete_windows(self):
        self.env.common_step_counter = 1
        self.env.termination_manager.success[:] = True
        state = self._compute()
        self.assertEqual(state["stage"], 1.0)
        self.assertEqual(self.env.event_manager.cfg.params["position_range"]["x"], (-0.025, 0.025))

        state = self._compute()
        self.assertEqual(state["stage"], 2.0)
        self.assertEqual(self.env.event_manager.cfg.params["yaw_range"], (-0.8, 0.8))

        self.env.termination_manager.success[:] = False
        state = self._compute()
        self.assertEqual(state["stage"], 1.0)

    def test_checkpoint_state_restores_stage_and_partial_window(self):
        self.env.common_step_counter = 1
        self.env.termination_manager.success[:] = torch.tensor([True, True, True, False])
        self.params["min_evaluated_episodes"] = 8
        self._compute()
        saved = self.term.state_dict()

        restored_env = SimpleNamespace(
            num_envs=4,
            device="cpu",
            common_step_counter=0,
            event_manager=_EventManager(),
            termination_manager=_TerminationManager(4),
        )
        restored = curriculums.climb_box_pose_curriculum(self.cfg, restored_env)
        restored.load_state_dict(saved)

        self.assertEqual(restored.state_dict(), saved)
        self.assertEqual(restored_env.event_manager.cfg.params["position_range"]["x"], (-0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
