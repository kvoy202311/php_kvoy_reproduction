from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_platform_foot_support_module():
    """Load the pure load-gating helper without requiring Isaac Sim."""

    stubs = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.assets": types.ModuleType("isaaclab.assets"),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.sensors": types.ModuleType("isaaclab.sensors"),
        "php_kvoy_reproduction.tasks.tracking.mdp.obstacle": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.obstacle"
        ),
        "php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry"
        ),
    }
    stubs["isaaclab.assets"].Articulation = object
    stubs["isaaclab.assets"].RigidObject = object
    stubs["isaaclab.managers"].SceneEntityCfg = object
    stubs["isaaclab.sensors"].ContactSensor = object
    obstacle = stubs["php_kvoy_reproduction.tasks.tracking.mdp.obstacle"]
    obstacle.get_climb_box_sizes = lambda *_args, **_kwargs: None
    geometry = stubs["php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry"]
    geometry.filtered_platform_force_score = lambda *_args, **_kwargs: None
    geometry.foot_sole_corners_world = lambda *_args, **_kwargs: None
    geometry.foothold_safety_score = lambda *_args, **_kwargs: None

    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        path = (
            Path(__file__).parents[1]
            / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/platform_foot_support.py"
        )
        spec = importlib.util.spec_from_file_location("php_kvoy_reproduction_platform_foot_support", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, previous in saved.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


platform_foot_support = _load_platform_foot_support_module()


class PlatformFootLoadScoreTest(unittest.TestCase):
    def _robot_with_cpu_default_mass(self, default_mass: torch.Tensor | None = None):
        if default_mass is None:
            default_mass = torch.tensor([4.0, 6.0], device="cpu")
        return SimpleNamespace(data=SimpleNamespace(default_mass=default_mass))

    def test_computes_expected_score_with_cpu_mass(self):
        state = SimpleNamespace(upward_forces=torch.tensor([[10.0, 20.0], [30.0, 30.0]]))

        score = platform_foot_support.platform_foot_load_score(
            state,
            self._robot_with_cpu_default_mass(),
            min_total_load_fraction=0.5,
        )

        torch.testing.assert_close(score, torch.tensor([30.0 / 49.05, 1.0]))

    def test_computes_expected_score_with_per_environment_cpu_masses(self):
        state = SimpleNamespace(upward_forces=torch.tensor([[10.0, 20.0], [30.0, 30.0]]))
        robot = self._robot_with_cpu_default_mass(torch.tensor([[4.0, 6.0], [8.0, 12.0]], device="cpu"))

        score = platform_foot_support.platform_foot_load_score(
            state,
            robot,
            min_total_load_fraction=0.5,
        )

        torch.testing.assert_close(score, torch.tensor([30.0 / 49.05, 60.0 / 98.1]))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA to exercise the CPU/CUDA boundary")
    def test_moves_cpu_default_mass_to_contact_force_device(self):
        state = SimpleNamespace(
            upward_forces=torch.tensor([[10.0, 20.0], [30.0, 30.0]], device="cuda:0")
        )

        score = platform_foot_support.platform_foot_load_score(
            state,
            self._robot_with_cpu_default_mass(),
            min_total_load_fraction=0.5,
        )

        self.assertEqual(score.device, state.upward_forces.device)
        torch.testing.assert_close(score.cpu(), torch.tensor([30.0 / 49.05, 1.0]))


if __name__ == "__main__":
    unittest.main()
