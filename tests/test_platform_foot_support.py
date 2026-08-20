from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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
    geometry.sole_surface_alignment_score = lambda *_args, **_kwargs: None
    geometry.sole_surface_shaping_score = lambda *_args, **_kwargs: None

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


class PlatformFootSupportStateTest(unittest.TestCase):
    @staticmethod
    def _params():
        return {
            "foot_body_names": ("left", "right"),
            "platform_contact_sensor_names": ("left_sensor", "right_sensor"),
            "sole_corners_b": ((-0.1, -0.04, 0.0), (-0.1, 0.04, 0.0), (0.1, -0.04, 0.0), (0.1, 0.04, 0.0)),
            "approach_side": -1.0,
            "max_heel_overhang": 0.05,
            "min_forefoot_inside": 0.04,
            "far_edge_margin": 0.04,
            "lateral_margin": 0.02,
            "surface_tilt_scale": 0.05,
            "surface_height_scale": 0.06,
        }

    @staticmethod
    def _state(upward_force: float):
        platform = SimpleNamespace(
            device="cpu",
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.0, 0.0, 0.5]]),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            ),
        )
        sensors = {}
        for name in ("left_sensor", "right_sensor"):
            sensors[name] = SimpleNamespace(
                data=SimpleNamespace(
                    force_matrix_w=torch.tensor([[[[0.0, 0.0, upward_force]]]])
                )
            )
        class _Scene:
            def __init__(self):
                self.sensors = sensors

            def __getitem__(self, name):
                return {"platform": platform}[name]

        env = SimpleNamespace(num_envs=1, scene=_Scene())
        robot = SimpleNamespace(
            body_names=["left", "right"],
            data=SimpleNamespace(
                body_pos_w=torch.zeros(1, 2, 3),
                body_quat_w=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]]),
            ),
        )
        sole_corners = torch.zeros(1, 2, 4, 3)
        sole_corners[..., 2] = 1.0
        with (
            patch.object(platform_foot_support, "get_climb_box_sizes", return_value=torch.ones(1, 3)),
            patch.object(platform_foot_support, "foot_sole_corners_world", return_value=sole_corners),
            patch.object(
                platform_foot_support,
                "foothold_safety_score",
                return_value=(torch.ones(1, 2), torch.tensor([[False, True]])),
            ),
            patch.object(
                platform_foot_support,
                "sole_surface_alignment_score",
                return_value=(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bool), torch.zeros(1, 2)),
            ),
            patch.object(
                platform_foot_support,
                "sole_surface_shaping_score",
                return_value=(torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2)),
            ),
        ):
            return platform_foot_support.platform_foot_support_state(
                env,
                robot,
                "cpu",
                SimpleNamespace(name="platform"),
                (1.0, 1.0, 1.0),
                PlatformFootSupportStateTest._params(),
                min_upward_force=10.0,
                sole_height_tolerance=0.03,
            )

    def test_unsafe_top_contact_can_complete_handoff_but_not_strict_support(self):
        state = self._state(20.0)

        self.assertEqual(state.contact_support.tolist(), [[True, True]])
        self.assertEqual(state.active_support.tolist(), [[False, True]])

    def test_flat_hovering_without_platform_force_is_not_support(self):
        state = self._state(0.0)

        self.assertEqual(state.contact_support.tolist(), [[False, False]])
        self.assertEqual(state.active_support.tolist(), [[False, False]])

    def test_stale_contact_time_cannot_reward_an_unsafe_foot(self):
        state = self._state(20.0)
        with patch.object(
            platform_foot_support,
            "filtered_platform_force_score",
            return_value=torch.ones(1, 2),
        ):
            score = platform_foot_support.platform_foot_support_score(
                state,
                torch.ones(1, 2),
                min_upward_force=10.0,
                contact_time_scale=0.06,
                sole_height_tolerance=0.03,
            )

        torch.testing.assert_close(score, torch.tensor([[0.0, 1.0]]))


if __name__ == "__main__":
    unittest.main()
