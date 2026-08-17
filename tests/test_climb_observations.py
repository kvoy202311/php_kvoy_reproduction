from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


def _load_observations_module():
    stubs = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.assets": types.ModuleType("isaaclab.assets"),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.sensors": types.ModuleType("isaaclab.sensors"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": types.ModuleType("isaaclab.utils.math"),
        "php_kvoy_reproduction.tasks.tracking.mdp.commands": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.commands"
        ),
        "php_kvoy_reproduction.tasks.tracking.mdp.obstacle": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.obstacle"
        ),
    }
    stubs["isaaclab.assets"].RigidObject = object
    stubs["isaaclab.managers"].SceneEntityCfg = object
    stubs["isaaclab.sensors"].RayCaster = object
    stubs["isaaclab.utils.math"].matrix_from_quat = lambda *_: None
    stubs["isaaclab.utils.math"].subtract_frame_transforms = lambda *_: (None, None)
    stubs["php_kvoy_reproduction.tasks.tracking.mdp.commands"].MotionCommand = object
    obstacle = stubs["php_kvoy_reproduction.tasks.tracking.mdp.obstacle"]
    obstacle.climb_box_top_height = lambda *_args, **_kwargs: (None, None)
    obstacle.get_climb_box_sizes = lambda *_args, **_kwargs: None

    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        path = (
            Path(__file__).parents[1]
            / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/observations.py"
        )
        spec = importlib.util.spec_from_file_location("php_kvoy_reproduction_climb_observations", path)
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


observations = _load_observations_module()


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "motion"
        return self.command


class FirstFootholdHeightOffsetObservationTest(unittest.TestCase):
    def _env(self, offsets=None):
        command = SimpleNamespace(
            cfg=SimpleNamespace(body_names=["torso", "left_foot", "right_foot"]),
            time_steps=torch.zeros(2, dtype=torch.long),
            joint_pos=torch.zeros(2, 3),
        )
        if offsets is not None:
            command.first_foothold_height_offsets = torch.tensor(offsets, dtype=torch.float32)
        return SimpleNamespace(num_envs=2, command_manager=_CommandManager(command))

    def test_returns_exact_configured_foot_offsets_in_requested_order(self):
        env = self._env([[0.0, 0.0914, 0.0], [0.0, 0.0, -0.0977]])

        output = observations.first_foothold_height_offsets(
            env,
            command_name="motion",
            foot_body_names=("right_foot", "left_foot"),
        )

        torch.testing.assert_close(output, torch.tensor([[0.0, 0.0914], [-0.0977, 0.0]]))

    def test_generic_command_without_alignment_returns_fixed_shape_zeros(self):
        env = self._env()

        output = observations.first_foothold_height_offsets(
            env,
            command_name="motion",
            foot_body_names=("left_foot", "right_foot"),
        )

        torch.testing.assert_close(output, torch.zeros(2, 2))

    def test_rejects_invalid_command_offset_shape(self):
        env = self._env([[0.0, 0.1], [0.0, -0.1]])

        with self.assertRaisesRegex(RuntimeError, "first_foothold_height_offsets must have shape"):
            observations.first_foothold_height_offsets(
                env,
                command_name="motion",
                foot_body_names=("left_foot", "right_foot"),
            )


if __name__ == "__main__":
    unittest.main()
