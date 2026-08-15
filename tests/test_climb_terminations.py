from __future__ import annotations

import importlib.util
import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


class _ManagerTermBase:
    def __init__(self, cfg, env):
        self.cfg = cfg
        self._env = env


class _SceneEntityCfg:
    def __init__(self, name: str, body_ids: list[int] | None = None):
        self.name = name
        self.body_ids = body_ids


def _quat_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Identity-quaternion implementation sufficient for these CPU tests."""

    if vector.ndim == 1:
        return vector.expand(quaternion.shape[0], -1)
    return vector


def _get_climb_box_sizes(platform, base_size, device):
    return torch.tensor(base_size, dtype=torch.float32, device=device).expand(platform.num_envs, -1).clone()


def _points_inside_oriented_box_xy(points, box_positions, box_quaternions, box_sizes):
    del box_quaternions
    relative_xy = points[..., :2] - box_positions[:, None, :2]
    return torch.all(torch.abs(relative_xy) <= 0.5 * box_sizes[:, None, :2], dim=-1)


def _motion_clip_timeout_mask(motion_finished, terminated):
    return motion_finished & ~terminated


def _load_terminations_module():
    """Load the production module without requiring an Isaac Sim process."""

    stub_modules = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": types.ModuleType("isaaclab.utils.math"),
        "isaaclab.assets": types.ModuleType("isaaclab.assets"),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.sensors": types.ModuleType("isaaclab.sensors"),
        "php_kvoy_reproduction.tasks.tracking.mdp.commands": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.commands"
        ),
        "php_kvoy_reproduction.tasks.tracking.mdp.motion_data": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.motion_data"
        ),
        "php_kvoy_reproduction.tasks.tracking.mdp.obstacle": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.obstacle"
        ),
        "php_kvoy_reproduction.tasks.tracking.mdp.rewards": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.rewards"
        ),
    }
    stub_modules["isaaclab.utils.math"].quat_apply_inverse = _quat_apply_inverse
    stub_modules["isaaclab.assets"].Articulation = object
    stub_modules["isaaclab.assets"].RigidObject = object
    stub_modules["isaaclab.managers"].ManagerTermBase = _ManagerTermBase
    stub_modules["isaaclab.managers"].SceneEntityCfg = _SceneEntityCfg
    stub_modules["isaaclab.sensors"].ContactSensor = object
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.commands"].MotionCommand = object
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.motion_data"].motion_clip_timeout_mask = (
        _motion_clip_timeout_mask
    )
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.obstacle"].get_climb_box_sizes = (
        _get_climb_box_sizes
    )
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.obstacle"].points_inside_oriented_box_xy = (
        _points_inside_oriented_box_xy
    )
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.rewards"]._get_body_indexes = lambda *_: []

    module_names = tuple(stub_modules)
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    try:
        sys.modules.update(stub_modules)
        module_path = (
            Path(__file__).parents[1]
            / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/terminations.py"
        )
        spec = importlib.util.spec_from_file_location("php_kvoy_reproduction_climb_terminations", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for module_name, saved_module in saved_modules.items():
            if saved_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = saved_module


terminations = _load_terminations_module()


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "motion"
        return self.command


class _TerminationManager:
    def __init__(self, num_envs):
        self.terminated = torch.zeros(num_envs, dtype=torch.bool)


class _DiagnosticContactSensor:
    def __init__(self, num_envs):
        self.data = SimpleNamespace(
            net_forces_w=torch.zeros(num_envs, 2, 3),
            current_contact_time=torch.zeros(num_envs, 2),
        )

    def find_bodies(self, names, preserve_order=False):
        assert preserve_order
        return [0, 1], list(names)


class _SuccessScene:
    def __init__(self, num_envs):
        self.sensors = {"contact_forces": _DiagnosticContactSensor(num_envs)}
        self.platform = SimpleNamespace(
            _climb_box_nominal_geometry_mask=torch.tensor([index == 0 for index in range(num_envs)])
        )

    def __getitem__(self, name):
        assert name == "platform"
        return self.platform


def _make_success_term(num_envs=3, step_dt=0.02, min_stable_time=0.05):
    command = SimpleNamespace(
        cfg=SimpleNamespace(terminate_on_motion_end=True),
        metrics={},
        motion=SimpleNamespace(motion_end_idx=torch.tensor([10], dtype=torch.long)),
        robot=SimpleNamespace(
            joint_names=["waist_y_joint", "l_hip_y_joint", "l_wrist_z_joint"],
        ),
        device="cpu",
        robot_joint_vel=torch.zeros(num_envs, 3),
        robot_anchor_ang_vel_w=torch.zeros(num_envs, 3),
        motion_ids=torch.zeros(num_envs, dtype=torch.long),
        time_steps=torch.full((num_envs,), 9, dtype=torch.long),
        motion_finished=torch.ones(num_envs, dtype=torch.bool),
    )
    params = {
        "command_name": "motion",
        "contact_sensor_cfg": _SceneEntityCfg("contact_forces", body_ids=[0, 1]),
        "min_stable_time": min_stable_time,
    }
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        step_dt=step_dt,
        scene=_SuccessScene(num_envs),
        command_manager=_CommandManager(command),
        termination_manager=_TerminationManager(num_envs),
    )
    term = terminations.motion_end_success(SimpleNamespace(params=params), env)
    return term, env, command


def _call_success_term(term, env, min_stable_time=0.05):
    return term(
        env,
        command_name="motion",
        platform_cfg=_SceneEntityCfg("platform"),
        contact_sensor_cfg=_SceneEntityCfg("contact_forces", body_ids=[0, 1]),
        base_size=(1.0, 1.0, 0.65),
        foot_body_names=["left_foot", "right_foot"],
        footprint_inset=0.05,
        foot_height_range=(-0.02, 0.08),
        min_foot_contact_force=20.0,
        min_foot_contact_time=0.05,
        max_root_height_error=0.15,
        max_root_linear_speed=0.25,
        max_root_angular_speed=0.5,
        max_joint_speed=1.0,
        max_torso_tilt=0.35,
        min_stable_time=min_stable_time,
    )


class MotionEndSuccessTest(unittest.TestCase):
    def _standing_result(self, valid, invalid_condition="feet_inside"):
        conditions = {
            "feet_inside": torch.ones_like(valid),
            "foot_height_valid": torch.ones_like(valid),
            "foot_contact_valid": torch.ones_like(valid),
            "upright": torch.ones_like(valid),
            "root_height_valid": torch.ones_like(valid),
            "root_linear_speed_valid": torch.ones_like(valid),
            "root_angular_speed_valid": torch.ones_like(valid),
            "joint_speed_valid": torch.ones_like(valid),
        }
        conditions[invalid_condition] = valid.clone()
        return conditions, torch.zeros(valid.shape[0])

    def test_success_requires_the_complete_continuous_window(self):
        term, env, _ = _make_success_term()
        # ceil(0.05 / 0.02) == 3: two good steps must not pass.
        with patch.object(
            terminations,
            "_climb_standing_conditions",
            return_value=self._standing_result(torch.ones(3, dtype=torch.bool)),
        ):
            self.assertFalse(torch.any(_call_success_term(term, env)))
            self.assertFalse(torch.any(_call_success_term(term, env)))
            self.assertTrue(torch.all(_call_success_term(term, env)))
        torch.testing.assert_close(term._stable_steps, torch.full((3,), 3, dtype=torch.long))
        torch.testing.assert_close(term._longest_stable_steps, torch.full((3,), 3, dtype=torch.long))

    def test_any_failed_condition_resets_only_that_environment(self):
        condition_names = (
            "feet_inside",
            "foot_height_valid",
            "foot_contact_valid",
            "upright",
            "root_height_valid",
            "root_linear_speed_valid",
            "root_angular_speed_valid",
            "joint_speed_valid",
        )
        for condition_name in condition_names:
            with self.subTest(condition=condition_name):
                term, env, _ = _make_success_term()
                all_valid = self._standing_result(torch.ones(3, dtype=torch.bool))
                with patch.object(terminations, "_climb_standing_conditions", return_value=all_valid):
                    _call_success_term(term, env)
                    _call_success_term(term, env)

                one_invalid = self._standing_result(
                    torch.tensor([True, False, True]),
                    invalid_condition=condition_name,
                )
                with patch.object(terminations, "_climb_standing_conditions", return_value=one_invalid):
                    result = _call_success_term(term, env)

                torch.testing.assert_close(term._stable_steps, torch.tensor([3, 0, 3]))
                torch.testing.assert_close(result, torch.tensor([True, False, True]))

                # Its next valid step starts again at one rather than resuming at three.
                with patch.object(terminations, "_climb_standing_conditions", return_value=all_valid):
                    result = _call_success_term(term, env)
                torch.testing.assert_close(term._stable_steps, torch.tensor([4, 1, 4]))
                torch.testing.assert_close(result, torch.tensor([True, False, True]))

    def test_reset_supports_selected_environments_and_all_environments(self):
        term, env, _ = _make_success_term()
        all_valid = self._standing_result(torch.ones(3, dtype=torch.bool))
        with patch.object(terminations, "_climb_standing_conditions", return_value=all_valid):
            _call_success_term(term, env)
            _call_success_term(term, env)

        term.reset(torch.tensor([0, 2]))
        torch.testing.assert_close(term._stable_steps, torch.tensor([0, 2, 0]))
        torch.testing.assert_close(term._longest_stable_steps, torch.tensor([0, 2, 0]))

        term.reset()
        torch.testing.assert_close(term._stable_steps, torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(term._longest_stable_steps, torch.zeros(3, dtype=torch.long))

    def test_not_at_final_frame_cannot_accumulate_stability(self):
        term, env, command = _make_success_term()
        command.time_steps[:] = 8
        all_valid = self._standing_result(torch.ones(3, dtype=torch.bool))
        with patch.object(terminations, "_climb_standing_conditions", return_value=all_valid):
            result = _call_success_term(term, env)
        self.assertFalse(torch.any(result))
        torch.testing.assert_close(term._stable_steps, torch.zeros(3, dtype=torch.long))

    def test_success_window_waits_for_terminal_platform_alignment(self):
        term, env, command = _make_success_term()
        command.terminal_platform_alignment_complete = torch.tensor([False, False, False])
        all_valid = self._standing_result(torch.ones(3, dtype=torch.bool))
        with patch.object(terminations, "_climb_standing_conditions", return_value=all_valid):
            result = _call_success_term(term, env)
        self.assertFalse(torch.any(result))
        torch.testing.assert_close(term._stable_steps, torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(command.metrics["final_standing_terminal_alignment_complete"], torch.zeros(3))

        command.terminal_platform_alignment_complete[:] = True
        with patch.object(terminations, "_climb_standing_conditions", return_value=all_valid):
            _call_success_term(term, env)
            _call_success_term(term, env)
            result = _call_success_term(term, env)
        self.assertTrue(torch.all(result))
        torch.testing.assert_close(command.metrics["final_standing_terminal_alignment_complete"], torch.ones(3))

    def test_speed_and_contact_diagnostics_record_actual_terminal_values(self):
        term, env, command = _make_success_term()
        command.robot_joint_vel[:] = torch.tensor(
            [[0.1, 0.2, 0.3], [0.6, 0.8, 2.1], [0.4, 0.2, 1.1]]
        )
        command.robot_anchor_ang_vel_w[:] = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.6, 0.8], [0.0, 0.0, 0.0]])
        sensor = env.scene.sensors["contact_forces"]
        sensor.data.net_forces_w[:, :, 2] = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        sensor.data.current_contact_time[:] = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        all_valid = self._standing_result(torch.ones(3, dtype=torch.bool))
        with patch.object(terminations, "_climb_standing_conditions", return_value=all_valid):
            _call_success_term(term, env)

        metrics = command.metrics
        torch.testing.assert_close(metrics["final_standing_final_frame_fraction"], torch.ones(3))
        torch.testing.assert_close(metrics["final_standing_max_joint_speed"], torch.tensor([0.3, 2.1, 1.1]))
        torch.testing.assert_close(
            metrics["final_standing_joint_speed_rms"],
            torch.sqrt(torch.tensor([(0.01 + 0.04 + 0.09) / 3, (0.36 + 0.64 + 4.41) / 3, (0.16 + 0.04 + 1.21) / 3])),
        )
        torch.testing.assert_close(metrics["final_standing_joints_over_0_5"], torch.tensor([0.0, 3.0, 1.0]))
        torch.testing.assert_close(metrics["final_standing_joints_over_0_75"], torch.tensor([0.0, 2.0, 1.0]))
        torch.testing.assert_close(metrics["final_standing_joints_over_1_0"], torch.tensor([0.0, 1.0, 1.0]))
        torch.testing.assert_close(metrics["final_standing_joints_over_2_0"], torch.tensor([0.0, 1.0, 0.0]))
        torch.testing.assert_close(metrics["final_standing_root_angular_speed"], torch.tensor([0.2, 1.0, 0.0]))
        torch.testing.assert_close(metrics["final_standing_arm_max_joint_speed"], torch.tensor([0.3, 2.1, 1.1]))
        torch.testing.assert_close(metrics["final_standing_waist_max_joint_speed"], torch.tensor([0.1, 0.6, 0.4]))
        torch.testing.assert_close(metrics["final_standing_leg_max_joint_speed"], torch.tensor([0.2, 0.8, 0.2]))
        torch.testing.assert_close(metrics["final_standing_left_wrist_contact_force"], torch.tensor([5.0, 7.0, 9.0]))
        torch.testing.assert_close(metrics["final_standing_right_wrist_contact_force"], torch.tensor([6.0, 8.0, 10.0]))
        torch.testing.assert_close(metrics["final_standing_left_wrist_contact_time"], torch.tensor([0.1, 0.3, 0.5]))
        torch.testing.assert_close(metrics["final_standing_right_wrist_contact_time"], torch.tensor([0.2, 0.4, 0.6]))
        torch.testing.assert_close(metrics["final_standing_nominal_geometry"], torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(metrics["final_standing_longest_stable_time"], torch.full((3,), 0.02))


class ClimbStandingConditionsTest(unittest.TestCase):
    def test_real_condition_function_accepts_the_configured_signature(self):
        num_envs = 2
        platform = SimpleNamespace(
            num_envs=num_envs,
            device="cpu",
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.0, 0.0, 0.325], [0.0, 0.0, 0.325]]),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            ),
        )
        foot_positions = torch.tensor(
            [
                [[-0.15, 0.0, 0.67], [0.15, 0.0, 0.67]],
                [[-0.15, 0.0, 0.67], [0.15, 0.0, 0.67]],
            ]
        )
        robot = SimpleNamespace(
            body_names=["left_foot", "right_foot"],
            data=SimpleNamespace(
                body_pos_w=foot_positions,
                GRAVITY_VEC_W=torch.tensor([0.0, 0.0, -1.0]),
                default_joint_pos=torch.zeros(num_envs, 3),
            ),
        )
        command = SimpleNamespace(
            robot=robot,
            device="cpu",
            robot_anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(num_envs, -1),
            anchor_pos_w=torch.tensor([[0.0, 0.0, 1.30], [0.0, 0.0, 1.30]]),
            robot_anchor_pos_w=torch.tensor([[0.0, 0.0, 1.30], [0.0, 0.0, 1.30]]),
            robot_anchor_lin_vel_w=torch.zeros(num_envs, 3),
            robot_anchor_ang_vel_w=torch.zeros(num_envs, 3),
            robot_joint_vel=torch.zeros(num_envs, 3),
            robot_joint_pos=torch.tensor([[0.0, 0.0, 0.0], [0.1, -0.1, 0.1]]),
        )
        contact_sensor = SimpleNamespace(
            data=SimpleNamespace(
                net_forces_w=torch.tensor(
                    [
                        [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
                        [[0.0, 0.0, 100.0], [0.0, 0.0, 100.0]],
                    ]
                ),
                current_contact_time=torch.full((num_envs, 2), 0.10),
            )
        )
        # Special methods are resolved on a type rather than an instance.
        class _Scene:
            sensors = {"contact_forces": contact_sensor}

            def __getitem__(self, name):
                return {"platform": platform}[name]

        env = SimpleNamespace(scene=_Scene())
        conditions, default_pose_rms = terminations._climb_standing_conditions(
            env=env,
            command=command,
            platform_cfg=_SceneEntityCfg("platform"),
            contact_sensor_cfg=_SceneEntityCfg("contact_forces", body_ids=[0, 1]),
            base_size=(1.0, 1.0, 0.65),
            foot_body_names=["left_foot", "right_foot"],
            footprint_inset=0.05,
            foot_height_range=(-0.02, 0.08),
            min_foot_contact_force=20.0,
            min_foot_contact_time=0.05,
            max_root_height_error=0.15,
            max_root_linear_speed=0.25,
            max_root_angular_speed=0.5,
            max_joint_speed=1.0,
            max_torso_tilt=math.radians(20.0),
        )

        self.assertEqual(
            set(conditions),
            {
                "feet_inside",
                "foot_height_valid",
                "foot_contact_valid",
                "upright",
                "root_height_valid",
                "root_linear_speed_valid",
                "root_angular_speed_valid",
                "joint_speed_valid",
            },
        )
        for condition in conditions.values():
            self.assertTrue(torch.all(condition))
        torch.testing.assert_close(default_pose_rms, torch.tensor([0.0, 0.1]))


if __name__ == "__main__":
    unittest.main()
