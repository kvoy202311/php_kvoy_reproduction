from __future__ import annotations

import importlib.util
import inspect
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


def _quat_error_magnitude(quaternion_a: torch.Tensor, quaternion_b: torch.Tensor) -> torch.Tensor:
    dot = torch.sum(quaternion_a * quaternion_b, dim=1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _get_climb_box_sizes(platform, base_size, device):
    return torch.tensor(base_size, dtype=torch.float32, device=device).expand(platform.num_envs, -1).clone()


def _points_inside_oriented_box_xy(points, box_positions, box_quaternions, box_sizes):
    del box_quaternions
    relative_xy = points[..., :2] - box_positions[:, None, :2]
    return torch.all(torch.abs(relative_xy) <= 0.5 * box_sizes[:, None, :2], dim=-1)


def _motion_clip_boundary_mask(motion_finished, terminated):
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
        "php_kvoy_reproduction.tasks.tracking.mdp.platform_foot_support": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.platform_foot_support"
        ),
    }
    stub_modules["isaaclab.utils.math"].quat_apply_inverse = _quat_apply_inverse
    stub_modules["isaaclab.utils.math"].quat_error_magnitude = _quat_error_magnitude
    stub_modules["isaaclab.assets"].Articulation = object
    stub_modules["isaaclab.assets"].RigidObject = object
    stub_modules["isaaclab.managers"].ManagerTermBase = _ManagerTermBase
    stub_modules["isaaclab.managers"].SceneEntityCfg = _SceneEntityCfg
    stub_modules["isaaclab.sensors"].ContactSensor = object
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.commands"].MotionCommand = object
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.motion_data"].motion_clip_boundary_mask = (
        _motion_clip_boundary_mask
    )
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.obstacle"].get_climb_box_sizes = (
        _get_climb_box_sizes
    )
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.obstacle"].points_inside_oriented_box_xy = (
        _points_inside_oriented_box_xy
    )
    stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.rewards"]._get_body_indexes = lambda *_: []
    platform_foot_support = stub_modules["php_kvoy_reproduction.tasks.tracking.mdp.platform_foot_support"]
    platform_foot_support.platform_foot_load_valid = lambda *_args, **_kwargs: None
    platform_foot_support.platform_foot_support_state = lambda *_args, **_kwargs: None

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
        self.term_values = {}

    def get_term(self, name):
        return self.term_values[name]


class MotionClipEndTerminationTest(unittest.TestCase):
    def _env(self, *, terminate_on_motion_end: bool, motion_finished: torch.Tensor, terminated: torch.Tensor):
        command = SimpleNamespace(
            cfg=SimpleNamespace(terminate_on_motion_end=terminate_on_motion_end),
            motion_finished=motion_finished,
        )
        env = SimpleNamespace(
            num_envs=motion_finished.numel(),
            device=motion_finished.device,
            command_manager=_CommandManager(command),
            termination_manager=_TerminationManager(motion_finished.numel()),
        )
        env.termination_manager.terminated[:] = terminated
        return env

    def test_completed_motion_is_a_boundary_only_when_physics_has_not_terminated(self):
        env = self._env(
            terminate_on_motion_end=True,
            motion_finished=torch.tensor([True, True, False]),
            terminated=torch.tensor([False, True, False]),
        )

        result = terminations.motion_clip_end(env, "motion")

        self.assertTrue(torch.equal(result, torch.tensor([True, False, False])))

    def test_disabled_motion_end_never_requests_a_clip_boundary(self):
        env = self._env(
            terminate_on_motion_end=False,
            motion_finished=torch.tensor([True, True]),
            terminated=torch.tensor([False, False]),
        )

        result = terminations.motion_clip_end(env, "motion")

        self.assertTrue(torch.equal(result, torch.tensor([False, False])))

    def test_timeout_classifiers_remain_mutually_exclusive_with_generic_boundary(self):
        env = self._env(
            terminate_on_motion_end=True,
            motion_finished=torch.tensor([True, True, True]),
            terminated=torch.tensor([False, False, False]),
        )
        env.termination_manager.term_values = {
            "motion_end_success": torch.tensor([True, False, False]),
            "motion_end_failure": torch.tensor([False, True, False]),
        }

        result = terminations.motion_clip_end(
            env,
            "motion",
            ("motion_end_success", "motion_end_failure"),
        )

        self.assertTrue(torch.equal(result, torch.tensor([False, False, True])))


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
            _climb_box_random_phase_env_mask=torch.tensor([index == 0 for index in range(num_envs)])
        )

    def __getitem__(self, name):
        assert name == "platform"
        return self.platform


_QUALITY_JOINT_GROUPS = {
    "waist": ["waist_y_joint"],
    "left_arm": ["l_shoulder_y_joint", "l_elbow_y_joint", "l_wrist_z_joint"],
    "right_arm": ["r_shoulder_y_joint", "r_elbow_y_joint", "r_wrist_z_joint"],
    "left_leg": ["l_hip_y_joint", "l_knee_y_joint", "l_ankle_y_joint"],
    "right_leg": ["r_hip_y_joint", "r_knee_y_joint", "r_ankle_y_joint"],
}
_QUALITY_JOINT_NAMES = [name for names in _QUALITY_JOINT_GROUPS.values() for name in names]
_POSE_RMS_THRESHOLDS = {name: 0.50 for name in _QUALITY_JOINT_GROUPS}
_POSE_MAX_THRESHOLDS = {name: 0.90 for name in _QUALITY_JOINT_GROUPS}
_VELOCITY_RMS_THRESHOLDS = {name: 0.60 for name in _QUALITY_JOINT_GROUPS}


def _success_quality_params(min_stable_time=0.05, static_window_time_s=0.10):
    return {
        "command_name": "motion",
        "contact_sensor_cfg": _SceneEntityCfg("contact_forces", body_ids=[0, 1]),
        "min_stable_time": min_stable_time,
        "static_window_time_s": static_window_time_s,
        "reference_max_joint_speed": 0.10,
        "joint_groups": _QUALITY_JOINT_GROUPS,
        "group_pose_rms_thresholds": _POSE_RMS_THRESHOLDS,
        "group_pose_max_thresholds": _POSE_MAX_THRESHOLDS,
        "group_velocity_rms_thresholds": _VELOCITY_RMS_THRESHOLDS,
        "max_torso_orientation_error": 0.20,
    }


def _make_success_term(
    num_envs=3,
    step_dt=0.02,
    min_stable_time=0.05,
    static_window_time_s=0.10,
    max_expert_joint_pos_rms=None,
    params_override=None,
):
    num_joints = len(_QUALITY_JOINT_NAMES)
    command = SimpleNamespace(
        cfg=SimpleNamespace(
            terminate_on_motion_end=True,
            motion_end_hold_time_s=0.0,
            terminal_default_pose_enabled=False,
        ),
        metrics={},
        motion=SimpleNamespace(
            motion_end_idx=torch.tensor([20], dtype=torch.long),
            joint_vel=torch.zeros(20, num_joints),
        ),
        robot=SimpleNamespace(
            joint_names=_QUALITY_JOINT_NAMES,
        ),
        device="cpu",
        source_joint_pos=torch.zeros(num_envs, num_joints),
        robot_joint_pos=torch.zeros(num_envs, num_joints),
        robot_joint_vel=torch.zeros(num_envs, num_joints),
        joint_vel=torch.zeros(num_envs, num_joints),
        anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(num_envs, -1).clone(),
        robot_anchor_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(num_envs, -1).clone(),
        robot_anchor_lin_vel_w=torch.zeros(num_envs, 3),
        robot_anchor_ang_vel_w=torch.zeros(num_envs, 3),
        motion_ids=torch.zeros(num_envs, dtype=torch.long),
        time_steps=torch.full((num_envs,), 14, dtype=torch.long),
        motion_finished=torch.zeros(num_envs, dtype=torch.bool),
        episode_started_at_motion_beginning=torch.ones(num_envs, dtype=torch.bool),
    )
    params = _success_quality_params(min_stable_time, static_window_time_s)
    if max_expert_joint_pos_rms is not None:
        params["max_expert_joint_pos_rms"] = max_expert_joint_pos_rms
    if params_override:
        params.update(params_override)
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


def _call_success_term(
    term,
    env,
    min_stable_time=0.05,
    static_window_time_s=0.10,
    max_expert_joint_pos_rms=None,
    **overrides,
):
    call_params = {
        "command_name": "motion",
        "platform_cfg": _SceneEntityCfg("platform"),
        "contact_sensor_cfg": _SceneEntityCfg("contact_forces", body_ids=[0, 1]),
        "base_size": (1.0, 1.0, 0.65),
        "foot_body_names": ["left_foot", "right_foot"],
        "footprint_inset": 0.05,
        "foot_height_range": (-0.02, 0.08),
        "min_foot_contact_force": 20.0,
        "min_foot_contact_time": 0.05,
        "max_root_height_error": 0.15,
        "max_root_linear_speed": 0.25,
        "max_root_angular_speed": 0.5,
        "max_joint_speed": 1.0,
        "max_torso_tilt": 0.35,
        "min_stable_time": min_stable_time,
        "static_window_time_s": static_window_time_s,
        "reference_max_joint_speed": 0.10,
        "joint_groups": _QUALITY_JOINT_GROUPS,
        "group_pose_rms_thresholds": _POSE_RMS_THRESHOLDS,
        "group_pose_max_thresholds": _POSE_MAX_THRESHOLDS,
        "group_velocity_rms_thresholds": _VELOCITY_RMS_THRESHOLDS,
        "expert_pose_exempt_groups": tuple(term._expert_pose_exempt_groups),
        "max_torso_orientation_error": 0.20,
        "max_expert_joint_pos_rms": max_expert_joint_pos_rms,
    }
    call_params.update(overrides)
    return term(env, **call_params)


class MotionEndSuccessTest(unittest.TestCase):
    def test_manager_call_signature_accepts_expert_pose_exempt_groups(self):
        parameter = inspect.signature(terminations.motion_end_success.__call__).parameters[
            "expert_pose_exempt_groups"
        ]

        self.assertEqual(parameter.default, ())

    def test_runtime_exempt_groups_cannot_change_after_construction(self):
        term, env, _ = _make_success_term()

        with self.assertRaisesRegex(ValueError, "changed after construction"):
            _call_success_term(term, env, expert_pose_exempt_groups=("right_leg",))

    def _standing_result(self, valid, invalid_condition="feet_inside"):
        conditions = {
            "feet_inside": torch.ones_like(valid),
            "foot_height_valid": torch.ones_like(valid),
            "foot_contact_valid": torch.ones_like(valid),
            "foot_load_valid": torch.ones_like(valid),
            "upright": torch.ones_like(valid),
            "root_height_valid": torch.ones_like(valid),
            "root_linear_speed_valid": torch.ones_like(valid),
            "root_angular_speed_valid": torch.ones_like(valid),
            "joint_speed_valid": torch.ones_like(valid),
            "default_joint_pos_valid": torch.ones_like(valid),
        }
        conditions[invalid_condition] = valid.clone()
        return conditions, torch.zeros(valid.shape[0])

    def _call_frame(self, term, env, command, frame, *, finished=False, standing=None, **overrides):
        command.time_steps[:] = frame
        if isinstance(finished, torch.Tensor):
            command.motion_finished.copy_(finished)
        else:
            command.motion_finished[:] = finished
        if standing is None:
            standing = self._standing_result(torch.ones(env.num_envs, dtype=torch.bool))
        with patch.object(terminations, "_climb_standing_conditions", return_value=standing):
            return _call_success_term(term, env, **overrides)

    def test_accumulates_only_in_static_tail_and_returns_only_at_boundary(self):
        term, env, command = _make_success_term()

        self.assertFalse(torch.any(self._call_frame(term, env, command, 14)))
        torch.testing.assert_close(term._stable_steps, torch.zeros(3, dtype=torch.long))
        for frame, expected_count in ((15, 1), (16, 2), (17, 3), (18, 4)):
            result = self._call_frame(term, env, command, frame)
            self.assertFalse(torch.any(result))
            torch.testing.assert_close(term._stable_steps, torch.full((3,), expected_count))

        result = self._call_frame(term, env, command, 19, finished=True)
        self.assertTrue(torch.all(result))
        torch.testing.assert_close(term._stable_steps, torch.full((3,), 5, dtype=torch.long))

    def test_each_real_standing_failure_resets_only_the_bad_environment(self):
        condition_names = (
            "feet_inside",
            "foot_height_valid",
            "foot_contact_valid",
            "foot_load_valid",
            "upright",
            "root_height_valid",
            "root_linear_speed_valid",
            "root_angular_speed_valid",
            "joint_speed_valid",
            "default_joint_pos_valid",
        )
        for condition_name in condition_names:
            with self.subTest(condition=condition_name):
                term, env, command = _make_success_term()
                self._call_frame(term, env, command, 15)
                self._call_frame(term, env, command, 16)
                one_invalid = self._standing_result(
                    torch.tensor([True, False, True]), invalid_condition=condition_name
                )
                self._call_frame(term, env, command, 17, standing=one_invalid)
                torch.testing.assert_close(term._stable_steps, torch.tensor([3, 0, 3]))
                self._call_frame(term, env, command, 18)
                result = self._call_frame(term, env, command, 19, finished=True)
                torch.testing.assert_close(term._stable_steps, torch.tensor([5, 2, 5]))
                torch.testing.assert_close(result, torch.tensor([True, False, True]))

    def test_bad_final_sample_clears_an_already_sufficient_run(self):
        term, env, command = _make_success_term()
        for frame in range(15, 19):
            self._call_frame(term, env, command, frame)
        bad_final = self._standing_result(torch.zeros(3, dtype=torch.bool))

        result = self._call_frame(term, env, command, 19, finished=True, standing=bad_final)

        self.assertFalse(torch.any(result))
        torch.testing.assert_close(term._stable_steps, torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(term._longest_stable_steps, torch.full((3,), 4, dtype=torch.long))

    def test_reference_motion_must_be_static_on_every_counted_sample(self):
        term, env, command = _make_success_term()
        command.motion.joint_vel[16, 0] = 0.11

        self._call_frame(term, env, command, 15)
        self._call_frame(term, env, command, 16)
        torch.testing.assert_close(term._stable_steps, torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(command.metrics["final_standing_reference_static"], torch.zeros(3))
        self._call_frame(term, env, command, 17)
        self._call_frame(term, env, command, 18)
        result = self._call_frame(term, env, command, 19, finished=True)
        self.assertTrue(torch.all(result))

    def test_group_max_rejects_one_bad_arm_joint_even_when_group_rms_passes(self):
        rms_thresholds = dict(_POSE_RMS_THRESHOLDS)
        rms_thresholds["left_arm"] = 0.60
        term, env, command = _make_success_term(
            params_override={"group_pose_rms_thresholds": rms_thresholds}
        )
        bad_joint_id = _QUALITY_JOINT_NAMES.index("l_shoulder_y_joint")
        command.robot_joint_pos[1, bad_joint_id] = 0.91

        self._call_frame(
            term,
            env,
            command,
            15,
            group_pose_rms_thresholds=rms_thresholds,
        )

        torch.testing.assert_close(term._stable_steps, torch.tensor([1, 0, 1]))
        self.assertGreater(command.metrics["final_standing_left_arm_pose_rms"][1], 0.0)
        self.assertEqual(command.metrics["final_standing_left_arm_pose_rms_valid"][1], 1.0)
        self.assertEqual(command.metrics["final_standing_left_arm_pose_max_valid"][1], 0.0)

    def test_adaptive_ankle_group_ignores_expert_pose_but_still_requires_low_velocity(self):
        term, env, command = _make_success_term(
            params_override={"expert_pose_exempt_groups": ("right_leg",)}
        )
        right_leg_ids = [
            _QUALITY_JOINT_NAMES.index(name) for name in _QUALITY_JOINT_GROUPS["right_leg"]
        ]
        command.robot_joint_pos[:, right_leg_ids] = 2.0

        self._call_frame(term, env, command, 15)
        torch.testing.assert_close(term._stable_steps, torch.ones(3, dtype=torch.long))
        torch.testing.assert_close(
            command.metrics["final_standing_right_leg_pose_max_valid"], torch.ones(3)
        )

        command.robot_joint_vel[1, right_leg_ids] = 0.61
        self._call_frame(term, env, command, 16)
        torch.testing.assert_close(term._stable_steps, torch.tensor([2, 0, 2]))

    def test_each_explicit_quality_gate_rejects_only_its_bad_environment(self):
        cases = (
            "group_pose_rms",
            "group_velocity_rms",
            "global_joint_speed",
            "torso_orientation",
            "root_linear_speed",
            "root_angular_speed",
        )
        left_arm_ids = [_QUALITY_JOINT_NAMES.index(name) for name in _QUALITY_JOINT_GROUPS["left_arm"]]
        for case in cases:
            with self.subTest(case=case):
                term, env, command = _make_success_term()
                call_overrides = {}
                if case == "group_pose_rms":
                    command.robot_joint_pos[1, left_arm_ids] = 0.51
                elif case == "group_velocity_rms":
                    command.robot_joint_vel[1, left_arm_ids] = 0.61
                elif case == "global_joint_speed":
                    command.robot_joint_vel[1, left_arm_ids[0]] = 1.01
                elif case == "torso_orientation":
                    angle = 0.21
                    command.robot_anchor_quat_w[1] = torch.tensor(
                        [math.cos(angle / 2.0), math.sin(angle / 2.0), 0.0, 0.0]
                    )
                elif case == "root_linear_speed":
                    command.robot_anchor_lin_vel_w[1, 0] = 0.26
                elif case == "root_angular_speed":
                    command.robot_anchor_ang_vel_w[1, 0] = 0.51

                self._call_frame(term, env, command, 15, **call_overrides)
                torch.testing.assert_close(term._stable_steps, torch.tensor([1, 0, 1]))

    def test_reset_supports_selected_environments_and_all_environments(self):
        term, env, command = _make_success_term()
        self._call_frame(term, env, command, 15)
        self._call_frame(term, env, command, 16)

        term.reset(torch.tensor([0, 2]))
        torch.testing.assert_close(term._stable_steps, torch.tensor([0, 2, 0]))
        torch.testing.assert_close(term._longest_stable_steps, torch.tensor([0, 2, 0]))
        term.reset()
        torch.testing.assert_close(term._stable_steps, torch.zeros(3, dtype=torch.long))
        torch.testing.assert_close(term._longest_stable_steps, torch.zeros(3, dtype=torch.long))

    def test_alignment_gate_prevents_tail_samples_from_counting(self):
        term, env, command = _make_success_term()
        command.terminal_platform_alignment_complete = torch.zeros(3, dtype=torch.bool)
        self._call_frame(term, env, command, 15)
        torch.testing.assert_close(term._stable_steps, torch.zeros(3, dtype=torch.long))

        command.terminal_platform_alignment_complete[:] = True
        for frame in (16, 17, 18):
            self._call_frame(term, env, command, frame)
        result = self._call_frame(term, env, command, 19, finished=True)
        self.assertTrue(torch.all(result))

    def test_success_and_failure_are_exact_boundary_complements(self):
        term, env, command = _make_success_term()
        for frame in (15, 16, 17, 18):
            self._call_frame(term, env, command, frame)
        bad_final = self._standing_result(torch.tensor([True, False, True]))
        success = self._call_frame(term, env, command, 19, finished=True, standing=bad_final)
        env.termination_manager.get_term = lambda name: success
        env.termination_manager.terminated |= success
        failure = terminations.motion_end_failure(env, "motion", "motion_end_success")
        env.termination_manager.terminated |= failure
        generic_boundary = terminations.motion_clip_end(env, "motion")

        torch.testing.assert_close(success, torch.tensor([True, False, True]))
        torch.testing.assert_close(failure, torch.tensor([False, True, False]))
        torch.testing.assert_close(success ^ failure, command.motion_finished)
        torch.testing.assert_close(generic_boundary, torch.zeros(3, dtype=torch.bool))

    def test_physical_termination_is_neither_boundary_success_nor_failure(self):
        term, env, command = _make_success_term()
        for frame in (15, 16, 17, 18):
            self._call_frame(term, env, command, frame)
        env.termination_manager.terminated[1] = True
        success = self._call_frame(term, env, command, 19, finished=True)
        env.termination_manager.get_term = lambda name: success
        env.termination_manager.terminated |= success
        failure = terminations.motion_end_failure(env, "motion", "motion_end_success")
        env.termination_manager.terminated |= failure
        generic_boundary = terminations.motion_clip_end(env, "motion")

        torch.testing.assert_close(success, torch.tensor([True, False, True]))
        torch.testing.assert_close(failure, torch.zeros(3, dtype=torch.bool))
        torch.testing.assert_close(generic_boundary, torch.zeros(3, dtype=torch.bool))

    def test_random_phase_clip_end_is_neither_quality_success_nor_failure(self):
        term, env, command = _make_success_term()
        command.episode_started_at_motion_beginning[:] = torch.tensor([True, False, True])
        for frame in (15, 16, 17, 18):
            self._call_frame(term, env, command, frame)

        success = self._call_frame(term, env, command, 19, finished=True)
        env.termination_manager.get_term = lambda name: success
        env.termination_manager.terminated |= success
        failure = terminations.motion_end_failure(env, "motion", "motion_end_success")
        env.termination_manager.terminated |= failure
        generic_boundary = terminations.motion_clip_end(env, "motion")

        torch.testing.assert_close(success, torch.tensor([True, False, True]))
        torch.testing.assert_close(failure, torch.zeros(3, dtype=torch.bool))
        torch.testing.assert_close(generic_boundary, torch.tensor([False, True, False]))
        torch.testing.assert_close(success | failure | generic_boundary, command.motion_finished)

    def test_terminal_quality_requires_the_complete_trial_marker(self):
        term, env, command = _make_success_term()
        delattr(command, "episode_started_at_motion_beginning")

        with self.assertRaisesRegex(RuntimeError, "episode_started_at_motion_beginning"):
            self._call_frame(term, env, command, 15)

    def test_records_group_pose_speed_torso_root_and_contact_diagnostics(self):
        term, env, command = _make_success_term()
        left_arm_ids = [_QUALITY_JOINT_NAMES.index(name) for name in _QUALITY_JOINT_GROUPS["left_arm"]]
        command.robot_joint_pos[0, left_arm_ids] = torch.tensor([0.1, 0.2, 0.3])
        command.robot_joint_vel[0, left_arm_ids] = torch.tensor([0.2, 0.4, 0.6])
        command.robot_anchor_lin_vel_w[0, 0] = 0.12
        command.robot_anchor_ang_vel_w[0, 2] = 0.20
        sensor = env.scene.sensors["contact_forces"]
        sensor.data.net_forces_w[:, :, 2] = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        sensor.data.current_contact_time[:] = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

        self._call_frame(term, env, command, 19, finished=True)
        metrics = command.metrics
        torch.testing.assert_close(metrics["final_standing_final_frame_fraction"], torch.ones(3))
        torch.testing.assert_close(
            metrics["final_standing_left_arm_pose_rms"][0],
            torch.sqrt(torch.tensor((0.01 + 0.04 + 0.09) / 3)),
        )
        torch.testing.assert_close(metrics["final_standing_left_arm_max_pose_error"][0], torch.tensor(0.3))
        torch.testing.assert_close(
            metrics["final_standing_left_arm_joint_speed_rms"][0],
            torch.sqrt(torch.tensor((0.04 + 0.16 + 0.36) / 3)),
        )
        torch.testing.assert_close(metrics["final_standing_root_linear_speed"][0], torch.tensor(0.12))
        torch.testing.assert_close(metrics["final_standing_root_angular_speed"][0], torch.tensor(0.20))
        torch.testing.assert_close(metrics["final_standing_left_wrist_contact_force"], torch.tensor([5.0, 7.0, 9.0]))
        torch.testing.assert_close(metrics["final_standing_right_wrist_contact_time"], torch.tensor([0.2, 0.4, 0.6]))
        torch.testing.assert_close(metrics["final_standing_random_phase_allowed"], torch.tensor([1.0, 0.0, 0.0]))

    def test_optional_legacy_expert_rms_gate_remains_supported(self):
        term, env, command = _make_success_term(max_expert_joint_pos_rms=0.5)
        standing = self._standing_result(torch.ones(3, dtype=torch.bool))
        standing[0]["expert_joint_pos_valid"] = torch.ones(3, dtype=torch.bool)
        expert_rms = torch.tensor([0.1, 0.2, 0.3])
        command.time_steps[:] = 15
        with (
            patch.object(terminations, "_climb_standing_conditions", return_value=standing),
            patch.object(terminations, "_expert_joint_position_rms", return_value=expert_rms),
        ):
            _call_success_term(term, env, max_expert_joint_pos_rms=0.5)
        torch.testing.assert_close(command.metrics["final_standing_expert_joint_pos_valid"], torch.ones(3))
        torch.testing.assert_close(command.metrics["final_standing_expert_joint_pos_rms"], expert_rms)

    def test_configuration_validation_is_strict_and_actionable(self):
        _, env, command = _make_success_term()
        cases = []
        missing = _success_quality_params()
        missing.pop("static_window_time_s")
        cases.append((missing, "missing required quality parameters"))
        mismatched_keys = _success_quality_params()
        mismatched_keys["group_pose_rms_thresholds"] = {"waist": 0.5}
        cases.append((mismatched_keys, "keys must exactly match"))
        non_positive = _success_quality_params()
        non_positive["group_velocity_rms_thresholds"] = dict(_VELOCITY_RMS_THRESHOLDS, waist=0.0)
        cases.append((non_positive, "must be positive"))
        impossible_duration = _success_quality_params(min_stable_time=0.12, static_window_time_s=0.10)
        cases.append((impossible_duration, "cannot exceed the available static_window_time_s"))
        repeated = _success_quality_params()
        repeated_groups = {name: list(names) for name, names in _QUALITY_JOINT_GROUPS.items()}
        repeated_groups["right_arm"][0] = repeated_groups["left_arm"][0]
        repeated["joint_groups"] = repeated_groups
        cases.append((repeated, "must be disjoint"))
        unknown = _success_quality_params()
        unknown_groups = {name: list(names) for name, names in _QUALITY_JOINT_GROUPS.items()}
        unknown_groups["right_leg"][-1] = "unknown_joint"
        unknown["joint_groups"] = unknown_groups
        cases.append((unknown, "unknown robot joints"))
        incomplete = _success_quality_params()
        incomplete_groups = {name: list(names) for name, names in _QUALITY_JOINT_GROUPS.items()}
        incomplete_groups["right_leg"].pop()
        incomplete["joint_groups"] = incomplete_groups
        cases.append((incomplete, "cover every robot joint"))

        for params, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex((ValueError, RuntimeError), message):
                    terminations.motion_end_success(SimpleNamespace(params=params), env)

        command.cfg.motion_end_hold_time_s = 0.10
        with self.assertRaisesRegex(ValueError, "motion_end_hold_time_s=0"):
            terminations.motion_end_success(SimpleNamespace(params=_success_quality_params()), env)
        command.cfg.motion_end_hold_time_s = 0.0
        command.cfg.terminal_default_pose_enabled = True
        with self.assertRaisesRegex(ValueError, "terminal_default_pose_enabled"):
            terminations.motion_end_success(SimpleNamespace(params=_success_quality_params()), env)

    def test_legacy_phase_mask_attribute_remains_supported(self):
        term, env, command = _make_success_term()
        legacy_mask = torch.tensor([False, True, False])
        delattr(env.scene.platform, "_climb_box_random_phase_env_mask")
        env.scene.platform._climb_box_nominal_geometry_mask = legacy_mask
        self._call_frame(term, env, command, 15)
        torch.testing.assert_close(command.metrics["final_standing_random_phase_allowed"], legacy_mask.float())


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

        env = SimpleNamespace(num_envs=num_envs, device="cpu", scene=_Scene())
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
            max_default_joint_pos_rms=0.09,
        )

        self.assertEqual(
            set(conditions),
            {
                "feet_inside",
                "foot_height_valid",
                "foot_contact_valid",
                "foot_load_valid",
                "upright",
                "root_height_valid",
                "root_linear_speed_valid",
                "root_angular_speed_valid",
                "joint_speed_valid",
                "default_joint_pos_valid",
            },
        )
        for name, condition in conditions.items():
            if name == "default_joint_pos_valid":
                torch.testing.assert_close(condition, torch.tensor([True, False]))
                continue
            self.assertTrue(torch.all(condition))
        torch.testing.assert_close(default_pose_rms, torch.tensor([0.0, 0.1]))


if __name__ == "__main__":
    unittest.main()
