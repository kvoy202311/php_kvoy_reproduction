from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


class _SceneEntityCfg:
    def __init__(self, name, body_ids=None, joint_ids=None):
        self.name = name
        self.body_ids = body_ids
        self.joint_ids = joint_ids


def _load_rewards_module():
    stubs = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": types.ModuleType("isaaclab.utils.math"),
        "isaaclab.assets": types.ModuleType("isaaclab.assets"),
        "isaaclab.managers": types.ModuleType("isaaclab.managers"),
        "isaaclab.sensors": types.ModuleType("isaaclab.sensors"),
        "php_kvoy_reproduction.tasks.tracking.mdp.commands": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.commands"
        ),
        "php_kvoy_reproduction.tasks.tracking.mdp.joint_settling": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.joint_settling"
        ),
        "php_kvoy_reproduction.tasks.tracking.mdp.obstacle": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.obstacle"
        ),
        "php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry"
        ),
        "php_kvoy_reproduction.tasks.tracking.mdp.platform_foot_support": types.ModuleType(
            "php_kvoy_reproduction.tasks.tracking.mdp.platform_foot_support"
        ),
    }
    stubs["isaaclab.assets"].RigidObject = object
    stubs["isaaclab.assets"].Articulation = object
    stubs["isaaclab.managers"].SceneEntityCfg = _SceneEntityCfg
    stubs["isaaclab.sensors"].ContactSensor = object
    stubs["isaaclab.utils.math"].quat_error_magnitude = lambda first, *_: torch.zeros(first.shape[0])
    stubs["php_kvoy_reproduction.tasks.tracking.mdp.commands"].MotionCommand = object
    stubs["php_kvoy_reproduction.tasks.tracking.mdp.joint_settling"].joint_settling_score = (
        lambda velocity, **_: 1.0 / (1.0 + torch.max(torch.abs(velocity), dim=1).values)
    )
    stubs["php_kvoy_reproduction.tasks.tracking.mdp.obstacle"].get_climb_box_sizes = lambda *_args, **_kwargs: None
    stubs["php_kvoy_reproduction.tasks.tracking.mdp.obstacle"].points_inside_oriented_box_xy = (
        lambda *_args, **_kwargs: None
    )
    obstacle_geometry = stubs["php_kvoy_reproduction.tasks.tracking.mdp.obstacle_geometry"]
    obstacle_geometry.filtered_platform_contact_score = lambda *_args, **_kwargs: None
    obstacle_geometry.filtered_platform_force_score = lambda *_args, **_kwargs: None
    obstacle_geometry.first_foothold_reference_gate = lambda *_args, **_kwargs: None
    obstacle_geometry.foot_sole_corners_world = lambda *_args, **_kwargs: None
    obstacle_geometry.foothold_precontact_score = lambda *_args, **_kwargs: None
    obstacle_geometry.foothold_safety_score = lambda *_args, **_kwargs: None
    obstacle_geometry.foothold_safety_violation = lambda *_args, **_kwargs: None
    obstacle_geometry.sole_top_height_score = lambda *_args, **_kwargs: None
    platform_foot_support = stubs["php_kvoy_reproduction.tasks.tracking.mdp.platform_foot_support"]
    platform_foot_support.platform_foot_load_score = lambda *_args, **_kwargs: None
    platform_foot_support.platform_foot_support_score = lambda *_args, **_kwargs: None
    platform_foot_support.platform_foot_support_state = lambda *_args, **_kwargs: None
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        module_path = (
            Path(__file__).parents[1]
            / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/rewards.py"
        )
        spec = importlib.util.spec_from_file_location("php_kvoy_reproduction_rewards", module_path)
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


rewards = _load_rewards_module()


class _CommandManager:
    def __init__(self, command):
        self.command = command

    def get_term(self, name):
        assert name == "motion"
        return self.command


class FirstFootholdTrackingFadeTest(unittest.TestCase):
    def test_fade_is_side_specific_and_never_below_existing_terminal_floor(self):
        command = SimpleNamespace(
            cfg=SimpleNamespace(
                body_names=[
                    "l_knee_y_link",
                    "l_ankle_x_link",
                    "r_knee_y_link",
                    "r_ankle_x_link",
                    "torso_link",
                ]
            )
        )
        body_indexes = [0, 1, 2, 3, 4]
        gates = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        body_weights = {
            "l_knee_y_link": 0.65,
            "l_ankle_x_link": 0.25,
            "r_knee_y_link": 0.65,
            "r_ankle_x_link": 0.25,
        }
        weights = rewards._apply_first_foothold_body_tracking_weights(
            torch.ones(2, 5),
            command,
            body_indexes,
            gates,
            ("l_ankle_x_link", "r_ankle_x_link"),
            body_weights,
        )

        torch.testing.assert_close(weights[0], torch.tensor([0.65, 0.25, 1.0, 1.0, 1.0]))
        torch.testing.assert_close(weights[1], torch.tensor([1.0, 1.0, 0.65, 0.25, 1.0]))

        terminal_weights = torch.ones(1, 5)
        terminal_weights[0, 1] = 0.15
        preserved_floor = rewards._apply_first_foothold_body_tracking_weights(
            terminal_weights,
            command,
            body_indexes,
            gates[:1],
            ("l_ankle_x_link", "r_ankle_x_link"),
            body_weights,
        )
        self.assertAlmostEqual(preserved_floor[0, 1].item(), 0.15, places=6)


class TerminalDefaultPoseModeGateTest(unittest.TestCase):
    def test_latched_terminal_mode_disables_expert_credit_and_enables_single_q_credit(self):
        command = SimpleNamespace(
            joint_pos=torch.zeros(2, 1),
            time_steps=torch.zeros(2, dtype=torch.long),
            terminal_default_pose_expert_tracking_factor=torch.tensor([1.0, 0.0]),
            terminal_default_pose_latched=torch.tensor([False, True]),
        )

        torch.testing.assert_close(rewards._terminal_expert_tracking_factor(command), torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(rewards._terminal_default_pose_latched_gate(command), torch.tensor([0.0, 1.0]))

    def test_default_pose_reward_gate_stays_off_without_terminal_mode(self):
        command = SimpleNamespace(
            joint_pos=torch.zeros(2, 1),
            time_steps=torch.zeros(2, dtype=torch.long),
        )

        torch.testing.assert_close(rewards._terminal_default_pose_latched_gate(command), torch.zeros(2))

    def test_disabled_default_q_waits_for_the_final_source_frame_before_settling(self):
        command = SimpleNamespace(
            cfg=SimpleNamespace(terminal_default_pose_enabled=False),
            joint_pos=torch.zeros(2, 1),
            time_steps=torch.tensor([98, 99], dtype=torch.long),
            motion_ids=torch.zeros(2, dtype=torch.long),
            motion=SimpleNamespace(motion_end_idx=torch.tensor([100], dtype=torch.long)),
            terminal_default_pose_complete=torch.ones(2, dtype=torch.bool),
            terminal_default_pose_latched=torch.zeros(2, dtype=torch.bool),
            terminal_default_pose_static_tail=torch.ones(2, dtype=torch.bool),
        )

        torch.testing.assert_close(rewards._terminal_stationary_target_gate(command), torch.tensor([0.0, 1.0]))


class FinalJointSettlingRewardTest(unittest.TestCase):
    def _call(
        self,
        reference_velocity,
        robot_velocity,
        support_scores,
        *,
        time_step=80,
        alignment_complete=None,
        default_pose_complete=None,
        default_pose_latched=None,
        default_pose_static_tail=None,
    ):
        command = SimpleNamespace(
            joint_vel=torch.tensor(reference_velocity, dtype=torch.float32),
            robot_joint_vel=torch.tensor(robot_velocity, dtype=torch.float32),
            motion=SimpleNamespace(
                motion_start_idx=torch.tensor([0]),
                motion_end_idx=torch.tensor([101]),
                motion_lengths=torch.tensor([101]),
            ),
            motion_ids=torch.zeros(len(reference_velocity), dtype=torch.long),
            time_steps=torch.full((len(reference_velocity),), time_step, dtype=torch.long),
        )
        if alignment_complete is not None:
            command.terminal_platform_alignment_complete = torch.tensor(alignment_complete, dtype=torch.bool)
        if default_pose_complete is not None:
            command.terminal_default_pose_complete = torch.tensor(default_pose_complete, dtype=torch.bool)
        if default_pose_latched is not None:
            command.terminal_default_pose_latched = torch.tensor(default_pose_latched, dtype=torch.bool)
        if default_pose_static_tail is not None:
            command.terminal_default_pose_static_tail = torch.tensor(default_pose_static_tail, dtype=torch.bool)
        env = SimpleNamespace(command_manager=_CommandManager(command))
        with patch.object(
            rewards,
            "_platform_foot_contact_scores",
            return_value=torch.tensor(support_scores, dtype=torch.float32),
        ):
            return rewards.final_joint_settling(
                env,
                command_name="motion",
                platform_cfg=_SceneEntityCfg("platform"),
                contact_sensor_cfg=_SceneEntityCfg("contact_forces", body_ids=[0, 1]),
                base_size=(0.46, 0.8, 0.65),
                foot_body_names=["left", "right"],
                footprint_inset=0.02,
                foot_height_std=0.08,
                min_contact_force=10.0,
                contact_time_scale=0.25,
                reference_max_joint_speed=0.1,
                rms_speed_tolerance=0.35,
                max_speed_tolerance=0.5,
                rms_speed_scale=1.0,
                max_speed_scale=2.0,
                fine_max_speed_scale=0.5,
                score_weights=(0.4, 0.3, 0.3),
            )

    def test_requires_stationary_reference_and_two_foot_support(self):
        reward = self._call(
            reference_velocity=[[0.0, 0.0], [0.2, 0.0], [0.0, 0.0]],
            robot_velocity=[[0.2, 0.1], [0.2, 0.1], [0.2, 0.1]],
            support_scores=[[1.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        )
        self.assertGreater(reward[0].item(), 0.0)
        self.assertEqual(reward[1].item(), 0.0)
        self.assertEqual(reward[2].item(), 0.0)

    def test_does_not_require_torso_or_hand_state(self):
        reward = self._call(
            reference_velocity=[[0.0, 0.0]],
            robot_velocity=[[0.3, 0.2]],
            support_scores=[[0.8, 0.9]],
        )
        self.assertGreater(reward.item(), 0.0)

    def test_remains_active_while_the_final_reference_frame_is_held(self):
        reward = self._call(
            reference_velocity=[[0.0, 0.0]],
            robot_velocity=[[0.4, 0.2]],
            support_scores=[[1.0, 1.0]],
            time_step=100,
        )
        self.assertGreater(reward.item(), 0.0)

    def test_static_initial_pose_does_not_open_terminal_settling(self):
        command = SimpleNamespace(
            joint_vel=torch.zeros(1, 2),
            robot_joint_vel=torch.tensor([[0.3, 0.2]]),
            motion=SimpleNamespace(
                motion_start_idx=torch.tensor([0]),
                motion_end_idx=torch.tensor([101]),
                motion_lengths=torch.tensor([101]),
            ),
            motion_ids=torch.zeros(1, dtype=torch.long),
            time_steps=torch.tensor([10]),
        )
        env = SimpleNamespace(command_manager=_CommandManager(command))
        with patch.object(
            rewards,
            "_platform_foot_contact_scores",
            return_value=torch.ones(1, 2),
        ):
            reward = rewards.final_joint_settling(
                env,
                command_name="motion",
                platform_cfg=_SceneEntityCfg("platform"),
                contact_sensor_cfg=_SceneEntityCfg("contact_forces", body_ids=[0, 1]),
                base_size=(0.46, 0.8, 0.65),
                foot_body_names=["left", "right"],
                footprint_inset=0.02,
                foot_height_std=0.08,
                min_contact_force=10.0,
                contact_time_scale=0.25,
                reference_max_joint_speed=0.1,
                rms_speed_tolerance=0.35,
                max_speed_tolerance=0.5,
                rms_speed_scale=1.0,
                max_speed_scale=2.0,
                fine_max_speed_scale=0.5,
                score_weights=(0.4, 0.3, 0.3),
            )
        self.assertEqual(reward.item(), 0.0)

    def test_terminal_settling_waits_for_platform_relative_reference_alignment(self):
        reward = self._call(
            reference_velocity=[[0.0, 0.0], [0.0, 0.0]],
            robot_velocity=[[0.1, 0.1], [0.1, 0.1]],
            support_scores=[[1.0, 1.0], [1.0, 1.0]],
            alignment_complete=[False, True],
        )

        self.assertEqual(reward[0].item(), 0.0)
        self.assertGreater(reward[1].item(), 0.0)

    def test_terminal_settling_waits_for_default_pose_completion(self):
        reward = self._call(
            reference_velocity=[[0.0, 0.0], [0.0, 0.0]],
            robot_velocity=[[0.1, 0.1], [0.1, 0.1]],
            support_scores=[[1.0, 1.0], [1.0, 1.0]],
            default_pose_complete=[False, True],
        )

        self.assertEqual(reward[0].item(), 0.0)
        self.assertGreater(reward[1].item(), 0.0)

    def test_terminal_settling_bridges_static_source_but_not_moving_default_transition(self):
        reward = self._call(
            reference_velocity=[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            robot_velocity=[[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]],
            support_scores=[[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            default_pose_complete=[False, False, True],
            default_pose_latched=[False, True, True],
            default_pose_static_tail=[True, True, True],
        )

        self.assertGreater(reward[0].item(), 0.0)
        self.assertEqual(reward[1].item(), 0.0)
        self.assertGreater(reward[2].item(), 0.0)


class FinalExpertUpperBodyPoseRewardTest(unittest.TestCase):
    def _call(
        self,
        *,
        reference_joint_pos,
        robot_joint_pos,
        reference_joint_vel,
        two_foot_contact,
        time_step=100,
        alignment_complete=None,
    ):
        reference_joint_pos = torch.tensor(reference_joint_pos, dtype=torch.float32)
        robot_joint_pos = torch.tensor(robot_joint_pos, dtype=torch.float32)
        reference_joint_vel = torch.tensor(reference_joint_vel, dtype=torch.float32)
        num_envs = reference_joint_pos.shape[0]
        command = SimpleNamespace(
            joint_pos=reference_joint_pos,
            joint_vel=reference_joint_vel,
            motion=SimpleNamespace(
                motion_start_idx=torch.zeros(num_envs, dtype=torch.long),
                motion_end_idx=torch.full((num_envs,), 101, dtype=torch.long),
            ),
            motion_ids=torch.zeros(num_envs, dtype=torch.long),
            time_steps=torch.full((num_envs,), time_step, dtype=torch.long),
            # MotionCommand always exposes this value.  Keep the mock faithful
            # so the terminal gate can also be exercised during final-frame hold.
            final_hold_progress=torch.zeros(num_envs, dtype=torch.float32),
        )
        if alignment_complete is not None:
            command.terminal_platform_alignment_complete = torch.tensor(alignment_complete, dtype=torch.bool)
        asset = SimpleNamespace(data=SimpleNamespace(joint_pos=robot_joint_pos))
        env = SimpleNamespace(
            command_manager=_CommandManager(command),
            scene={"robot": asset},
            step_dt=0.02,
        )
        with patch.object(
            rewards,
            "_terminal_platform_contact_gate",
            return_value=torch.tensor(two_foot_contact, dtype=torch.float32),
        ):
            return rewards.final_expert_upper_body_joint_position_error_exp(
                env,
                command_name="motion",
                asset_cfg=_SceneEntityCfg("robot", joint_ids=[0, 1, 2]),
                platform_cfg=_SceneEntityCfg("platform"),
                contact_sensor_cfg=_SceneEntityCfg("contact_forces", body_ids=[0, 1]),
                base_size=(0.46, 0.8, 0.65),
                foot_body_names=["left", "right"],
                footprint_inset=0.02,
                foot_height_std=0.08,
                min_contact_force=10.0,
                contact_time_scale=0.25,
                reference_max_joint_speed=0.1,
                static_window_time_s=0.5,
                std=0.35,
            )

    def test_rewards_matching_expert_upper_body_during_static_tail(self):
        reward = self._call(
            reference_joint_pos=[[0.1, -0.2, 0.3, 1.0, -1.0]],
            robot_joint_pos=[[0.1, -0.2, 0.3, 5.0, -5.0]],
            reference_joint_vel=[[0.0, 0.0, 0.0, 0.0, 0.0]],
            two_foot_contact=[1.0],
        )
        self.assertAlmostEqual(reward.item(), 1.0, places=6)

    def test_ignores_leg_error_but_penalizes_selected_upper_body_error(self):
        matching = self._call(
            reference_joint_pos=[[0.1, -0.2, 0.3, 1.0, -1.0]],
            robot_joint_pos=[[0.1, -0.2, 0.3, 0.0, 0.0]],
            reference_joint_vel=[[0.0] * 5],
            two_foot_contact=[1.0],
        )
        upper_body_error = self._call(
            reference_joint_pos=[[0.1, -0.2, 0.3, 1.0, -1.0]],
            robot_joint_pos=[[0.5, -0.2, 0.3, 0.0, 0.0]],
            reference_joint_vel=[[0.0] * 5],
            two_foot_contact=[1.0],
        )
        self.assertAlmostEqual(matching.item(), 1.0, places=6)
        self.assertLess(upper_body_error.item(), matching.item())

    def test_requires_static_tail_and_two_foot_contact(self):
        common = dict(
            reference_joint_pos=[[0.0] * 5],
            robot_joint_pos=[[0.0] * 5],
        )
        initial_static = self._call(
            **common,
            reference_joint_vel=[[0.0] * 5],
            two_foot_contact=[1.0],
            time_step=10,
        )
        moving_tail = self._call(
            **common,
            reference_joint_vel=[[0.11, 0.0, 0.0, 0.0, 0.0]],
            two_foot_contact=[1.0],
        )
        single_foot = self._call(
            **common,
            reference_joint_vel=[[0.0] * 5],
            two_foot_contact=[0.0],
        )
        alignment_in_progress = self._call(
            **common,
            reference_joint_vel=[[0.0] * 5],
            two_foot_contact=[1.0],
            alignment_complete=[False],
        )
        self.assertEqual(initial_static.item(), 0.0)
        self.assertEqual(moving_tail.item(), 0.0)
        self.assertEqual(single_foot.item(), 0.0)
        self.assertEqual(alignment_in_progress.item(), 0.0)


class GroupedTerminalExpertRewardTest(unittest.TestCase):
    def _command(self):
        joint_names = ("waist", "left_arm_a", "left_arm_b", "left_leg_a", "left_leg_b")
        robot_joint_pos = torch.zeros(2, len(joint_names), dtype=torch.float32)
        source_joint_pos = torch.zeros_like(robot_joint_pos)
        # Compare equal-sized errors in a low-weight leg group and a
        # high-weight arm group.  The latter must reduce the aggregate more.
        robot_joint_pos[0, 3:] = 1.0
        robot_joint_pos[1, 1:3] = 1.0
        robot_joint_vel = robot_joint_pos.clone()
        return SimpleNamespace(
            robot=SimpleNamespace(joint_names=joint_names),
            robot_joint_pos=robot_joint_pos,
            source_joint_pos=source_joint_pos,
            joint_pos=source_joint_pos,
            robot_joint_vel=robot_joint_vel,
        )

    @staticmethod
    def _groups():
        return {
            "waist": ["waist"],
            "left_arm": ["left_arm_a", "left_arm_b"],
            "left_leg": ["left_leg_a", "left_leg_b"],
        }

    @staticmethod
    def _stds():
        return {"waist": 1.0, "left_arm": 1.0, "left_leg": 1.0}

    @staticmethod
    def _weights():
        return {"waist": 2.0, "left_arm": 2.0, "left_leg": 0.5}

    def test_grouped_pose_does_not_dilute_arm_error_into_all_joints(self):
        score, rms, maximum = rewards._grouped_expert_joint_pose_score(
            self._command(), self._groups(), self._stds(), self._weights()
        )
        self.assertLess(score[1].item(), score[0].item())
        torch.testing.assert_close(rms["left_arm"], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(rms["left_leg"], torch.tensor([1.0, 0.0]))
        torch.testing.assert_close(maximum["left_arm"], torch.tensor([0.0, 1.0]))

    def test_grouped_velocity_uses_measured_robot_velocity(self):
        score, rms, maximum = rewards._grouped_actual_joint_speed_score(
            self._command(), self._groups(), self._stds(), self._weights()
        )
        self.assertLess(score[1].item(), score[0].item())
        torch.testing.assert_close(rms["left_arm"], torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(maximum["left_leg"], torch.tensor([1.0, 0.0]))

    def test_joint_groups_must_be_disjoint_and_known(self):
        command = self._command()
        with self.assertRaisesRegex(ValueError, "disjoint"):
            rewards._grouped_expert_joint_pose_score(
                command,
                {"a": ["waist"], "b": ["waist"]},
                {"a": 1.0, "b": 1.0},
                {"a": 1.0, "b": 1.0},
            )
        with self.assertRaisesRegex(RuntimeError, "unknown robot joints"):
            rewards._grouped_expert_joint_pose_score(
                command,
                {"a": ["missing"]},
                {"a": 1.0},
                {"a": 1.0},
            )

    def test_inverse_quadratic_pose_retains_signal_for_large_arm_error(self):
        command = SimpleNamespace(
            robot=SimpleNamespace(joint_names=("left_arm",)),
            robot_joint_pos=torch.tensor([[1.7], [1.6], [0.0]]),
            source_joint_pos=torch.zeros(3, 1),
        )
        score, rms, maximum = rewards._grouped_expert_joint_pose_score(
            command,
            {"left_arm": ["left_arm"]},
            {"left_arm": 0.45},
            {"left_arm": 1.0},
        )
        expected = torch.reciprocal(1.0 + torch.square(torch.tensor([1.7, 1.6, 0.0]) / 0.45))
        torch.testing.assert_close(score, expected)
        torch.testing.assert_close(rms, {"left_arm": torch.tensor([1.7, 1.6, 0.0])})
        torch.testing.assert_close(maximum, {"left_arm": torch.tensor([1.7, 1.6, 0.0])})
        self.assertGreater(score[1].item() - score[0].item(), 0.005)
        self.assertGreater(score[0].item(), 0.05)

    def test_inverse_quadratic_velocity_is_monotonic_and_normalized(self):
        command = SimpleNamespace(
            robot=SimpleNamespace(joint_names=("left_arm",)),
            robot_joint_pos=torch.zeros(3, 1),
            robot_joint_vel=torch.tensor([[1.24], [0.62], [0.0]]),
        )
        score, rms, maximum = rewards._grouped_actual_joint_speed_score(
            command,
            {"left_arm": ["left_arm"]},
            {"left_arm": 0.55},
            {"left_arm": 1.0},
        )
        expected = torch.reciprocal(1.0 + torch.square(torch.tensor([1.24, 0.62, 0.0]) / 0.55))
        torch.testing.assert_close(score, expected)
        torch.testing.assert_close(rms["left_arm"], torch.tensor([1.24, 0.62, 0.0]))
        torch.testing.assert_close(maximum["left_arm"], torch.tensor([1.24, 0.62, 0.0]))
        self.assertLess(score[0].item(), score[1].item())
        self.assertLess(score[1].item(), score[2].item())
        self.assertEqual(score[2].item(), 1.0)

    def test_static_tail_reaches_full_strength_after_short_ramp(self):
        command = SimpleNamespace(
            motion=SimpleNamespace(motion_end_idx=torch.tensor([101], dtype=torch.long)),
            motion_ids=torch.zeros(6, dtype=torch.long),
            time_steps=torch.tensor([75, 76, 77, 80, 90, 100], dtype=torch.long),
            final_hold_progress=torch.zeros(6),
            joint_vel=torch.zeros(6, 2),
            joint_pos=torch.zeros(6, 2),
        )
        gate = rewards._expert_static_tail_gate(
            command,
            reference_max_joint_speed=0.1,
            static_window_time_s=0.5,
            step_dt=0.02,
            ramp_time_s=0.1,
        )
        torch.testing.assert_close(gate, torch.tensor([0.0, 0.0, 0.25, 1.0, 1.0, 1.0]))

    def test_soft_support_retains_guidance_without_replacing_strict_support(self):
        command = SimpleNamespace()
        env = SimpleNamespace()
        with (
            patch.object(
                rewards,
                "_platform_foot_contact_scores",
                return_value=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            ),
            patch.object(
                rewards,
                "_platform_foot_load_score",
                return_value=torch.tensor([0.0, 1.0]),
            ),
        ):
            soft, strict = rewards._terminal_expert_support_gate(
                env,
                command,
                _SceneEntityCfg("platform"),
                _SceneEntityCfg("contact_forces"),
                (0.51, 0.8, 0.66),
                ["left", "right"],
                0.02,
                0.04,
                10.0,
                0.25,
                None,
                0.5,
                0.25,
            )
        torch.testing.assert_close(strict, torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(soft, torch.tensor([0.25, 1.0]))

    def test_top_level_grouped_rewards_apply_tail_support_and_publish_metrics(self):
        command = self._command()
        command.time_steps = torch.tensor([99, 100], dtype=torch.long)
        command.metrics = {}
        command.anchor_quat_w = torch.zeros(2, 4)
        command.robot_anchor_quat_w = torch.zeros(2, 4)
        command.robot_anchor_ang_vel_w = torch.zeros(2, 3)
        env = SimpleNamespace(command_manager=_CommandManager(command), step_dt=0.02)
        static_tail = torch.ones(2)
        soft_support = torch.tensor([0.25, 1.0])
        strict_support = torch.tensor([0.0, 1.0])
        common = dict(
            env=env,
            command_name="motion",
            platform_cfg=_SceneEntityCfg("platform"),
            contact_sensor_cfg=_SceneEntityCfg("contact_forces"),
            base_size=(0.51, 0.8, 0.66),
            foot_body_names=["left", "right"],
            footprint_inset=0.02,
            foot_height_std=0.04,
            min_contact_force=10.0,
            contact_time_scale=0.25,
            reference_max_joint_speed=0.1,
            static_window_time_s=0.5,
            ramp_time_s=0.1,
            joint_groups=self._groups(),
            group_stds=self._stds(),
            group_weights=self._weights(),
            support_floor=0.25,
            min_total_load_fraction=0.5,
        )
        with (
            patch.object(
                rewards,
                "_terminal_expert_support_gate",
                return_value=(soft_support, strict_support),
            ),
            patch.object(rewards, "_expert_static_tail_gate", return_value=static_tail),
        ):
            pose_score, _, _ = rewards._grouped_expert_joint_pose_score(
                command, self._groups(), self._stds(), self._weights()
            )
            velocity_score, _, _ = rewards._grouped_actual_joint_speed_score(
                command, self._groups(), self._stds(), self._weights()
            )
            pose_reward = rewards.final_grouped_expert_joint_position_error_exp(**common)
            velocity_reward = rewards.final_grouped_actual_joint_velocity_exp(**common)

        torch.testing.assert_close(pose_reward, soft_support * pose_score)
        torch.testing.assert_close(velocity_reward, soft_support * velocity_score)
        torch.testing.assert_close(command.metrics["final_tail_strict_support"], strict_support)
        self.assertIn("final_tail_left_arm_pose_rms", command.metrics)
        self.assertIn("final_tail_left_arm_max_pose_error", command.metrics)
        self.assertIn("final_tail_left_leg_joint_speed_rms", command.metrics)
        self.assertIn("final_tail_left_leg_max_joint_speed", command.metrics)
        self.assertIn("final_tail_torso_orientation_error", command.metrics)
        self.assertIn("final_tail_torso_angular_speed", command.metrics)


class FinalExpertFullJointPoseRewardTest(unittest.TestCase):
    """Behavioral coverage for the terminal full-joint expert-pose terms."""

    def test_pose_score_uses_immutable_source_pose_not_mutable_command_target(self):
        command = SimpleNamespace(
            robot_joint_pos=torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32),
            source_joint_pos=torch.tensor([[0.2, -0.3, 0.4]], dtype=torch.float32),
            # This deliberately disagrees with the source pose.  A terminal
            # alignment/interpolation target must not replace the expert pose.
            joint_pos=torch.tensor([[1.2, -1.3, 1.4]], dtype=torch.float32),
        )

        score = rewards._expert_joint_pose_score(command, std=0.35)

        torch.testing.assert_close(score, torch.ones(1))

    def test_pose_score_is_gaussian_of_mean_squared_all_joint_error(self):
        std = 0.4
        source_joint_pos = torch.zeros(2, 4)
        robot_joint_pos = torch.tensor(
            [
                [0.4, 0.0, 0.0, 0.0],
                [0.4, -0.4, 0.4, -0.4],
            ],
            dtype=torch.float32,
        )
        command = SimpleNamespace(
            robot_joint_pos=robot_joint_pos,
            source_joint_pos=source_joint_pos,
            # Keep an incompatible mutable target present to ensure the
            # calculation still covers every source-joint coordinate.
            joint_pos=torch.full_like(source_joint_pos, 5.0),
        )

        score = rewards._expert_joint_pose_score(command, std=std)
        expected = torch.exp(-torch.mean(torch.square(robot_joint_pos - source_joint_pos), dim=1) / std**2)

        torch.testing.assert_close(score, expected)
        self.assertGreater(score[0].item(), score[1].item())

    def test_full_joint_reward_requires_static_bilateral_loaded_support(self):
        std = 0.4
        source_joint_pos = torch.zeros(5, 3)
        robot_joint_pos = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.2, -0.2, 0.4],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        command = SimpleNamespace(
            robot_joint_pos=robot_joint_pos,
            source_joint_pos=source_joint_pos,
            # This must not be used as the pose target even when the static
            # tail, contacts, and load gates all permit reward.
            joint_pos=torch.full_like(source_joint_pos, 3.0),
        )
        env = SimpleNamespace(
            command_manager=_CommandManager(command),
            step_dt=0.02,
        )
        support_scores = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        foot_load_scores = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0], dtype=torch.float32)
        static_tail = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        with (
            patch.object(rewards, "_platform_foot_contact_scores", return_value=support_scores),
            patch.object(rewards, "_platform_foot_load_score", return_value=foot_load_scores),
            patch.object(rewards, "_expert_static_tail_gate", return_value=static_tail),
        ):
            reward = rewards.final_expert_joint_position_error_exp(
                env,
                command_name="motion",
                platform_cfg=_SceneEntityCfg("platform"),
                contact_sensor_cfg=_SceneEntityCfg("contact_forces", body_ids=[0, 1]),
                base_size=(0.51, 0.8, 0.66),
                foot_body_names=["left", "right"],
                footprint_inset=0.02,
                foot_height_std=0.08,
                min_contact_force=10.0,
                contact_time_scale=0.25,
                reference_max_joint_speed=0.1,
                static_window_time_s=0.5,
                std=std,
                min_total_load_fraction=0.5,
            )

        expected_second = torch.exp(-torch.mean(torch.square(robot_joint_pos[1])) / std**2)
        torch.testing.assert_close(reward[:2], torch.tensor([1.0, expected_second]))
        torch.testing.assert_close(reward[2:], torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
