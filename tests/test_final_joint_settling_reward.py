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
    stubs["isaaclab.utils.math"].quat_error_magnitude = lambda *_: torch.zeros(1)
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


if __name__ == "__main__":
    unittest.main()
