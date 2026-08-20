from __future__ import annotations

import ast
import unittest
from pathlib import Path


_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/config/elf3/climb_env_cfg.py"
)
_COMMANDS_PATH = (
    Path(__file__).parents[1]
    / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/commands.py"
)


def _top_level_assignments(tree: ast.Module) -> dict[str, ast.expr]:
    assignments: dict[str, ast.expr] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = node.value
    return assignments


class ClimbTerminalConfigTest(unittest.TestCase):
    def test_source_motion_ends_without_an_extra_terminal_hold(self):
        tree = ast.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
        assignments = _top_level_assignments(tree)
        self.assertIn("ELF3_CLIMB_FINAL_HOLD_TIME_S", assignments)
        self.assertIsInstance(assignments["ELF3_CLIMB_FINAL_HOLD_TIME_S"], ast.Constant)
        self.assertEqual(assignments["ELF3_CLIMB_FINAL_HOLD_TIME_S"].value, 0.0)
        self.assertNotIn("ELF3_CLIMB_TERMINAL_PLATFORM_ALIGNMENT_RAMP_TIME_S", assignments)
        self.assertNotIn("ELF3_CLIMB_MIN_STABLE_TIME_S", assignments)

        commands_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ELF3ClimbCommandsCfg"
        )
        motion_assignment = next(
            node
            for node in commands_class.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "motion" for target in node.targets)
        )
        self.assertIsInstance(motion_assignment.value, ast.Call)
        keywords = {keyword.arg: keyword.value for keyword in motion_assignment.value.keywords}
        self.assertIsInstance(keywords["motion_end_hold_time_s"], ast.Name)
        self.assertEqual(keywords["motion_end_hold_time_s"].id, "ELF3_CLIMB_FINAL_HOLD_TIME_S")
        self.assertIsInstance(keywords["terminate_on_motion_end"], ast.Constant)
        self.assertTrue(keywords["terminate_on_motion_end"].value)
        self.assertIsInstance(keywords["adaptive_failure_term_names"], ast.Tuple)
        self.assertEqual(
            [ast.literal_eval(value) for value in keywords["adaptive_failure_term_names"].elts],
            ["motion_end_failure"],
        )
        for keyword_name in (
            "terminal_platform_alignment_foot_body_names",
            "terminal_platform_alignment_sole_corners_b",
            "terminal_platform_alignment_base_size",
            "terminal_platform_alignment_clearance",
            "terminal_platform_alignment_ramp_time_s",
            "terminal_support_confirmation_time_s",
            "terminal_stable_time_s",
        ):
            with self.subTest(keyword=keyword_name):
                self.assertNotIn(keyword_name, keywords)

        self.assertIsInstance(keywords["terminal_default_pose_enabled"], ast.Constant)
        self.assertFalse(keywords["terminal_default_pose_enabled"].value)
        self.assertIsInstance(keywords["random_phase_env_mask_attr"], ast.Constant)
        self.assertEqual(keywords["random_phase_env_mask_attr"].value, "_climb_box_random_phase_env_mask")
        self.assertNotIn("first_foothold_height_alignment_params", keywords)
        self.assertNotIn("terminal_default_pose_transition_time_s", keywords)
        self.assertNotIn("terminal_platform_xy_alignment_enabled", keywords)

        rewards_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ELF3ClimbRewardsCfg"
        )
        reward_names = {
            target.id
            for node in rewards_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertNotIn("final_standing_stability", reward_names)
        self.assertNotIn("final_joint_settling", reward_names)
        self.assertIn("final_expert_joint_pose", reward_names)
        self.assertIn("final_actual_joint_velocity", reward_names)
        self.assertIn("final_ankle_surface_settling", reward_names)
        self.assertIn("final_expert_root_orientation", reward_names)
        self.assertIn("final_expert_root_linear_velocity", reward_names)
        self.assertIn("final_expert_root_angular_velocity", reward_names)
        self.assertIn("first_foothold_surface_alignment", reward_names)
        self.assertIn("platform_foot_surface_alignment", reward_names)

        reward_functions = {
            target.id: ast.unparse(
                next(keyword.value for keyword in node.value.keywords if keyword.arg == "func")
            )
            for node in rewards_class.body
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
            for target in node.targets
            if isinstance(target, ast.Name)
            and any(keyword.arg == "func" for keyword in node.value.keywords)
        }
        self.assertEqual(
            reward_functions["final_expert_joint_pose"],
            "mdp.final_grouped_expert_joint_position_error_exp",
        )
        self.assertEqual(
            reward_functions["final_actual_joint_velocity"],
            "mdp.final_grouped_actual_joint_velocity_exp",
        )
        self.assertEqual(
            reward_functions["final_ankle_surface_settling"],
            "mdp.final_ankle_surface_settling",
        )
        self.assertEqual(
            reward_functions["final_expert_root_orientation"],
            "mdp.final_expert_root_orientation_error_exp",
        )
        self.assertEqual(
            reward_functions["final_expert_root_linear_velocity"],
            "mdp.final_expert_root_linear_velocity_error_exp",
        )
        self.assertEqual(
            reward_functions["final_expert_root_angular_velocity"],
            "mdp.final_expert_root_angular_velocity_error_exp",
        )
        for reward_name in ("final_expert_joint_pose", "final_actual_joint_velocity"):
            with self.subTest(reward=reward_name):
                assignment = next(
                    node
                    for node in rewards_class.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == reward_name
                        for target in node.targets
                    )
                )
                reward_keywords = {
                    keyword.arg: keyword.value for keyword in assignment.value.keywords
                }
                reward_params = {
                    ast.literal_eval(key): value
                    for key, value in zip(
                        reward_keywords["params"].keys,
                        reward_keywords["params"].values,
                        strict=True,
                    )
                }
                self.assertEqual(
                    ast.unparse(reward_params["worst_group_weight"]),
                    "ELF3_CLIMB_FINAL_WORST_GROUP_WEIGHT",
                )

        terminations_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ELF3ClimbTerminationsCfg"
        )
        termination_names = {
            target.id
            for node in terminations_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertIn("motion_clip_end", termination_names)
        self.assertIn("motion_end_success", termination_names)
        self.assertIn("motion_end_failure", termination_names)
        boundary_order = [
            target.id
            for node in terminations_class.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"motion_end_success", "motion_end_failure", "motion_clip_end"}
        ]
        self.assertEqual(
            boundary_order,
            ["motion_end_success", "motion_end_failure", "motion_clip_end"],
        )
        for term_name, function_name in (
            ("motion_end_success", "mdp.motion_end_success"),
            ("motion_end_failure", "mdp.motion_end_failure"),
        ):
            with self.subTest(termination=term_name):
                assignment = next(
                    node
                    for node in terminations_class.body
                    if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == term_name for target in node.targets)
                )
                term_keywords = {keyword.arg: keyword.value for keyword in assignment.value.keywords}
                self.assertEqual(ast.unparse(term_keywords["func"]), function_name)
                self.assertTrue(ast.literal_eval(term_keywords["time_out"]))

        success_assignment = next(
            node
            for node in terminations_class.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "motion_end_success" for target in node.targets)
        )
        success_keywords = {keyword.arg: keyword.value for keyword in success_assignment.value.keywords}
        success_params = {
            ast.literal_eval(key): value
            for key, value in zip(
                success_keywords["params"].keys,
                success_keywords["params"].values,
                strict=True,
            )
        }
        for parameter_name in (
            "static_window_time_s",
            "reference_max_joint_speed",
            "joint_groups",
            "group_pose_rms_thresholds",
            "group_pose_max_thresholds",
            "group_velocity_rms_thresholds",
            "max_torso_orientation_error",
        ):
            with self.subTest(success_parameter=parameter_name):
                self.assertIn(parameter_name, success_params)
        motion_clip_end_assignment = next(
            node
            for node in terminations_class.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "motion_clip_end" for target in node.targets)
        )
        self.assertIsInstance(motion_clip_end_assignment.value, ast.Call)
        self.assertEqual(ast.unparse(motion_clip_end_assignment.value.func), "DoneTerm")
        motion_clip_end_keywords = {
            keyword.arg: keyword.value for keyword in motion_clip_end_assignment.value.keywords
        }
        self.assertEqual(ast.unparse(motion_clip_end_keywords["func"]), "mdp.motion_clip_end")
        self.assertIsInstance(motion_clip_end_keywords["time_out"], ast.Constant)
        self.assertTrue(motion_clip_end_keywords["time_out"].value)
        self.assertIsInstance(motion_clip_end_keywords["params"], ast.Dict)
        motion_clip_end_params = {
            ast.literal_eval(key): ast.literal_eval(value)
            for key, value in zip(
                motion_clip_end_keywords["params"].keys,
                motion_clip_end_keywords["params"].values,
                strict=True,
            )
        }
        self.assertEqual(motion_clip_end_params, {"command_name": "motion"})

        curriculum_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ELF3ClimbCurriculumCfg"
        )
        curriculum_assignment = next(
            node
            for node in curriculum_class.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "platform_pose" for target in node.targets)
        )
        curriculum_keywords = {keyword.arg: keyword.value for keyword in curriculum_assignment.value.keywords}
        self.assertIsInstance(curriculum_keywords["params"], ast.Dict)
        curriculum_params = {
            ast.literal_eval(key): value
            for key, value in zip(curriculum_keywords["params"].keys, curriculum_keywords["params"].values, strict=True)
        }
        self.assertEqual(ast.literal_eval(curriculum_params["success_term_name"]), "motion_end_success")

        observations_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ELF3ClimbObservationsCfg"
        )
        for group_name in ("PolicyCfg", "PrivilegedCfg"):
            with self.subTest(observation_group=group_name):
                group = next(
                    node for node in observations_class.body if isinstance(node, ast.ClassDef) and node.name == group_name
                )
                term_names = {
                    target.id
                    for node in group.body
                    if isinstance(node, ast.Assign)
                    for target in node.targets
                    if isinstance(target, ast.Name)
                }
                self.assertNotIn("terminal_platform_xy_offset", term_names)
                self.assertNotIn("first_foothold_height_offsets", term_names)

    def test_terminal_joint_groups_cover_robot_once_and_keep_legs_looser(self):
        tree = ast.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
        assignments = _top_level_assignments(tree)
        robot_joint_names = ast.literal_eval(assignments["ELF3_CLIMB_JOINT_NAMES"])
        groups = ast.literal_eval(assignments["ELF3_CLIMB_FINAL_EXPERT_JOINT_GROUPS"])
        grouped_joint_names = [name for names in groups.values() for name in names]

        self.assertCountEqual(grouped_joint_names, robot_joint_names)
        self.assertEqual(len(grouped_joint_names), len(set(grouped_joint_names)))

        pose_stds = ast.literal_eval(assignments["ELF3_CLIMB_FINAL_EXPERT_POSE_GROUP_STDS"])
        pose_weights = ast.literal_eval(assignments["ELF3_CLIMB_FINAL_EXPERT_POSE_GROUP_WEIGHTS"])
        velocity_stds = ast.literal_eval(assignments["ELF3_CLIMB_FINAL_ACTUAL_VELOCITY_GROUP_STDS"])
        velocity_weights = ast.literal_eval(assignments["ELF3_CLIMB_FINAL_ACTUAL_VELOCITY_GROUP_WEIGHTS"])
        for side in ("left", "right"):
            self.assertGreater(pose_stds[f"{side}_leg"], pose_stds[f"{side}_arm"])
            self.assertLess(pose_weights[f"{side}_leg"], pose_weights[f"{side}_arm"])
            self.assertGreater(velocity_stds[f"{side}_leg"], velocity_stds[f"{side}_arm"])
            self.assertLess(velocity_weights[f"{side}_leg"], velocity_weights[f"{side}_arm"])
            self.assertEqual(pose_weights[f"{side}_ankle"], 0.0)
            self.assertEqual(velocity_weights[f"{side}_ankle"], 0.0)

        exempt_groups = ast.literal_eval(
            assignments["ELF3_CLIMB_FINAL_QUALITY_EXPERT_POSE_EXEMPT_GROUPS"]
        )
        self.assertEqual(set(exempt_groups), {"left_ankle", "right_ankle"})
        self.assertEqual(
            ast.literal_eval(
                assignments["ELF3_CLIMB_TERMINAL_FOOT_VERTICAL_POSITION_TRACKING_WEIGHT"]
            ),
            0.0,
        )
        vertical_position_weights = ast.literal_eval(
            assignments["ELF3_CLIMB_FIRST_FOOTHOLD_VERTICAL_POSITION_TRACKING_WEIGHTS"]
        )
        self.assertEqual(vertical_position_weights, {"l_ankle_x_link": 0.0, "r_ankle_x_link": 0.0})
        vertical_precontact_weights = ast.literal_eval(
            assignments["ELF3_CLIMB_FIRST_FOOTHOLD_VERTICAL_PRECONTACT_TRACKING_WEIGHTS"]
        )
        orientation_precontact_weights = ast.literal_eval(
            assignments["ELF3_CLIMB_FIRST_FOOTHOLD_ORIENTATION_PRECONTACT_TRACKING_WEIGHTS"]
        )
        self.assertEqual(vertical_precontact_weights, {"l_ankle_x_link": 0.25, "r_ankle_x_link": 0.25})
        self.assertEqual(orientation_precontact_weights, {"l_ankle_x_link": 0.15, "r_ankle_x_link": 0.15})
        generic_velocity_weight = ast.literal_eval(
            assignments["ELF3_CLIMB_FINAL_ACTUAL_JOINT_VELOCITY_REWARD_WEIGHT"]
        )
        ankle_velocity_weight = ast.literal_eval(
            assignments["ELF3_CLIMB_FINAL_ANKLE_SURFACE_SETTLING_REWARD_WEIGHT"]
        )
        self.assertEqual(generic_velocity_weight + ankle_velocity_weight, 3.0)

        self.assertEqual(
            ast.literal_eval(assignments["ELF3_CLIMB_FINAL_EXPERT_REWARD_RAMP_TIME_S"]),
            0.1,
        )
        self.assertEqual(
            ast.literal_eval(assignments["ELF3_CLIMB_FINAL_EXPERT_SUPPORT_FLOOR"]),
            1.0,
        )
        self.assertEqual(ast.literal_eval(assignments["ELF3_CLIMB_FINAL_SCORE_EXPONENT"]), 0.5)
        self.assertEqual(ast.literal_eval(assignments["ELF3_CLIMB_FINAL_GROUP_AGGREGATION"]), "harmonic")
        self.assertEqual(ast.literal_eval(assignments["ELF3_CLIMB_FINAL_WORST_JOINT_WEIGHT"]), 0.75)
        self.assertEqual(ast.literal_eval(assignments["ELF3_CLIMB_FINAL_WORST_GROUP_WEIGHT"]), 0.75)
        worst_counts = ast.literal_eval(assignments["ELF3_CLIMB_FINAL_WORST_JOINT_COUNT"])
        for group_name in groups:
            with self.subTest(worst_group=group_name):
                self.assertEqual(worst_counts[group_name], 1)

        pose_rms_thresholds = ast.literal_eval(
            assignments["ELF3_CLIMB_FINAL_QUALITY_POSE_RMS_THRESHOLDS"]
        )
        pose_max_thresholds = ast.literal_eval(
            assignments["ELF3_CLIMB_FINAL_QUALITY_POSE_MAX_THRESHOLDS"]
        )
        velocity_rms_thresholds = ast.literal_eval(
            assignments["ELF3_CLIMB_FINAL_QUALITY_VELOCITY_RMS_THRESHOLDS"]
        )
        self.assertEqual(set(pose_rms_thresholds), set(groups))
        self.assertEqual(set(pose_max_thresholds), set(groups))
        self.assertEqual(set(velocity_rms_thresholds), set(groups))
        self.assertLessEqual(
            ast.literal_eval(assignments["ELF3_CLIMB_FINAL_QUALITY_MIN_STABLE_TIME_S"]),
            ast.literal_eval(assignments["ELF3_CLIMB_FINAL_EXPERT_JOINT_POSE_WINDOW_S"]),
        )

    def test_length_range_is_fixed_to_the_source_aligned_platform(self):
        tree = ast.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
        assignments = _top_level_assignments(tree)
        length_range = assignments["ELF3_CLIMB_PLATFORM_LENGTH_RANGE"]
        self.assertIsInstance(length_range, ast.Tuple)
        self.assertEqual([ast.literal_eval(value) for value in length_range.elts], [0.51, 0.51])

    def test_default_pose_handoff_starts_from_the_real_supported_joint_state(self):
        tree = ast.parse(_COMMANDS_PATH.read_text(encoding="utf-8"))
        motion_command = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand")
        handoff_method = next(
            node
            for node in motion_command.body
            if isinstance(node, ast.FunctionDef) and node.name == "_update_terminal_default_pose_mode"
        )
        assignments = [node for node in ast.walk(handoff_method) if isinstance(node, ast.Assign)]
        handoff_assignment = next(
            node
            for node in assignments
            if any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "_terminal_default_pose_start_joint_pos"
                for target in node.targets
            )
        )
        assigned_expression = ast.unparse(handoff_assignment.value)
        self.assertIn("self.robot.data.joint_pos", assigned_expression)
        self.assertNotIn("self.source_joint_pos", assigned_expression)

    def test_platform_contact_timer_is_updated_independently_of_default_q_handoff(self):
        tree = ast.parse(_COMMANDS_PATH.read_text(encoding="utf-8"))
        motion_command = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand")
        method_names = {
            node.name for node in motion_command.body if isinstance(node, ast.FunctionDef)
        }
        self.assertIn("_update_platform_foot_support_timer", method_names)

        update_command = next(
            node for node in motion_command.body if isinstance(node, ast.FunctionDef) and node.name == "_update_command"
        )
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(update_command)
            if isinstance(node, ast.Call)
        ]
        self.assertIn("self._update_platform_foot_support_timer", calls)
        self.assertIn("self._update_terminal_default_pose_mode", calls)
        self.assertLess(
            calls.index("self._update_platform_foot_support_timer"),
            calls.index("self._update_terminal_default_pose_mode"),
        )

    def test_optional_motion_command_hold_budget_validation_remains_available(self):
        tree = ast.parse(_COMMANDS_PATH.read_text(encoding="utf-8"))
        motion_command = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionCommand")
        method_names = {node.name for node in motion_command.body if isinstance(node, ast.FunctionDef)}
        self.assertIn("_validate_terminal_physical_hold_budget", method_names)

        init = next(node for node in motion_command.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
        ]
        self.assertIn("self._initialize_terminal_platform_z_alignment", calls)
        self.assertIn("self._validate_terminal_physical_hold_budget", calls)
        self.assertLess(
            calls.index("self._initialize_terminal_platform_z_alignment"),
            calls.index("self._validate_terminal_physical_hold_budget"),
        )

    def test_optional_terminal_mechanisms_default_to_disabled(self):
        tree = ast.parse(_COMMANDS_PATH.read_text(encoding="utf-8"))
        motion_command_cfg = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MotionCommandCfg"
        )
        defaults = {
            node.target.id: node.value
            for node in motion_command_cfg.body
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.value is not None
        }
        for name in (
            "terminal_platform_alignment_ramp_time_s",
            "terminal_support_confirmation_time_s",
            "terminal_stable_time_s",
        ):
            with self.subTest(name=name):
                self.assertIsInstance(defaults[name], ast.Constant)
                self.assertEqual(defaults[name].value, 0.0)


if __name__ == "__main__":
    unittest.main()
