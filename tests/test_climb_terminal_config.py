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
    def test_short_terminal_hold_keeps_strict_support_and_disables_default_q_handoff(self):
        tree = ast.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
        assignments = _top_level_assignments(tree)
        expected_values = {
            "ELF3_CLIMB_FINAL_HOLD_TIME_S": 1.30,
            "ELF3_CLIMB_TERMINAL_PLATFORM_ALIGNMENT_RAMP_TIME_S": 0.5,
        }
        for name, expected in expected_values.items():
            with self.subTest(name=name):
                self.assertIn(name, assignments)
                self.assertIsInstance(assignments[name], ast.Constant)
                self.assertEqual(assignments[name].value, expected)
        self.assertGreaterEqual(
            assignments["ELF3_CLIMB_FINAL_HOLD_TIME_S"].value,
            assignments["ELF3_CLIMB_TERMINAL_PLATFORM_ALIGNMENT_RAMP_TIME_S"].value
            + assignments["ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S"].value
            + assignments["ELF3_CLIMB_MIN_STABLE_TIME_S"].value,
        )

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
        expected_wiring = {
            "motion_end_hold_time_s": "ELF3_CLIMB_FINAL_HOLD_TIME_S",
            "terminal_platform_alignment_ramp_time_s": "ELF3_CLIMB_TERMINAL_PLATFORM_ALIGNMENT_RAMP_TIME_S",
            "terminal_support_confirmation_time_s": "ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S",
            "terminal_stable_time_s": "ELF3_CLIMB_MIN_STABLE_TIME_S",
        }
        for keyword_name, expected_name in expected_wiring.items():
            with self.subTest(keyword=keyword_name):
                self.assertIsInstance(keywords[keyword_name], ast.Name)
                self.assertEqual(keywords[keyword_name].id, expected_name)

        self.assertIsInstance(keywords["terminal_default_pose_enabled"], ast.Constant)
        self.assertFalse(keywords["terminal_default_pose_enabled"].value)
        self.assertIsInstance(keywords["random_phase_env_mask_attr"], ast.Constant)
        self.assertEqual(keywords["random_phase_env_mask_attr"].value, "_climb_box_random_phase_env_mask")
        self.assertNotIn("first_foothold_height_alignment_params", keywords)
        self.assertNotIn("terminal_default_pose_transition_time_s", keywords)
        self.assertNotIn("terminal_platform_xy_alignment_enabled", keywords)

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

    def test_short_hold_has_an_independent_physical_budget_validation(self):
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


if __name__ == "__main__":
    unittest.main()
