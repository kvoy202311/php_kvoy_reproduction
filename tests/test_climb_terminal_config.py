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
    def test_terminal_handoff_constants_are_physically_budgeted_and_wired(self):
        tree = ast.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
        assignments = _top_level_assignments(tree)
        expected_values = {
            "ELF3_CLIMB_FINAL_HOLD_TIME_S": 2.25,
            "ELF3_CLIMB_TERMINAL_PLATFORM_ALIGNMENT_RAMP_TIME_S": 0.5,
            "ELF3_CLIMB_FIRST_FOOTHOLD_HEIGHT_ALIGNMENT_MAX_OFFSET": 0.10,
            "ELF3_CLIMB_TERMINAL_DEFAULT_POSE_TRANSITION_TIME_S": 0.8,
            "ELF3_CLIMB_MAX_DEFAULT_JOINT_POS_RMS": 0.25,
        }
        for name, expected in expected_values.items():
            with self.subTest(name=name):
                self.assertIn(name, assignments)
                self.assertIsInstance(assignments[name], ast.Constant)
                self.assertEqual(assignments[name].value, expected)

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
            "first_foothold_height_alignment_max_offset": "ELF3_CLIMB_FIRST_FOOTHOLD_HEIGHT_ALIGNMENT_MAX_OFFSET",
            "terminal_default_pose_transition_time_s": "ELF3_CLIMB_TERMINAL_DEFAULT_POSE_TRANSITION_TIME_S",
        }
        for keyword_name, expected_name in expected_wiring.items():
            with self.subTest(keyword=keyword_name):
                self.assertIsInstance(keywords[keyword_name], ast.Name)
                self.assertEqual(keywords[keyword_name].id, expected_name)

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

    def test_length_range_preserves_nominal_geometry_and_removes_invalid_short_boxes(self):
        tree = ast.parse(_CONFIG_PATH.read_text(encoding="utf-8"))
        assignments = _top_level_assignments(tree)
        length_range = assignments["ELF3_CLIMB_PLATFORM_LENGTH_RANGE"]
        self.assertIsInstance(length_range, ast.Tuple)
        self.assertEqual([ast.literal_eval(value) for value in length_range.elts], [0.46, 0.51])

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


if __name__ == "__main__":
    unittest.main()
