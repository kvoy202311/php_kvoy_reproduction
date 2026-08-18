from __future__ import annotations

import ast
import unittest
from pathlib import Path


_PLAY_PATH = Path(__file__).parents[1] / "scripts/rsl_rl/play.py"


def _is_attribute(node: ast.AST, base_name: str, attribute: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == base_name
        and node.attr == attribute
    )


class PlayDebugVisualizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(_PLAY_PATH.read_text(encoding="utf-8"))

    def test_debug_vis_argument_is_an_opt_in_flag(self):
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and any(isinstance(argument, ast.Constant) and argument.value == "--debug_vis" for argument in node.args)
        ]
        self.assertEqual(len(calls), 1)

        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertEqual(getattr(keywords.get("action"), "value", None), "store_true")
        self.assertIs(getattr(keywords.get("default"), "value", None), False)

    def test_stop_at_motion_end_defaults_to_the_motion_boundary(self):
        calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and any(
                isinstance(argument, ast.Constant) and argument.value == "--stop_at_motion_end"
                for argument in node.args
            )
        ]
        self.assertEqual(len(calls), 1)

        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        self.assertTrue(_is_attribute(keywords["action"], "argparse", "BooleanOptionalAction"))
        self.assertIs(getattr(keywords.get("default"), "value", None), True)

    def test_full_clip_configuration_uses_the_debug_vis_flag(self):
        configure = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_configure_playback"
        )
        assignments = [
            node
            for node in ast.walk(configure)
            if isinstance(node, ast.Assign)
            and any(_is_attribute(target, "motion_cfg", "debug_vis") for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertTrue(_is_attribute(assignments[0].value, "args_cli", "debug_vis"))

    def test_full_clip_configuration_does_not_add_a_new_final_hold(self):
        """Playback preserves the task horizon instead of introducing a viewer-only hold."""

        configure = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_configure_playback"
        )
        assignments = [
            node
            for node in ast.walk(configure)
            if isinstance(node, ast.Assign)
            and any(_is_attribute(target, "motion_cfg", "motion_end_hold_time_s") for target in node.targets)
        ]
        self.assertEqual(assignments, [])

    def test_full_clip_configuration_keeps_termination_disabled_and_motion_end_enabled(self):
        configure = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_configure_playback"
        )
        termination_assignments = [
            node
            for node in ast.walk(configure)
            if isinstance(node, ast.Assign)
            and any(_is_attribute(target, "motion_cfg", "terminate_on_motion_end") for target in node.targets)
        ]
        self.assertEqual(len(termination_assignments), 1)
        self.assertIsInstance(termination_assignments[0].value, ast.Constant)
        self.assertTrue(termination_assignments[0].value.value)

        disable_calls = [
            node
            for node in ast.walk(configure)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_disable_all_config_terms"
            and len(node.args) == 1
            and _is_attribute(node.args[0], "env_cfg", "terminations")
        ]
        self.assertEqual(len(disable_calls), 1)

    def test_default_full_clip_playback_executes_the_final_source_frame_once(self):
        main = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        play_loop = next(
            node
            for node in ast.walk(main)
            if isinstance(node, ast.While)
            and isinstance(node.test, ast.Call)
            and _is_attribute(node.test.func, "simulation_app", "is_running")
        )
        step_calls = [
            node
            for node in ast.walk(play_loop)
            if isinstance(node, ast.Call) and _is_attribute(node.func, "env", "step")
        ]
        self.assertEqual(len(step_calls), 1)

        final_frame_assignments = [
            node
            for node in play_loop.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "execute_final_source_frame" for target in node.targets)
        ]
        self.assertEqual(len(final_frame_assignments), 1)
        self.assertLess(final_frame_assignments[0].lineno, step_calls[0].lineno)
        self.assertIn("motion_command.motion_finished", ast.unparse(final_frame_assignments[0].value))

        stop_conditions = [
            node
            for node in play_loop.body
            if isinstance(node, ast.If)
            and "execute_final_source_frame" in ast.unparse(node.test)
            and "args_cli.stop_at_motion_end" not in ast.unparse(node.test)
        ]
        self.assertEqual(len(stop_conditions), 1)
        stop_condition = stop_conditions[0]
        self.assertGreater(stop_condition.lineno, step_calls[0].lineno)
        self.assertTrue(any(isinstance(node, ast.Break) for node in ast.walk(stop_condition)))

    def test_training_mode_does_not_install_a_motion_end_stop_check(self):
        main = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
        assignments = [
            node
            for node in ast.walk(main)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "motion_command" for target in node.targets)
        ]
        # One neutral initialization is needed for the shared loop; the only
        # real command lookup must remain inside the non-training guard.
        self.assertEqual(len(assignments), 2)
        self.assertTrue(
            any(
                isinstance(assignment.value, ast.Constant) and assignment.value.value is None
                for assignment in assignments
            )
        )
        guarded = next(
            node
            for node in main.body
            if isinstance(node, ast.If)
            and any(isinstance(child, ast.Assign) for child in node.body)
            and "args_cli.stop_at_motion_end" in ast.unparse(node.test)
        )
        condition = ast.unparse(guarded.test)
        self.assertIn("args_cli.playback_mode !=", condition)
        self.assertIn("training", condition)
        guarded_motion_assignments = [
            node
            for node in ast.walk(guarded)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "motion_command" for target in node.targets)
        ]
        self.assertEqual(len(guarded_motion_assignments), 1)


if __name__ == "__main__":
    unittest.main()
