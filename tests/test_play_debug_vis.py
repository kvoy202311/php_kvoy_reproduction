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

    def test_full_clip_configuration_preserves_the_task_final_hold(self):
        """Playback must not disable a configured terminal alignment window."""

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


if __name__ == "__main__":
    unittest.main()
