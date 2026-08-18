from __future__ import annotations

import ast
import unittest
from pathlib import Path


class FirstFootholdHeightOffsetRemovalTest(unittest.TestCase):
    def test_observation_module_no_longer_exposes_a_per_foot_z_reference_override(self):
        path = (
            Path(__file__).parents[1]
            / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/observations.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function_names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertNotIn("first_foothold_height_offsets", function_names)


if __name__ == "__main__":
    unittest.main()
