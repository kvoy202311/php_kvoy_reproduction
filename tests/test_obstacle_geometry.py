from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/obstacle_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location("php_kvoy_reproduction_obstacle", _MODULE_PATH)
obstacle = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(obstacle)


class ClimbBoxGeometryTest(unittest.TestCase):
    def test_nominal_box_top_is_exactly_point_six_five_metres(self):
        hits = torch.tensor([[[-0.95, 0.0, 0.0], [-0.50, 0.0, 0.0]]], dtype=torch.float32)
        centers = torch.tensor([[-0.95, 0.0, 0.325]], dtype=torch.float32)
        orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        sizes = torch.tensor([[0.46, 0.80, 0.65]], dtype=torch.float32)

        heights, mask = obstacle.climb_box_top_height(hits, centers, orientations, sizes)

        self.assertEqual(mask.tolist(), [[True, False]])
        self.assertTrue(torch.equal(heights, torch.tensor([[0.65, 0.0]], dtype=torch.float32)))

    def test_yaw_rotates_the_footprint_without_changing_top_height(self):
        half_yaw = math.pi / 4.0
        orientations = torch.tensor([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]], dtype=torch.float32)
        centers = torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32)
        sizes = torch.tensor([[0.40, 1.00, 0.60]], dtype=torch.float32)
        # At 90 degrees yaw, the local short x axis lies along world y.
        hits = torch.tensor([[[0.0, 0.19, 0.0], [0.0, 0.21, 0.0]]], dtype=torch.float32)

        heights, mask = obstacle.climb_box_top_height(hits, centers, orientations, sizes)

        self.assertEqual(mask.tolist(), [[True, False]])
        self.assertTrue(torch.equal(heights, torch.tensor([[0.60, 0.0]], dtype=torch.float32)))

    def test_dimension_sampling_stays_inside_all_configured_ranges(self):
        torch.manual_seed(5)
        sizes = obstacle.sample_climb_box_sizes(
            10_000,
            length_range=(0.41, 0.51),
            width_range=(0.80, 1.50),
            height_range=(0.60, 0.70),
        )

        self.assertTrue(torch.all((sizes[:, 0] >= 0.41) & (sizes[:, 0] <= 0.51)))
        self.assertTrue(torch.all((sizes[:, 1] >= 0.80) & (sizes[:, 1] <= 1.50)))
        self.assertTrue(torch.all((sizes[:, 2] >= 0.60) & (sizes[:, 2] <= 0.70)))

    def test_nominal_partition_has_an_exact_reproducible_size(self):
        env_ids = torch.arange(10, dtype=torch.long)
        mask = obstacle.nominal_environment_mask(env_ids, num_envs=10, nominal_fraction=0.5)

        self.assertEqual(mask.tolist(), [True, True, True, True, True, False, False, False, False, False])


if __name__ == "__main__":
    unittest.main()
