from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/climb_progress.py"
)
_SPEC = importlib.util.spec_from_file_location("php_kvoy_reproduction_climb_progress", _MODULE_PATH)
climb_progress = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(climb_progress)


class BoundedEpisodeProgressIncrementTest(unittest.TestCase):
    def test_first_sample_establishes_baseline_without_reward(self):
        increment, episode_max = climb_progress.bounded_episode_progress_increment(
            torch.tensor([0.4, 0.8]),
            torch.tensor([False, False]),
            torch.zeros(2),
            max_delta_per_step=0.05,
        )

        self.assertTrue(torch.equal(increment, torch.zeros(2)))
        self.assertTrue(torch.allclose(episode_max, torch.tensor([0.4, 0.8])))

    def test_only_new_episode_maximum_is_rewarded(self):
        initialized = torch.tensor([True])
        episode_max = torch.tensor([0.2])

        increment, episode_max = climb_progress.bounded_episode_progress_increment(
            torch.tensor([0.24]), initialized, episode_max, max_delta_per_step=0.05
        )
        self.assertTrue(torch.allclose(increment, torch.tensor([0.04])))

        increment, episode_max = climb_progress.bounded_episode_progress_increment(
            torch.tensor([0.10]), initialized, episode_max, max_delta_per_step=0.05
        )
        self.assertTrue(torch.equal(increment, torch.zeros(1)))

        # Returning to the previous best cannot collect the same reward again.
        increment, episode_max = climb_progress.bounded_episode_progress_increment(
            torch.tensor([0.24]), initialized, episode_max, max_delta_per_step=0.05
        )
        self.assertTrue(torch.equal(increment, torch.zeros(1)))

        increment, episode_max = climb_progress.bounded_episode_progress_increment(
            torch.tensor([0.27]), initialized, episode_max, max_delta_per_step=0.05
        )
        self.assertTrue(torch.allclose(increment, torch.tensor([0.03])))
        self.assertTrue(torch.allclose(episode_max, torch.tensor([0.27])))

    def test_large_jump_is_capped_and_discarded_instead_of_deferred(self):
        initialized = torch.tensor([True])
        increment, episode_max = climb_progress.bounded_episode_progress_increment(
            torch.tensor([0.9]), initialized, torch.tensor([0.1]), max_delta_per_step=0.05
        )
        self.assertTrue(torch.allclose(increment, torch.tensor([0.05])))
        self.assertTrue(torch.allclose(episode_max, torch.tensor([0.9])))

        # Holding, losing, or regaining the same progress never releases the
        # clipped excess from the first jump.
        for potential in (0.9, 0.0, 0.9):
            increment, episode_max = climb_progress.bounded_episode_progress_increment(
                torch.tensor([potential]), initialized, episode_max, max_delta_per_step=0.05
            )
            self.assertTrue(torch.equal(increment, torch.zeros(1)))

    def test_parallel_environments_keep_independent_maxima(self):
        increment, episode_max = climb_progress.bounded_episode_progress_increment(
            torch.tensor([0.3, 0.7]),
            torch.tensor([True, False]),
            torch.tensor([0.2, 0.0]),
            max_delta_per_step=0.05,
        )
        self.assertTrue(torch.allclose(increment, torch.tensor([0.05, 0.0])))
        self.assertTrue(torch.allclose(episode_max, torch.tensor([0.3, 0.7])))

    def test_rejects_invalid_state(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            climb_progress.bounded_episode_progress_increment(
                torch.zeros(1), torch.zeros(1, dtype=torch.bool), torch.zeros(1), 0.0
            )
        with self.assertRaisesRegex(ValueError, "identical shapes"):
            climb_progress.bounded_episode_progress_increment(
                torch.zeros(2), torch.zeros(1, dtype=torch.bool), torch.zeros(2), 0.05
            )
        with self.assertRaisesRegex(TypeError, "boolean"):
            climb_progress.bounded_episode_progress_increment(
                torch.zeros(1), torch.zeros(1), torch.zeros(1), 0.05
            )


if __name__ == "__main__":
    unittest.main()
