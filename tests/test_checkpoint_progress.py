from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


module_path = (
    Path(__file__).parents[1]
    / "source/php_kvoy_reproduction/php_kvoy_reproduction/utils/checkpoint_progress.py"
)
spec = importlib.util.spec_from_file_location("php_kvoy_reproduction_checkpoint_progress", module_path)
checkpoint_progress = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(checkpoint_progress)


class CheckpointProgressTest(unittest.TestCase):
    def test_legacy_checkpoint_resumes_after_last_completed_iteration(self):
        self.assertEqual(checkpoint_progress.resolve_resume_iteration({}, loaded_iteration=14), 15)
        self.assertEqual(checkpoint_progress.resolve_resume_iteration(None, loaded_iteration=0), 1)

    def test_versioned_progress_round_trip(self):
        state = checkpoint_progress.build_runner_progress_state(14)
        infos = {checkpoint_progress.RUNNER_PROGRESS_CHECKPOINT_KEY: state}

        self.assertEqual(
            state,
            {"version": 1, "last_completed_iteration": 14, "next_iteration": 15},
        )
        self.assertEqual(checkpoint_progress.resolve_resume_iteration(infos, loaded_iteration=14), 15)

    def test_rejects_metadata_that_disagrees_with_rsl_rl_iteration(self):
        infos = {
            checkpoint_progress.RUNNER_PROGRESS_CHECKPOINT_KEY: {
                "version": 1,
                "last_completed_iteration": 13,
                "next_iteration": 14,
            }
        }

        with self.assertRaisesRegex(ValueError, "disagrees"):
            checkpoint_progress.resolve_resume_iteration(infos, loaded_iteration=14)

    def test_rejects_skipped_or_repeated_next_iteration(self):
        for invalid_next in (14, 16):
            with self.subTest(next_iteration=invalid_next):
                infos = {
                    checkpoint_progress.RUNNER_PROGRESS_CHECKPOINT_KEY: {
                        "version": 1,
                        "last_completed_iteration": 14,
                        "next_iteration": invalid_next,
                    }
                }
                with self.assertRaisesRegex(ValueError, "resume immediately"):
                    checkpoint_progress.resolve_resume_iteration(infos, loaded_iteration=14)

    def test_rejects_invalid_types_and_versions(self):
        with self.assertRaises(TypeError):
            checkpoint_progress.build_runner_progress_state(True)
        with self.assertRaises(ValueError):
            checkpoint_progress.build_runner_progress_state(-1)

        infos = {
            checkpoint_progress.RUNNER_PROGRESS_CHECKPOINT_KEY: {
                "version": 2,
                "last_completed_iteration": 14,
                "next_iteration": 15,
            }
        }
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            checkpoint_progress.resolve_resume_iteration(infos, loaded_iteration=14)


if __name__ == "__main__":
    unittest.main()
