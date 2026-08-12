from __future__ import annotations

import csv
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path

import torch


_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = (
    _ROOT
    / "source/php_kvoy_reproduction/php_kvoy_reproduction/utils/climb_evaluation_report.py"
)
_SPEC = importlib.util.spec_from_file_location("php_kvoy_reproduction_climb_evaluation_report", _MODULE_PATH)
reporting = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(reporting)


class TerminalOutcomeClassificationTest(unittest.TestCase):
    def test_only_three_canonical_contracts_receive_expected_labels(self):
        canonical = {
            (True, False, False, True): reporting.OUTCOME_SUCCESS,
            (False, True, False, True): reporting.OUTCOME_STANDING_FAILURE,
            (False, False, True, False): reporting.OUTCOME_TRACKING_FAILURE,
        }
        for success in (False, True):
            for standing_failure in (False, True):
                for terminated in (False, True):
                    for completed in (False, True):
                        flags = (success, standing_failure, terminated, completed)
                        expected = canonical.get(flags, reporting.OUTCOME_UNEXPECTED_FAILURE)
                        with self.subTest(flags=flags):
                            self.assertEqual(
                                reporting.classify_terminal_outcome(
                                    motion_end_success=success,
                                    motion_end_failure=standing_failure,
                                    physically_terminated=terminated,
                                    completed_motion_end=completed,
                                ),
                                expected,
                            )


class TerminalSnapshotRecorderTest(unittest.TestCase):
    def test_captures_pre_reset_state_once_and_restores_original_method(self):
        class _TerminationManager:
            def __init__(self):
                self.terminated = torch.tensor([False, True, False])
                self.terms = {"success": torch.tensor([True, False, False])}

            def get_term(self, name):
                return self.terms[name]

        class _Env:
            num_envs = 3
            device = "cpu"

            def __init__(self):
                self.termination_manager = _TerminationManager()
                self.reset_calls = []

            def _reset_idx(self, env_ids):
                self.reset_calls.append(torch.as_tensor(env_ids).tolist())
                self.termination_manager.terminated[:] = False
                self.termination_manager.terms["success"][:] = False
                command.metrics["metric"][:] = -1.0
                command.motion_ids[:] = 99

        env = _Env()
        command = types.SimpleNamespace(
            motion_ids=torch.tensor([0, 1, 2]),
            metrics={"metric": torch.tensor([1.0, 2.0, 3.0])},
        )
        original_reset = env._reset_idx
        recorder = reporting.TerminalSnapshotRecorder(
            env,
            command,
            termination_term_names=("success",),
            metric_keys=("metric",),
        )
        recorder.install()
        env._reset_idx(torch.tensor([0, 1]))

        self.assertTrue(torch.equal(recorder.captured, torch.tensor([True, True, False])))
        self.assertTrue(torch.equal(recorder.motion_ids[:2], torch.tensor([0, 1])))
        self.assertTrue(torch.equal(recorder.physically_terminated[:2], torch.tensor([False, True])))
        self.assertTrue(torch.equal(recorder.termination_terms["success"][:2], torch.tensor([True, False])))
        self.assertTrue(torch.equal(recorder.metrics["metric"][:2], torch.tensor([1.0, 2.0])))

        # A later reset of an already-recorded environment must not overwrite
        # its first-episode snapshot.
        env.termination_manager.terminated[0] = True
        env.termination_manager.terms["success"][0] = False
        command.motion_ids[0] = 7
        command.metrics["metric"][0] = 7.0
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(recorder.motion_ids[0].item(), 0)
        self.assertEqual(recorder.metrics["metric"][0].item(), 1.0)

        recorder.uninstall()
        self.assertEqual(env._reset_idx, original_reset)


class ClimbEvaluationReportTest(unittest.TestCase):
    def setUp(self):
        self.motion_files = (Path("/motions/a.npz"), Path("/motions/b.npz"))
        self.conditions = ("feet_inside", "upright", "continuous_stability")

    def _trial(
        self,
        env_id: int,
        motion_id: int,
        outcome: str,
        conditions: tuple[bool, bool, bool],
        default_rms: float = 0.0,
        stable_time: float = 0.0,
    ) -> dict:
        return {
            "env_id": env_id,
            "motion_id": motion_id,
            "outcome": outcome,
            "standing_condition_pass": dict(zip(self.conditions, conditions, strict=True)),
            "default_joint_pos_rms": default_rms,
            "stable_time_s": stable_time,
        }

    def _build(self):
        trials = [
            self._trial(0, 0, reporting.OUTCOME_SUCCESS, (True, True, True), 0.4, 0.26),
            self._trial(2, 0, reporting.OUTCOME_STANDING_FAILURE, (False, True, False), 0.6, 0.10),
            self._trial(1, 1, reporting.OUTCOME_TRACKING_FAILURE, (False, False, False)),
            self._trial(3, 1, reporting.OUTCOME_UNEXPECTED_FAILURE, (False, False, False)),
        ]
        return reporting.build_climb_evaluation_report(
            task="Tracking-Climb-ELF3-v0",
            checkpoint=Path("/checkpoints/model.pt"),
            motion_dir=Path("/motions"),
            motion_files=self.motion_files,
            trials_per_motion=2,
            randomized_obstacles=False,
            seed=42,
            min_success_rate=0.5,
            required_stable_time_s=0.25,
            condition_names=self.conditions,
            trials=trials,
        )

    def test_aggregates_per_motion_outcomes_and_overlapping_conditions(self):
        report = self._build()
        first, second = report["motions"]

        self.assertTrue(first["accepted"])
        self.assertEqual(first["successes"], 1)
        self.assertEqual(first["standing_failures"], 1)
        self.assertEqual(
            first["standing_condition_failures"],
            {"feet_inside": 1, "upright": 0, "continuous_stability": 1},
        )
        self.assertEqual(first["standing_condition_failure_denominator"], 1)
        self.assertEqual(first["terminal_diagnostics"]["completed_motion_ends"], 2)
        self.assertAlmostEqual(first["terminal_diagnostics"]["default_joint_pos_rms"]["mean"], 0.5)
        self.assertAlmostEqual(first["terminal_diagnostics"]["stable_time_s"]["min"], 0.10)

        self.assertFalse(second["accepted"])
        self.assertEqual(second["tracking_failures"], 1)
        self.assertEqual(second["unexpected_failures"], 1)
        self.assertEqual(second["terminal_diagnostics"]["completed_motion_ends"], 0)
        self.assertIsNone(second["terminal_diagnostics"]["default_joint_pos_rms"]["mean"])
        self.assertFalse(report["accepted"])
        self.assertEqual(
            report["totals"],
            {
                "trials": 4,
                "successes": 1,
                "tracking_failures": 1,
                "standing_failures": 1,
                "unexpected_failures": 1,
            },
        )

    def test_json_and_csv_reports_are_strict_and_per_motion(self):
        report = self._build()
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "nested/report.json"
            csv_path = Path(directory) / "nested/report.csv"
            reporting.write_climb_evaluation_json(report, json_path)
            reporting.write_climb_evaluation_csv(report, csv_path)

            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(loaded["schema_version"], reporting.CLIMB_EVALUATION_REPORT_VERSION)
            self.assertEqual(len(loaded["motions"]), 2)
            with csv_path.open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["motion_file"] for row in rows], ["a.npz", "b.npz"])
            self.assertEqual(rows[0]["standing_condition_failure__feet_inside"], "1")
            self.assertEqual(rows[1]["completed_motion_ends"], "0")

    def test_rejects_missing_or_duplicate_terminal_snapshots(self):
        with self.assertRaisesRegex(ValueError, "Expected 4 terminal snapshots"):
            reporting.build_climb_evaluation_report(
                task="task",
                checkpoint=Path("model.pt"),
                motion_dir=Path("motions"),
                motion_files=self.motion_files,
                trials_per_motion=2,
                randomized_obstacles=False,
                seed=1,
                min_success_rate=1.0,
                required_stable_time_s=0.25,
                condition_names=self.conditions,
                trials=[],
            )

        duplicated_environment = [
            self._trial(0, 0, reporting.OUTCOME_SUCCESS, (True, True, True)),
            self._trial(0, 0, reporting.OUTCOME_SUCCESS, (True, True, True)),
            self._trial(1, 1, reporting.OUTCOME_SUCCESS, (True, True, True)),
            self._trial(3, 1, reporting.OUTCOME_SUCCESS, (True, True, True)),
        ]
        with self.assertRaisesRegex(ValueError, "Environment 0 has more than one terminal snapshot"):
            reporting.build_climb_evaluation_report(
                task="task",
                checkpoint=Path("model.pt"),
                motion_dir=Path("motions"),
                motion_files=self.motion_files,
                trials_per_motion=2,
                randomized_obstacles=False,
                seed=1,
                min_success_rate=1.0,
                required_stable_time_s=0.25,
                condition_names=self.conditions,
                trials=duplicated_environment,
            )

        wrong_motion = [
            self._trial(0, 1, reporting.OUTCOME_SUCCESS, (True, True, True)),
            self._trial(1, 1, reporting.OUTCOME_SUCCESS, (True, True, True)),
            self._trial(2, 0, reporting.OUTCOME_SUCCESS, (True, True, True)),
            self._trial(3, 1, reporting.OUTCOME_SUCCESS, (True, True, True)),
        ]
        with self.assertRaisesRegex(ValueError, "Environment 0 should evaluate motion 0"):
            reporting.build_climb_evaluation_report(
                task="task",
                checkpoint=Path("model.pt"),
                motion_dir=Path("motions"),
                motion_files=self.motion_files,
                trials_per_motion=2,
                randomized_obstacles=False,
                seed=1,
                min_success_rate=1.0,
                required_stable_time_s=0.25,
                condition_names=self.conditions,
                trials=wrong_motion,
            )


if __name__ == "__main__":
    unittest.main()
