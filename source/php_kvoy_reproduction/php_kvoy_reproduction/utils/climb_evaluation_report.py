"""Classification and structured reports for deterministic ELF3 climb evaluation."""

from __future__ import annotations

import csv
import json
import math
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch


CLIMB_EVALUATION_REPORT_VERSION = 1

OUTCOME_SUCCESS = "success"
OUTCOME_STANDING_FAILURE = "standing_failure"
OUTCOME_TRACKING_FAILURE = "tracking_failure"
OUTCOME_UNEXPECTED_FAILURE = "unexpected_failure"
VALID_OUTCOMES = (
    OUTCOME_SUCCESS,
    OUTCOME_STANDING_FAILURE,
    OUTCOME_TRACKING_FAILURE,
    OUTCOME_UNEXPECTED_FAILURE,
)


class TerminalSnapshotRecorder:
    """Capture the first pre-reset terminal state for every vector environment."""

    def __init__(
        self,
        env,
        command,
        *,
        termination_term_names: Sequence[str],
        metric_keys: Sequence[str],
    ):
        self._env = env
        self._command = command
        self._termination_term_names = tuple(termination_term_names)
        self._metric_keys = tuple(metric_keys)
        self._original_reset_idx = env._reset_idx
        self._installed = False
        self.captured = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.motion_ids = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.physically_terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.termination_terms = {
            name: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            for name in self._termination_term_names
        }
        self.metrics = {
            key: torch.full((env.num_envs,), torch.nan, dtype=torch.float, device=env.device)
            for key in self._metric_keys
        }

    def install(self) -> None:
        """Install the pre-reset hook exactly once."""

        if self._installed:
            raise RuntimeError("Terminal snapshot recorder is already installed.")

        def capture_then_reset(env_instance, env_ids):
            self.capture(env_ids)
            return self._original_reset_idx(env_ids)

        self._env._reset_idx = types.MethodType(capture_then_reset, self._env)
        self._installed = True

    def capture(self, env_ids) -> None:
        """Capture only environments that have not ended previously."""

        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self._env.device).flatten()
        new_env_ids = env_ids[~self.captured[env_ids]]
        if new_env_ids.numel() == 0:
            return
        manager = self._env.termination_manager
        self.motion_ids[new_env_ids] = self._command.motion_ids[new_env_ids]
        self.physically_terminated[new_env_ids] = manager.terminated[new_env_ids]
        for name in self._termination_term_names:
            self.termination_terms[name][new_env_ids] = manager.get_term(name)[new_env_ids]
        for key in self._metric_keys:
            self.metrics[key][new_env_ids] = self._command.metrics[key][new_env_ids]
        self.captured[new_env_ids] = True

    def uninstall(self) -> None:
        """Restore the exact bound reset method that existed before installation."""

        if self._installed:
            self._env._reset_idx = self._original_reset_idx
            self._installed = False


def classify_terminal_outcome(
    *,
    motion_end_success: bool,
    motion_end_failure: bool,
    physically_terminated: bool,
    completed_motion_end: bool,
) -> str:
    """Classify one reset using strict, mutually exclusive terminal contracts.

    A normal completed clip must set exactly one of ``motion_end_success`` and
    ``motion_end_failure`` as well as ``completed_motion_end``.  A physical
    tracking termination must set none of the clip-end flags.  Any overlap or
    incomplete combination is deliberately reported as unexpected instead of
    silently assigning it to one of the expected categories.
    """

    success = bool(motion_end_success)
    standing_failure = bool(motion_end_failure)
    tracking_failure = bool(physically_terminated)
    completed = bool(completed_motion_end)

    if success and completed and not standing_failure and not tracking_failure:
        return OUTCOME_SUCCESS
    if standing_failure and completed and not success and not tracking_failure:
        return OUTCOME_STANDING_FAILURE
    if tracking_failure and not success and not standing_failure and not completed:
        return OUTCOME_TRACKING_FAILURE
    return OUTCOME_UNEXPECTED_FAILURE


def _finite_values(values: Sequence[float]) -> list[float]:
    result = [float(value) for value in values]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("Terminal diagnostics must contain only finite values.")
    return result


def _summary(values: Sequence[float], *, include_min: bool) -> dict[str, float | None]:
    finite_values = _finite_values(values)
    if not finite_values:
        result: dict[str, float | None] = {"mean": None, "max": None}
        if include_min:
            result["min"] = None
        return result
    result = {
        "mean": sum(finite_values) / len(finite_values),
        "max": max(finite_values),
    }
    if include_min:
        result["min"] = min(finite_values)
    return result


def _percentile_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Return deterministic nearest-rank-style P50/P90/P95 summaries."""

    finite_values = sorted(_finite_values(values))
    if not finite_values:
        return {"p50": None, "p90": None, "p95": None}

    def percentile(fraction: float) -> float:
        index = round(fraction * (len(finite_values) - 1))
        return finite_values[index]

    return {"p50": percentile(0.50), "p90": percentile(0.90), "p95": percentile(0.95)}


def build_climb_evaluation_report(
    *,
    task: str,
    checkpoint: Path,
    motion_dir: Path,
    motion_files: Sequence[Path],
    trials_per_motion: int,
    randomized_obstacles: bool,
    seed: int,
    min_success_rate: float,
    required_stable_time_s: float,
    condition_names: Sequence[str],
    trials: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate exactly one terminal snapshot per environment into a report."""

    if trials_per_motion <= 0:
        raise ValueError("trials_per_motion must be positive.")
    if not 0.0 <= min_success_rate <= 1.0:
        raise ValueError("min_success_rate must lie in [0, 1].")
    if not math.isfinite(required_stable_time_s) or required_stable_time_s <= 0.0:
        raise ValueError("required_stable_time_s must be finite and positive.")

    resolved_files = tuple(Path(path).resolve() for path in motion_files)
    if not resolved_files:
        raise ValueError("motion_files must not be empty.")
    normalized_conditions = tuple(str(name) for name in condition_names)
    if not normalized_conditions or any(not name for name in normalized_conditions):
        raise ValueError("condition_names must contain non-empty names.")
    if len(set(normalized_conditions)) != len(normalized_conditions):
        raise ValueError("condition_names must be unique.")

    expected_total = len(resolved_files) * trials_per_motion
    grouped: list[list[dict[str, Any]]] = [[] for _ in resolved_files]
    seen_env_ids: set[int] = set()
    for trial in trials:
        env_id = int(trial["env_id"])
        if env_id < 0 or env_id >= expected_total:
            raise ValueError(f"Trial env_id {env_id} is outside [0, {expected_total}).")
        if env_id in seen_env_ids:
            raise ValueError(f"Environment {env_id} has more than one terminal snapshot.")
        seen_env_ids.add(env_id)
        motion_id = int(trial["motion_id"])
        if motion_id < 0 or motion_id >= len(resolved_files):
            raise ValueError(f"Trial motion_id {motion_id} is outside [0, {len(resolved_files)}).")
        expected_motion_id = env_id % len(resolved_files)
        if motion_id != expected_motion_id:
            raise ValueError(
                f"Environment {env_id} should evaluate motion {expected_motion_id}, got motion {motion_id}."
            )
        outcome = str(trial["outcome"])
        if outcome not in VALID_OUTCOMES:
            raise ValueError(f"Unknown terminal outcome: {outcome!r}.")
        condition_pass = trial["standing_condition_pass"]
        if set(condition_pass) != set(normalized_conditions):
            raise ValueError(
                "Every trial must contain exactly the configured standing conditions; "
                f"expected {normalized_conditions}, got {tuple(condition_pass)}."
            )
        grouped[motion_id].append(trial)

    if len(trials) != expected_total:
        raise ValueError(f"Expected {expected_total} terminal snapshots, got {len(trials)}.")

    motion_reports: list[dict[str, Any]] = []
    total_counts = {outcome: 0 for outcome in VALID_OUTCOMES}
    for motion_id, (motion_file, motion_trials) in enumerate(zip(resolved_files, grouped, strict=True)):
        if len(motion_trials) != trials_per_motion:
            raise ValueError(
                f"Motion {motion_id} ({motion_file.name}) expected {trials_per_motion} trials, "
                f"got {len(motion_trials)}."
            )

        counts = {outcome: 0 for outcome in VALID_OUTCOMES}
        for trial in motion_trials:
            counts[str(trial["outcome"])] += 1
        for outcome in VALID_OUTCOMES:
            total_counts[outcome] += counts[outcome]

        standing_trials = [
            trial for trial in motion_trials if trial["outcome"] == OUTCOME_STANDING_FAILURE
        ]
        condition_failures = {
            name: sum(not bool(trial["standing_condition_pass"][name]) for trial in standing_trials)
            for name in normalized_conditions
        }
        completed_trials = [
            trial
            for trial in motion_trials
            if trial["outcome"] in (OUTCOME_SUCCESS, OUTCOME_STANDING_FAILURE)
        ]
        default_rms = [float(trial["default_joint_pos_rms"]) for trial in completed_trials]
        stable_times = [float(trial["stable_time_s"]) for trial in completed_trials]
        diagnostic_names = tuple(
            sorted(
                set().union(
                    *(set(trial.get("terminal_diagnostics", {})) for trial in completed_trials)
                )
            )
        )
        for trial in completed_trials:
            if set(trial.get("terminal_diagnostics", {})) != set(diagnostic_names):
                raise ValueError("Completed trials must expose identical terminal diagnostic fields.")
        speed_diagnostics = {
            name: _percentile_summary(
                [float(trial["terminal_diagnostics"][name]) for trial in completed_trials]
            )
            for name in diagnostic_names
        }
        geometry_groups: dict[str, dict[str, Any]] = {}
        if "nominal_geometry" in diagnostic_names:
            for group_name, is_nominal in (("nominal", True), ("randomized", False)):
                group_trials = [
                    trial
                    for trial in completed_trials
                    if (float(trial["terminal_diagnostics"]["nominal_geometry"]) >= 0.5) == is_nominal
                ]
                geometry_groups[group_name] = {
                    "completed_motion_ends": len(group_trials),
                    "max_joint_speed": _percentile_summary(
                        [float(trial["terminal_diagnostics"]["max_joint_speed"]) for trial in group_trials]
                    ),
                    "joint_speed_rms": _percentile_summary(
                        [float(trial["terminal_diagnostics"]["joint_speed_rms"]) for trial in group_trials]
                    ),
                    "root_angular_speed": _percentile_summary(
                        [float(trial["terminal_diagnostics"]["root_angular_speed"]) for trial in group_trials]
                    ),
                }

        successes = counts[OUTCOME_SUCCESS]
        success_rate = successes / trials_per_motion
        motion_reports.append(
            {
                "motion_id": motion_id,
                "motion_file": motion_file.name,
                "motion_path": str(motion_file),
                "trials": trials_per_motion,
                "successes": successes,
                "success_rate": success_rate,
                "tracking_failures": counts[OUTCOME_TRACKING_FAILURE],
                "standing_failures": counts[OUTCOME_STANDING_FAILURE],
                "unexpected_failures": counts[OUTCOME_UNEXPECTED_FAILURE],
                "standing_condition_failures": condition_failures,
                "standing_condition_failure_denominator": len(standing_trials),
                "terminal_diagnostics": {
                    "completed_motion_ends": len(completed_trials),
                    "default_joint_pos_rms": _summary(default_rms, include_min=False),
                    "stable_time_s": _summary(stable_times, include_min=True),
                    "speed_and_contact_percentiles": speed_diagnostics,
                    "geometry_groups": geometry_groups,
                    "required_stable_time_s": float(required_stable_time_s),
                },
                "accepted": success_rate >= min_success_rate,
            }
        )

    accepted = all(motion["accepted"] for motion in motion_reports)
    return {
        "schema_version": CLIMB_EVALUATION_REPORT_VERSION,
        "task": task,
        "checkpoint": str(Path(checkpoint).resolve()),
        "motion_dir": str(Path(motion_dir).resolve()),
        "randomized_obstacles": bool(randomized_obstacles),
        "seed": int(seed),
        "trials_per_motion": int(trials_per_motion),
        "min_success_rate": float(min_success_rate),
        "required_stable_time_s": float(required_stable_time_s),
        "standing_condition_names": list(normalized_conditions),
        "accepted": accepted,
        "totals": {
            "trials": expected_total,
            "successes": total_counts[OUTCOME_SUCCESS],
            "tracking_failures": total_counts[OUTCOME_TRACKING_FAILURE],
            "standing_failures": total_counts[OUTCOME_STANDING_FAILURE],
            "unexpected_failures": total_counts[OUTCOME_UNEXPECTED_FAILURE],
        },
        "motions": motion_reports,
    }


def write_climb_evaluation_json(report: dict[str, Any], path: Path) -> None:
    """Write a standards-compliant, human-readable JSON report."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, allow_nan=False)
        file.write("\n")


def write_climb_evaluation_csv(
    report: dict[str, Any],
    path: Path,
    condition_names: Sequence[str] | None = None,
) -> None:
    """Write one flattened CSV row per evaluated NPZ."""

    if condition_names is None:
        condition_names = report["standing_condition_names"]
    condition_names = tuple(condition_names)
    condition_fields = [f"standing_condition_failure__{name}" for name in condition_names]
    fieldnames = [
        "schema_version",
        "task",
        "checkpoint",
        "motion_dir",
        "randomized_obstacles",
        "seed",
        "min_success_rate",
        "overall_accepted",
        "motion_file",
        "motion_path",
        "trials",
        "successes",
        "success_rate",
        "tracking_failures",
        "standing_failures",
        "unexpected_failures",
        "accepted",
        "standing_condition_failure_denominator",
        *condition_fields,
        "completed_motion_ends",
        "default_joint_pos_rms_mean",
        "default_joint_pos_rms_max",
        "stable_time_s_mean",
        "stable_time_s_min",
        "stable_time_s_max",
        "required_stable_time_s",
    ]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for motion in report["motions"]:
            diagnostic = motion["terminal_diagnostics"]
            default_rms = diagnostic["default_joint_pos_rms"]
            stable_time = diagnostic["stable_time_s"]
            row = {
                "schema_version": report["schema_version"],
                "task": report["task"],
                "checkpoint": report["checkpoint"],
                "motion_dir": report["motion_dir"],
                "randomized_obstacles": report["randomized_obstacles"],
                "seed": report["seed"],
                "min_success_rate": report["min_success_rate"],
                "overall_accepted": report["accepted"],
                "motion_file": motion["motion_file"],
                "motion_path": motion["motion_path"],
                "trials": motion["trials"],
                "successes": motion["successes"],
                "success_rate": motion["success_rate"],
                "tracking_failures": motion["tracking_failures"],
                "standing_failures": motion["standing_failures"],
                "unexpected_failures": motion["unexpected_failures"],
                "accepted": motion["accepted"],
                "standing_condition_failure_denominator": motion["standing_condition_failure_denominator"],
                "completed_motion_ends": diagnostic["completed_motion_ends"],
                "default_joint_pos_rms_mean": default_rms["mean"],
                "default_joint_pos_rms_max": default_rms["max"],
                "stable_time_s_mean": stable_time["mean"],
                "stable_time_s_min": stable_time["min"],
                "stable_time_s_max": stable_time["max"],
                "required_stable_time_s": diagnostic["required_stable_time_s"],
            }
            row.update(
                {
                    field: motion["standing_condition_failures"][name]
                    for field, name in zip(condition_fields, condition_names, strict=True)
                }
            )
            writer.writerow(row)
