"""Deterministic, per-NPZ acceptance evaluation for the ELF3 climb expert."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate every ELF3 climb NPZ independently from frame zero.")
parser.add_argument("--task", type=str, default="Tracking-Climb-ELF3-v0", help="Registered Isaac Lab task.")
parser.add_argument("--motion_dir", type=Path, required=True, help="Directory containing one expert's NPZ clips.")
parser.add_argument("--checkpoint", type=Path, required=True, help="RSL-RL checkpoint to evaluate.")
parser.add_argument("--trials_per_motion", type=int, default=10, help="Independent environments assigned to each NPZ.")
parser.add_argument("--seed", type=int, default=42, help="Seed controlling all evaluation inputs.")
parser.add_argument(
    "--randomized_obstacles",
    action="store_true",
    help="Use the configured size/position/yaw ranges instead of the fixed source-aligned platform.",
)
parser.add_argument(
    "--min_success_rate",
    type=float,
    default=None,
    help="Per-NPZ acceptance threshold. Defaults to 1.0 fixed or 0.9 randomized.",
)
parser.add_argument(
    "--platform_tolerance", type=float, default=1.0e-5, help="Maximum platform/height-map consistency error."
)
parser.add_argument("--json_output", type=Path, default=None, help="Optional path for a structured JSON report.")
parser.add_argument("--csv_output", type=Path, default=None, help="Optional path for a per-NPZ CSV report.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.motion_dir = args_cli.motion_dir.expanduser().resolve()
args_cli.checkpoint = args_cli.checkpoint.expanduser().resolve()
if not args_cli.motion_dir.is_dir():
    parser.error(f"Motion directory does not exist: {args_cli.motion_dir}")
motion_files = tuple(sorted(args_cli.motion_dir.glob("*.npz")))
if not motion_files:
    parser.error(f"Motion directory contains no NPZ files: {args_cli.motion_dir}")
if not args_cli.checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {args_cli.checkpoint}")
if args_cli.trials_per_motion <= 0:
    parser.error("--trials_per_motion must be positive.")
if args_cli.platform_tolerance <= 0.0:
    parser.error("--platform_tolerance must be positive.")
if args_cli.min_success_rate is not None and not 0.0 <= args_cli.min_success_rate <= 1.0:
    parser.error("--min_success_rate must be in [0, 1].")
for output_name in ("json_output", "csv_output"):
    output_path = getattr(args_cli, output_name)
    if output_path is not None:
        output_path = output_path.expanduser().resolve()
        if output_path.is_dir():
            parser.error(f"--{output_name} must name a file, got directory: {output_path}")
        setattr(args_cli, output_name, output_path)

sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import php_kvoy_reproduction.tasks  # noqa: F401
import php_kvoy_reproduction.tasks.tracking.mdp as mdp
from php_kvoy_reproduction.tasks.tracking.config.elf3.climb_env_cfg import (
    ELF3_CLIMB_HEIGHT_SCAN_VALUE_OFFSET,
    ELF3_CLIMB_RANDOM_PHASE_ENV_FRACTION,
    ELF3_CLIMB_PLATFORM_CENTER,
    ELF3_CLIMB_PLATFORM_HEIGHT_RANGE,
    ELF3_CLIMB_PLATFORM_LENGTH_RANGE,
    ELF3_CLIMB_PLATFORM_SIZE,
    ELF3_CLIMB_PLATFORM_WIDTH_RANGE,
    ELF3_CLIMB_PLATFORM_X_OFFSET_RANGE,
    ELF3_CLIMB_PLATFORM_Y_OFFSET_RANGE,
    ELF3_CLIMB_PLATFORM_YAW_RANGE,
)
from php_kvoy_reproduction.utils.climb_evaluation_report import (
    COMPLETION_ONLY_CONDITION_NAME,
    TerminalSnapshotRecorder,
    build_climb_evaluation_report,
    build_completion_only_trial_record,
    write_climb_evaluation_csv,
    write_climb_evaluation_json,
)


_TERMINAL_TERM_NAMES = ("motion_clip_end",)


def _read_motion_metadata(files: tuple[Path, ...]) -> tuple[float, list[int]]:
    fps = None
    lengths = []
    for motion_file in files:
        with np.load(motion_file, allow_pickle=False) as data:
            if "fps" not in data.files or "joint_pos" not in data.files:
                raise KeyError(f"{motion_file} is missing fps or joint_pos.")
            file_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            frame_count = int(np.asarray(data["joint_pos"]).shape[0])
        if frame_count < 2:
            raise ValueError(f"{motion_file} contains fewer than two frames.")
        if fps is None:
            fps = file_fps
        elif not np.isclose(fps, file_fps, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"FPS mismatch in {motion_file}: expected {fps:g}, got {file_fps:g}.")
        lengths.append(frame_count)
    assert fps is not None
    return fps, lengths


def _configure_obstacles(env_cfg: ManagerBasedRLEnvCfg, randomized: bool) -> None:
    if randomized:
        return
    length, width, height = ELF3_CLIMB_PLATFORM_SIZE
    env_cfg.events.platform_geometry.params["length_range"] = (length, length)
    env_cfg.events.platform_geometry.params["width_range"] = (width, width)
    env_cfg.events.platform_geometry.params["height_range"] = (height, height)
    env_cfg.events.platform_geometry.params["nominal_size_fraction"] = 1.0
    env_cfg.events.platform_pose.params["position_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
    env_cfg.events.platform_pose.params["yaw_range"] = (0.0, 0.0)


def _assert_in_range(name: str, values: torch.Tensor, value_range: tuple[float, float], tolerance: float) -> None:
    if torch.any(values < value_range[0] - tolerance) or torch.any(values > value_range[1] + tolerance):
        raise RuntimeError(
            f"Platform audit failed: {name} lies outside {value_range}; "
            f"observed [{values.min().item():.8f}, {values.max().item():.8f}]."
        )


def _audit_platform_and_height_map(env, randomized: bool, tolerance: float) -> None:
    platform_cfg = SceneEntityCfg("platform")
    platform: RigidObject = env.scene["platform"]
    errors = mdp.climb_box_geometry_errors(
        env,
        asset_cfg=platform_cfg,
        base_size=ELF3_CLIMB_PLATFORM_SIZE,
    )
    for name, values in errors.items():
        maximum = values.max().item()
        if maximum > tolerance:
            raise RuntimeError(f"Platform audit failed: {name}={maximum:.9g} exceeds {tolerance:g}.")

    sizes = mdp.get_climb_box_sizes(platform, base_size=ELF3_CLIMB_PLATFORM_SIZE, device=env.device)
    local_positions = platform.data.root_pos_w - env.scene.env_origins
    _, _, yaw = euler_xyz_from_quat(platform.data.root_quat_w)
    if randomized:
        _assert_in_range("length", sizes[:, 0], ELF3_CLIMB_PLATFORM_LENGTH_RANGE, tolerance)
        _assert_in_range("width", sizes[:, 1], ELF3_CLIMB_PLATFORM_WIDTH_RANGE, tolerance)
        _assert_in_range("height", sizes[:, 2], ELF3_CLIMB_PLATFORM_HEIGHT_RANGE, tolerance)
        _assert_in_range(
            "x offset",
            local_positions[:, 0] - ELF3_CLIMB_PLATFORM_CENTER[0],
            ELF3_CLIMB_PLATFORM_X_OFFSET_RANGE,
            tolerance,
        )
        _assert_in_range(
            "y offset",
            local_positions[:, 1] - ELF3_CLIMB_PLATFORM_CENTER[1],
            ELF3_CLIMB_PLATFORM_Y_OFFSET_RANGE,
            tolerance,
        )
        _assert_in_range("yaw", yaw, ELF3_CLIMB_PLATFORM_YAW_RANGE, tolerance)

        random_phase_mask = getattr(platform, "_climb_box_random_phase_env_mask", None)
        if random_phase_mask is None:
            # Accept snapshots/configurations created before the explicit
            # random-phase attribute was introduced.
            random_phase_mask = getattr(platform, "_climb_box_nominal_geometry_mask", None)
        if random_phase_mask is None:
            raise RuntimeError("Platform audit failed: random-phase environment mask is missing.")
        random_phase_mask = random_phase_mask.to(device=env.device, dtype=torch.bool)
        expected_random_phase_count = int(env.num_envs * ELF3_CLIMB_RANDOM_PHASE_ENV_FRACTION + 0.5)
        actual_random_phase_count = int(torch.count_nonzero(random_phase_mask).item())
        if actual_random_phase_count != expected_random_phase_count:
            raise RuntimeError(
                f"Platform audit failed: expected {expected_random_phase_count} random-phase environments, "
                f"got {actual_random_phase_count}."
            )
        if actual_random_phase_count:
            expected_random_phase_sizes = torch.tensor(
                ELF3_CLIMB_PLATFORM_SIZE, device=env.device
            ).expand(actual_random_phase_count, -1)
            random_phase_size_error = torch.max(
                torch.abs(sizes[random_phase_mask] - expected_random_phase_sizes)
            ).item()
            if random_phase_size_error > tolerance:
                raise RuntimeError(
                    "Platform audit failed: random-phase group size error "
                    f"{random_phase_size_error:.9g} exceeds {tolerance:g}."
                )
    else:
        expected_sizes = torch.tensor(ELF3_CLIMB_PLATFORM_SIZE, device=env.device).expand_as(sizes)
        expected_positions = torch.tensor(ELF3_CLIMB_PLATFORM_CENTER, device=env.device).expand_as(local_positions)
        fixed_error = max(
            torch.max(torch.abs(sizes - expected_sizes)).item(),
            torch.max(torch.abs(local_positions - expected_positions)).item(),
            torch.max(torch.abs(yaw)).item(),
        )
        if fixed_error > tolerance:
            raise RuntimeError(f"Fixed-platform audit failed: maximum pose/size error is {fixed_error:.9g}.")

    sensor_cfg = SceneEntityCfg("height_scanner")
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    scan = mdp.box_obstacle_height_scan(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=platform_cfg,
        base_size=ELF3_CLIMB_PLATFORM_SIZE,
        offset=ELF3_CLIMB_HEIGHT_SCAN_VALUE_OFFSET,
    )
    reconstructed_height = sensor.data.pos_w[:, 2, None] - scan - ELF3_CLIMB_HEIGHT_SCAN_VALUE_OFFSET
    expected_height, _ = mdp.climb_box_top_height(
        sensor.data.ray_hits_w,
        platform.data.root_pos_w,
        platform.data.root_quat_w,
        sizes,
    )
    scan_error = torch.max(torch.abs(reconstructed_height - expected_height)).item()
    if not np.isfinite(scan_error) or scan_error > tolerance:
        raise RuntimeError(f"Height-map audit failed: maximum reconstructed height error is {scan_error:.9g}.")

    center_points = platform.data.root_pos_w[:, None, :].clone()
    center_points[..., 2] = env.scene.env_origins[:, None, 2]
    center_height, center_mask = mdp.climb_box_top_height(
        center_points,
        platform.data.root_pos_w,
        platform.data.root_quat_w,
        sizes,
    )
    physical_top = platform.data.root_pos_w[:, 2] + 0.5 * sizes[:, 2]
    center_error = torch.max(torch.abs(center_height[:, 0] - physical_top)).item()
    if not torch.all(center_mask) or center_error > tolerance:
        raise RuntimeError(f"Platform-top audit failed: maximum top-height error is {center_error:.9g}.")
    print(f"[PASS] Platform collider/pose/height-map consistency (max error <= {tolerance:g}).")


def _audit_motion_contract(command, expected_files: tuple[Path, ...]) -> None:
    loaded_files = tuple(Path(path).resolve() for path in command.motion.motion_files)
    if loaded_files != tuple(path.resolve() for path in expected_files):
        raise RuntimeError(f"Motion ordering mismatch. Expected {expected_files}, loaded {loaded_files}.")

    expected_ids = torch.arange(command.num_envs, device=command.device) % command.motion.num_motions
    expected_starts = command.motion.motion_start_idx[expected_ids]
    if not torch.equal(command.motion_ids, expected_ids):
        raise RuntimeError("Round-robin motion assignment does not match env_id % num_motions.")
    if not torch.equal(command.time_steps, expected_starts):
        raise RuntimeError("At least one evaluation environment did not start at its NPZ frame zero.")

    if command.motion_end_hold_steps != 0:
        raise RuntimeError(
            "ELF3 completion-only evaluation requires motion_end_hold_steps=0; "
            f"got {command.motion_end_hold_steps}."
        )

    # Start one source frame before each end.  With zero configured hold, that
    # same update must arrive at the final frame and mark the clip complete;
    # it must not require a repeated final-source-frame update to complete.
    penultimate_frames = command.motion.motion_end_idx - 2
    _, hold_counts, completed = mdp.advance_motion_frames_with_final_hold(
        torch.arange(command.motion.num_motions, device=command.device),
        penultimate_frames,
        command.motion.motion_end_idx,
        torch.zeros_like(penultimate_frames),
        command.motion_end_hold_steps,
    )
    if not torch.all(completed) or torch.any(hold_counts != 0):
        raise RuntimeError("Motion boundary contract requires completion on the first final-source-frame update.")
    print("[PASS] NPZ sort order, frame-zero starts, and zero-hold end-of-clip boundary contract.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    fps, motion_lengths = _read_motion_metadata(motion_files)
    num_motions = len(motion_files)
    env_cfg.scene.num_envs = num_motions * args_cli.trials_per_motion
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.episode_length_s = (max(motion_lengths) + 5) / fps + env_cfg.commands.motion.motion_end_hold_time_s

    env_cfg.commands.motion.motion_file = None
    env_cfg.commands.motion.motion_dir = str(args_cli.motion_dir)
    env_cfg.commands.motion.motion_sampling_mode = "round_robin"
    env_cfg.commands.motion.start_at_motion_beginning = True
    env_cfg.commands.motion.use_adaptive_sampling = False
    env_cfg.commands.motion.terminate_on_motion_end = True
    env_cfg.commands.motion.debug_vis = False
    env_cfg.terminations.time_out = None
    # Acceptance evaluates either the exact source-aligned platform or the
    # complete configured randomization range; a training curriculum must not
    # narrow it.
    env_cfg.curriculum.platform_pose = None
    env_cfg.scene.contact_forces.debug_vis = False
    env_cfg.scene.height_scanner.debug_vis = False
    _configure_obstacles(env_cfg, args_cli.randomized_obstacles)

    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    agent_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        agent_cfg.device = args_cli.device

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env)
    try:
        command = env.unwrapped.command_manager.get_term("motion")
        _audit_motion_contract(command, motion_files)
        _audit_platform_and_height_map(env.unwrapped, args_cli.randomized_obstacles, args_cli.platform_tolerance)

        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # Evaluation only needs policy/value and observation-normalizer state.
        # Do not allocate/restore optimizer state from a training checkpoint.
        runner.load(str(args_cli.checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs, _ = env.get_observations()

        expected_ids = torch.arange(env.num_envs, device=env.device) % num_motions
        # This task accepts an episode when it reaches the authored source
        # boundary without a preceding physical failure.  No post-expert
        # standing interval or final-standing metric contract is evaluated.
        report_condition_names = (COMPLETION_ONLY_CONDITION_NAME,)
        metric_keys: tuple[str, ...] = ()
        # ManagerBasedRLEnv clears termination terms and command metrics inside
        # step() before returning, so take the terminal snapshot in _reset_idx.
        recorder = TerminalSnapshotRecorder(
            env.unwrapped,
            command,
            termination_term_names=_TERMINAL_TERM_NAMES,
            metric_keys=metric_keys,
        )
        maximum_steps = max(motion_lengths) + 2
        threshold = args_cli.min_success_rate
        if threshold is None:
            threshold = 0.9 if args_cli.randomized_obstacles else 1.0
        recorder.install()
        try:
            for _ in range(maximum_steps):
                with torch.inference_mode():
                    obs, _, _, _ = env.step(policy(obs))
                if torch.all(recorder.captured):
                    break
        finally:
            recorder.uninstall()

        if not torch.all(recorder.captured):
            missing = torch.where(~recorder.captured)[0].cpu().tolist()
            raise RuntimeError(f"Evaluation did not terminate all environments within {maximum_steps} steps: {missing}")
        if not torch.equal(recorder.motion_ids, expected_ids):
            mismatched = torch.where(recorder.motion_ids != expected_ids)[0].cpu().tolist()
            raise RuntimeError(
                "Evaluation environment changed NPZ before its first terminal snapshot; "
                f"mismatched environment ids: {mismatched}."
            )

        trial_records = []
        for env_id in range(env.num_envs):
            physically_terminated = bool(recorder.physically_terminated[env_id].item())
            completed_motion_end = bool(recorder.termination_terms["motion_clip_end"][env_id].item())
            trial_records.append(
                build_completion_only_trial_record(
                    env_id=env_id,
                    motion_id=int(recorder.motion_ids[env_id].item()),
                    completed_motion_end=completed_motion_end,
                    physically_terminated=physically_terminated,
                )
            )

        report = build_climb_evaluation_report(
            task=args_cli.task,
            checkpoint=args_cli.checkpoint,
            motion_dir=args_cli.motion_dir,
            motion_files=motion_files,
            trials_per_motion=args_cli.trials_per_motion,
            randomized_obstacles=args_cli.randomized_obstacles,
            seed=args_cli.seed,
            min_success_rate=threshold,
            required_stable_time_s=0.0,
            condition_names=report_condition_names,
            trials=trial_records,
        )

        print("\nfile | trials | completed | rate | tracking_fail | unexpected | acceptance")
        print("--- | ---: | ---: | ---: | ---: | ---: | ---")
        for motion in report["motions"]:
            print(
                f"{motion['motion_file']} | {motion['trials']} | {motion['successes']} | "
                f"{motion['success_rate']:.1%} | {motion['tracking_failures']} | "
                f"{motion['unexpected_failures']} | "
                f"{'PASS' if motion['accepted'] else 'FAIL'}"
            )

        if args_cli.json_output is not None:
            write_climb_evaluation_json(report, args_cli.json_output)
            print(f"[INFO] JSON report: {args_cli.json_output}")
        if args_cli.csv_output is not None:
            write_climb_evaluation_csv(report, args_cli.csv_output)
            print(f"[INFO] CSV report: {args_cli.csv_output}")

        print(f"\nAcceptance threshold: every NPZ >= {threshold:.1%}.")
        if not report["accepted"]:
            raise RuntimeError("ELF3 climb acceptance failed: at least one NPZ is below the required success rate.")
        print("[PASS] Every NPZ satisfies the independent success-rate acceptance threshold.")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
