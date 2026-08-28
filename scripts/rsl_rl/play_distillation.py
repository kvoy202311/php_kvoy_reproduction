"""Play one routed skill using a unified visual-student checkpoint."""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Mapping
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play an ELF3 multi-skill distillation checkpoint.")
parser.add_argument("--task", type=str, default="Distillation-MultiSkill-ELF3-v0")
parser.add_argument("--num_envs", type=int, default=10)
parser.add_argument("--climb_motion_dir", type=Path, required=True)
parser.add_argument("--down_roll_motion_dir", type=Path, required=True)
parser.add_argument("--locomotion_manifest", type=Path, required=True)
parser.add_argument("--climb_manifest", type=Path, required=True)
parser.add_argument("--down_roll_manifest", type=Path, required=True)
parser.add_argument("--checkpoint_path", type=Path, required=True)
parser.add_argument(
    "--playback_mode",
    choices=("fixed_skill", "composed"),
    default="fixed_skill",
    help="Play one nominal-geometry skill or the visual climb/top-walk/down-roll composition.",
)
parser.add_argument("--skill", choices=("locomotion", "climb", "down_roll"), default=None)
parser.add_argument("--motion_id", type=int, default=0)
parser.add_argument("--vx", type=float, default=0.6)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop cleanly after this many control steps; omit for interactive playback.",
)
parser.add_argument("--no_depth_noise", action="store_true", default=False)
parser.add_argument(
    "--quiet_reset_log",
    action="store_true",
    default=False,
    help="Suppress per-reset termination summaries during interactive playback.",
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.playback_mode == "fixed_skill" and args_cli.skill is None:
    parser.error("--skill is required with --playback_mode fixed_skill")
if args_cli.playback_mode == "composed" and args_cli.skill is not None:
    parser.error("--skill must be omitted with --playback_mode composed")
if args_cli.max_steps is not None and args_cli.max_steps <= 0:
    parser.error("--max_steps must be positive when provided")


def _directory_with_npz(value: Path, option: str) -> Path:
    result = value.expanduser().resolve()
    if not result.is_dir():
        parser.error(f"{option} does not exist or is not a directory: {result}")
    if not any(result.glob("*.npz")):
        parser.error(f"{option} contains no direct .npz files: {result}")
    return result


args_cli.climb_motion_dir = _directory_with_npz(args_cli.climb_motion_dir, "--climb_motion_dir")
args_cli.down_roll_motion_dir = _directory_with_npz(
    args_cli.down_roll_motion_dir,
    "--down_roll_motion_dir",
)
for name in ("locomotion_manifest", "climb_manifest", "down_roll_manifest", "checkpoint_path"):
    value = getattr(args_cli, name).expanduser().resolve()
    if not value.is_file():
        parser.error(f"--{name} does not exist or is not a file: {value}")
    setattr(args_cli, name, value)
if args_cli.motion_id < 0:
    parser.error("--motion_id must be non-negative")
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

simulation_app = None


def _close_simulation_app() -> None:
    """Close Kit exactly once, including when a post-launch import fails."""

    global simulation_app
    if simulation_app is not None:
        app = simulation_app
        simulation_app = None
        app.close()


app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
atexit.register(_close_simulation_app)

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

import php_kvoy_reproduction.tasks  # noqa: F401
from php_kvoy_reproduction.distillation.action_contract import validate_runtime_action_contract
from php_kvoy_reproduction.distillation.runner import DistillationRunner
from php_kvoy_reproduction.distillation.teacher_manifest import TeacherManifest
from php_kvoy_reproduction.distillation.teacher_policy import TeacherPolicy
from php_kvoy_reproduction.distillation.teacher_router import TeacherRouter
from php_kvoy_reproduction.distillation.training_contract import student_policy_input_contract
from php_kvoy_reproduction.distillation.training_stage import configure_training_stage


def _router(device: str) -> TeacherRouter:
    paths = {
        "locomotion": args_cli.locomotion_manifest,
        "climb": args_cli.climb_manifest,
        "down_roll": args_cli.down_roll_manifest,
    }
    teachers = {}
    for skill, path in paths.items():
        manifest = TeacherManifest.load(path)
        manifest.verify_assets()
        if manifest.skill_name != skill:
            raise ValueError(f"manifest {path} is not the {skill} teacher")
        if manifest.control_dt != 0.02:
            raise ValueError(f"teacher {skill!r} control_dt must be exactly 0.02 s")
        teachers[skill] = TeacherPolicy.from_manifest(manifest, device=device)
    return TeacherRouter(teachers, {"locomotion": 0, "climb": 1, "down_roll": 2}).to(device)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # Match the environment curriculum to the checkpoint before constructing
    # the scene.  In particular, a nominal transition checkpoint must not be
    # silently evaluated with full-stage randomized geometry.
    checkpoint_metadata = torch.load(
        args_cli.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint_metadata, Mapping):
        raise TypeError("Distillation checkpoint payload must be a mapping.")
    checkpoint_stage = checkpoint_metadata.get("training_stage")
    if checkpoint_stage not in ("atomic", "transition", "full"):
        raise ValueError(
            "Distillation playback requires a checkpoint with one verified "
            f"training_stage, got {checkpoint_stage!r}."
        )
    configure_training_stage(env_cfg, agent_cfg, checkpoint_stage)

    command = env_cfg.commands.multi_skill
    command.climb_motion_dir = str(args_cli.climb_motion_dir)
    command.climb_motion_file = None
    command.down_roll_motion_dir = str(args_cli.down_roll_motion_dir)
    command.down_roll_motion_file = None
    command.motion_sampling_mode = "fixed"
    command.fixed_motion_id = args_cli.motion_id
    command.start_at_motion_beginning = True
    if args_cli.playback_mode == "fixed_skill":
        command.forced_skill_id = {"locomotion": 0, "climb": 1, "down_roll": 2}[args_cli.skill]
        command.composed_episode_fraction = 0.0
        # Non-nominal geometry is deliberately composed during training.  A
        # fixed-skill inspection must therefore also make every platform
        # nominal; setting only composed_episode_fraction=0 would still mix
        # direct and composed playback across environments.
        nominal_length, nominal_width, nominal_height = command.platform_size
        geometry = env_cfg.events.platform_geometry.params
        geometry["length_range"] = (nominal_length, nominal_length)
        geometry["width_range"] = (nominal_width, nominal_width)
        geometry["height_range"] = (nominal_height, nominal_height)
        geometry["nominal_size_fraction"] = 1.0
    else:
        # A composed evaluation always starts from climb.  The privileged
        # command state machine may then retain top locomotion on a long
        # platform or switch to down-roll near the real far edge; the Actor
        # still receives neither this route nor a skill identifier.
        command.forced_skill_id = 1
        command.composed_episode_fraction = 1.0
    command.forced_world_command = (args_cli.vx, args_cli.vy)
    if args_cli.no_depth_noise:
        env_cfg.observations.policy.depth.params["noise_enabled"] = False
        extrinsics = env_cfg.events.camera_extrinsics.params
        extrinsics["translation_range_m"] = (0.0, 0.0)
        extrinsics["rotation_range_rad"] = (0.0, 0.0)

    env = gym.make(args_cli.task, cfg=env_cfg)
    runner = None
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        router = _router(agent_cfg.device)
        validate_runtime_action_contract(
            router,
            env.unwrapped.action_manager.get_term("joint_pos"),
        )
        runner = DistillationRunner(
            env,
            agent_cfg.to_dict(),
            router,
            device=agent_cfg.device,
            policy_input_contract=student_policy_input_contract(
                env_cfg,
                task=args_cli.task,
            ),
        )
        runner.load(
            args_cli.checkpoint_path,
            load_optimizer=False,
            restore_training_state=True,
        )
        command_term = env.unwrapped.command_manager.get_term("multi_skill")
        completed_iteration = max(0, runner.current_learning_iteration - 1)
        termination_scale = float(command_term.student_termination_scale)
        print(
            "[INFO] Inference checkpoint: "
            f"training_stage={runner.loaded_training_stage or 'unspecified'}, "
            f"completed_iteration={completed_iteration}, "
            f"motion_termination_scale={termination_scale:.4f}, "
            "thresholds=("
            f"anchor_z={command_term.cfg.teacher_anchor_z_threshold * termination_scale:.4f}, "
            f"orientation={command_term.cfg.teacher_orientation_threshold * termination_scale:.4f}, "
            "end_effector_z="
            f"{command_term.cfg.teacher_end_effector_z_threshold * termination_scale:.4f})"
        )
        policy = runner.get_inference_policy(device=agent_cfg.device)
        observations, _ = env.get_observations()
        step_count = 0
        while simulation_app.is_running() and (
            args_cli.max_steps is None or step_count < args_cli.max_steps
        ):
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, extras = env.step(actions)
            if not args_cli.quiet_reset_log and torch.any(dones):
                log = extras.get("log", extras.get("episode", {}))
                reasons = []
                if isinstance(log, dict):
                    for key, value in sorted(log.items()):
                        if "Termination/" not in key:
                            continue
                        tensor = torch.as_tensor(value, dtype=torch.float32)
                        if tensor.numel() == 0:
                            continue
                        scalar = float(tensor.mean().item())
                        if scalar > 0.0:
                            reasons.append(f"{key.rsplit('/', 1)[-1]}={scalar:.4f}")
                suffix = ", ".join(reasons) if reasons else "reason unavailable in environment log"
                print(f"[INFO] Reset {int(torch.count_nonzero(dones).item())} env(s): {suffix}")
            step_count += 1
    finally:
        if runner is not None:
            runner.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        _close_simulation_app()
