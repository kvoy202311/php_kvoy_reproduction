"""Train the unified ELF3 visual student with online DAgger and PPO."""

from __future__ import annotations

import argparse
import atexit
from datetime import datetime
from pathlib import Path
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Train PHP-style three-teacher visual distillation.")
parser.add_argument("--task", type=str, default="Distillation-MultiSkill-ELF3-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--climb_motion_dir", type=Path, required=True)
parser.add_argument("--down_roll_motion_dir", type=Path, required=True)
parser.add_argument("--locomotion_manifest", type=Path, required=True)
parser.add_argument("--climb_manifest", type=Path, required=True)
parser.add_argument("--down_roll_manifest", type=Path, required=True)
parser.add_argument(
    "--training_stage",
    choices=("atomic", "transition", "full"),
    required=True,
    help=(
        "Required curriculum stage: atomic learns nominal independent skills, "
        "transition learns nominal continuous compositions, and full adds randomized geometry."
    ),
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()


def _directory_with_npz(value: Path, option: str) -> Path:
    result = value.expanduser().resolve()
    if not result.is_dir():
        parser.error(f"{option} does not exist or is not a directory: {result}")
    if not any(result.glob("*.npz")):
        parser.error(f"{option} contains no direct .npz files: {result}")
    return result


args_cli.climb_motion_dir = _directory_with_npz(args_cli.climb_motion_dir, "--climb_motion_dir")
args_cli.down_roll_motion_dir = _directory_with_npz(args_cli.down_roll_motion_dir, "--down_roll_motion_dir")
for option in ("locomotion_manifest", "climb_manifest", "down_roll_manifest"):
    value = getattr(args_cli, option).expanduser().resolve()
    if not value.is_file():
        parser.error(f"--{option} does not exist or is not a file: {value}")
    setattr(args_cli, option, value)

# The student always consumes an RTX depth tensor, even without video.
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
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import php_kvoy_reproduction.tasks  # noqa: F401
from php_kvoy_reproduction.distillation.action_contract import validate_runtime_action_contract
from php_kvoy_reproduction.distillation.runner import DistillationRunner
from php_kvoy_reproduction.distillation.teacher_manifest import TeacherManifest
from php_kvoy_reproduction.distillation.teacher_policy import TeacherPolicy
from php_kvoy_reproduction.distillation.teacher_router import TeacherRouter
from php_kvoy_reproduction.distillation.training_contract import (
    environment_training_contract,
    student_policy_input_contract,
)
from php_kvoy_reproduction.distillation.training_stage import (
    configure_training_stage,
    validate_training_stage_checkpoint_mode,
)


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _teacher_router(device: str) -> TeacherRouter:
    manifest_paths = {
        "locomotion": args_cli.locomotion_manifest,
        "climb": args_cli.climb_manifest,
        "down_roll": args_cli.down_roll_manifest,
    }
    teachers = {}
    for skill, path in manifest_paths.items():
        manifest = TeacherManifest.load(path)
        if manifest.skill_name != skill:
            raise ValueError(f"{path} declares skill {manifest.skill_name!r}, expected {skill!r}")
        manifest.verify_assets()
        if manifest.control_dt != 0.02:
            raise ValueError(f"teacher {skill!r} control_dt must be exactly 0.02 s")
        teachers[skill] = TeacherPolicy.from_manifest(manifest, device=device)
    return TeacherRouter(
        teachers,
        {"locomotion": 0, "climb": 1, "down_roll": 2},
    ).to(device)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs or env_cfg.scene.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed

    command_cfg = env_cfg.commands.multi_skill
    command_cfg.climb_motion_file = None
    command_cfg.climb_motion_dir = str(args_cli.climb_motion_dir)
    command_cfg.down_roll_motion_file = None
    command_cfg.down_roll_motion_dir = str(args_cli.down_roll_motion_dir)
    configure_training_stage(env_cfg, agent_cfg, args_cli.training_stage)
    validate_training_stage_checkpoint_mode(
        args_cli.training_stage,
        resume=bool(agent_cfg.resume),
        warm_start=bool(args_cli.warm_start),
    )

    log_root = Path("logs/rsl_rl") / agent_cfg.experiment_name
    log_root = log_root.resolve()
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_name += f"_{agent_cfg.run_name}"
    log_dir = log_root / run_name
    print(f"[INFO] Distillation training stage: {args_cli.training_stage}")
    print(f"[INFO] Distillation log directory: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    runner = None
    try:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        router = _teacher_router(agent_cfg.device)
        validate_runtime_action_contract(
            router,
            env.unwrapped.action_manager.get_term("joint_pos"),
        )
        runner = DistillationRunner(
            env,
            agent_cfg.to_dict(),
            router,
            log_dir=log_dir,
            device=agent_cfg.device,
            environment_contract=environment_training_contract(
                env_cfg,
                task=args_cli.task,
                climb_motion_dir=args_cli.climb_motion_dir,
                down_roll_motion_dir=args_cli.down_roll_motion_dir,
            ),
            policy_input_contract=student_policy_input_contract(
                env_cfg,
                task=args_cli.task,
            ),
            training_stage=args_cli.training_stage,
        )
        runner.add_git_repo_to_log(__file__)

        if agent_cfg.resume or args_cli.warm_start:
            checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)
            mode = "Warm-starting from" if args_cli.warm_start else "Resuming"
            print(f"[INFO] {mode} distillation checkpoint: {checkpoint}")
            runner.load(checkpoint, load_optimizer=not args_cli.warm_start)

        # Isaac Lab 4.5's IO helpers predate pathlib support and call
        # ``endswith`` directly on their filename argument.
        dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
        dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)
        dump_pickle(str(log_dir / "params" / "env.pkl"), env_cfg)
        dump_pickle(str(log_dir / "params" / "agent.pkl"), agent_cfg)
        manifest_record = {
            "locomotion": str(args_cli.locomotion_manifest),
            "climb": str(args_cli.climb_manifest),
            "down_roll": str(args_cli.down_roll_manifest),
        }
        dump_yaml(str(log_dir / "params" / "teachers.yaml"), manifest_record)

        runner.learn(agent_cfg.max_iterations, init_at_random_ep_len=False)
    finally:
        if runner is not None:
            runner.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        _close_simulation_app()
