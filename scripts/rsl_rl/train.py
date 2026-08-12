# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
motion_source_group = parser.add_mutually_exclusive_group(required=True)
motion_source_group.add_argument("--motion_file", type=Path, help="Path to a local motion NPZ file.")
motion_source_group.add_argument(
    "--motion_dir", type=Path, help="Directory containing the motion NPZ clips for one expert."
)
motion_source_group.add_argument("--registry_name", type=str, help="The name of the W&B motion registry artifact.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Validate a local motion path before launching Isaac Sim so invalid input does
# not initialize the simulator or allocate GPU resources.
if args_cli.motion_file is not None:
    args_cli.motion_file = args_cli.motion_file.expanduser().resolve()
    if not args_cli.motion_file.is_file():
        parser.error(f"Local motion file does not exist or is not a file: {args_cli.motion_file}")
    if args_cli.motion_file.suffix.lower() != ".npz":
        parser.error(f"Local motion file must have a .npz suffix: {args_cli.motion_file}")
elif args_cli.motion_dir is not None:
    args_cli.motion_dir = args_cli.motion_dir.expanduser().resolve()
    if not args_cli.motion_dir.is_dir():
        parser.error(f"Local motion directory does not exist or is not a directory: {args_cli.motion_dir}")
    motion_files = sorted(args_cli.motion_dir.glob("*.npz"))
    if not motion_files:
        parser.error(f"Local motion directory contains no .npz files: {args_cli.motion_dir}")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import php_kvoy_reproduction.tasks  # noqa: F401
from php_kvoy_reproduction.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    # Select exactly one motion source. argparse enforces that a local file and
    # a W&B registry artifact cannot be supplied at the same time.
    registry_name = None
    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = str(args_cli.motion_file)
        env_cfg.commands.motion.motion_dir = None
        print(f"[INFO] Using local motion file: {args_cli.motion_file}")
    elif args_cli.motion_dir is not None:
        env_cfg.commands.motion.motion_file = None
        env_cfg.commands.motion.motion_dir = str(args_cli.motion_dir)
        motion_count = len(list(args_cli.motion_dir.glob("*.npz")))
        print(f"[INFO] Using {motion_count} local motion files from: {args_cli.motion_dir}")
    else:
        registry_name = args_cli.registry_name
        if ":" not in registry_name:  # Check if the registry name includes alias, if not, append ":latest"
            registry_name += ":latest"

        import wandb

        api = wandb.Api()
        artifact = api.artifact(registry_name)
        env_cfg.commands.motion.motion_file = str(Path(artifact.download()) / "motion.npz")
        env_cfg.commands.motion.motion_dir = None

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # create runner from rsl-rl
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device, registry_name=registry_name
    )
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # Resolve checkpoint paths against the source experiment. The new log
    # directory remains separate for both resume and warm-start modes.
    if agent_cfg.resume or args_cli.warm_start:
        checkpoint_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        if args_cli.warm_start:
            print(f"[INFO]: Warm-starting policy/value/normalizers from: {checkpoint_path}")
            print("[INFO]: Optimizer, adaptive sampler, curriculum, and iteration state will start fresh.")
            runner.load_policy_only(checkpoint_path)
        else:
            print(f"[INFO]: Loading model checkpoint from: {checkpoint_path}")
            runner.load(checkpoint_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    # Motion clips define the episode boundaries for expert tracking. Randomizing
    # the generic episode counter would truncate part of the first motion batch
    # before its NPZ boundary and must remain disabled for this training entry.
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
