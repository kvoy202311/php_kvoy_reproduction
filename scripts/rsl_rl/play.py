"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
motion_source_group = parser.add_mutually_exclusive_group()
motion_source_group.add_argument("--motion_file", type=Path, help="Path to a local motion NPZ file.")
motion_source_group.add_argument(
    "--motion_dir", type=Path, help="Directory containing the motion NPZ clips used by one policy."
)
parser.add_argument(
    "--playback_mode",
    type=str,
    choices=("training", "full_clip", "fixed_clip"),
    default="training",
    help=(
        "Motion playback behavior. 'training' preserves the task configuration; 'full_clip' assigns clips "
        "round-robin from frame zero; 'fixed_clip' plays --motion_id from frame zero. Full-clip modes hold "
        "the last reference frame without resetting the episode."
    ),
)
parser.add_argument(
    "--motion_id",
    type=int,
    default=None,
    help="Zero-based clip index used by --playback_mode fixed_clip.",
)
parser.add_argument(
    "--free_camera",
    action="store_true",
    default=False,
    help="Use a world-frame camera so manual viewport movement is not overwritten by asset tracking.",
)
parser.add_argument(
    "--debug_vis",
    action="store_true",
    default=False,
    help="Show current-robot and expert-target tracked-body 3-D coordinate frames during full/fixed-clip playback.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

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
    if not any(args_cli.motion_dir.glob("*.npz")):
        parser.error(f"Local motion directory contains no .npz files: {args_cli.motion_dir}")
if args_cli.playback_mode == "fixed_clip":
    if args_cli.motion_id is None:
        parser.error("--playback_mode fixed_clip requires --motion_id.")
    if args_cli.motion_id < 0:
        parser.error(f"--motion_id must be non-negative, got {args_cli.motion_id}.")
elif args_cli.motion_id is not None:
    parser.error("--motion_id is valid only with --playback_mode fixed_clip.")
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
import pathlib
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import php_kvoy_reproduction.tasks  # noqa: F401
from php_kvoy_reproduction.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


def _configured_motion_count(motion_file: str | Path | None, motion_dir: str | Path | None) -> int | None:
    """Return the configured clip count when it is knowable before environment creation."""

    if motion_file is not None:
        return 1
    if motion_dir is not None:
        path = Path(motion_dir).expanduser()
        if path.is_dir():
            return len(list(path.glob("*.npz")))
    return None


def _disable_all_config_terms(cfg: object | None) -> None:
    """Disable every public manager term while preserving its configuration class."""

    if cfg is None:
        return
    for term_name in vars(cfg):
        if not term_name.startswith("_"):
            setattr(cfg, term_name, None)


def _configure_playback(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Apply viewer-only overrides without changing the registered training configuration."""

    if args_cli.free_camera:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None

    if args_cli.playback_mode == "training":
        return

    # Make every nested object we mutate private to this invocation. Some
    # configclass defaults are class-level instances and shallow Hydra config
    # copies must never leak playback overrides into another task instance.
    env_cfg.commands = copy.deepcopy(env_cfg.commands)
    env_cfg.terminations = copy.deepcopy(env_cfg.terminations)
    env_cfg.events = copy.deepcopy(env_cfg.events)
    motion_cfg = env_cfg.commands.motion
    motion_cfg.motion_sampling_mode = "fixed" if args_cli.playback_mode == "fixed_clip" else "round_robin"
    motion_cfg.fixed_motion_id = args_cli.motion_id if args_cli.motion_id is not None else 0
    motion_cfg.start_at_motion_beginning = True
    motion_cfg.use_adaptive_sampling = False
    # Keeping this true makes MotionCommand clamp permanently to the last NPZ
    # frame. Terminations are disabled below, so motion_finished cannot reset
    # or teleport the robot after the clip completes.
    motion_cfg.terminate_on_motion_end = True
    # Preserve the task's configured final hold.  Terminal reference alignment
    # may use this interval to move smoothly onto the sampled platform before
    # the stationary final pose is evaluated.  With play terminations disabled
    # below, the final NPZ frame still remains displayed indefinitely after
    # that configured interval has elapsed.
    motion_cfg.adaptive_failure_term_names = ()
    motion_cfg.random_phase_env_mask_attr = None
    # Remove reset-state perturbations as well as random phase selection. This
    # makes repeated visual inspection of the same frame-zero rollout exact.
    motion_cfg.pose_range = {key: (0.0, 0.0) for key in motion_cfg.pose_range}
    motion_cfg.velocity_range = {key: (0.0, 0.0) for key in motion_cfg.velocity_range}
    motion_cfg.joint_position_range = (0.0, 0.0)
    # Full/fixed-clip playback is normally kept visually clean.  Opt in to
    # MotionCommand's marker callback when inspecting current-vs-target poses.
    motion_cfg.debug_vis = args_cli.debug_vis

    # Playback is an inference mode: no episode boundary, early tracking
    # reset, end-of-clip classification, or training curriculum may run.
    _disable_all_config_terms(env_cfg.terminations)
    env_cfg.curriculum = None

    # The reference motions were authored against the nominal platform. Make
    # both its prestartup dimensions and reset pose deterministic and exact.
    platform_geometry = getattr(env_cfg.events, "platform_geometry", None)
    if platform_geometry is not None:
        base_size = tuple(platform_geometry.params["base_size"])
        platform_geometry.params["length_range"] = (base_size[0], base_size[0])
        platform_geometry.params["width_range"] = (base_size[1], base_size[1])
        platform_geometry.params["height_range"] = (base_size[2], base_size[2])
        platform_geometry.params["nominal_size_fraction"] = 1.0

    platform_pose = getattr(env_cfg.events, "platform_pose", None)
    if platform_pose is not None:
        platform_pose.params["position_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        platform_pose.params["yaw_range"] = (0.0, 0.0)

    mode_label = "one fixed clip" if args_cli.playback_mode == "fixed_clip" else "all clips round-robin"
    print(f"[INFO]: Playback mode: {args_cli.playback_mode} ({mode_label}, frame zero, nominal platform).")
    print("[INFO]: The final NPZ frame will be held indefinitely; all terminations and curricula are disabled.")
    if args_cli.debug_vis:
        print("[INFO]: Motion debug visualization enabled: current-robot and expert-target body frames are shown.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
        agent_cfg.device = args_cli.device

    if not isinstance(env_cfg, ManagerBasedRLEnvCfg) and (
        args_cli.playback_mode != "training" or args_cli.free_camera
    ):
        raise TypeError("Playback-mode and free-camera overrides currently require a manager-based environment.")

    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        if args_cli.playback_mode != "training" and (
            env_cfg.commands is None or not hasattr(env_cfg.commands, "motion")
        ):
            raise TypeError("Full-clip playback requires a manager-based task with a 'motion' command term.")
        _configure_playback(env_cfg)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path

        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        # loop over files in the run
        files = [file.name for file in wandb_run.files() if "model" in file.name]
        # files are all model_xxx.pt find the largest filename
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is None and args_cli.motion_dir is None:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                print("[WARN] No motion artifact found in the run and no local motion source was supplied.")
            else:
                env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")
                env_cfg.commands.motion.motion_dir = None

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = str(args_cli.motion_file)
        env_cfg.commands.motion.motion_dir = None
        print(f"[INFO]: Using local motion file: {args_cli.motion_file}")
    elif args_cli.motion_dir is not None:
        env_cfg.commands.motion.motion_file = None
        env_cfg.commands.motion.motion_dir = str(args_cli.motion_dir)
        motion_count = len(list(args_cli.motion_dir.glob("*.npz")))
        print(f"[INFO]: Using {motion_count} local motion files from: {args_cli.motion_dir}")

    if args_cli.playback_mode == "fixed_clip":
        motion_count = _configured_motion_count(
            env_cfg.commands.motion.motion_file,
            env_cfg.commands.motion.motion_dir,
        )
        if motion_count is not None and args_cli.motion_id >= motion_count:
            parser.error(f"--motion_id must be in [0, {motion_count - 1}], got {args_cli.motion_id}.")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    log_dir = os.path.dirname(resume_path)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording playback video.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # Playback only needs policy/value and observation-normalizer state. Avoid
    # allocating and restoring optimizer tensors that inference never uses.
    ppo_runner.load(resume_path, load_optimizer=False)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.policy,
        normalizer=ppo_runner.obs_normalizer,
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
