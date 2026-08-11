import os
import warnings

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from php_kvoy_reproduction.utils.checkpoint_progress import (
    RUNNER_PROGRESS_CHECKPOINT_KEY,
    build_runner_progress_state,
    resolve_resume_iteration,
)
from php_kvoy_reproduction.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


_MOTION_SAMPLER_CHECKPOINT_KEY = "whole_body_tracking_motion_sampler_v1"
_CURRICULUM_CHECKPOINT_KEY = "whole_body_tracking_curriculum_v1"
_UPSTREAM_INFOS_KEY = "rsl_rl_infos"


def _stateful_curriculum_terms(env):
    """Return curriculum callable objects that expose checkpoint state."""

    manager = env.unwrapped.curriculum_manager
    return {
        name: term_cfg.func
        for name, term_cfg in zip(manager._term_names, manager._term_cfgs, strict=True)
        if hasattr(term_cfg.func, "state_dict") and hasattr(term_cfg.func, "load_state_dict")
    }


class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_policy_as_onnx(self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save the policy together with adaptive motion-sampling state."""
        motion_command = self.env.unwrapped.command_manager.get_term("motion")
        curriculum_state = {name: term.state_dict() for name, term in _stateful_curriculum_terms(self.env).items()}
        checkpoint_infos = {
            _MOTION_SAMPLER_CHECKPOINT_KEY: motion_command.motion_sampler.state_dict(),
            _CURRICULUM_CHECKPOINT_KEY: curriculum_state,
            RUNNER_PROGRESS_CHECKPOINT_KEY: build_runner_progress_state(self.current_learning_iteration),
            _UPSTREAM_INFOS_KEY: infos,
        }
        super().save(path, checkpoint_infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            export_motion_policy_as_onnx(
                self.env.unwrapped, self.alg.policy, normalizer=self.obs_normalizer, path=policy_path, filename=filename
            )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run
            if self.registry_name is not None:
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None

    def load(self, path: str, load_optimizer: bool = True):
        """Restore training state and resume after the last completed update."""

        checkpoint_infos = super().load(path, load_optimizer=load_optimizer)
        if load_optimizer:
            last_completed_iteration = self.current_learning_iteration
            self.current_learning_iteration = resolve_resume_iteration(
                checkpoint_infos,
                loaded_iteration=last_completed_iteration,
            )
            print(
                f"[INFO]: Checkpoint completed iteration {last_completed_iteration}; "
                f"resuming from iteration {self.current_learning_iteration}."
            )
        if isinstance(checkpoint_infos, dict) and _MOTION_SAMPLER_CHECKPOINT_KEY in checkpoint_infos:
            motion_command = self.env.unwrapped.command_manager.get_term("motion")
            motion_command.motion_sampler.load_state_dict(checkpoint_infos[_MOTION_SAMPLER_CHECKPOINT_KEY])
        elif load_optimizer:
            warnings.warn(
                "Checkpoint has no adaptive motion-sampler state; policy training can resume, "
                "but phase-failure statistics will restart from zero.",
                stacklevel=2,
            )

        stateful_terms = _stateful_curriculum_terms(self.env)
        if isinstance(checkpoint_infos, dict) and _CURRICULUM_CHECKPOINT_KEY in checkpoint_infos:
            saved_terms = checkpoint_infos[_CURRICULUM_CHECKPOINT_KEY]
            if set(saved_terms) != set(stateful_terms):
                raise ValueError(
                    "Checkpoint curriculum terms do not match the current task: "
                    f"saved={sorted(saved_terms)}, current={sorted(stateful_terms)}."
                )
            for name, term in stateful_terms.items():
                term.load_state_dict(saved_terms[name])
        elif load_optimizer and stateful_terms:
            warnings.warn(
                "Checkpoint has no terrain-curriculum state; policy training can resume, "
                "but the curriculum will restart from stage zero.",
                stacklevel=2,
            )

        if isinstance(checkpoint_infos, dict) and _UPSTREAM_INFOS_KEY in checkpoint_infos:
            return checkpoint_infos[_UPSTREAM_INFOS_KEY]
        return checkpoint_infos
