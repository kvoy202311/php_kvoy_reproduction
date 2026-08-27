"""Independent runner for three-teacher PHP visuomotor distillation."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
import math
import os
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from .dagger_ppo import DAggerPPO
from .observation import BlockwiseObservationNormalizer, RunningMeanStd, VisionObservationLayout
from .training_contract import contract_fingerprint
from .vision_actor_critic import VisionActorCritic


_CHECKPOINT_FORMAT = "php_multi_teacher_student_v2"


@dataclass(frozen=True)
class RunnerObservationKeys:
    """Names of the observation groups produced by the unified environment."""

    critic: str = "critic"
    route: str = "skill_id"
    distill_mask: str = "teacher_valid"
    teacher_locomotion: str = "locomotion_teacher"
    teacher_climb: str = "motion_teacher"
    teacher_down_roll: str = "motion_teacher"

    def teacher_groups(self) -> dict[str, str]:
        return {
            "locomotion": self.teacher_locomotion,
            "climb": self.teacher_climb,
            "down_roll": self.teacher_down_roll,
        }


@dataclass
class _RolloutState:
    actor_observations: torch.Tensor
    critic_observations: torch.Tensor
    observation_groups: Mapping[str, torch.Tensor]


def _require_flat_group(groups: Mapping[str, Any], key: str, num_envs: int) -> torch.Tensor:
    if key not in groups:
        raise KeyError(f"Environment did not provide required observation group {key!r}.")
    value = groups[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Observation group {key!r} must be a concatenated tensor; got {type(value).__name__}."
        )
    if value.ndim != 2 or value.shape[0] != num_envs:
        raise ValueError(
            f"Observation group {key!r} must have shape [{num_envs}, features], "
            f"got {tuple(value.shape)}."
        )
    if value.dtype.is_floating_point and not torch.isfinite(value).all():
        raise ValueError(f"Observation group {key!r} contains NaN or infinity.")
    return value


def _cpu_cuda_rng_states(value: Any) -> list[torch.Tensor]:
    """Validate serialized CUDA RNG states and return CPU byte tensors."""

    if not isinstance(value, (list, tuple)) or not value:
        raise TypeError("Checkpoint cuda_rng_state_all must be a non-empty list or tuple.")
    result: list[torch.Tensor] = []
    for index, state in enumerate(value):
        if not isinstance(state, torch.Tensor) or state.dtype != torch.uint8 or state.ndim != 1:
            raise TypeError(
                f"Checkpoint CUDA RNG state {index} must be a one-dimensional torch.uint8 tensor."
            )
        result.append(state.detach().cpu())
    return result


class DistillationRunner:
    """Collect student rollouts and optimize the joint DAgger-PPO objective.

    This class intentionally does not inherit RSL-RL's ``OnPolicyRunner``.
    That runner dispatches only the built-in PPO and pure-distillation storage
    types, neither of which can represent the combined objective.
    """

    def __init__(
        self,
        env,
        train_cfg: dict[str, Any],
        teacher_router,
        log_dir: str | os.PathLike[str] | None = None,
        device: str | torch.device = "cpu",
        environment_contract: Mapping[str, Any] | None = None,
    ) -> None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size != 1:
            raise RuntimeError(
                "The first multi-teacher runner is single-process only; "
                f"WORLD_SIZE={world_size} would route teachers without synchronized normalization."
            )

        self.cfg = deepcopy(train_cfg)
        runner_contract_keys = (
            "algorithm",
            "clip_actions",
            "environment_iteration_command",
            "num_steps_per_env",
            "observation_keys",
            "observation_layout",
            "policy",
        )
        runner_contract = {
            key: deepcopy(train_cfg.get(key))
            for key in runner_contract_keys
            if key in train_cfg
        }
        if environment_contract is not None and not isinstance(environment_contract, Mapping):
            raise TypeError("environment_contract must be a mapping or None")
        self.resume_contract_fingerprints = {
            "runner": contract_fingerprint(runner_contract),
            "environment": contract_fingerprint(dict(environment_contract or {})),
        }
        self.device = torch.device(device)
        self.env = env
        self.teacher_router = teacher_router
        self.log_dir = None if log_dir is None else Path(log_dir).expanduser().resolve()
        self.writer = None
        self.git_status_repos: list[str] = []

        if int(env.num_actions) != 29:
            raise ValueError(f"The ELF3 student action dimension must be 29, got {env.num_actions}.")
        if self.cfg.get("empirical_normalization", False):
            raise ValueError(
                "Set empirical_normalization=False. DistillationRunner applies block-wise "
                "proprioception normalization and must not normalize depth pixels as one flat vector."
            )

        layout_cfg = dict(self.cfg.pop("observation_layout", {}))
        self.layout = VisionObservationLayout(**layout_cfg)
        keys_cfg = dict(self.cfg.pop("observation_keys", {}))
        self.observation_keys = RunnerObservationKeys(**keys_cfg)
        iteration_command = self.cfg.pop("environment_iteration_command", None)
        if iteration_command is not None and (
            not isinstance(iteration_command, str) or not iteration_command
        ):
            raise ValueError("environment_iteration_command must be null or a non-empty string")
        self.environment_iteration_command = iteration_command

        raw_actor_obs, initial_extras = self.env.get_observations()
        initial_groups = self._groups_from_extras(initial_extras)
        raw_actor_obs = raw_actor_obs.to(self.device)
        self.layout.validate(raw_actor_obs)
        critic_sample = _require_flat_group(
            initial_groups,
            self.observation_keys.critic,
            self.env.num_envs,
        ).to(self.device)
        self._validate_route_groups(initial_groups)

        policy_cfg = dict(self.cfg["policy"])
        policy_class_name = policy_cfg.pop("class_name", "VisionActorCritic")
        if policy_class_name != "VisionActorCritic":
            raise ValueError(
                f"DistillationRunner requires VisionActorCritic, got {policy_class_name!r}."
            )
        self.policy = VisionActorCritic(
            self.layout.actor_obs_dim,
            critic_sample.shape[1],
            env.num_actions,
            layout=self.layout,
            **policy_cfg,
        ).to(self.device)

        algorithm_cfg = dict(self.cfg["algorithm"])
        algorithm_class_name = algorithm_cfg.pop("class_name", "DAggerPPO")
        if algorithm_class_name != "DAggerPPO":
            raise ValueError(f"DistillationRunner requires DAggerPPO, got {algorithm_class_name!r}.")
        for unsupported_key in ("symmetry_cfg", "rnd_cfg"):
            unsupported_value = algorithm_cfg.pop(unsupported_key, None)
            if unsupported_value is not None:
                raise ValueError(
                    f"{unsupported_key} is not implemented by DAggerPPO and must be None."
                )
        configured_skills = tuple(
            algorithm_cfg.get("skill_names", ("locomotion", "climb", "down_roll"))
        )
        router_skills = tuple(getattr(self.teacher_router, "skill_names", ()))
        if router_skills != configured_skills:
            raise ValueError(
                "Teacher-router skill order must exactly match the algorithm: "
                f"router={router_skills}, algorithm={configured_skills}."
            )
        self.alg = DAggerPPO(self.policy, device=self.device, **algorithm_cfg)

        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg.get("save_interval", 1000))
        self.log_interval = int(self.cfg.get("log_interval", 1))
        if self.num_steps_per_env <= 0 or self.save_interval <= 0 or self.log_interval <= 0:
            raise ValueError("num_steps_per_env, save_interval, and log_interval must be positive.")
        rollout_size = self.env.num_envs * self.num_steps_per_env
        if rollout_size % self.alg.num_mini_batches != 0:
            raise ValueError(
                f"Rollout size {rollout_size} is not divisible by "
                f"num_mini_batches={self.alg.num_mini_batches}."
            )

        self.actor_normalizer = BlockwiseObservationNormalizer(self.layout).to(self.device)
        self.critic_normalizer = RunningMeanStd(critic_sample.shape[1]).to(self.device)
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.layout.actor_obs_dim],
            [critic_sample.shape[1]],
            [self.env.num_actions],
        )

        self.current_learning_iteration = 0  # absolute next iteration
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self._initial_pack = (raw_actor_obs, initial_groups)
        self._rollout_state: _RolloutState | None = None
        self._last_infos: dict[str, Any] | None = None

        self.logger_type = str(self.cfg.get("logger", "tensorboard")).lower()
        if self.logger_type != "tensorboard":
            raise ValueError(
                "DistillationRunner currently supports logger='tensorboard' only; "
                f"got {self.logger_type!r}."
            )

    def _groups_from_extras(self, extras: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
        groups = extras.get("observations")
        if not isinstance(groups, Mapping):
            raise KeyError("Environment extras must contain a mapping at extras['observations'].")
        return groups

    def _validate_route_groups(self, groups: Mapping[str, Any]) -> None:
        route = _require_flat_group(groups, self.observation_keys.route, self.env.num_envs)
        mask = _require_flat_group(groups, self.observation_keys.distill_mask, self.env.num_envs)
        if route.shape[1] != 1 or mask.shape[1] != 1:
            raise ValueError("teacher_route and distill_mask observation groups must each have one column.")
        for group_name in self.observation_keys.teacher_groups().values():
            _require_flat_group(groups, group_name, self.env.num_envs)

    def _normalize_pack(
        self,
        raw_actor_obs: torch.Tensor,
        groups: Mapping[str, torch.Tensor],
        *,
        update: bool,
    ) -> _RolloutState:
        raw_actor_obs = raw_actor_obs.to(self.device)
        actor_obs = self.actor_normalizer.transform(raw_actor_obs, update=update)
        critic_raw = _require_flat_group(
            groups,
            self.observation_keys.critic,
            self.env.num_envs,
        ).to(self.device)
        if update:
            self.critic_normalizer.update(critic_raw)
        critic_obs = self.critic_normalizer(critic_raw)
        return _RolloutState(actor_obs, critic_obs, groups)

    def _initial_state(self) -> _RolloutState:
        if self._rollout_state is None:
            raw_actor_obs, groups = self._initial_pack
            self._rollout_state = self._normalize_pack(raw_actor_obs, groups, update=True)
            self._initial_pack = (torch.empty(0), {})
        return self._rollout_state

    def _set_environment_training_iteration(self, iteration: int) -> None:
        if self.environment_iteration_command is None:
            return
        raw_env = getattr(self.env, "unwrapped", None)
        manager = getattr(raw_env, "command_manager", None)
        if manager is None:
            raise RuntimeError(
                "environment_iteration_command is configured but the environment exposes no command manager"
            )
        command = manager.get_term(self.environment_iteration_command)
        setter = getattr(command, "set_training_iteration", None)
        if not callable(setter):
            raise RuntimeError(
                f"command {self.environment_iteration_command!r} cannot receive the training iteration"
            )
        setter(iteration)

    def _route_and_mask(self, groups: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        route_raw = _require_flat_group(groups, self.observation_keys.route, self.env.num_envs).to(self.device)
        rounded = route_raw.round()
        if not torch.equal(route_raw, rounded):
            raise ValueError("teacher_route must contain exact integer-valued skill IDs.")
        skill_ids = rounded.to(dtype=torch.long)
        if ((skill_ids < 0) | (skill_ids >= len(self.alg.skill_names))).any():
            raise ValueError(
                f"teacher_route IDs must lie in [0, {len(self.alg.skill_names)})."
            )

        mask_raw = _require_flat_group(groups, self.observation_keys.distill_mask, self.env.num_envs).to(self.device)
        if not torch.isfinite(mask_raw).all() or torch.any((mask_raw < 0.0) | (mask_raw > 1.0)):
            raise ValueError("distill_mask must contain finite confidence weights in [0, 1].")
        return skill_ids, mask_raw

    def _teacher_labels(
        self,
        groups: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        skill_ids, environment_mask = self._route_and_mask(groups)
        teacher_observations = {
            skill: _require_flat_group(groups, group_name, self.env.num_envs).to(self.device)
            for skill, group_name in self.observation_keys.teacher_groups().items()
        }
        labels = self.teacher_router.act(
            teacher_observations,
            skill_ids,
            validity_mask=environment_mask,
        )
        actions = labels.actions
        valid_mask = labels.valid_mask
        routed_ids = labels.skill_ids
        expected_action_shape = (self.env.num_envs, self.env.num_actions)
        if actions.shape != expected_action_shape:
            raise ValueError(
                f"Teacher router returned actions {tuple(actions.shape)}; expected {expected_action_shape}."
            )
        if valid_mask.shape != (self.env.num_envs, 1):
            raise ValueError("Teacher router valid_mask must have shape [num_envs, 1].")
        if routed_ids.shape != (self.env.num_envs, 1) or not torch.equal(routed_ids, skill_ids):
            raise RuntimeError("Teacher router changed or reordered the environment skill IDs.")
        if not torch.isfinite(actions).all():
            raise ValueError("Teacher router produced NaN or infinity actions.")
        if torch.any((valid_mask > 0.0) & (environment_mask <= 0.0)):
            raise RuntimeError("Teacher router marked an environment valid outside the environment distill mask.")
        return actions, valid_mask, skill_ids

    def _initialize_writer(self) -> None:
        if self.log_dir is None or self.writer is not None:
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=10)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run ``num_learning_iterations`` updates from the current checkpoint."""

        if num_learning_iterations <= 0:
            raise ValueError("num_learning_iterations must be positive.")
        if init_at_random_ep_len:
            raise ValueError(
                "Random generic episode lengths are incompatible with independent routed motion episodes."
            )
        self._initialize_writer()
        self.policy.train()
        if hasattr(self.teacher_router, "eval"):
            self.teacher_router.eval()

        state = self._initial_state()
        episode_infos: list[dict[str, Any]] = []
        reward_buffer: deque[float] = deque(maxlen=100)
        length_buffer: deque[float] = deque(maxlen=100)
        current_rewards = torch.zeros(self.env.num_envs, device=self.device)
        current_lengths = torch.zeros(self.env.num_envs, device=self.device)

        start_iteration = self.current_learning_iteration
        end_iteration = start_iteration + num_learning_iterations
        for iteration in range(start_iteration, end_iteration):
            self._set_environment_training_iteration(iteration)
            collection_start = time.perf_counter()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    teacher_actions, dagger_mask, skill_ids = self._teacher_labels(state.observation_groups)
                    student_actions = self.alg.act(
                        state.actor_observations,
                        state.critic_observations,
                        teacher_actions,
                        dagger_mask,
                        skill_ids,
                    )
                    # This is the central online-DAgger invariant: only the
                    # sampled student action advances physics.
                    raw_next_obs, rewards, dones, infos = self.env.step(
                        student_actions.to(self.env.device)
                    )
                    next_groups = self._groups_from_extras(infos)
                    self._validate_route_groups(next_groups)
                    next_state = self._normalize_pack(raw_next_obs, next_groups, update=True)

                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    state = next_state

                    current_rewards += rewards.reshape(-1)
                    current_lengths += 1
                    completed = torch.nonzero(dones.reshape(-1) > 0, as_tuple=False).squeeze(-1)
                    if completed.numel() > 0:
                        reward_buffer.extend(current_rewards[completed].cpu().tolist())
                        length_buffer.extend(current_lengths[completed].cpu().tolist())
                        current_rewards[completed] = 0.0
                        current_lengths[completed] = 0.0
                    if "episode" in infos and isinstance(infos["episode"], Mapping):
                        episode_infos.append(dict(infos["episode"]))
                    elif "log" in infos and isinstance(infos["log"], Mapping):
                        episode_infos.append(dict(infos["log"]))
                    self._last_infos = infos

                self.alg.compute_returns(state.critic_observations)
            collection_time = time.perf_counter() - collection_start

            learning_start = time.perf_counter()
            loss_metrics = self.alg.update(iteration)
            learning_time = time.perf_counter() - learning_start
            self.current_learning_iteration = iteration + 1
            self._rollout_state = state
            self.tot_timesteps += self.env.num_envs * self.num_steps_per_env
            self.tot_time += collection_time + learning_time

            if iteration % self.log_interval == 0:
                self._log_iteration(
                    iteration,
                    end_iteration,
                    collection_time,
                    learning_time,
                    loss_metrics,
                    episode_infos,
                    reward_buffer,
                    length_buffer,
                )
            episode_infos.clear()

            if self.log_dir is not None and iteration % self.save_interval == 0:
                self.save(self.log_dir / f"model_{iteration}.pt")

        if self.log_dir is not None:
            final_completed_iteration = self.current_learning_iteration - 1
            self.save(self.log_dir / f"model_{final_completed_iteration}.pt")

    def _log_iteration(
        self,
        iteration: int,
        end_iteration: int,
        collection_time: float,
        learning_time: float,
        loss_metrics: Mapping[str, float],
        episode_infos: list[dict[str, Any]],
        reward_buffer: deque[float],
        length_buffer: deque[float],
    ) -> None:
        iteration_time = collection_time + learning_time
        fps = int(self.env.num_envs * self.num_steps_per_env / max(iteration_time, 1.0e-9))
        if self.writer is not None:
            for name, value in loss_metrics.items():
                self.writer.add_scalar(f"Loss/{name}", value, iteration)
            self.writer.add_scalar("Perf/total_fps", fps, iteration)
            self.writer.add_scalar("Perf/collection_time", collection_time, iteration)
            self.writer.add_scalar("Perf/learning_time", learning_time, iteration)
            self.writer.add_scalar("Policy/mean_noise_std", self.policy.action_std.mean().item(), iteration)
            if reward_buffer:
                self.writer.add_scalar("Train/mean_reward", statistics.mean(reward_buffer), iteration)
                self.writer.add_scalar("Train/mean_episode_length", statistics.mean(length_buffer), iteration)
            for episode_info in episode_infos:
                for name, value in episode_info.items():
                    tensor = torch.as_tensor(value, device=self.device, dtype=torch.float32)
                    if tensor.numel() > 0:
                        self.writer.add_scalar(name, tensor.mean().item(), iteration)

        mean_reward = statistics.mean(reward_buffer) if reward_buffer else float("nan")
        mean_length = statistics.mean(length_buffer) if length_buffer else float("nan")
        print(
            f"Iteration {iteration}/{end_iteration - 1} | {fps} steps/s | "
            f"reward {mean_reward:.2f} | episode length {mean_length:.2f} | "
            f"DAgger {loss_metrics['dagger']:.6f} (w={loss_metrics['dagger_weight']:.3f}) | "
            f"PPO {loss_metrics['ppo_total']:.6f} (w={loss_metrics['ppo_weight']:.3f})"
        )

    def _teacher_fingerprints(self) -> Mapping[str, str]:
        fingerprints = self.teacher_router.fingerprints()
        if not isinstance(fingerprints, Mapping):
            raise TypeError("TeacherRouter.fingerprints() must return a mapping.")
        result = {str(key): str(value) for key, value in fingerprints.items()}
        if tuple(result) != tuple(self.alg.skill_names):
            raise ValueError(
                "Teacher fingerprints must preserve algorithm skill order: "
                f"got {tuple(result)}, expected {self.alg.skill_names}."
            )
        return result

    def save(self, path: str | os.PathLike[str], infos: Mapping[str, Any] | None = None) -> None:
        """Save student/optimizer/normalizers and the absolute next iteration."""

        destination = Path(path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "format": _CHECKPOINT_FORMAT,
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "actor_norm_state_dict": self.actor_normalizer.state_dict(),
            "critic_norm_state_dict": self.critic_normalizer.state_dict(),
            "next_iteration": self.current_learning_iteration,
            "total_timesteps": self.tot_timesteps,
            "total_time": self.tot_time,
            "teacher_fingerprints": dict(self._teacher_fingerprints()),
            "resume_contract_fingerprints": dict(self.resume_contract_fingerprints),
            "torch_rng_state": torch.get_rng_state(),
            "infos": dict(infos or {}),
        }
        if torch.cuda.is_available():
            checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
        torch.save(checkpoint, destination)

    def load(self, path: str | os.PathLike[str], load_optimizer: bool = True) -> Mapping[str, Any]:
        """Strictly resume a student checkpoint without resetting the PHP schedule."""

        if self._rollout_state is not None:
            raise RuntimeError("Load a checkpoint before learn(); an active rollout cannot be replaced safely.")
        source = Path(path).expanduser().resolve()
        checkpoint = torch.load(source, map_location=self.device, weights_only=True)
        if checkpoint.get("format") != _CHECKPOINT_FORMAT:
            raise ValueError(f"Unsupported distillation checkpoint format in {source}.")
        saved_fingerprints = checkpoint.get("teacher_fingerprints")
        current_fingerprints = dict(self._teacher_fingerprints())
        if saved_fingerprints != current_fingerprints:
            raise ValueError(
                "Frozen teachers differ from the checkpoint; refusing to continue one schedule with "
                f"different labels. saved={saved_fingerprints}, current={current_fingerprints}."
            )
        if load_optimizer:
            saved_contract = checkpoint.get("resume_contract_fingerprints")
            if saved_contract != self.resume_contract_fingerprints:
                raise ValueError(
                    "Training or environment semantics differ from the checkpoint; refusing a strict resume. "
                    f"saved={saved_contract}, current={self.resume_contract_fingerprints}. "
                    "Use warm_start only for an intentional new training schedule."
                )

        self.policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.actor_normalizer.load_state_dict(checkpoint["actor_norm_state_dict"], strict=True)
        self.critic_normalizer.load_state_dict(checkpoint["critic_norm_state_dict"], strict=True)
        if load_optimizer:
            self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            optimizer_learning_rates = {
                float(parameter_group["lr"])
                for parameter_group in self.alg.optimizer.param_groups
            }
            if len(optimizer_learning_rates) != 1:
                raise ValueError(
                    "DAggerPPO checkpoints must contain one common optimizer learning rate; "
                    f"got {sorted(optimizer_learning_rates)}."
                )
            restored_learning_rate = optimizer_learning_rates.pop()
            if not math.isfinite(restored_learning_rate) or restored_learning_rate <= 0.0:
                raise ValueError(
                    "Checkpoint optimizer learning rate must be finite and positive; "
                    f"got {restored_learning_rate}."
                )
            # Adaptive KL updates read this scalar before writing every
            # optimizer parameter group.  Restoring only the optimizer state
            # would therefore reset a previously adapted rate to the config
            # default on the first update after resume.
            self.alg.learning_rate = restored_learning_rate
            self.current_learning_iteration = int(checkpoint["next_iteration"])
            self.tot_timesteps = int(checkpoint.get("total_timesteps", 0))
            self.tot_time = float(checkpoint.get("total_time", 0.0))
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
            if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
                # ``map_location=self.device`` also moves serialized RNG byte
                # tensors to CUDA, while PyTorch's RNG API requires CPU
                # ByteTensors even when restoring CUDA generators.
                torch.cuda.set_rng_state_all(_cpu_cuda_rng_states(checkpoint["cuda_rng_state_all"]))
        else:
            self.current_learning_iteration = 0
            self.tot_timesteps = 0
            self.tot_time = 0.0
        infos = checkpoint.get("infos", {})
        if not isinstance(infos, Mapping):
            raise TypeError("Checkpoint infos must be a mapping.")
        return infos

    def get_inference_policy(self, device: str | torch.device | None = None):
        """Return deterministic student inference with frozen actor moments."""

        inference_device = self.device if device is None else torch.device(device)
        self.policy.eval().to(inference_device)
        self.actor_normalizer.eval().to(inference_device)

        def inference(raw_observations: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                normalized = self.actor_normalizer(raw_observations.to(inference_device))
                return self.policy.act_inference(normalized)

        return inference

    def close(self) -> None:
        """Flush and close the optional TensorBoard writer."""

        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None

    def add_git_repo_to_log(self, repo_file_path: str | os.PathLike[str]) -> None:
        """Retain the familiar runner hook for training-script compatibility."""

        self.git_status_repos.append(str(Path(repo_file_path).expanduser().resolve()))
