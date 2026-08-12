from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class climb_box_pose_curriculum(ManagerTermBase):
    """Performance-gated curriculum for platform position and yaw randomization.

    Platform collider dimensions are authored before PhysX starts and cannot be
    changed safely by a reset curriculum.  This term therefore leaves the
    startup size distribution untouched and progressively expands only the
    reset-time x/y/yaw ranges.  A stage changes after a complete success-rate
    window, rather than merely as a function of elapsed training steps.
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        stage_scales = tuple(float(value) for value in cfg.params["stage_scales"])
        if not stage_scales or stage_scales[0] != 0.0 or stage_scales[-1] != 1.0:
            raise ValueError("stage_scales must start at 0.0 and end at 1.0.")
        if any(left >= right for left, right in zip(stage_scales, stage_scales[1:], strict=False)):
            raise ValueError(f"stage_scales must be strictly increasing, got {stage_scales}.")

        advance_threshold = float(cfg.params["advance_success_rate"])
        regress_threshold = float(cfg.params["regress_success_rate"])
        if not 0.0 <= regress_threshold < advance_threshold <= 1.0:
            raise ValueError(
                "Curriculum thresholds must satisfy 0 <= regress_success_rate < "
                "advance_success_rate <= 1."
            )
        if int(cfg.params["min_evaluated_episodes"]) <= 0:
            raise ValueError("min_evaluated_episodes must be positive.")

        self._stage_scales = stage_scales
        self._stage = 0
        self._window_successes = 0
        self._window_episodes = 0
        self._last_window_success_rate = 0.0
        self._applied_stage = -1
        self._pose_event_cfg = env.event_manager.get_term_cfg(cfg.params["event_term_name"])
        self._apply_stage(
            env,
            cfg.params["event_term_name"],
            cfg.params["full_position_range"],
            cfg.params["full_yaw_range"],
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        event_term_name: str,
        success_term_name: str,
        command_name: str,
        full_position_range: dict[str, tuple[float, float]],
        full_yaw_range: tuple[float, float],
        stage_scales: tuple[float, ...],
        advance_success_rate: float,
        regress_success_rate: float,
        min_evaluated_episodes: int,
    ) -> dict[str, float]:
        # The initial environment reset has no preceding episode and must not
        # be interpreted as a failed curriculum trial.
        if env.common_step_counter > 0:
            if isinstance(env_ids, slice):
                selected_env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
            else:
                selected_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
            command = env.command_manager.get_term(command_name)
            complete_trials = command.episode_started_at_motion_beginning[selected_env_ids]
            successful = env.termination_manager.get_term(success_term_name)[selected_env_ids] & complete_trials
            self._window_successes += int(torch.count_nonzero(successful).item())
            self._window_episodes += int(torch.count_nonzero(complete_trials).item())

        if self._window_episodes >= min_evaluated_episodes:
            success_rate = self._window_successes / self._window_episodes
            self._last_window_success_rate = success_rate
            if success_rate >= advance_success_rate and self._stage < len(self._stage_scales) - 1:
                self._stage += 1
            elif success_rate < regress_success_rate and self._stage > 0:
                self._stage -= 1
            self._window_successes = 0
            self._window_episodes = 0

        self._apply_stage(env, event_term_name, full_position_range, full_yaw_range)
        partial_rate = (
            self._window_successes / self._window_episodes
            if self._window_episodes > 0
            else self._last_window_success_rate
        )
        return {
            "stage": float(self._stage),
            "pose_range_scale": self._stage_scales[self._stage],
            "window_success_rate": partial_rate,
            "window_episodes": float(self._window_episodes),
        }

    def _apply_stage(
        self,
        env: ManagerBasedRLEnv,
        event_term_name: str,
        full_position_range: dict[str, tuple[float, float]],
        full_yaw_range: tuple[float, float],
    ) -> None:
        if self._applied_stage == self._stage:
            return
        scale = self._stage_scales[self._stage]
        self._pose_event_cfg.params["position_range"] = {
            axis: (float(value_range[0]) * scale, float(value_range[1]) * scale)
            for axis, value_range in full_position_range.items()
        }
        self._pose_event_cfg.params["yaw_range"] = (
            float(full_yaw_range[0]) * scale,
            float(full_yaw_range[1]) * scale,
        )
        env.event_manager.set_term_cfg(event_term_name, self._pose_event_cfg)
        self._applied_stage = self._stage

    def state_dict(self) -> dict[str, Any]:
        """Return the performance window and stage needed for exact resume."""

        return {
            "version": 1,
            "stage": self._stage,
            "window_successes": self._window_successes,
            "window_episodes": self._window_episodes,
            "last_window_success_rate": self._last_window_success_rate,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore a compatible curriculum state from a training checkpoint."""

        if state_dict.get("version") != 1:
            raise ValueError(f"Unsupported climb curriculum state version: {state_dict.get('version')!r}.")
        stage = int(state_dict["stage"])
        if not 0 <= stage < len(self._stage_scales):
            raise ValueError(f"Saved curriculum stage {stage} is outside the configured stages.")
        successes = int(state_dict["window_successes"])
        episodes = int(state_dict["window_episodes"])
        last_rate = float(state_dict["last_window_success_rate"])
        if successes < 0 or episodes < successes or not 0.0 <= last_rate <= 1.0:
            raise ValueError("Saved climb curriculum counters are invalid.")

        self._stage = stage
        self._window_successes = successes
        self._window_episodes = episodes
        self._last_window_success_rate = last_rate
        self._applied_stage = -1
        self._apply_stage(
            self._env,
            self.cfg.params["event_term_name"],
            self.cfg.params["full_position_range"],
            self.cfg.params["full_yaw_range"],
        )
