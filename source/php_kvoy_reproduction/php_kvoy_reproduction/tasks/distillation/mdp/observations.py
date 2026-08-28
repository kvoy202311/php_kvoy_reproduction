"""Observation terms for the unified multi-skill distillation environment."""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import TiledCamera

from php_kvoy_reproduction.tasks.tracking.mdp.observations import box_obstacle_height_scan

from .commands import MultiSkillCommand
from .depth_buffer import PeriodicCaptureSchedule, TimestampedDepthBuffer, preprocess_depth_image


def _command(env, command_name: str) -> MultiSkillCommand:
    command = env.command_manager.get_term(command_name)
    if not isinstance(command, MultiSkillCommand):
        raise TypeError(f"command {command_name!r} must be a MultiSkillCommand")
    return command


def student_proprio_frame(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One 93-D frame in the immutable student-history order."""

    robot = env.scene[asset_cfg.name]
    frame = torch.cat(
        (
            robot.data.projected_gravity_b,
            robot.data.root_ang_vel_b,
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel - robot.data.default_joint_vel,
            env.action_manager.action,
        ),
        dim=1,
    )
    if frame.shape != (env.num_envs, 93):
        raise RuntimeError(f"student proprio frame must be [N, 93], got {tuple(frame.shape)}")
    return frame


def world_planar_command(env, command_name: str) -> torch.Tensor:
    return _command(env, command_name).world_command.clone()


def body_planar_command(env, command_name: str) -> torch.Tensor:
    """Express the public world-frame command in the observable body frame.

    The deployment API remains ``(vx, vy)`` in world coordinates.  Feeding
    that vector directly to a yaw-invariant proprioceptive policy is
    ambiguous, however, because absolute heading is not observed.  This
    rotation exposes exactly the forward/lateral correction the student must
    execute without adding privileged yaw.
    """

    command = _command(env, command_name)
    heading = command.robot.data.heading_w
    cosine = torch.cos(heading)
    sine = torch.sin(heading)
    # This is the latest deployment request, not the privileged
    # locomotion-teacher command.  It may keep changing during a committed
    # climb/down-roll and does not become zero during an internally scheduled
    # settle, so the Actor must infer when to ignore it from vision and
    # proprioceptive history just as it will on hardware.
    velocity = command.requested_world_command
    return torch.stack(
        (
            cosine * velocity[:, 0] + sine * velocity[:, 1],
            -sine * velocity[:, 0] + cosine * velocity[:, 1],
        ),
        dim=1,
    )


def skill_id(env, command_name: str) -> torch.Tensor:
    return _command(env, command_name).skill_ids.to(dtype=torch.float32).unsqueeze(1)


def skill_one_hot(env, command_name: str) -> torch.Tensor:
    ids = _command(env, command_name).skill_ids
    return torch.nn.functional.one_hot(ids, num_classes=3).to(dtype=torch.float32)


def teacher_valid(env, command_name: str) -> torch.Tensor:
    return _command(env, command_name).teacher_valid.to(dtype=torch.float32).unsqueeze(1)


def locomotion_teacher_frame(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Exact 102-D per-step contract of the frozen TienKung locomotion actor."""

    robot = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    phase = command.gait_phase
    frame = torch.cat(
        (
            robot.data.root_ang_vel_b,
            robot.data.projected_gravity_b,
            command.locomotion_teacher_command_b,
            robot.data.joint_pos - robot.data.default_joint_pos,
            robot.data.joint_vel - robot.data.default_joint_vel,
            env.action_manager.action,
            torch.sin(2.0 * torch.pi * phase),
            torch.cos(2.0 * torch.pi * phase),
            command.phase_ratio,
        ),
        dim=1,
    )
    if frame.shape != (env.num_envs, 102):
        raise RuntimeError(f"locomotion teacher frame must be [N, 102], got {tuple(frame.shape)}")
    return frame


class DelayedDepthObservation(ManagerTermBase):
    """Grid-independent depth capture with metric noise and timestamp delay."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        params = cfg.params
        sensor_cfg = params.get("sensor_cfg")
        if not isinstance(sensor_cfg, SceneEntityCfg):
            raise TypeError("DelayedDepthObservation requires sensor_cfg=SceneEntityCfg(...)")
        sensor = env.scene[sensor_cfg.name]
        if not isinstance(sensor, TiledCamera):
            raise TypeError(f"sensor {sensor_cfg.name!r} must be a TiledCamera")
        # Do not retain the camera itself here.  ManagerTermBase already keeps
        # the environment alive and ObservationManager stores class terms in a
        # reference cycle.  A second strong reference to the TiledCamera then
        # postpones its destructor until SimulationContext.clear_all_callbacks,
        # after Replicator's weak callbacks have become invalid (Isaac Lab 4.5).
        # Looking it up through the scene keeps normal ownership with the scene,
        # so env.close() destroys the camera before clearing simulator callbacks.
        self.sensor_name = sensor_cfg.name
        self.data_type = str(params.get("data_type", "distance_to_image_plane"))
        self.height = int(params.get("height", 58))
        self.width = int(params.get("width", 87))
        self.near_clip = float(params.get("near_clip", 0.15))
        self.far_clip = float(params.get("far_clip", 2.0))
        self.image_offset_range = tuple(params.get("image_offset_range", (-0.03, 0.03)))
        self.pixel_noise_std = float(params.get("pixel_noise_std", 0.03))
        self.delay_range_s = tuple(params.get("delay_range_s", (0.06, 0.08)))
        self.capture_frequency_hz = float(params.get("capture_frequency_hz", 30.0))
        self.noise_enabled = bool(params.get("noise_enabled", True))
        if sensor.cfg.update_period != 0.0:
            raise ValueError("depth camera must use update_period=0 for observation-controlled capture")
        if not 0.0 < self.capture_frequency_hz <= 1.0 / env.step_dt:
            raise ValueError("capture_frequency_hz must be positive and no faster than the control rate")
        if tuple(sensor.cfg.data_types) != (self.data_type,):
            raise ValueError(f"depth camera must expose only {self.data_type!r}")
        if (sensor.cfg.height, sensor.cfg.width) != (self.height, self.width):
            raise ValueError("depth observation shape does not match TiledCameraCfg")
        if not (
            len(self.image_offset_range) == 2
            and self.image_offset_range[0] <= self.image_offset_range[1]
            and len(self.delay_range_s) == 2
            and 0.0 <= self.delay_range_s[0] <= self.delay_range_s[1]
        ):
            raise ValueError("depth noise and delay ranges must be ordered")
        if self.pixel_noise_std < 0.0 or not math.isfinite(self.pixel_noise_std):
            raise ValueError("pixel_noise_std must be finite and non-negative")

        max_delay = self.delay_range_s[1]
        capture_period = 1.0 / self.capture_frequency_hz
        capacity = max(3, math.ceil(max_delay / capture_period) + 3)
        self.buffer = TimestampedDepthBuffer(
            env.num_envs,
            self.height,
            self.width,
            capacity,
            device=env.device,
        )
        self.capture_schedule = PeriodicCaptureSchedule(
            env.num_envs,
            self.capture_frequency_hz,
            device=env.device,
        )
        self.delay = torch.empty(env.num_envs, device=env.device)
        self.delay.uniform_(*self.delay_range_s)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = self._ids(env_ids)
        self.buffer.reset(ids)
        self.delay[ids].uniform_(*self.delay_range_s)
        now = float(self._env.common_step_counter) * self._env.step_dt
        # The post-command reset render will provide an immediate first frame;
        # no frame from the preceding episode remains eligible.
        self.capture_schedule.reset(ids, now=now)

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,  # noqa: ARG002 - part of manager signature contract
        data_type: str = "distance_to_image_plane",  # noqa: ARG002
        height: int = 58,  # noqa: ARG002
        width: int = 87,  # noqa: ARG002
        near_clip: float = 0.15,  # noqa: ARG002
        far_clip: float = 2.0,  # noqa: ARG002
        image_offset_range: tuple[float, float] = (-0.03, 0.03),  # noqa: ARG002
        pixel_noise_std: float = 0.03,  # noqa: ARG002
        delay_range_s: tuple[float, float] = (0.06, 0.08),  # noqa: ARG002
        capture_frequency_hz: float = 30.0,  # noqa: ARG002
        noise_enabled: bool = True,  # noqa: ARG002
    ) -> torch.Tensor:
        now = float(env.common_step_counter) * env.step_dt
        new_ids = self.capture_schedule.pop_due(now)
        if new_ids.numel() > 0:
            # update_period=0 keeps the RTX sensor ready for an on-demand read;
            # only scheduled environments enter the timestamped delay buffer.
            raw = self._sensor().data.output[self.data_type]
            if raw.ndim == 4 and raw.shape[-1] == 1:
                raw = raw[..., 0]
            if raw.shape != (env.num_envs, self.height, self.width):
                raise RuntimeError(
                    f"camera depth must be [N, {self.height}, {self.width}], got {tuple(raw.shape)}"
                )
            selected = raw[new_ids]
            if self.noise_enabled:
                offset = torch.empty(new_ids.numel(), device=env.device).uniform_(*self.image_offset_range)
                pixel_noise = torch.randn_like(selected) * self.pixel_noise_std
            else:
                offset = None
                pixel_noise = None
            processed = preprocess_depth_image(
                selected,
                near_clip=self.near_clip,
                far_clip=self.far_clip,
                image_offset=offset,
                pixel_noise=pixel_noise,
            )
            now = float(env.common_step_counter) * env.step_dt
            timestamps = torch.full(
                (new_ids.numel(),), now, device=env.device, dtype=torch.float64
            )
            self.buffer.append(processed, timestamps, new_ids)
        return self.buffer.select(now, self.delay).reshape(env.num_envs, -1)

    def _ids(self, env_ids: Sequence[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)

    def _sensor(self) -> TiledCamera:
        sensor = self._env.scene[self.sensor_name]
        if not isinstance(sensor, TiledCamera):
            raise TypeError(f"sensor {self.sensor_name!r} must be a TiledCamera")
        return sensor


__all__ = [
    "DelayedDepthObservation",
    "box_obstacle_height_scan",
    "locomotion_teacher_frame",
    "skill_id",
    "skill_one_hot",
    "student_proprio_frame",
    "teacher_valid",
    "body_planar_command",
    "world_planar_command",
]
