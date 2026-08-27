"""Pure-PyTorch depth preprocessing and timestamp-delay buffering.

The simulator-facing observation term lives in :mod:`observations`.  Keeping
the numerical contract here independent of Isaac Lab makes its edge cases
unit-testable without launching Omniverse.
"""

from __future__ import annotations

import math

import torch


class PeriodicCaptureSchedule:
    """Per-environment wall-clock schedule independent of the control grid.

    A 30 Hz source sampled by a 50 Hz control loop cannot use one fixed
    integer number of control steps.  Advancing an absolute deadline produces
    the required 20/40 ms interval pattern without accumulating phase drift.
    """

    def __init__(
        self,
        num_envs: int,
        frequency_hz: float,
        *,
        device: str | torch.device,
    ) -> None:
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
            raise ValueError("frequency_hz must be finite and positive")
        self.num_envs = num_envs
        self.period_s = 1.0 / float(frequency_hz)
        self.device = torch.device(device)
        self.next_capture_time = torch.zeros(num_envs, device=self.device, dtype=torch.float64)

    def reset(
        self,
        env_ids: torch.Tensor | list[int] | None = None,
        *,
        now: float,
    ) -> None:
        if not math.isfinite(float(now)):
            raise ValueError("now must be finite")
        ids = self._ids(env_ids)
        self.next_capture_time[ids] = float(now)

    def pop_due(self, now: float) -> torch.Tensor:
        """Return due environments and advance each deadline beyond ``now``."""

        if not math.isfinite(float(now)):
            raise ValueError("now must be finite")
        # The tolerance is far below a simulation step and only absorbs binary
        # representation error when a deadline lies exactly on the time grid.
        tolerance_s = 1.0e-9
        due = torch.where(self.next_capture_time <= float(now) + tolerance_s)[0]
        if due.numel() > 0:
            elapsed = float(now) - self.next_capture_time[due]
            periods = torch.floor(elapsed / self.period_s + tolerance_s).clamp_min(0.0) + 1.0
            self.next_capture_time[due] += periods * self.period_s
        return due

    def _ids(self, env_ids: torch.Tensor | list[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise IndexError("environment index is outside the capture-schedule range")
        return ids


def preprocess_depth_image(
    raw_depth: torch.Tensor,
    *,
    near_clip: float = 0.15,
    far_clip: float = 2.0,
    image_offset: torch.Tensor | None = None,
    pixel_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sanitize, corrupt, clamp and linearly normalize one depth batch.

    Invalid values (NaN, either infinity, and values at or behind the near
    plane) represent a missing return and are therefore mapped to the far
    plane.  Noise is expressed in metres and is applied before clipping.
    """

    if not isinstance(raw_depth, torch.Tensor) or not raw_depth.is_floating_point():
        raise TypeError("raw_depth must be a floating-point torch.Tensor")
    if raw_depth.ndim != 3:
        raise ValueError(f"raw_depth must have shape [N, H, W], got {tuple(raw_depth.shape)}")
    if not math.isfinite(near_clip) or not math.isfinite(far_clip) or not 0.0 < near_clip < far_clip:
        raise ValueError("depth clipping bounds must satisfy 0 < near_clip < far_clip")

    depth = raw_depth.clone()
    invalid = ~torch.isfinite(depth) | (depth <= near_clip)
    depth = torch.where(invalid, torch.full_like(depth, far_clip), depth)

    if image_offset is not None:
        if image_offset.shape not in {(depth.shape[0],), (depth.shape[0], 1, 1)}:
            raise ValueError("image_offset must have shape [N] or [N, 1, 1]")
        depth = depth + image_offset.reshape(-1, 1, 1).to(device=depth.device, dtype=depth.dtype)
    if pixel_noise is not None:
        if pixel_noise.shape != depth.shape:
            raise ValueError("pixel_noise must have the same shape as raw_depth")
        depth = depth + pixel_noise.to(device=depth.device, dtype=depth.dtype)

    depth.clamp_(near_clip, far_clip)
    return (depth - near_clip) / (far_clip - near_clip)


class TimestampedDepthBuffer:
    """Per-environment ring buffer selecting the newest sufficiently old image.

    After an episode reset, old frames are invalidated.  Until the configured
    delay has elapsed, the first newly captured frame is repeated rather than
    exposing pixels from the preceding episode.
    """

    def __init__(
        self,
        num_envs: int,
        height: int,
        width: int,
        capacity: int,
        *,
        device: str | torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        dimensions = (num_envs, height, width, capacity)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("depth-buffer dimensions and capacity must be positive integers")
        self.num_envs = num_envs
        self.height = height
        self.width = width
        self.capacity = capacity
        self.device = torch.device(device)
        self.images = torch.ones(num_envs, capacity, height, width, device=self.device, dtype=dtype)
        self.timestamps = torch.full((num_envs, capacity), -torch.inf, device=self.device, dtype=torch.float64)
        self.write_index = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.valid_count = torch.zeros(num_envs, device=self.device, dtype=torch.long)

    def reset(self, env_ids: torch.Tensor | list[int] | None = None) -> None:
        ids = self._ids(env_ids)
        self.timestamps[ids] = -torch.inf
        self.images[ids] = 1.0
        self.write_index[ids] = 0
        self.valid_count[ids] = 0

    def append(
        self,
        images: torch.Tensor,
        timestamps: torch.Tensor,
        env_ids: torch.Tensor | list[int] | None = None,
    ) -> None:
        ids = self._ids(env_ids)
        expected = (ids.numel(), self.height, self.width)
        if images.shape != expected:
            raise ValueError(f"images must have shape {expected}, got {tuple(images.shape)}")
        if timestamps.shape != (ids.numel(),):
            raise ValueError("timestamps must have shape [len(env_ids)]")
        if not images.is_floating_point() or not bool(torch.isfinite(images).all()):
            raise ValueError("buffered images must be finite floating-point tensors")
        if not timestamps.is_floating_point() or not bool(torch.isfinite(timestamps).all()):
            raise ValueError("timestamps must be finite floating-point tensors")

        slots = self.write_index[ids]
        self.images[ids, slots] = images.to(device=self.device, dtype=self.images.dtype)
        self.timestamps[ids, slots] = timestamps.to(device=self.device, dtype=self.timestamps.dtype)
        self.write_index[ids] = (slots + 1) % self.capacity
        self.valid_count[ids] = torch.clamp(self.valid_count[ids] + 1, max=self.capacity)

    def select(self, now: float | torch.Tensor, delay: torch.Tensor) -> torch.Tensor:
        if isinstance(now, torch.Tensor):
            if now.ndim == 0:
                now_tensor = now.to(device=self.device, dtype=torch.float64).expand(self.num_envs)
            elif now.shape == (self.num_envs,):
                now_tensor = now.to(device=self.device, dtype=torch.float64)
            else:
                raise ValueError("now must be a scalar or shape [N]")
        else:
            if not math.isfinite(float(now)):
                raise ValueError("now must be finite")
            now_tensor = torch.full((self.num_envs,), float(now), device=self.device, dtype=torch.float64)
        if delay.shape != (self.num_envs,) or not delay.is_floating_point():
            raise ValueError("delay must be a floating-point tensor with shape [N]")
        delay = delay.to(device=self.device, dtype=torch.float64)
        if not bool(torch.isfinite(delay).all()) or bool((delay < 0.0).any()):
            raise ValueError("delay must contain finite non-negative values")

        valid = self.timestamps > -torch.inf
        # ``delay`` commonly originates from a float32 tensor.  Converting it
        # to float64 cannot recover the decimal value rounded at construction
        # time (for example 0.07 becomes slightly larger than 0.07).  Without
        # a tiny tolerance, a capture exactly on the requested delay boundary
        # can therefore be rejected for numerical rather than temporal
        # reasons.  This tolerance is over five orders of magnitude below the
        # 30 Hz capture period and cannot admit a genuinely newer frame.
        timestamp_tolerance_s = 1.0e-7
        eligible = valid & (
            self.timestamps <= (now_tensor - delay + timestamp_tolerance_s).unsqueeze(1)
        )
        eligible_times = torch.where(eligible, self.timestamps, -torch.inf)
        selected_slot = eligible_times.argmax(dim=1)

        # Warm-up fallback: choose the earliest frame from this episode.  No
        # valid frame means the normalized far image initialized at slot zero.
        no_eligible = ~eligible.any(dim=1)
        valid_times = torch.where(valid, self.timestamps, torch.inf)
        earliest_slot = valid_times.argmin(dim=1)
        selected_slot = torch.where(no_eligible, earliest_slot, selected_slot)
        env_ids = torch.arange(self.num_envs, device=self.device)
        return self.images[env_ids, selected_slot].clone()

    def _ids(self, env_ids: torch.Tensor | list[int] | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if bool(((ids < 0) | (ids >= self.num_envs)).any()):
            raise IndexError("environment index is outside the depth-buffer range")
        return ids


__all__ = ["PeriodicCaptureSchedule", "TimestampedDepthBuffer", "preprocess_depth_image"]
