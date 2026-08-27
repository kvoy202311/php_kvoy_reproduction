"""Explicit student-observation layout and safe block-wise normalization."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class VisionObservationLayout:
    """Flat actor-observation contract used by the vision policy.

    The order is deliberately fixed to ``proprio history, command, depth``.
    Keeping the layout as data rather than scattering literal offsets through
    the policy makes an observation-term reorder fail immediately.
    """

    proprio_frame_dim: int = 93
    proprio_history_length: int = 8
    command_dim: int = 2
    depth_height: int = 58
    depth_width: int = 87

    def __post_init__(self) -> None:
        for name, value in (
            ("proprio_frame_dim", self.proprio_frame_dim),
            ("proprio_history_length", self.proprio_history_length),
            ("command_dim", self.command_dim),
            ("depth_height", self.depth_height),
            ("depth_width", self.depth_width),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")

    @property
    def proprio_dim(self) -> int:
        return self.proprio_frame_dim * self.proprio_history_length

    @property
    def depth_dim(self) -> int:
        return self.depth_height * self.depth_width

    @property
    def actor_obs_dim(self) -> int:
        return self.proprio_dim + self.command_dim + self.depth_dim

    @property
    def proprio_slice(self) -> slice:
        return slice(0, self.proprio_dim)

    @property
    def command_slice(self) -> slice:
        return slice(self.proprio_dim, self.proprio_dim + self.command_dim)

    @property
    def depth_slice(self) -> slice:
        return slice(self.proprio_dim + self.command_dim, self.actor_obs_dim)

    def validate(self, observations: torch.Tensor) -> None:
        if observations.ndim != 2:
            raise ValueError(
                "Actor observations must have shape [batch, features], "
                f"got {tuple(observations.shape)}."
            )
        if observations.shape[1] != self.actor_obs_dim:
            raise ValueError(
                f"Actor observation dimension mismatch: expected {self.actor_obs_dim} "
                f"({self.proprio_dim} proprio + {self.command_dim} command + "
                f"{self.depth_dim} depth), got {observations.shape[1]}."
            )

    def split(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.validate(observations)
        proprio = observations[:, self.proprio_slice]
        command = observations[:, self.command_slice]
        depth = observations[:, self.depth_slice].reshape(
            observations.shape[0], 1, self.depth_height, self.depth_width
        )
        return proprio, command, depth


class RunningMeanStd(nn.Module):
    """Numerically stable running moments with explicit update control."""

    def __init__(self, feature_dim: int, *, epsilon: float = 1.0e-5, clip: float | None = 10.0):
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive.")
        if clip is not None and clip <= 0.0:
            raise ValueError("clip must be positive when provided.")
        self.feature_dim = int(feature_dim)
        self.epsilon = float(epsilon)
        self.clip = clip
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("variance", torch.ones(feature_dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected values with shape [batch, {self.feature_dim}], got {tuple(values.shape)}."
            )
        if values.shape[0] == 0:
            return
        if not torch.isfinite(values).all():
            raise ValueError("Cannot update normalization statistics from non-finite observations.")

        values = values.detach()
        batch_count = values.shape[0]
        batch_mean = values.mean(dim=0)
        batch_var = values.var(dim=0, unbiased=False)
        old_count = int(self.count.item())
        if old_count == 0:
            self.mean.copy_(batch_mean)
            self.variance.copy_(batch_var)
            self.count.fill_(batch_count)
            return

        total_count = old_count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total_count)
        old_m2 = self.variance * old_count
        batch_m2 = batch_var * batch_count
        correction = delta.square() * (old_count * batch_count / total_count)
        self.mean.copy_(new_mean)
        self.variance.copy_((old_m2 + batch_m2 + correction) / total_count)
        self.count.fill_(total_count)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.feature_dim:
            raise ValueError(
                f"Expected final dimension {self.feature_dim}, got {values.shape[-1]}."
            )
        normalized = (values - self.mean) * torch.rsqrt(self.variance + self.epsilon)
        if self.clip is not None:
            normalized = normalized.clamp(-self.clip, self.clip)
        return normalized


class BlockwiseObservationNormalizer(nn.Module):
    """Normalize proprioception while leaving command and depth unchanged.

    ``transform(..., update=True)`` is intended to be called exactly once
    during rollout collection.  The returned tensor, rather than the raw
    tensor, is stored in PPO memory so changing moments cannot invalidate old
    action log-probabilities.
    """

    def __init__(
        self,
        layout: VisionObservationLayout,
        *,
        epsilon: float = 1.0e-5,
        clip: float | None = 10.0,
    ):
        super().__init__()
        self.layout = layout
        self.proprio = RunningMeanStd(layout.proprio_dim, epsilon=epsilon, clip=clip)

    def transform(self, observations: torch.Tensor, *, update: bool = False) -> torch.Tensor:
        self.layout.validate(observations)
        proprio = observations[:, self.layout.proprio_slice]
        if update:
            self.proprio.update(proprio)
        normalized_proprio = self.proprio(proprio)
        # Concatenation preserves command/depth values exactly and avoids an
        # in-place write into an environment-owned observation buffer.
        return torch.cat(
            (
                normalized_proprio,
                observations[:, self.layout.command_slice],
                observations[:, self.layout.depth_slice],
            ),
            dim=-1,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.transform(observations, update=False)
