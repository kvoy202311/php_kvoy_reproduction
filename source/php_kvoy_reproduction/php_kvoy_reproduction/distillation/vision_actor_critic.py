"""Feed-forward visuomotor actor-critic for PHP-style distillation.

The public surface intentionally matches the feed-forward ``ActorCritic`` in
RSL-RL 2.3.x.  Unlike the upstream flat MLP, the actor decodes the explicit
student-observation layout and embeds depth before applying its policy MLP.
The critic remains a flat MLP over privileged observations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, Callable

import torch
from torch import nn
from torch.distributions import Normal

from .observation import VisionObservationLayout


_DEPTH_FEATURE_DIM = 32
_MINIMUM_DEPTH_SIDE = 15
_SUPPORTED_ACTION_DIM = 29


def _positive_dimension(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _hidden_dimensions(values: Sequence[int], name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of positive integers.")
    result = tuple(values)
    if not result:
        raise ValueError(f"{name} must contain at least one layer.")
    for index, value in enumerate(result):
        _positive_dimension(value, f"{name}[{index}]")
    return result


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    if not isinstance(name, str):
        raise TypeError(f"activation must be a string, got {type(name).__name__}.")
    factories: dict[str, Callable[[], nn.Module]] = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "crelu": nn.CELU,
        "leaky_relu": nn.LeakyReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "identity": nn.Identity,
    }
    key = name.lower()
    if key not in factories:
        supported = ", ".join(sorted(factories))
        raise ValueError(f"Unknown activation {name!r}; expected one of: {supported}.")
    return factories[key]


def _make_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation_factory: Callable[[], nn.Module],
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(activation_factory())
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


def _reference_parameter(module: nn.Module) -> nn.Parameter:
    try:
        return next(module.parameters())
    except StopIteration as exc:  # pragma: no cover - construction guarantees parameters.
        raise RuntimeError("The network has no parameters for device/dtype validation.") from exc


def _validate_matrix(
    values: torch.Tensor,
    *,
    name: str,
    feature_dim: int,
    reference_module: nn.Module,
) -> None:
    if not isinstance(values, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(values).__name__}.")
    if values.ndim != 2:
        raise ValueError(
            f"{name} must have shape [batch, {feature_dim}], got {tuple(values.shape)}."
        )
    if values.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one batch element.")
    if values.shape[1] != feature_dim:
        raise ValueError(
            f"{name} feature dimension mismatch: expected {feature_dim}, "
            f"got {values.shape[1]}."
        )
    if not values.dtype.is_floating_point:
        raise TypeError(f"{name} must use a floating dtype, got {values.dtype}.")

    reference = _reference_parameter(reference_module)
    if values.device != reference.device:
        raise ValueError(
            f"{name} is on {values.device}, but the policy is on {reference.device}; "
            "move both to the same device."
        )
    if values.dtype != reference.dtype:
        raise TypeError(
            f"{name} has dtype {values.dtype}, but the policy uses {reference.dtype}; "
            "convert the observation or the policy explicitly."
        )
    if not torch.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinity.")


def _require_finite(values: torch.Tensor, name: str) -> None:
    if not torch.isfinite(values).all():
        raise FloatingPointError(f"{name} contains NaN or infinity.")


class DepthEncoder(nn.Module):
    """Encode one configured depth image into a fixed 32-dimensional vector."""

    output_dim = _DEPTH_FEATURE_DIM

    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        if height < _MINIMUM_DEPTH_SIDE or width < _MINIMUM_DEPTH_SIDE:
            raise ValueError(
                "Depth dimensions must both be at least "
                f"{_MINIMUM_DEPTH_SIDE} for three unpadded stride-2 convolutions, "
                f"got {height}x{width}."
            )
        self.height = height
        self.width = width
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        expected = (depth.shape[0], 1, self.height, self.width) if depth.ndim >= 1 else None
        if depth.ndim != 4 or tuple(depth.shape[1:]) != (1, self.height, self.width):
            raise ValueError(
                f"Depth input must have shape [batch, 1, {self.height}, {self.width}], "
                f"got {tuple(depth.shape)}; expected {expected}."
            )
        encoded = self.network(depth)
        _require_finite(encoded, "Depth embedding")
        return encoded


class VisionActor(nn.Module):
    """Actor that owns the complete flat-observation-to-action mapping."""

    def __init__(
        self,
        layout: VisionObservationLayout,
        num_actions: int,
        hidden_dims: Sequence[int],
        activation_factory: Callable[[], nn.Module],
    ) -> None:
        super().__init__()
        self.layout = layout
        self.num_actions = num_actions
        self.depth_encoder = DepthEncoder(layout.depth_height, layout.depth_width)
        actor_input_dim = layout.proprio_dim + layout.command_dim + self.depth_encoder.output_dim
        self.mlp = _make_mlp(actor_input_dim, hidden_dims, num_actions, activation_factory)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        _validate_matrix(
            observations,
            name="Actor observations",
            feature_dim=self.layout.actor_obs_dim,
            reference_module=self,
        )
        # ``split`` is the single source of truth for offsets and depth shape.
        proprio, command, depth = self.layout.split(observations)
        depth_features = self.depth_encoder(depth)
        action_mean = self.mlp(torch.cat((proprio, command, depth_features), dim=-1))
        _require_finite(action_mean, "Actor action mean")
        return action_mean


class VisionActorCritic(nn.Module):
    """RSL-RL-compatible visuomotor policy and privileged value function."""

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        layout: VisionObservationLayout | None = None,
        actor_hidden_dims: Sequence[int] = (2048, 1024, 512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 0.01,
        noise_std_type: str = "scalar",
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(
                "VisionActorCritic received unsupported configuration keys: "
                f"{sorted(kwargs)}."
            )
        super().__init__()

        num_actor_obs = _positive_dimension(num_actor_obs, "num_actor_obs")
        num_critic_obs = _positive_dimension(num_critic_obs, "num_critic_obs")
        num_actions = _positive_dimension(num_actions, "num_actions")
        if num_actions != _SUPPORTED_ACTION_DIM:
            raise ValueError(
                f"ELF3 policies require {_SUPPORTED_ACTION_DIM} actions, got {num_actions}."
            )
        if layout is None:
            layout = VisionObservationLayout()
        if not isinstance(layout, VisionObservationLayout):
            raise TypeError(
                "layout must be a VisionObservationLayout, "
                f"got {type(layout).__name__}."
            )
        if num_actor_obs != layout.actor_obs_dim:
            raise ValueError(
                "num_actor_obs does not match the configured observation layout: "
                f"expected {layout.actor_obs_dim}, got {num_actor_obs}."
            )

        actor_hidden_dims = _hidden_dimensions(actor_hidden_dims, "actor_hidden_dims")
        critic_hidden_dims = _hidden_dimensions(critic_hidden_dims, "critic_hidden_dims")
        activation_factory = _activation_factory(activation)
        if isinstance(init_noise_std, bool) or not isinstance(init_noise_std, (int, float)):
            raise TypeError("init_noise_std must be a positive finite number.")
        if not math.isfinite(float(init_noise_std)) or float(init_noise_std) <= 0.0:
            raise ValueError(
                f"init_noise_std must be positive and finite, got {init_noise_std!r}."
            )
        if noise_std_type not in ("scalar", "log"):
            raise ValueError(
                f"Unknown standard deviation type {noise_std_type!r}; expected 'scalar' or 'log'."
            )

        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        self.layout = layout
        self.noise_std_type = noise_std_type
        self.actor = VisionActor(layout, num_actions, actor_hidden_dims, activation_factory)
        self.critic = _make_mlp(
            num_critic_obs,
            critic_hidden_dims,
            1,
            activation_factory,
        )

        initial_std = torch.full((num_actions,), float(init_noise_std))
        if noise_std_type == "scalar":
            self.std = nn.Parameter(initial_std)
        else:
            self.log_std = nn.Parameter(initial_std.log())

        self.distribution: Normal | None = None

    @staticmethod
    def init_weights(sequential: nn.Sequential, scales: Sequence[float]) -> None:
        """Retain the optional upstream RSL-RL orthogonal initializer."""

        linear_layers = [module for module in sequential if isinstance(module, nn.Linear)]
        if len(scales) < len(linear_layers):
            raise ValueError(
                f"Expected at least {len(linear_layers)} initialization scales, got {len(scales)}."
            )
        for index, module in enumerate(linear_layers):
            nn.init.orthogonal_(module.weight, gain=scales[index])

    def _require_distribution(self) -> Normal:
        if self.distribution is None:
            raise RuntimeError("Action distribution is unavailable; call act() first.")
        return self.distribution

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> VisionActorCritic:
        # A cached Normal is not a registered submodule and would otherwise
        # retain tensors on the old device/dtype after ``to``/``double``.
        result = super()._apply(fn)
        self.distribution = None
        return result

    def reset(self, dones: torch.Tensor | None = None) -> None:
        """Feed-forward policies have no hidden state to reset."""

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError("Use act(), act_inference(), or evaluate() explicitly.")

    @property
    def action_mean(self) -> torch.Tensor:
        mean = self._require_distribution().mean
        _require_finite(mean, "Action mean")
        return mean

    @property
    def action_std(self) -> torch.Tensor:
        std = self._require_distribution().stddev
        _require_finite(std, "Action standard deviation")
        if (std <= 0.0).any():
            raise FloatingPointError("Action standard deviation must remain strictly positive.")
        return std

    @property
    def entropy(self) -> torch.Tensor:
        entropy = self._require_distribution().entropy().sum(dim=-1)
        _require_finite(entropy, "Action entropy")
        return entropy

    def update_distribution(self, observations: torch.Tensor) -> None:
        mean = self.actor(observations)
        if self.noise_std_type == "scalar":
            std_parameter = self.std
        else:
            std_parameter = self.log_std.exp()
        _require_finite(std_parameter, "Action standard deviation")
        if (std_parameter <= 0.0).any():
            raise FloatingPointError("Action standard deviation must remain strictly positive.")
        std = std_parameter.expand_as(mean)
        self.distribution = Normal(mean, std, validate_args=False)

    def act(self, observations: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.update_distribution(observations)
        actions = self._require_distribution().sample()
        _require_finite(actions, "Sampled actions")
        return actions

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        distribution = self._require_distribution()
        _validate_matrix(
            actions,
            name="Actions",
            feature_dim=self.num_actions,
            reference_module=self,
        )
        if actions.shape != distribution.mean.shape:
            raise ValueError(
                "Actions must match the initialized distribution shape "
                f"{tuple(distribution.mean.shape)}, got {tuple(actions.shape)}."
            )
        log_probability = distribution.log_prob(actions).sum(dim=-1)
        _require_finite(log_probability, "Action log probability")
        return log_probability

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(observations)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        _validate_matrix(
            critic_observations,
            name="Critic observations",
            feature_dim=self.num_critic_obs,
            reference_module=self.critic,
        )
        value = self.critic(critic_observations)
        _require_finite(value, "Critic value")
        return value

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
    ) -> bool:
        """Strictly load finite parameters and return RSL-RL's resume flag."""

        if not isinstance(state_dict, Mapping):
            raise TypeError(f"state_dict must be a mapping, got {type(state_dict).__name__}.")
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point:
                if not torch.isfinite(value).all():
                    raise ValueError(f"State tensor {name!r} contains NaN or infinity.")
        super().load_state_dict(state_dict, strict=strict)
        self.distribution = None
        return True


__all__ = ["DepthEncoder", "VisionActor", "VisionActorCritic"]
