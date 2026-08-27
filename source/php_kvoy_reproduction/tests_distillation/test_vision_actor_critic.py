from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import pytest
import torch
from torch import nn


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / "php_kvoy_reproduction/distillation"
)
_PACKAGE_NAME = "_vision_actor_critic_test_package"
_PACKAGE = types.ModuleType(_PACKAGE_NAME)
_PACKAGE.__path__ = [str(_MODULE_DIR)]
sys.modules[_PACKAGE_NAME] = _PACKAGE
observation = _load_module(f"{_PACKAGE_NAME}.observation", _MODULE_DIR / "observation.py")
vision_actor_critic = _load_module(
    f"{_PACKAGE_NAME}.vision_actor_critic",
    _MODULE_DIR / "vision_actor_critic.py",
)

VisionObservationLayout = observation.VisionObservationLayout
VisionActorCritic = vision_actor_critic.VisionActorCritic


def _small_policy(
    *,
    layout: VisionObservationLayout | None = None,
    critic_dim: int = 37,
    noise_std_type: str = "scalar",
) -> VisionActorCritic:
    if layout is None:
        layout = VisionObservationLayout(
            proprio_frame_dim=7,
            proprio_history_length=3,
            command_dim=2,
            depth_height=31,
            depth_width=47,
        )
    return VisionActorCritic(
        layout.actor_obs_dim,
        critic_dim,
        29,
        layout=layout,
        actor_hidden_dims=(32, 16),
        critic_hidden_dims=(16, 8),
        init_noise_std=0.01,
        noise_std_type=noise_std_type,
    )


def _linear_dimensions(module: nn.Sequential) -> list[tuple[int, int]]:
    return [
        (layer.in_features, layer.out_features)
        for layer in module
        if isinstance(layer, nn.Linear)
    ]


def test_default_architecture_matches_the_configured_php_network() -> None:
    layout = VisionObservationLayout(
        proprio_frame_dim=3,
        proprio_history_length=2,
        command_dim=2,
        depth_height=31,
        depth_width=47,
    )
    policy = VisionActorCritic(layout.actor_obs_dim, 101, 29, layout=layout)

    convolution_layers = [
        layer
        for layer in policy.actor.depth_encoder.network
        if isinstance(layer, nn.Conv2d)
    ]
    assert [(layer.in_channels, layer.out_channels) for layer in convolution_layers] == [
        (1, 16),
        (16, 32),
        (32, 32),
    ]
    assert all(layer.kernel_size == (3, 3) for layer in convolution_layers)
    assert all(layer.stride == (2, 2) for layer in convolution_layers)
    assert all(layer.padding == (0, 0) for layer in convolution_layers)
    assert sum(isinstance(layer, nn.ELU) for layer in policy.actor.depth_encoder.network) == 3
    adaptive_pool = [
        layer
        for layer in policy.actor.depth_encoder.network
        if isinstance(layer, nn.AdaptiveAvgPool2d)
    ]
    assert len(adaptive_pool) == 1
    assert adaptive_pool[0].output_size == (1, 1)

    assert _linear_dimensions(policy.actor.mlp) == [
        (layout.proprio_dim + layout.command_dim + 32, 2048),
        (2048, 1024),
        (1024, 512),
        (512, 256),
        (256, 128),
        (128, 29),
    ]
    assert _linear_dimensions(policy.critic) == [
        (101, 512),
        (512, 256),
        (256, 128),
        (128, 1),
    ]


def test_rsl_rl_surface_shapes_and_distribution_properties() -> None:
    policy = _small_policy()
    actor_obs = torch.randn(4, policy.num_actor_obs)
    critic_obs = torch.randn(4, policy.num_critic_obs)

    actions = policy.act(actor_obs)
    assert actions.shape == (4, 29)
    assert policy.action_mean.shape == (4, 29)
    assert policy.action_std.shape == (4, 29)
    assert policy.entropy.shape == (4,)
    assert policy.get_actions_log_prob(actions).shape == (4,)
    assert policy.act_inference(actor_obs).shape == (4, 29)
    assert policy.evaluate(critic_obs).shape == (4, 1)
    assert torch.isfinite(actions).all()


def test_batch_size_one_is_preserved_without_squeezing() -> None:
    policy = _small_policy()
    actor_obs = torch.randn(1, policy.num_actor_obs)
    critic_obs = torch.randn(1, policy.num_critic_obs)

    assert policy.act(actor_obs).shape == (1, 29)
    assert policy.action_mean.shape == (1, 29)
    assert policy.action_std.shape == (1, 29)
    assert policy.entropy.shape == (1,)
    assert policy.evaluate(critic_obs).shape == (1, 1)


def test_default_58_by_87_layout_runs_without_literal_offsets() -> None:
    layout = VisionObservationLayout()
    policy = _small_policy(layout=layout)

    assert layout.actor_obs_dim == 5792
    assert policy.act_inference(torch.randn(1, layout.actor_obs_dim)).shape == (1, 29)


def test_custom_depth_dimensions_are_layout_driven() -> None:
    layout = VisionObservationLayout(
        proprio_frame_dim=5,
        proprio_history_length=4,
        command_dim=3,
        depth_height=35,
        depth_width=53,
    )
    policy = _small_policy(layout=layout)
    actor_obs = torch.randn(2, layout.actor_obs_dim)

    assert policy.act_inference(actor_obs).shape == (2, 29)
    assert policy.actor.depth_encoder.height == 35
    assert policy.actor.depth_encoder.width == 53


@pytest.mark.parametrize("height,width", [(14, 31), (31, 14), (1, 1)])
def test_depth_dimensions_too_small_for_three_convolutions_are_rejected(
    height: int,
    width: int,
) -> None:
    layout = VisionObservationLayout(
        proprio_frame_dim=3,
        proprio_history_length=2,
        command_dim=2,
        depth_height=height,
        depth_width=width,
    )
    with pytest.raises(ValueError, match="at least 15"):
        _small_policy(layout=layout)


def test_constructor_rejects_layout_and_action_dimension_mismatches() -> None:
    layout = VisionObservationLayout(
        proprio_frame_dim=3,
        proprio_history_length=2,
        command_dim=2,
        depth_height=31,
        depth_width=47,
    )
    with pytest.raises(ValueError, match="num_actor_obs"):
        VisionActorCritic(layout.actor_obs_dim + 1, 12, 29, layout=layout)
    with pytest.raises(ValueError, match="29 actions"):
        VisionActorCritic(layout.actor_obs_dim, 12, 28, layout=layout)
    with pytest.raises(TypeError, match="unsupported configuration keys"):
        VisionActorCritic(layout.actor_obs_dim, 12, 29, layout=layout, misspelled_option=True)


@pytest.mark.parametrize(
    "bad_observation",
    [
        lambda dimension: torch.zeros(dimension),
        lambda dimension: torch.zeros(2, dimension, 1),
        lambda dimension: torch.zeros(0, dimension),
        lambda dimension: torch.zeros(2, dimension + 1),
    ],
)
def test_actor_rejects_malformed_shapes(bad_observation) -> None:
    policy = _small_policy()
    with pytest.raises(ValueError):
        policy.act_inference(bad_observation(policy.num_actor_obs))


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_actor_rejects_non_finite_observations(invalid_value: float) -> None:
    policy = _small_policy()
    actor_obs = torch.zeros(2, policy.num_actor_obs)
    actor_obs[0, 0] = invalid_value
    with pytest.raises(ValueError, match="NaN or infinity"):
        policy.act(actor_obs)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_critic_rejects_non_finite_and_malformed_observations(invalid_value: float) -> None:
    policy = _small_policy()
    critic_obs = torch.zeros(2, policy.num_critic_obs)
    critic_obs[1, 2] = invalid_value
    with pytest.raises(ValueError, match="NaN or infinity"):
        policy.evaluate(critic_obs)
    with pytest.raises(ValueError, match="feature dimension"):
        policy.evaluate(torch.zeros(2, policy.num_critic_obs + 1))


def test_observation_dtype_and_device_mismatches_fail_actionably() -> None:
    policy = _small_policy()
    with pytest.raises(TypeError, match="floating dtype"):
        policy.act_inference(torch.zeros(2, policy.num_actor_obs, dtype=torch.long))
    with pytest.raises(TypeError, match="policy uses"):
        policy.act_inference(torch.zeros(2, policy.num_actor_obs, dtype=torch.float64))
    with pytest.raises(ValueError, match="same device"):
        policy.act_inference(torch.empty(2, policy.num_actor_obs, device="meta"))


def test_double_policy_accepts_double_observations_end_to_end() -> None:
    policy = _small_policy().double()
    actor_obs = torch.randn(2, policy.num_actor_obs, dtype=torch.float64)
    critic_obs = torch.randn(2, policy.num_critic_obs, dtype=torch.float64)

    actions = policy.act(actor_obs)
    assert actions.dtype == torch.float64
    assert policy.action_mean.dtype == torch.float64
    assert policy.action_std.dtype == torch.float64
    assert policy.evaluate(critic_obs).dtype == torch.float64


@pytest.mark.parametrize("noise_std_type", ["scalar", "log"])
def test_noise_standard_deviation_initializes_to_exact_requested_value(
    noise_std_type: str,
) -> None:
    policy = _small_policy(noise_std_type=noise_std_type)
    actor_obs = torch.zeros(3, policy.num_actor_obs)
    policy.act(actor_obs)

    assert torch.equal(policy.action_std, torch.full((3, 29), 0.01))
    if noise_std_type == "scalar":
        assert torch.equal(policy.std, torch.full((29,), 0.01))
        assert not hasattr(policy, "log_std")
    else:
        assert torch.equal(policy.log_std.exp(), torch.full((29,), 0.01))
        assert not hasattr(policy, "std")


@pytest.mark.parametrize("property_name", ["action_mean", "action_std", "entropy"])
def test_distribution_properties_require_prior_act(property_name: str) -> None:
    policy = _small_policy()
    with pytest.raises(RuntimeError, match=r"call act\(\) first"):
        getattr(policy, property_name)
    with pytest.raises(RuntimeError, match=r"call act\(\) first"):
        policy.get_actions_log_prob(torch.zeros(1, 29))


def test_action_log_probability_validates_shape_dtype_and_values() -> None:
    policy = _small_policy()
    policy.act(torch.zeros(2, policy.num_actor_obs))

    with pytest.raises(ValueError, match="distribution shape"):
        policy.get_actions_log_prob(torch.zeros(1, 29))
    with pytest.raises(ValueError, match="feature dimension"):
        policy.get_actions_log_prob(torch.zeros(2, 28))
    with pytest.raises(TypeError, match="floating dtype"):
        policy.get_actions_log_prob(torch.zeros(2, 29, dtype=torch.long))
    non_finite = torch.zeros(2, 29)
    non_finite[0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or infinity"):
        policy.get_actions_log_prob(non_finite)


def test_invalid_learned_standard_deviation_is_rejected() -> None:
    policy = _small_policy(noise_std_type="scalar")
    with torch.no_grad():
        policy.std[0] = 0.0
    with pytest.raises(FloatingPointError, match="strictly positive"):
        policy.act(torch.zeros(1, policy.num_actor_obs))

    policy = _small_policy(noise_std_type="scalar")
    policy.act(torch.zeros(1, policy.num_actor_obs))
    with torch.no_grad():
        policy.std[0] = -0.01
    with pytest.raises(FloatingPointError, match="strictly positive"):
        _ = policy.action_std


def test_strict_state_dict_round_trip_returns_resume_flag_and_clears_distribution() -> None:
    source = _small_policy()
    destination = _small_policy()
    source.act(torch.randn(2, source.num_actor_obs))
    destination.act(torch.randn(2, destination.num_actor_obs))

    assert destination.load_state_dict(source.state_dict()) is True
    with pytest.raises(RuntimeError, match="unavailable"):
        _ = destination.action_mean
    for name, value in source.state_dict().items():
        assert torch.equal(value, destination.state_dict()[name])


def test_strict_state_loading_rejects_missing_unexpected_and_wrong_shaped_entries() -> None:
    policy = _small_policy()
    state = {name: value.clone() for name, value in policy.state_dict().items()}

    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="Missing key"):
        policy.load_state_dict(missing)

    unexpected = dict(state)
    unexpected["not_a_parameter"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        policy.load_state_dict(unexpected)

    wrong_shape = dict(state)
    first_name = next(iter(wrong_shape))
    wrong_shape[first_name] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="size mismatch"):
        policy.load_state_dict(wrong_shape)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_state_loading_rejects_non_finite_tensors(invalid_value: float) -> None:
    policy = _small_policy()
    state = {name: value.clone() for name, value in policy.state_dict().items()}
    first_name = next(iter(state))
    state[first_name].reshape(-1)[0] = invalid_value

    with pytest.raises(ValueError, match=first_name):
        policy.load_state_dict(state)


def test_reset_is_harmless_and_forward_remains_explicit() -> None:
    policy = _small_policy()
    assert policy.reset() is None
    assert policy.reset(torch.tensor([True, False])) is None
    with pytest.raises(NotImplementedError, match="act_inference"):
        policy(torch.zeros(1, policy.num_actor_obs))


def test_dtype_conversion_invalidates_a_cached_distribution() -> None:
    policy = _small_policy()
    policy.act(torch.zeros(1, policy.num_actor_obs))
    policy.double()

    with pytest.raises(RuntimeError, match="unavailable"):
        _ = policy.action_mean
