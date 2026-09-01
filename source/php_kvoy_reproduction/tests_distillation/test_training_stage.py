from __future__ import annotations

from types import SimpleNamespace

import pytest

from php_kvoy_reproduction.distillation.training_stage import (
    configure_training_stage,
    expected_warm_start_source_stage,
    validate_training_stage_checkpoint_mode,
)


def _configs():
    env = SimpleNamespace(
        commands=SimpleNamespace(
            multi_skill=SimpleNamespace(
                platform_size=(0.51, 0.80, 0.66),
                composed_episode_fraction=0.5,
                atomic_motion_start_at_beginning_fraction=0.5,
                transition_settle_time_range_s=(0.2, 0.5),
            )
        ),
        events=SimpleNamespace(
            platform_geometry=SimpleNamespace(
                params={
                    "length_range": (0.51, 3.0),
                    "width_range": (0.75, 1.2),
                    "height_range": (0.60, 0.72),
                    "nominal_size_fraction": 0.25,
                }
            ),
            camera_extrinsics=SimpleNamespace(
                params={
                    "translation_range_m": (-0.025, 0.025),
                    "rotation_range_rad": (-0.04, 0.04),
                }
            ),
        ),
        observations=SimpleNamespace(
            policy=SimpleNamespace(
                depth=SimpleNamespace(
                    params={"noise_enabled": True, "delay_range_s": (0.06, 0.08)}
                )
            )
        ),
    )
    agent = SimpleNamespace(
        algorithm=SimpleNamespace(minimum_dagger_weight=0.1),
        option_control={
            "teacher_forcing_start": 1.0,
            "teacher_forcing_end": 0.0,
            "teacher_forcing_iterations": 20_000,
        },
    )
    return env, agent


@pytest.mark.parametrize(
    ("stage", "composed_fraction", "dagger_weight", "locked_resampling"),
    (
        ("atomic", 0.0, 0.8, True),
        ("transition", 1.0, 0.4, False),
        ("full", 1.0, 0.1, True),
    ),
)
def test_training_stage_routing_and_dagger_schedule(
    stage: str,
    composed_fraction: float,
    dagger_weight: float,
    locked_resampling: bool,
) -> None:
    env, agent = _configs()
    configure_training_stage(env, agent, stage)
    assert env.commands.multi_skill.composed_episode_fraction == composed_fraction
    assert env.commands.multi_skill.locked_command_resampling_enabled is locked_resampling
    assert agent.algorithm.minimum_dagger_weight == dagger_weight
    assert env.commands.multi_skill.transition_settle_time_range_s == (0.0, 0.0)


@pytest.mark.parametrize(
    ("stage", "forcing_start", "forcing_end"),
    (("atomic", 1.0, 1.0), ("transition", 1.0, 0.25), ("full", 0.25, 0.0)),
)
def test_training_stage_route_teacher_forcing_schedule(
    stage: str, forcing_start: float, forcing_end: float
) -> None:
    env, agent = _configs()
    configure_training_stage(env, agent, stage)
    assert agent.option_control["teacher_forcing_start"] == forcing_start
    assert agent.option_control["teacher_forcing_end"] == forcing_end


@pytest.mark.parametrize("stage", ("atomic", "transition"))
def test_early_stages_use_nominal_platform_geometry(stage: str) -> None:
    env, agent = _configs()
    configure_training_stage(env, agent, stage)
    geometry = env.events.platform_geometry.params
    assert geometry["length_range"] == (0.51, 0.51)
    assert geometry["width_range"] == (0.80, 0.80)
    assert geometry["height_range"] == (0.66, 0.66)
    assert geometry["nominal_size_fraction"] == 1.0


def test_atomic_stage_disables_visual_augmentation() -> None:
    env, agent = _configs()
    configure_training_stage(env, agent, "atomic")
    assert env.observations.policy.depth.params["noise_enabled"] is False
    assert env.observations.policy.depth.params["delay_range_s"] == (0.0, 0.0)
    assert env.events.camera_extrinsics.params["translation_range_m"] == (0.0, 0.0)
    assert env.events.camera_extrinsics.params["rotation_range_rad"] == (0.0, 0.0)


def test_atomic_stage_uses_only_complete_observable_motion_starts() -> None:
    env, agent = _configs()
    configure_training_stage(env, agent, "atomic")
    assert env.commands.multi_skill.atomic_motion_start_at_beginning_fraction == 1.0


def test_full_stage_preserves_default_geometry_and_augmentation() -> None:
    env, agent = _configs()
    configure_training_stage(env, agent, "full")
    assert env.events.platform_geometry.params["length_range"] == (0.51, 3.0)
    assert env.observations.policy.depth.params["noise_enabled"] is True
    assert env.events.camera_extrinsics.params["translation_range_m"] == (-0.025, 0.025)


def test_unknown_stage_is_rejected() -> None:
    env, agent = _configs()
    with pytest.raises(ValueError, match="unknown training stage"):
        configure_training_stage(env, agent, "invalid")


def test_warm_start_predecessor_is_strictly_sequential() -> None:
    assert expected_warm_start_source_stage("transition") == "atomic"
    assert expected_warm_start_source_stage("full") == "transition"
    with pytest.raises(ValueError, match="no warm-start predecessor"):
        expected_warm_start_source_stage("atomic")


@pytest.mark.parametrize(
    ("stage", "resume", "warm_start"),
    (
        ("atomic", False, False),
        ("atomic", True, False),
        ("transition", False, True),
        ("transition", True, False),
        ("full", False, True),
        ("full", True, False),
    ),
)
def test_valid_training_stage_checkpoint_modes(stage: str, resume: bool, warm_start: bool) -> None:
    validate_training_stage_checkpoint_mode(stage, resume=resume, warm_start=warm_start)


@pytest.mark.parametrize(
    ("stage", "resume", "warm_start", "message"),
    (
        ("atomic", False, True, "cannot warm-start"),
        ("transition", False, False, "must resume"),
        ("full", False, False, "must resume"),
        ("full", True, True, "mutually exclusive"),
    ),
)
def test_invalid_training_stage_checkpoint_modes(
    stage: str,
    resume: bool,
    warm_start: bool,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_training_stage_checkpoint_mode(stage, resume=resume, warm_start=warm_start)
