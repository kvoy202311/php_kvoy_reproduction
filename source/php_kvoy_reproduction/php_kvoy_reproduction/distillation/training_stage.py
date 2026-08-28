"""Explicit curriculum-stage configuration for multi-skill distillation."""

from __future__ import annotations

from typing import Literal


TrainingStage = Literal["atomic", "transition", "full"]


def expected_warm_start_source_stage(stage: TrainingStage) -> TrainingStage:
    """Return the only checkpoint stage allowed to initialize ``stage``."""

    if stage == "transition":
        return "atomic"
    if stage == "full":
        return "transition"
    if stage == "atomic":
        raise ValueError("atomic is the clean Student baseline and has no warm-start predecessor")
    raise ValueError(f"unknown training stage {stage!r}")


def validate_training_stage_checkpoint_mode(
    stage: TrainingStage,
    *,
    resume: bool,
    warm_start: bool,
) -> None:
    """Reject checkpoint modes that bypass or mix curriculum stages."""

    if stage not in ("atomic", "transition", "full"):
        raise ValueError(f"unknown training stage {stage!r}")
    if resume and warm_start:
        raise ValueError("resume and warm_start are mutually exclusive")
    if stage == "atomic" and warm_start:
        raise ValueError(
            "atomic is the clean Student baseline and cannot warm-start; use resume only "
            "to continue the same atomic run"
        )
    if stage != "atomic" and not (resume or warm_start):
        raise ValueError(
            f"training stage {stage!r} must resume its own run or warm-start "
            "from the preceding validated stage"
        )


def configure_training_stage(env_cfg, agent_cfg, stage: TrainingStage) -> None:
    """Apply one curriculum stage before constructing the Isaac Lab scene.

    The three stages intentionally share the Actor input/action contract.  Only
    task difficulty, visual augmentation and the DAgger schedule change, which
    makes policy-only warm-starts between consecutive stages well-defined.
    """

    command = env_cfg.commands.multi_skill
    geometry = env_cfg.events.platform_geometry.params
    depth = env_cfg.observations.policy.depth.params
    camera_extrinsics = env_cfg.events.camera_extrinsics.params
    nominal_length, nominal_width, nominal_height = command.platform_size

    if stage in ("atomic", "transition"):
        geometry["length_range"] = (nominal_length, nominal_length)
        geometry["width_range"] = (nominal_width, nominal_width)
        geometry["height_range"] = (nominal_height, nominal_height)
        geometry["nominal_size_fraction"] = 1.0

    if stage == "atomic":
        command.composed_episode_fraction = 0.0
        # Atomic motion clips terminate before returning to locomotion.  Varying
        # the still-visible joystick request is therefore a safe way to teach
        # climb/down-roll command invariance from the first stage.
        command.locked_command_resampling_enabled = True
        # First establish deterministic visual/proprioceptive imitation.  The
        # later transition stage restores deployment delay and noise.
        depth["noise_enabled"] = False
        depth["delay_range_s"] = (0.0, 0.0)
        camera_extrinsics["translation_range_m"] = (0.0, 0.0)
        camera_extrinsics["rotation_range_rad"] = (0.0, 0.0)
        agent_cfg.algorithm.minimum_dagger_weight = 1.0
    elif stage == "transition":
        command.composed_episode_fraction = 1.0
        # First learn the complete nominal locomotion -> motion -> locomotion
        # sequence deterministically.  Full training restores live command
        # changes after the transition itself is reliable.
        command.locked_command_resampling_enabled = False
        agent_cfg.algorithm.minimum_dagger_weight = 0.5
    elif stage == "full":
        # Environment defaults retain mixed direct/composed episodes and the
        # complete geometry plus camera randomization contract.
        command.composed_episode_fraction = 0.5
        command.locked_command_resampling_enabled = True
        agent_cfg.algorithm.minimum_dagger_weight = 0.1
    else:
        raise ValueError(f"unknown training stage {stage!r}")


__all__ = [
    "TrainingStage",
    "configure_training_stage",
    "expected_warm_start_source_stage",
    "validate_training_stage_checkpoint_mode",
]
