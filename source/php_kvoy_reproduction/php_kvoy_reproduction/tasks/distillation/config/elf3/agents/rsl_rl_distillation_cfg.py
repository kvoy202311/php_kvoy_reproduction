"""PHP paper hyperparameters adapted to the local RSL-RL 2.3 runner."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class ELF3VisionStudentCfg(RslRlPpoActorCriticCfg):
    class_name = "VisionActorCritic"
    init_noise_std = 0.01
    # A log parameter preserves positivity throughout optimization while the
    # resulting initial standard deviation remains exactly 0.01.
    noise_std_type = "log"
    num_skills = 3
    actor_hidden_dims = [2048, 1024, 512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = "elu"


@configclass
class ELF3DAggerPPOCfg(RslRlPpoAlgorithmCfg):
    class_name = "DAggerPPO"
    value_loss_coef = 1.0
    use_clipped_value_loss = True
    clip_param = 0.2
    entropy_coef = 0.001
    num_learning_epochs = 2
    num_mini_batches = 96
    learning_rate = 3.0e-4
    schedule = "adaptive"
    gamma = 0.99
    lam = 0.95
    desired_kl = 0.01
    max_grad_norm = 1.0
    normalize_advantage_per_mini_batch = False

    dagger_base_coef = 10.0
    selector_loss_coef = 1.0
    # PHP uses the per-sample sum across the fixed 29-DoF action vector.
    # Keeping coefficient 10 with a per-DoF mean would weaken imitation by 29x.
    dagger_reduction = "sum_per_sample"
    curriculum_iterations = 10_000
    minimum_dagger_weight = 0.1
    adaptive_lr_minimum_ppo_weight = 0.1
    balance_skill_losses = True
    maximum_action_magnitude = 1000.0
    skill_names = ("locomotion", "climb", "down_roll")

    # Unsupported upstream extensions must stay absent from this independent
    # algorithm rather than being silently ignored.
    symmetry_cfg = None
    rnd_cfg = None


@configclass
class ELF3MultiSkillDistillationRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Three frozen teachers distilled into one depth-conditioned student."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 20_000
    empirical_normalization = False
    policy: ELF3VisionStudentCfg = ELF3VisionStudentCfg()
    algorithm: ELF3DAggerPPOCfg = ELF3DAggerPPOCfg()
    clip_actions = None
    save_interval = 500
    log_interval = 1
    experiment_name = "elf3_multi_skill_distillation"
    run_name = ""
    logger = "tensorboard"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    observation_layout = {
        "proprio_frame_dim": 93,
        "proprio_history_length": 8,
        # The public command remains world-frame (vx, vy); the actor receives
        # its body-frame projection so heading control is fully observable.
        "command_dim": 2,
        "depth_height": 58,
        "depth_width": 87,
    }
    observation_keys = {
        "critic": "critic",
        "route": "skill_id",
        "distill_mask": "teacher_valid",
        "teacher_locomotion": "locomotion_teacher",
        "teacher_climb": "motion_teacher",
        "teacher_down_roll": "motion_teacher",
    }
    environment_iteration_command = "multi_skill"
    option_control = {
        # The selector must remain confident for several frames before a
        # discrete option changes.  Motion options cannot switch directly to
        # one another and remain committed for at least two seconds.
        "activation_probability": 0.60,
        "release_probability": 0.55,
        "activation_confirmation_steps": 3,
        "release_confirmation_steps": 5,
        "post_release_cooldown_steps": 25,
        "minimum_skill_duration_steps": {"climb": 100, "down_roll": 100},
        "maximum_skill_duration_steps": {"climb": 400, "down_roll": 400},
        # Stage configuration overwrites these three schedule values.
        "teacher_forcing_start": 1.0,
        "teacher_forcing_end": 0.0,
        "teacher_forcing_iterations": 20_000,
    }


__all__ = [
    "ELF3DAggerPPOCfg",
    "ELF3MultiSkillDistillationRunnerCfg",
    "ELF3VisionStudentCfg",
]
