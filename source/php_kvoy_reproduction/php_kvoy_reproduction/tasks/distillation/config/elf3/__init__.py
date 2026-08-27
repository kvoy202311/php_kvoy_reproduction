import gymnasium as gym

from . import agents, multi_skill_env_cfg


gym.register(
    id="Distillation-MultiSkill-ELF3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": multi_skill_env_cfg.ELF3MultiSkillEnvCfg,
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:ELF3MultiSkillDistillationRunnerCfg"
        ),
    },
)
