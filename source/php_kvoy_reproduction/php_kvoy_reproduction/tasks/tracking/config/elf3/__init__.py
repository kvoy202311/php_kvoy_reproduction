import gymnasium as gym

from . import agents, climb_env_cfg, down_roll_env_cfg, flat_env_cfg


gym.register(
    id="Tracking-Flat-ELF3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.ELF3FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ELF3FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Climb-ELF3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": climb_env_cfg.ELF3ClimbEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ELF3ClimbPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-DownRoll-ELF3-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": down_roll_env_cfg.ELF3DownRollEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:ELF3DownRollPPORunnerCfg",
    },
)
