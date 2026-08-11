from isaaclab.utils import configclass

from php_kvoy_reproduction.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class ELF3FlatPPORunnerCfg(G1FlatPPORunnerCfg):
    """Current tracking PPO baseline, logged separately for ELF3."""

    experiment_name = "elf3_flat"


@configclass
class ELF3ClimbPPORunnerCfg(G1FlatPPORunnerCfg):
    """PPO baseline for the ELF3 climb expert, logged separately from flat tracking."""

    experiment_name = "elf3_climb"
