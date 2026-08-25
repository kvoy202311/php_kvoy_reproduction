from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import php_kvoy_reproduction.tasks.tracking.mdp as mdp
from php_kvoy_reproduction.tasks.tracking.config.elf3.climb_env_cfg import (
    ELF3_CLIMB_EXPERT_JOINT_SCORE_EXPONENT,
    ELF3_CLIMB_EXPERT_JOINT_WORST_COUNT,
    ELF3_CLIMB_EXPERT_JOINT_WORST_GROUP_WEIGHT,
    ELF3_CLIMB_EXPERT_JOINT_WORST_WEIGHT,
    ELF3_CLIMB_FINAL_EXPERT_JOINT_GROUPS,
    ELF3_CLIMB_PLATFORM_CENTER,
    ELF3_CLIMB_TRACKED_BODY_NAMES,
    ELF3ClimbActionsCfg,
    ELF3ClimbEnvCfg,
    ELF3ClimbEventCfg,
    ELF3ClimbObservationsCfg,
    ELF3ClimbSceneCfg,
)


# Down-roll uses the exact same fixed 0.66 m physical platform and observation
# geometry as the climb expert.  Its offline-retargeted references already
# encode the vertical platform transition, so no runtime reference-Z bridge is
# enabled here.
ELF3_DOWN_ROLL_PLATFORM_HEIGHT = 0.66
ELF3_DOWN_ROLL_FINAL_HOLD_TIME_S = 0.0

# Dense joint imitation closes the null space left by sparse tracked bodies.
# Unlike climb, down-roll does not hand ankle pitch/roll to a platform-surface
# objective: the complete authored roll is the target, including both ankles.
ELF3_DOWN_ROLL_EXPERT_JOINT_TRACKING_GROUPS = ELF3_CLIMB_FINAL_EXPERT_JOINT_GROUPS
ELF3_DOWN_ROLL_EXPERT_JOINT_POSITION_GROUP_STDS = {
    "waist": 0.35,
    "left_arm": 0.45,
    "right_arm": 0.45,
    "left_leg": 0.70,
    "left_ankle": 0.50,
    "right_leg": 0.70,
    "right_ankle": 0.50,
}
ELF3_DOWN_ROLL_EXPERT_JOINT_VELOCITY_GROUP_STDS = {
    "waist": 1.0,
    "left_arm": 2.0,
    "right_arm": 2.0,
    "left_leg": 2.0,
    "left_ankle": 2.0,
    "right_leg": 2.0,
    "right_ankle": 2.0,
}
ELF3_DOWN_ROLL_EXPERT_JOINT_GROUP_WEIGHTS = {
    "waist": 1.0,
    "left_arm": 1.0,
    "right_arm": 1.0,
    "left_leg": 1.0,
    "left_ankle": 1.0,
    "right_leg": 1.0,
    "right_ankle": 1.0,
}


@configclass
class ELF3DownRollSceneCfg(ELF3ClimbSceneCfg):
    """The shared fixed-platform scene used for the inverse traversal."""

    pass


@configclass
class ELF3DownRollCommandsCfg:
    """Offline-retargeted down-roll reference command."""

    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        root_body_name="torso_link",
        anchor_body_name="torso_link",
        body_names=ELF3_CLIMB_TRACKED_BODY_NAMES,
        pose_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        velocity_range={
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        joint_position_range=(0.0, 0.0),
        terminate_on_motion_end=True,
        motion_end_hold_time_s=ELF3_DOWN_ROLL_FINAL_HOLD_TIME_S,
        use_adaptive_sampling=False,
        exclude_repeated_terminal_frames_from_random_starts=True,
        adaptive_failure_term_names=(),
        random_phase_env_mask_attr="_climb_box_random_phase_env_mask",
        reference_transform_asset_name="platform",
        reference_transform_nominal_xy=ELF3_CLIMB_PLATFORM_CENTER[:2],
        terminal_default_pose_enabled=False,
    )


@configclass
class ELF3DownRollObservationsCfg(ELF3ClimbObservationsCfg):
    """Actor/critic inputs including the shared platform height scan."""

    pass


@configclass
class ELF3DownRollActionsCfg(ELF3ClimbActionsCfg):
    """The same explicit ELF3 action order, scales, and hard limits."""

    pass


@configclass
class ELF3DownRollEventCfg(ELF3ClimbEventCfg):
    """Fixed geometry and deterministic dynamics for source expert training."""

    pass


@configclass
class ELF3DownRollRewardsCfg:
    """Pure expert imitation without climb-direction or terminal-platform terms."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3, "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4, "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0, "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14, "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES},
    )
    motion_joint_pos = RewTerm(
        func=mdp.motion_grouped_expert_joint_position_error_exp,
        weight=2.0,
        params={
            "command_name": "motion",
            "joint_groups": ELF3_DOWN_ROLL_EXPERT_JOINT_TRACKING_GROUPS,
            "group_stds": ELF3_DOWN_ROLL_EXPERT_JOINT_POSITION_GROUP_STDS,
            "group_weights": ELF3_DOWN_ROLL_EXPERT_JOINT_GROUP_WEIGHTS,
            "score_exponent": ELF3_CLIMB_EXPERT_JOINT_SCORE_EXPONENT,
            "worst_joint_count": ELF3_CLIMB_EXPERT_JOINT_WORST_COUNT,
            "worst_joint_weight": ELF3_CLIMB_EXPERT_JOINT_WORST_WEIGHT,
            "group_aggregation": "harmonic",
            "worst_group_weight": ELF3_CLIMB_EXPERT_JOINT_WORST_GROUP_WEIGHT,
        },
    )
    motion_joint_vel = RewTerm(
        func=mdp.motion_grouped_expert_joint_velocity_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "joint_groups": ELF3_DOWN_ROLL_EXPERT_JOINT_TRACKING_GROUPS,
            "group_stds": ELF3_DOWN_ROLL_EXPERT_JOINT_VELOCITY_GROUP_STDS,
            "group_weights": ELF3_DOWN_ROLL_EXPERT_JOINT_GROUP_WEIGHTS,
            "score_exponent": ELF3_CLIMB_EXPERT_JOINT_SCORE_EXPONENT,
            "worst_joint_count": ELF3_CLIMB_EXPERT_JOINT_WORST_COUNT,
            "worst_joint_weight": ELF3_CLIMB_EXPERT_JOINT_WORST_WEIGHT,
            "group_aggregation": "harmonic",
            "worst_group_weight": ELF3_CLIMB_EXPERT_JOINT_WORST_GROUP_WEIGHT,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class ELF3DownRollTerminationsCfg:
    """Tracking failures plus an unclassified, bootstrapped clip boundary."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.30,
            "body_names": ["l_ankle_x_link", "r_ankle_x_link", "l_wrist_z_link", "r_wrist_z_link"],
        },
    )
    motion_clip_end = DoneTerm(
        func=mdp.motion_clip_end,
        time_out=True,
        params={"command_name": "motion", "classified_term_names": ()},
    )


@configclass
class ELF3DownRollCurriculumCfg:
    """No platform curriculum for the fixed 0.66 m expert."""

    pass


@configclass
class ELF3DownRollEnvCfg(ELF3ClimbEnvCfg):
    """ELF3 0.66 m platform down-and-roll expert task."""

    scene: ELF3DownRollSceneCfg = ELF3DownRollSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=False)
    observations: ELF3DownRollObservationsCfg = ELF3DownRollObservationsCfg()
    actions: ELF3DownRollActionsCfg = ELF3DownRollActionsCfg()
    commands: ELF3DownRollCommandsCfg = ELF3DownRollCommandsCfg()
    rewards: ELF3DownRollRewardsCfg = ELF3DownRollRewardsCfg()
    terminations: ELF3DownRollTerminationsCfg = ELF3DownRollTerminationsCfg()
    events: ELF3DownRollEventCfg = ELF3DownRollEventCfg()
    curriculum: ELF3DownRollCurriculumCfg = ELF3DownRollCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        platform_height = float(self.scene.platform.spawn.size[2])
        if abs(platform_height - ELF3_DOWN_ROLL_PLATFORM_HEIGHT) > 1.0e-12:
            raise ValueError(
                f"Down-roll requires a {ELF3_DOWN_ROLL_PLATFORM_HEIGHT:.2f} m platform, got {platform_height:.6f} m."
            )
