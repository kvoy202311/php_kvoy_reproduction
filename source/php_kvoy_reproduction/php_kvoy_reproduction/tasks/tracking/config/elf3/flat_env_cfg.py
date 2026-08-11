from isaaclab.utils import configclass

from php_kvoy_reproduction.assets.elf3 import ELF3_CFG
from php_kvoy_reproduction.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


# Policy/action order used by Isaac for ELF3. This exact order is shared by the
# TienKung-ELF3 task and the deployment-side policy metadata/remapping.
ELF3_ISAAC_JOINT_NAMES = [
    "l_shoulder_y_joint",
    "r_shoulder_y_joint",
    "waist_y_joint",
    "l_shoulder_x_joint",
    "r_shoulder_x_joint",
    "waist_x_joint",
    "l_shoulder_z_joint",
    "r_shoulder_z_joint",
    "waist_z_joint",
    "l_elbow_y_joint",
    "r_elbow_y_joint",
    "l_hip_y_joint",
    "r_hip_y_joint",
    "l_wrist_x_joint",
    "r_wrist_x_joint",
    "l_hip_x_joint",
    "r_hip_x_joint",
    "l_wrist_y_joint",
    "r_wrist_y_joint",
    "l_hip_z_joint",
    "r_hip_z_joint",
    "l_wrist_z_joint",
    "r_wrist_z_joint",
    "l_knee_y_joint",
    "r_knee_y_joint",
    "l_ankle_y_joint",
    "r_ankle_y_joint",
    "l_ankle_x_joint",
    "r_ankle_x_joint",
]


# Joint-space action scales from TienKung-ELF3's elf3_walk_wb task. Keeping
# these in the task config avoids changing the ELF3 articulation/actuator asset.
ELF3_ACTION_SCALE = {
    "waist_y_joint": 0.231,
    "waist_x_joint": 0.154,
    "waist_z_joint": 0.213,
    ".*_hip_y_joint": 0.213,
    ".*_hip_x_joint": 0.213,
    ".*_hip_z_joint": 0.231,
    ".*_knee_y_joint": 0.213,
    ".*_ankle_y_joint": 0.373,
    ".*_ankle_x_joint": 0.230,
    ".*_shoulder_y_joint": 0.231,
    ".*_shoulder_x_joint": 0.231,
    ".*_shoulder_z_joint": 0.373,
    ".*_elbow_y_joint": 0.231,
    ".*_wrist_x_joint": 0.373,
    ".*_wrist_y_joint": 0.373,
    ".*_wrist_z_joint": 0.373,
}


@configclass
class ELF3FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Use the existing ELF3 asset unchanged.
        self.scene.robot = ELF3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Make the policy order explicit instead of relying on an implicit USD
        # traversal order. The joint names also make a missing/renamed DOF fail
        # during action-manager initialization.
        self.actions.joint_pos.joint_names = ELF3_ISAAC_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = ELF3_ACTION_SCALE

        # torso_link is the articulation root. It is not the first entry in
        # the authoritative ELF3 tracking-body order below, so declare it
        # explicitly for motion reset/resampling.
        self.commands.motion.root_body_name = "torso_link"
        self.commands.motion.anchor_body_name = "torso_link"
        self.commands.motion.body_names = [
            "waist_z_link",
            "l_hip_y_link",
            "l_knee_y_link",
            "l_ankle_x_link",
            "r_hip_y_link",
            "r_knee_y_link",
            "r_ankle_x_link",
            "torso_link",
            "l_shoulder_y_link",
            "l_elbow_y_link",
            "l_wrist_z_link",
            "r_shoulder_y_link",
            "r_elbow_y_link",
            "r_wrist_z_link",
        ]

        # ELF3 end-effectors are allowed to contact the environment. Contacts
        # from every other rigid body retain the base task's penalty.
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!l_ankle_x_link$)(?!r_ankle_x_link$)(?!l_wrist_z_link$)(?!r_wrist_z_link$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = [
            "l_ankle_x_link",
            "r_ankle_x_link",
            "l_wrist_z_link",
            "r_wrist_z_link",
        ]
