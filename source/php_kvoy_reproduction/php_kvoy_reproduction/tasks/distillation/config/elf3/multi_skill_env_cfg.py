"""ELF3 unified visual multi-teacher distillation task."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, TiledCameraCfg, patterns
from isaaclab.utils import configclass

import php_kvoy_reproduction.tasks.distillation.mdp as mdp
import php_kvoy_reproduction.tasks.tracking.mdp as tracking_mdp
from php_kvoy_reproduction.tasks.tracking.config.elf3.climb_env_cfg import (
    ELF3_CLIMB_ACTION_SCALE,
    ELF3_CLIMB_EXPERT_JOINT_SCORE_EXPONENT,
    ELF3_CLIMB_EXPERT_JOINT_GROUP_WEIGHTS,
    ELF3_CLIMB_EXPERT_JOINT_TRACKING_GROUPS,
    ELF3_CLIMB_EXPERT_JOINT_VELOCITY_GROUP_STDS,
    ELF3_CLIMB_EXPERT_JOINT_POSITION_GROUP_STDS,
    ELF3_CLIMB_EXPERT_JOINT_WORST_COUNT,
    ELF3_CLIMB_EXPERT_JOINT_WORST_GROUP_WEIGHT,
    ELF3_CLIMB_EXPERT_JOINT_WORST_WEIGHT,
    ELF3_CLIMB_HEIGHT_SCAN_VALUE_OFFSET,
    ELF3_CLIMB_HEIGHT_SCAN_RESOLUTION,
    ELF3_CLIMB_HEIGHT_SCAN_SIZE,
    ELF3_CLIMB_JOINT_NAMES,
    ELF3_CLIMB_JOINT_POSITION_TARGET_LIMITS,
    ELF3_CLIMB_PLATFORM_CENTER,
    ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
    ELF3_CLIMB_PLATFORM_SIZE,
    ELF3_CLIMB_TRACKED_BODY_NAMES,
    ELF3ClimbSceneCfg,
)
from php_kvoy_reproduction.tasks.tracking.config.elf3.down_roll_env_cfg import (
    ELF3_DOWN_ROLL_EXPERT_JOINT_GROUP_WEIGHTS,
    ELF3_DOWN_ROLL_EXPERT_JOINT_POSITION_GROUP_STDS,
    ELF3_DOWN_ROLL_EXPERT_JOINT_TRACKING_GROUPS,
    ELF3_DOWN_ROLL_EXPERT_JOINT_VELOCITY_GROUP_STDS,
)
from php_kvoy_reproduction.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


# The D435i publishes 848x480 depth at 30 Hz on hardware.  Rendering the
# student's 87x58 tensor directly avoids the prohibitive cost of a native-size
# camera per environment while retaining the measured hardware field of view.
DEPTH_HEIGHT = 58
DEPTH_WIDTH = 87
DEPTH_NEAR = 0.15
DEPTH_FAR = 2.0
DEPTH_CAPTURE_FREQUENCY_HZ = mdp.ELF3_D435I_REFERENCE_DEPTH_FPS
DEPTH_SENSOR_NEAR, DEPTH_SENSOR_FAR = mdp.ELF3_D435I_DEPLOYMENT_DEPTH_RANGE_M
DEPTH_FOCAL_LENGTH = 11.2
DEPTH_HORIZONTAL_APERTURE = 20.955
DEPTH_VERTICAL_FOV_RAD = mdp.ELF3_D435I_REFERENCE_DEPTH_VERTICAL_FOV_RAD
DEPTH_VERTICAL_APERTURE = 2.0 * DEPTH_FOCAL_LENGTH * math.tan(0.5 * DEPTH_VERTICAL_FOV_RAD)
CAMERA_TRANSLATION_RANDOMIZATION = 0.025
CAMERA_ROTATION_RANDOMIZATION = math.radians(2.5)

# The physical collider, rendered box and analytical teacher height scan all
# share these prestartup samples.  Length grows away from the source-aligned
# climb edge; the upper end is long enough that a 2--4 s top traversal does
# not necessarily reach the far edge.  A quarter of environments retain the
# exact source geometry for reliable full-strength teacher supervision.
ELF3_DISTILL_PLATFORM_LENGTH_RANGE = (0.51, 3.0)
ELF3_DISTILL_PLATFORM_WIDTH_RANGE = (0.75, 1.20)
ELF3_DISTILL_PLATFORM_HEIGHT_RANGE = (0.60, 0.72)
ELF3_DISTILL_NOMINAL_GEOMETRY_FRACTION = 0.25
ELF3_DISTILL_GROUND_COLLIDER_PRIM_PATH = (
    "/World/ground/terrain/GroundPlane/CollisionPlane"
)
ELF3_DISTILL_GROUND_CONTACT_FOOT_BODY_NAMES = tuple(
    ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS["foot_body_names"]
)
ELF3_DISTILL_GROUND_CONTACT_SENSOR_NAMES = (
    "l_foot_ground_contact",
    "r_foot_ground_contact",
)
ELF3_DISTILL_MIN_GROUND_UPWARD_FORCE_N = 20.0


@configclass
class ELF3MultiSkillSceneCfg(ELF3ClimbSceneCfg):
    """Randomized platform scene augmented with one batched head D435i camera."""

    # Ground and platform contacts are deliberately independent per foot.  A
    # climb state may therefore contain one platform-supported foot and one
    # ground-supported foot without either measurement suppressing the other.
    # TerrainImporterCfg inserts the standard Grid USD at
    # /World/ground/terrain; its actual filtered collision prim is the nested
    # GroundPlane/CollisionPlane, not either importer parent.
    l_foot_ground_contact = ContactSensorCfg(
        prim_path=(
            f"{{ENV_REGEX_NS}}/Robot/{ELF3_DISTILL_GROUND_CONTACT_FOOT_BODY_NAMES[0]}"
        ),
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        force_threshold=ELF3_DISTILL_MIN_GROUND_UPWARD_FORCE_N,
        filter_prim_paths_expr=[ELF3_DISTILL_GROUND_COLLIDER_PRIM_PATH],
        debug_vis=False,
    )
    r_foot_ground_contact = ContactSensorCfg(
        prim_path=(
            f"{{ENV_REGEX_NS}}/Robot/{ELF3_DISTILL_GROUND_CONTACT_FOOT_BODY_NAMES[1]}"
        ),
        update_period=0.0,
        history_length=0,
        track_air_time=False,
        force_threshold=ELF3_DISTILL_MIN_GROUND_UPWARD_FORCE_N,
        filter_prim_paths_expr=[ELF3_DISTILL_GROUND_COLLIDER_PRIM_PATH],
        debug_vis=False,
    )

    # The TienKung locomotion teacher was trained with its 1.6 x 1.0 m scan
    # centered on the torso (x=0).  The climb/down-roll teachers use the
    # inherited x=0.4 m scanner.  Equal 187-D shapes do not make these two
    # spatial contracts interchangeable, so each frozen teacher receives its
    # own exact ray origin while the analytical platform overlay remains
    # shared.
    locomotion_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(
            resolution=ELF3_CLIMB_HEIGHT_SCAN_RESOLUTION,
            size=ELF3_CLIMB_HEIGHT_SCAN_SIZE,
        ),
        mesh_prim_paths=["/World/ground"],
        update_period=0.02,
        debug_vis=False,
        drift_range=(0.0, 0.0),
        ray_cast_drift_range={"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
    )

    depth_camera = TiledCameraCfg(
        # The unmodified URDF contains d435i_link, but fixed-link merging folds
        # it into torso_link in the runtime USD.  This sensor prim therefore
        # remains a torso child while its offset is the exact URDF head-camera
        # transform evaluated at the fixed deployment head pitch.
        prim_path="{ENV_REGEX_NS}/Robot/torso_link/HeadD435iDepthCamera",
        # Rendering remains on the 50 Hz control grid.  The observation term
        # performs absolute-time 30 Hz scheduling and reads this sensor only
        # when a capture is due.
        update_period=0.0,
        width=DEPTH_WIDTH,
        height=DEPTH_HEIGHT,
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=DEPTH_FOCAL_LENGTH,
            horizontal_aperture=DEPTH_HORIZONTAL_APERTURE,
            # Resizing a physical D435i stream to 87x58 must retain its field
            # of view.  Setting this explicitly avoids deriving an incorrect
            # vertical FOV from the non-native output aspect ratio.
            vertical_aperture=DEPTH_VERTICAL_APERTURE,
            # Match the verified deployment simulator's raw operating range.
            # The observation term independently clips the network input to
            # 0.15--2.0 m before normalization.
            clipping_range=(DEPTH_SENSOR_NEAR, DEPTH_SENSOR_FAR),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=mdp.ELF3_HEAD_D435I_CAMERA_POS,
            rot=mdp.ELF3_HEAD_D435I_CAMERA_ROT,
            convention="world",
        ),
    )


@configclass
class ELF3MultiSkillCommandsCfg:
    multi_skill = mdp.MultiSkillCommandCfg(
        asset_name="robot",
        platform_asset_name="platform",
        anchor_body_name="torso_link",
        root_body_name="torso_link",
        body_names=ELF3_CLIMB_TRACKED_BODY_NAMES,
        platform_size=ELF3_CLIMB_PLATFORM_SIZE,
        platform_center=ELF3_CLIMB_PLATFORM_CENTER,
        platform_height=ELF3_CLIMB_PLATFORM_SIZE[2],
    )


@configclass
class ELF3MultiSkillActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ELF3_CLIMB_JOINT_NAMES,
        scale=ELF3_CLIMB_ACTION_SCALE,
        clip=ELF3_CLIMB_JOINT_POSITION_TARGET_LIMITS,
        use_default_offset=True,
        preserve_order=True,
    )


def _motion_teacher_terms() -> dict[str, ObsTerm]:
    """Create the ordered 347-D climb/down teacher observation terms."""

    return {
        "command": ObsTerm(func=mdp.generated_commands, params={"command_name": "multi_skill"}),
        "motion_anchor_pos_b": ObsTerm(
            func=tracking_mdp.motion_anchor_pos_b,
            params={"command_name": "multi_skill"},
        ),
        "motion_anchor_ori_b": ObsTerm(
            func=tracking_mdp.motion_anchor_ori_b,
            params={"command_name": "multi_skill"},
        ),
        "base_lin_vel": ObsTerm(func=mdp.base_lin_vel),
        "base_ang_vel": ObsTerm(func=mdp.base_ang_vel),
        "joint_pos": ObsTerm(func=mdp.joint_pos_rel),
        "joint_vel": ObsTerm(func=mdp.joint_vel_rel),
        "actions": ObsTerm(func=mdp.last_action),
        "height_scan": ObsTerm(
            func=mdp.box_obstacle_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("platform"),
                "base_size": ELF3_CLIMB_PLATFORM_SIZE,
                "offset": ELF3_CLIMB_HEIGHT_SCAN_VALUE_OFFSET,
            },
            clip=(-1.0, 1.0),
        ),
    }


@configclass
class ELF3MultiSkillObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # One 93-D term is historized as a unit, which guarantees frame-major
        # oldest-to-newest layout rather than five independently flattened histories.
        proprio_history = ObsTerm(
            func=mdp.student_proprio_frame,
            history_length=8,
            flatten_history_dim=True,
        )
        body_command = ObsTerm(
            func=mdp.body_planar_command,
            params={"command_name": "multi_skill"},
        )
        depth = ObsTerm(
            func=mdp.DelayedDepthObservation,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "data_type": "distance_to_image_plane",
                "height": DEPTH_HEIGHT,
                "width": DEPTH_WIDTH,
                "near_clip": DEPTH_NEAR,
                "far_clip": DEPTH_FAR,
                "image_offset_range": (-0.03, 0.03),
                "pixel_noise_std": 0.03,
                "delay_range_s": (0.06, 0.08),
                "capture_frequency_hz": DEPTH_CAPTURE_FREQUENCY_HZ,
                "noise_enabled": True,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        proprio = ObsTerm(func=mdp.student_proprio_frame)
        world_command = ObsTerm(
            func=mdp.world_planar_command,
            params={"command_name": "multi_skill"},
        )
        skill = ObsTerm(func=mdp.skill_one_hot, params={"command_name": "multi_skill"})
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "multi_skill"})
        motion_anchor_pos_b = ObsTerm(
            func=tracking_mdp.motion_anchor_pos_b,
            params={"command_name": "multi_skill"},
        )
        motion_anchor_ori_b = ObsTerm(
            func=tracking_mdp.motion_anchor_ori_b,
            params={"command_name": "multi_skill"},
        )
        body_pos = ObsTerm(
            func=tracking_mdp.robot_body_pos_b,
            params={"command_name": "multi_skill"},
        )
        body_ori = ObsTerm(
            func=tracking_mdp.robot_body_ori_b,
            params={"command_name": "multi_skill"},
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        height_scan = ObsTerm(
            func=mdp.box_obstacle_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("platform"),
                "base_size": ELF3_CLIMB_PLATFORM_SIZE,
                "offset": ELF3_CLIMB_HEIGHT_SCAN_VALUE_OFFSET,
            },
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class SkillIdCfg(ObsGroup):
        value = ObsTerm(func=mdp.skill_id, params={"command_name": "multi_skill"})

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class TeacherValidCfg(ObsGroup):
        value = ObsTerm(func=mdp.teacher_valid, params={"command_name": "multi_skill"})

        def __post_init__(self):
            self.concatenate_terms = True

    @configclass
    class LocomotionTeacherCfg(ObsGroup):
        history = ObsTerm(
            func=mdp.locomotion_teacher_frame,
            params={"command_name": "multi_skill"},
            history_length=10,
            flatten_history_dim=True,
        )
        height_scan = ObsTerm(
            func=mdp.box_obstacle_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("locomotion_height_scanner"),
                "asset_cfg": SceneEntityCfg("platform"),
                "base_size": ELF3_CLIMB_PLATFORM_SIZE,
                "offset": ELF3_CLIMB_HEIGHT_SCAN_VALUE_OFFSET,
            },
            clip=(-100.0, 100.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class MotionTeacherCfg(ObsGroup):
        command = _motion_teacher_terms()["command"]
        motion_anchor_pos_b = _motion_teacher_terms()["motion_anchor_pos_b"]
        motion_anchor_ori_b = _motion_teacher_terms()["motion_anchor_ori_b"]
        base_lin_vel = _motion_teacher_terms()["base_lin_vel"]
        base_ang_vel = _motion_teacher_terms()["base_ang_vel"]
        joint_pos = _motion_teacher_terms()["joint_pos"]
        joint_vel = _motion_teacher_terms()["joint_vel"]
        actions = _motion_teacher_terms()["actions"]
        height_scan = _motion_teacher_terms()["height_scan"]

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    skill_id: SkillIdCfg = SkillIdCfg()
    teacher_valid: TeacherValidCfg = TeacherValidCfg()
    locomotion_teacher: LocomotionTeacherCfg = LocomotionTeacherCfg()
    motion_teacher: MotionTeacherCfg = MotionTeacherCfg()


def _routed(
    skill: str,
    wrapped_func,
    *,
    weight: float,
    wrapped_params: dict[str, object],
) -> RewTerm:
    return RewTerm(
        func=mdp.routed_reward,
        weight=weight,
        params={
            "command_name": "multi_skill",
            "skill": skill,
            "wrapped_func": wrapped_func,
            "wrapped_params": wrapped_params,
        },
    )


@configclass
class ELF3MultiSkillRewardsCfg:
    locomotion_velocity = RewTerm(
        func=mdp.locomotion_world_velocity_tracking_exp,
        weight=5.0,
        params={"command_name": "multi_skill", "std": 0.5},
    )
    locomotion_heading = RewTerm(
        func=mdp.locomotion_heading_tracking_exp,
        weight=2.0,
        params={"command_name": "multi_skill", "std": 0.5},
    )
    locomotion_upright = RewTerm(
        func=mdp.locomotion_upright_l2,
        weight=-2.0,
        params={"command_name": "multi_skill"},
    )

    climb_anchor_pos = _routed(
        "climb",
        tracking_mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        wrapped_params={"command_name": "multi_skill", "std": 0.3},
    )
    climb_anchor_ori = _routed(
        "climb",
        tracking_mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        wrapped_params={"command_name": "multi_skill", "std": 0.4},
    )
    climb_body_pos = _routed(
        "climb",
        tracking_mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        wrapped_params={
            "command_name": "multi_skill",
            "std": 0.3,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
        },
    )
    climb_body_ori = _routed(
        "climb",
        tracking_mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        wrapped_params={
            "command_name": "multi_skill",
            "std": 0.4,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
        },
    )
    climb_body_lin_vel = _routed(
        "climb",
        tracking_mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        wrapped_params={
            "command_name": "multi_skill",
            "std": 1.0,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
        },
    )
    climb_body_ang_vel = _routed(
        "climb",
        tracking_mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        wrapped_params={
            "command_name": "multi_skill",
            "std": math.pi,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
        },
    )
    climb_joint_pos = _routed(
        "climb",
        tracking_mdp.motion_grouped_expert_joint_position_error_exp,
        weight=2.0,
        wrapped_params={
            "command_name": "multi_skill",
            "joint_groups": ELF3_CLIMB_EXPERT_JOINT_TRACKING_GROUPS,
            "group_stds": ELF3_CLIMB_EXPERT_JOINT_POSITION_GROUP_STDS,
            "group_weights": ELF3_CLIMB_EXPERT_JOINT_GROUP_WEIGHTS,
            "score_exponent": ELF3_CLIMB_EXPERT_JOINT_SCORE_EXPONENT,
            "worst_joint_count": ELF3_CLIMB_EXPERT_JOINT_WORST_COUNT,
            "worst_joint_weight": ELF3_CLIMB_EXPERT_JOINT_WORST_WEIGHT,
            "group_aggregation": "harmonic",
            "worst_group_weight": ELF3_CLIMB_EXPERT_JOINT_WORST_GROUP_WEIGHT,
        },
    )
    climb_joint_vel = _routed(
        "climb",
        tracking_mdp.motion_grouped_expert_joint_velocity_error_exp,
        weight=0.5,
        wrapped_params={
            "command_name": "multi_skill",
            "joint_groups": ELF3_CLIMB_EXPERT_JOINT_TRACKING_GROUPS,
            "group_stds": ELF3_CLIMB_EXPERT_JOINT_VELOCITY_GROUP_STDS,
            "group_weights": ELF3_CLIMB_EXPERT_JOINT_GROUP_WEIGHTS,
            "score_exponent": ELF3_CLIMB_EXPERT_JOINT_SCORE_EXPONENT,
            "worst_joint_count": ELF3_CLIMB_EXPERT_JOINT_WORST_COUNT,
            "worst_joint_weight": ELF3_CLIMB_EXPERT_JOINT_WORST_WEIGHT,
            "group_aggregation": "harmonic",
            "worst_group_weight": ELF3_CLIMB_EXPERT_JOINT_WORST_GROUP_WEIGHT,
        },
    )

    down_roll_anchor_pos = _routed(
        "down_roll",
        tracking_mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        wrapped_params={"command_name": "multi_skill", "std": 0.3},
    )
    down_roll_anchor_ori = _routed(
        "down_roll",
        tracking_mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        wrapped_params={"command_name": "multi_skill", "std": 0.4},
    )
    down_roll_body_pos = _routed(
        "down_roll",
        tracking_mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        wrapped_params={
            "command_name": "multi_skill",
            "std": 0.3,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
        },
    )
    down_roll_body_ori = _routed(
        "down_roll",
        tracking_mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        wrapped_params={
            "command_name": "multi_skill",
            "std": 0.4,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
        },
    )
    down_roll_body_lin_vel = _routed(
        "down_roll",
        tracking_mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        wrapped_params={
            "command_name": "multi_skill",
            "std": 1.0,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
        },
    )
    down_roll_body_ang_vel = _routed(
        "down_roll",
        tracking_mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        wrapped_params={
            "command_name": "multi_skill",
            "std": math.pi,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
        },
    )
    down_roll_joint_pos = _routed(
        "down_roll",
        tracking_mdp.motion_grouped_expert_joint_position_error_exp,
        weight=2.0,
        wrapped_params={
            "command_name": "multi_skill",
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
    down_roll_joint_vel = _routed(
        "down_roll",
        tracking_mdp.motion_grouped_expert_joint_velocity_error_exp,
        weight=0.5,
        wrapped_params={
            "command_name": "multi_skill",
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

    # Geometry-relative objectives remain valid when the fixed-height motion
    # teacher is down-weighted.  They use the sampled collider dimensions and
    # real contact sensor, never a shifted expert Z trajectory.
    climb_geometry_progress = RewTerm(
        func=mdp.climb_geometry_progress,
        weight=3.0,
        params={
            "command_name": "multi_skill",
            "platform_cfg": SceneEntityCfg("platform"),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "platform_support_params": ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
        },
    )
    down_roll_geometry_progress = RewTerm(
        func=mdp.down_roll_geometry_progress,
        weight=3.0,
        params={
            "command_name": "multi_skill",
            "foot_cfg": SceneEntityCfg(
                "robot",
                body_names=list(ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS["foot_body_names"]),
                preserve_order=True,
            ),
            "ground_contact_sensor_names": ELF3_DISTILL_GROUND_CONTACT_SENSOR_NAMES,
            "sole_corners_b": tuple(ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS["sole_corners_b"]),
            "minimum_ground_upward_force": ELF3_DISTILL_MIN_GROUND_UPWARD_FORCE_N,
        },
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class ELF3MultiSkillTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    transition_failure = DoneTerm(
        func=mdp.routed_transition_failure,
        params={"command_name": "multi_skill"},
    )
    motion_tracking_failure = DoneTerm(
        func=mdp.routed_motion_tracking_failure,
        params={"command_name": "multi_skill"},
    )
    motion_clip_end = DoneTerm(
        func=mdp.routed_motion_clip_end,
        time_out=True,
        params={"command_name": "multi_skill"},
    )
    # The source climb torso never drops below 0.93 m.  This conservative
    # threshold terminates a hard fall even when fixed-height teacher tracking
    # is disabled, without applying to the deliberately low down-roll motion.
    climb_low_root = DoneTerm(
        func=mdp.climb_low_root,
        params={"command_name": "multi_skill", "minimum_height": 0.55},
    )
    locomotion_low_root = DoneTerm(
        func=mdp.locomotion_low_root,
        params={"command_name": "multi_skill", "minimum_height": 0.55},
    )
    locomotion_illegal_contact = DoneTerm(
        func=mdp.locomotion_illegal_contact,
        params={
            "command_name": "multi_skill",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    ".*_hip_z.*",
                    ".*_shoulder_y.*",
                    ".*_shoulder_z.*",
                    ".*_wrist_z.*",
                    "waist_z.*",
                    "torso_link",
                ],
            ),
            "threshold": 3.0,
        },
    )
    non_finite = DoneTerm(func=mdp.non_finite_robot_state)


@configclass
class ELF3MultiSkillEventCfg:
    platform_geometry = EventTerm(
        func=tracking_mdp.randomize_climb_box_geometry,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "length_range": ELF3_DISTILL_PLATFORM_LENGTH_RANGE,
            "width_range": ELF3_DISTILL_PLATFORM_WIDTH_RANGE,
            "height_range": ELF3_DISTILL_PLATFORM_HEIGHT_RANGE,
            "nominal_size_fraction": ELF3_DISTILL_NOMINAL_GEOMETRY_FRACTION,
        },
    )

    camera_extrinsics = EventTerm(
        func=mdp.randomize_camera_extrinsics,
        mode="reset",
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            # d435i_link is merged into this rigid body by the checked-in USD;
            # randomization composes around the deployment-aligned nominal
            # sensor transform above, without changing the robot asset.
            "parent_asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
            "translation_range_m": (
                -CAMERA_TRANSLATION_RANDOMIZATION,
                CAMERA_TRANSLATION_RANDOMIZATION,
            ),
            "rotation_range_rad": (
                -CAMERA_ROTATION_RANDOMIZATION,
                CAMERA_ROTATION_RANDOMIZATION,
            ),
        },
    )


@configclass
class ELF3MultiSkillEnvCfg(TrackingEnvCfg):
    scene: ELF3MultiSkillSceneCfg = ELF3MultiSkillSceneCfg(
        num_envs=2048,
        env_spacing=4.0,
        replicate_physics=False,
    )
    observations: ELF3MultiSkillObservationsCfg = ELF3MultiSkillObservationsCfg()
    actions: ELF3MultiSkillActionsCfg = ELF3MultiSkillActionsCfg()
    commands: ELF3MultiSkillCommandsCfg = ELF3MultiSkillCommandsCfg()
    rewards: ELF3MultiSkillRewardsCfg = ELF3MultiSkillRewardsCfg()
    terminations: ELF3MultiSkillTerminationsCfg = ELF3MultiSkillTerminationsCfg()
    events: ELF3MultiSkillEventCfg = ELF3MultiSkillEventCfg()
    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 20.0
        self.is_finite_horizon = False
        self.rerender_on_reset = True
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.locomotion_height_scanner.update_period = self.decimation * self.sim.dt
        self.viewer.eye = (2.5, 2.5, 2.0)
        self.viewer.origin_type = "world"
        self.viewer.asset_name = None

        if self.scene.replicate_physics:
            raise ValueError("per-environment platform geometry requires replicate_physics=False")
        if tuple(self.scene.platform.spawn.size) != tuple(ELF3_CLIMB_PLATFORM_SIZE):
            raise ValueError("the distillation platform spawn size must remain the 0.66 m source geometry")


__all__ = ["ELF3MultiSkillEnvCfg"]
