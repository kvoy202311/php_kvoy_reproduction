from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import php_kvoy_reproduction.tasks.tracking.mdp as mdp
from php_kvoy_reproduction.assets.elf3 import ELF3_CFG
from php_kvoy_reproduction.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


##
# ELF3 climb constants
##


VELOCITY_RANGE = {
    "x": (0.0, 0.0),
    "y": (0.0, 0.0),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (0.0, 0.0),
}


# Policy/action order used by Isaac for ELF3. Keeping the order explicit makes
# action-manager initialization fail if a joint is missing or renamed.
ELF3_CLIMB_JOINT_NAMES = [
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


# Joint-space scales from TienKung-ELF3's elf3_walk_wb task. These belong to
# the task action configuration and do not modify the articulation asset.
ELF3_CLIMB_ACTION_SCALE = {
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


ELF3_CLIMB_TRACKED_BODY_NAMES = [
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


ELF3_CLIMB_END_EFFECTOR_NAMES = [
    "l_ankle_x_link",
    "r_ankle_x_link",
    "l_wrist_z_link",
    "r_wrist_z_link",
]

# Bodies allowed to provide physical support for the conservative progress
# hint.  Contact and platform geometry still gate the term; this is not a
# substitute for the expert motion target.
ELF3_CLIMB_PROGRESS_SUPPORT_BODY_NAMES = [
    "l_ankle_x_link",
    "r_ankle_x_link",
]

# During the final platform-contact window, ankle position/orientation
# tracking is deliberately softened.  The expert still controls the dynamic
# climb; physical platform contact controls the final foot placement.
ELF3_CLIMB_TERMINAL_FOOT_TRACKING_WEIGHT = 0.15

# The first platform support is evaluated from the physical sole geometry, not
# from the ankle-link origin.  The values are conservative bounds of the ELF3
# foot mesh in the ankle-link frame.  Keeping the sole samples explicit lets a
# pitched forefoot contact remain valid while enforcing the requested heel
# overhang limit on the real contact geometry.
ELF3_CLIMB_FIRST_FOOTHOLD_FOOT_BODY_NAMES = ["l_ankle_x_link", "r_ankle_x_link"]
ELF3_CLIMB_FIRST_FOOTHOLD_PLATFORM_CONTACT_SENSOR_NAMES = [
    "l_foot_platform_contact",
    "r_foot_platform_contact",
]
ELF3_CLIMB_FIRST_FOOTHOLD_SOLE_CORNERS_B = (
    (-0.09, -0.04, -0.041),
    (-0.09, 0.04, -0.041),
    (0.15, -0.04, -0.041),
    (0.15, 0.04, -0.041),
)
# User-approved physical limit: a rear-most sole point may extend at most
# 5 cm past the approach edge.  Beyond this limit the positive support credit
# is exactly zero and the first-foot term applies a bounded return-to-safety
# penalty; this is not a soft extra 5--7 cm band.
ELF3_CLIMB_FIRST_FOOTHOLD_MAX_HEEL_OVERHANG = 0.05
ELF3_CLIMB_FIRST_FOOTHOLD_MIN_FOREFOOT_INSIDE = 0.04
ELF3_CLIMB_FIRST_FOOTHOLD_FAR_EDGE_MARGIN = 0.04
ELF3_CLIMB_FIRST_FOOTHOLD_LATERAL_MARGIN = 0.02
ELF3_CLIMB_FIRST_FOOTHOLD_HEIGHT_STD = 0.06
ELF3_CLIMB_FIRST_FOOTHOLD_PRECONTACT_APPROACH_DISTANCE = 0.15
ELF3_CLIMB_FIRST_FOOTHOLD_PRECONTACT_HEIGHT_STD = 0.16
ELF3_CLIMB_FIRST_FOOTHOLD_REFERENCE_ACTIVATION_DISTANCE = 0.20
ELF3_CLIMB_FIRST_FOOTHOLD_REFERENCE_ACTIVATION_INSIDE = 0.03
ELF3_CLIMB_FIRST_FOOTHOLD_REFERENCE_RELEASE_DISTANCE = 0.08
ELF3_CLIMB_FIRST_FOOTHOLD_REFERENCE_RELEASE_INSIDE = 0.03
ELF3_CLIMB_FIRST_FOOTHOLD_PHASE_START = 0.28
ELF3_CLIMB_FIRST_FOOTHOLD_PHASE_RAMP = 0.08
ELF3_CLIMB_FIRST_FOOTHOLD_PHASE_END = 0.72
ELF3_CLIMB_FIRST_FOOTHOLD_PHASE_FADE = 0.10
ELF3_CLIMB_FIRST_FOOTHOLD_REWARD_WEIGHT = 4.0
ELF3_CLIMB_FIRST_FOOTHOLD_MIN_UPWARD_FORCE_N = 10.0
ELF3_CLIMB_FIRST_FOOTHOLD_CONTACT_TIME_S = 0.06

# Once the first foot has made safe, real platform contact, its ankle and a
# small amount of same-side knee tracking may soften to accommodate physical
# support. Hip and torso tracking stay intact. The gate is applied to
# position, orientation, and velocity objectives so the immutable expert
# motion cannot pull an already-supported foot back to a bad edge pose.
ELF3_CLIMB_FIRST_FOOTHOLD_POSITION_TRACKING_WEIGHTS = {
    "l_knee_y_link": 0.65,
    "r_knee_y_link": 0.65,
    "l_ankle_x_link": 0.25,
    "r_ankle_x_link": 0.25,
}
ELF3_CLIMB_FIRST_FOOTHOLD_ORIENTATION_TRACKING_WEIGHTS = {
    "l_knee_y_link": 0.80,
    "r_knee_y_link": 0.80,
    "l_ankle_x_link": 0.50,
    "r_ankle_x_link": 0.50,
}
ELF3_CLIMB_FIRST_FOOTHOLD_LINEAR_VELOCITY_TRACKING_WEIGHTS = {
    "l_knee_y_link": 0.70,
    "r_knee_y_link": 0.70,
    "l_ankle_x_link": 0.35,
    "r_ankle_x_link": 0.35,
}
ELF3_CLIMB_FIRST_FOOTHOLD_ANGULAR_VELOCITY_TRACKING_WEIGHTS = {
    "l_knee_y_link": 0.75,
    "r_knee_y_link": 0.75,
    "l_ankle_x_link": 0.45,
    "r_ankle_x_link": 0.45,
}
ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS = {
    "foot_body_names": ELF3_CLIMB_FIRST_FOOTHOLD_FOOT_BODY_NAMES,
    "platform_contact_sensor_names": ELF3_CLIMB_FIRST_FOOTHOLD_PLATFORM_CONTACT_SENSOR_NAMES,
    "sole_corners_b": ELF3_CLIMB_FIRST_FOOTHOLD_SOLE_CORNERS_B,
    "approach_side": -1.0,
    "max_heel_overhang": ELF3_CLIMB_FIRST_FOOTHOLD_MAX_HEEL_OVERHANG,
    "min_forefoot_inside": ELF3_CLIMB_FIRST_FOOTHOLD_MIN_FOREFOOT_INSIDE,
    "far_edge_margin": ELF3_CLIMB_FIRST_FOOTHOLD_FAR_EDGE_MARGIN,
    "lateral_margin": ELF3_CLIMB_FIRST_FOOTHOLD_LATERAL_MARGIN,
}
ELF3_CLIMB_FIRST_FOOTHOLD_PARAMS = {
    **ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
    "foot_height_std": ELF3_CLIMB_FIRST_FOOTHOLD_HEIGHT_STD,
    "precontact_approach_distance": ELF3_CLIMB_FIRST_FOOTHOLD_PRECONTACT_APPROACH_DISTANCE,
    "precontact_height_std": ELF3_CLIMB_FIRST_FOOTHOLD_PRECONTACT_HEIGHT_STD,
    "reference_activation_distance": ELF3_CLIMB_FIRST_FOOTHOLD_REFERENCE_ACTIVATION_DISTANCE,
    "reference_activation_inside": ELF3_CLIMB_FIRST_FOOTHOLD_REFERENCE_ACTIVATION_INSIDE,
    "reference_release_distance": ELF3_CLIMB_FIRST_FOOTHOLD_REFERENCE_RELEASE_DISTANCE,
    "reference_release_inside": ELF3_CLIMB_FIRST_FOOTHOLD_REFERENCE_RELEASE_INSIDE,
    "phase_start": ELF3_CLIMB_FIRST_FOOTHOLD_PHASE_START,
    "phase_ramp": ELF3_CLIMB_FIRST_FOOTHOLD_PHASE_RAMP,
    "phase_end": ELF3_CLIMB_FIRST_FOOTHOLD_PHASE_END,
    "phase_fade": ELF3_CLIMB_FIRST_FOOTHOLD_PHASE_FADE,
    "min_upward_force": ELF3_CLIMB_FIRST_FOOTHOLD_MIN_UPWARD_FORCE_N,
    "contact_time_scale": ELF3_CLIMB_FIRST_FOOTHOLD_CONTACT_TIME_S,
}


# Start from the physical geometry closest to the source demonstrations.  The
# 0.51 m length moves only the platform edge, not the immutable reference XY,
# and gives the first heel useful margin inside the approved 5 cm overhang
# limit.  Height generalization is introduced by later training stages rather
# than mixing a 0.60--0.70 m source-Z mismatch into the first run.
ELF3_CLIMB_PLATFORM_SIZE = (0.51, 0.80, 0.66)
ELF3_CLIMB_PLATFORM_CENTER = (-0.95, 0.0, 0.33)
ELF3_CLIMB_PLATFORM_LENGTH_RANGE = (0.51, 0.51)
ELF3_CLIMB_PLATFORM_WIDTH_RANGE = (0.80, 0.80)
ELF3_CLIMB_PLATFORM_HEIGHT_RANGE = (0.66, 0.66)
# Geometry is fixed in this source-aligned run.  The deterministic 10% mask
# selects environments that may start at a random motion phase; the other 90%
# are forced to start at the clip beginning.  It is an initialization partition,
# not a nominal-versus-randomized geometry partition.
ELF3_CLIMB_RANDOM_PHASE_ENV_FRACTION = 0.1
# Backward-compatible import alias for external configurations written before
# the phase mask was named explicitly.  New code must use the name above.
ELF3_CLIMB_NOMINAL_GEOMETRY_ENV_FRACTION = ELF3_CLIMB_RANDOM_PHASE_ENV_FRACTION
ELF3_CLIMB_PLATFORM_X_OFFSET_RANGE = (-0.05, 0.05)
ELF3_CLIMB_PLATFORM_Y_OFFSET_RANGE = (-0.05, 0.05)
ELF3_CLIMB_PLATFORM_YAW_RANGE = (-math.pi / 4.0, math.pi / 4.0)


# A 0.1 m grid resolves the 0.51 m-long platform with multiple samples while
# keeping the observation compact: 17 longitudinal x 11 lateral = 187 points.
ELF3_CLIMB_HEIGHT_SCAN_RESOLUTION = 0.1
ELF3_CLIMB_HEIGHT_SCAN_SIZE = (1.6, 1.0)
ELF3_CLIMB_HEIGHT_SCAN_OFFSET = (0.4, 0.0, 20.0)
ELF3_CLIMB_HEIGHT_SCAN_VALUE_OFFSET = 0.5


# The source clips already contain a motionless 0.5 s tail.  Keep only a
# compact final hold: 0.5 s for the terminal whole-body Z bridge, 0.25 s for
# real dual-foot contact confirmation, and 0.5 s of continuous strict
# standing.  At 50 Hz, 1.30 s gives 65 policy steps versus the 63-step worst
# case, without reintroducing a second terminal joint-pose target.
ELF3_CLIMB_FINAL_HOLD_TIME_S = 1.30
# The final source soles are not at the same height as every sampled box. In
# the static hold, smoothly translate the whole source-body reference only in
# Z to the actual platform top. This preserves the source x/y footprint and
# full source joint/body geometry throughout the short terminal hold.
ELF3_CLIMB_TERMINAL_PLATFORM_ALIGNMENT_RAMP_TIME_S = 0.5
ELF3_CLIMB_TERMINAL_PLATFORM_ALIGNMENT_CLEARANCE = 0.0
ELF3_CLIMB_MIN_STABLE_TIME_S = 0.5
ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S = 0.25
ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N = 10.0
ELF3_CLIMB_FOOTPRINT_INSET = 0.02
ELF3_CLIMB_FOOT_HEIGHT_RANGE = (-0.03, 0.15)
ELF3_CLIMB_TERMINAL_REWARD_WINDOW_S = 1.5
# These shared strict platform-foot support tolerances retain their existing
# names for configuration compatibility.  They are used by rewards and final
# success even though the optional default-q terminal handoff is disabled.
ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE = 0.04
ELF3_CLIMB_TERMINAL_DEFAULT_POSE_MIN_TOTAL_LOAD_FRACTION = 0.50
# Bounded, non-repeatable physical progress shaping requested for the climb.
# Expert tracking remains active throughout the dynamic trajectory.
ELF3_CLIMB_PROGRESS_REWARD_WEIGHT = 3.2
ELF3_CLIMB_PROGRESS_APPROACH_DISTANCE = 0.45
ELF3_CLIMB_PROGRESS_APPROACH_LATERAL_MARGIN = 0.15
ELF3_CLIMB_PROGRESS_SUPPORT_XY_MARGIN = 0.12
ELF3_CLIMB_PROGRESS_SUPPORT_HEIGHT_STD = 0.12
ELF3_CLIMB_PROGRESS_LIFT_HEIGHT_WINDOW = 0.25
ELF3_CLIMB_PROGRESS_APPROACH_SIDE = -1.0
ELF3_CLIMB_PROGRESS_APPROACH_PHASE_END = 0.70
ELF3_CLIMB_PROGRESS_LIFT_PHASE_START = 0.30
ELF3_CLIMB_PROGRESS_LIFT_PHASE_END = 0.75
ELF3_CLIMB_PROGRESS_APPROACH_WEIGHT = 0.65
ELF3_CLIMB_PROGRESS_LIFT_WEIGHT = 0.35
ELF3_CLIMB_PROGRESS_MAX_DELTA_PER_STEP = 0.05
ELF3_CLIMB_FOOT_HEIGHT_REWARD_STD = 0.08
ELF3_CLIMB_ROOT_LINEAR_SPEED_REWARD_STD = 0.15
ELF3_CLIMB_ROOT_ANGULAR_SPEED_REWARD_STD = 0.5
ELF3_CLIMB_JOINT_SPEED_REWARD_STD = 0.5
ELF3_CLIMB_TORSO_TILT_REWARD_STD = 0.35
# Ordered as root linear speed, root angular speed, joint speed, torso tilt.
# The weighted average keeps a temporarily poor individual signal from
# collapsing the complete settling reward to zero.
ELF3_CLIMB_STABILITY_REWARD_WEIGHTS = (0.20, 0.35, 0.35, 0.10)
# A separate terminal settling signal addresses the strict all-joint speed
# success condition. Its maximum weight is close to the 4.2 effective joint-
# speed share of final_standing_stability (12.0 * 0.35), while remaining gated
# by a stationary expert target and sustained two-foot platform support.
ELF3_CLIMB_MAX_ROOT_HEIGHT_ERROR = 0.15
ELF3_CLIMB_MAX_ROOT_LINEAR_SPEED = 0.15
ELF3_CLIMB_MAX_ROOT_ANGULAR_SPEED = 0.5
ELF3_CLIMB_MAX_JOINT_SPEED = 0.5
ELF3_CLIMB_MAX_TORSO_TILT = 0.35
ELF3_CLIMB_FINAL_JOINT_SETTLING_REWARD_WEIGHT = 4.0
ELF3_CLIMB_REFERENCE_STATIC_MAX_JOINT_SPEED = 0.10
# Score excess over the actual success limits, rather than merely preferring
# an arbitrary low raw velocity.  This makes every all-joint <=0.5 rad/s
# state strongly preferable to the 1--3 rad/s terminal oscillations seen in
# the previous runs while retaining non-zero gradients above the threshold.
ELF3_CLIMB_SETTLING_RMS_SPEED_TOLERANCE = 0.35
ELF3_CLIMB_SETTLING_MAX_SPEED_TOLERANCE = ELF3_CLIMB_MAX_JOINT_SPEED
ELF3_CLIMB_SETTLING_RMS_SPEED_SCALE = 0.35
ELF3_CLIMB_SETTLING_MAX_SPEED_SCALE = 0.35
ELF3_CLIMB_SETTLING_FINE_MAX_SPEED_SCALE = 0.25
# Ordered as RMS broad, maximum-speed broad, maximum-speed fine.
ELF3_CLIMB_SETTLING_SCORE_WEIGHTS = (0.25, 0.60, 0.15)
# The final source pose is already a natural stable posture.  A broad all-joint
# target prevents a stationary crossed-leg/folded-arm pose from becoming a
# local optimum while still allowing small physical deviations from the clip.
ELF3_CLIMB_FINAL_EXPERT_JOINT_POSE_REWARD_WEIGHT = 6.0
ELF3_CLIMB_FINAL_EXPERT_JOINT_POSE_STD = 0.50
ELF3_CLIMB_FINAL_EXPERT_JOINT_POSE_WINDOW_S = 0.50
ELF3_CLIMB_MAX_EXPERT_JOINT_POS_RMS = 0.50


# Three performance-gated reset-pose stages: fixed, half range, full range.
# Collider sizes remain distributed at prestartup because PhysX collider scale
# cannot be changed safely by a reset-time curriculum.
ELF3_CLIMB_TERRAIN_CURRICULUM_STAGE_SCALES = (0.0, 0.5, 1.0)
ELF3_CLIMB_TERRAIN_CURRICULUM_ADVANCE_SUCCESS_RATE = 0.80
ELF3_CLIMB_TERRAIN_CURRICULUM_REGRESS_SUCCESS_RATE = 0.50
ELF3_CLIMB_TERRAIN_CURRICULUM_MIN_EPISODES = 8192


##
# Scene configuration
##


@configclass
class ELF3ClimbSceneCfg(InteractiveSceneCfg):
    """Complete scene configuration for the ELF3 climb expert."""

    # Infinite ground plane shared by all environments.
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )

    # Kinematic expert-training platform. It remains physically fixed during
    # each episode, while its per-environment size and reset pose can differ.
    platform = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ClimbPlatform",
        init_state=RigidObjectCfg.InitialStateCfg(pos=ELF3_CLIMB_PLATFORM_CENTER),
        spawn=sim_utils.CuboidCfg(
            size=ELF3_CLIMB_PLATFORM_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 0.90)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        ),
    )

    # ELF3 robot asset. The asset itself is reused without modification.
    robot = ELF3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting.
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )

    # Robot contact sensor used by contact rewards and air-time information.
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=10.0,
        debug_vis=True,
    )

    # Filtered, one-body sensors are required for a trustworthy first-foot
    # platform-support reward.  Isaac Lab's contact-force matrix is only
    # reliable with a single sensor body per filter, hence one sensor per
    # ankle instead of reusing the all-robot contact sensor above.
    l_foot_platform_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/l_ankle_x_link",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        force_threshold=ELF3_CLIMB_FIRST_FOOTHOLD_MIN_UPWARD_FORCE_N,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/ClimbPlatform"],
        debug_vis=False,
    )
    r_foot_platform_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/r_ankle_x_link",
        update_period=0.0,
        history_length=3,
        track_air_time=True,
        force_threshold=ELF3_CLIMB_FIRST_FOOTHOLD_MIN_UPWARD_FORCE_N,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/ClimbPlatform"],
        debug_vis=False,
    )

    # Yaw-aligned terrain scan attached to the torso. Isaac Lab 4.5 can ray
    # cast against only one static mesh, so this sensor measures the ground;
    # the observation term below overlays the separately spawned platform.
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=ELF3_CLIMB_HEIGHT_SCAN_OFFSET),
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


##
# MDP configuration
##


@configclass
class ELF3ClimbCommandsCfg:
    """Reference-motion command configuration."""

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
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(0.0, 0.0),
        terminate_on_motion_end=True,
        motion_end_hold_time_s=ELF3_CLIMB_FINAL_HOLD_TIME_S,
        adaptive_failure_term_names=("motion_end_failure",),
        random_phase_env_mask_attr="_climb_box_random_phase_env_mask",
        reference_transform_asset_name="platform",
        reference_transform_nominal_xy=ELF3_CLIMB_PLATFORM_CENTER[:2],
        terminal_platform_alignment_foot_body_names=tuple(ELF3_CLIMB_FIRST_FOOTHOLD_FOOT_BODY_NAMES),
        terminal_platform_alignment_sole_corners_b=ELF3_CLIMB_FIRST_FOOTHOLD_SOLE_CORNERS_B,
        terminal_platform_alignment_base_size=ELF3_CLIMB_PLATFORM_SIZE,
        terminal_platform_alignment_clearance=ELF3_CLIMB_TERMINAL_PLATFORM_ALIGNMENT_CLEARANCE,
        terminal_platform_alignment_ramp_time_s=ELF3_CLIMB_TERMINAL_PLATFORM_ALIGNMENT_RAMP_TIME_S,
        terminal_support_confirmation_time_s=ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S,
        terminal_stable_time_s=ELF3_CLIMB_MIN_STABLE_TIME_S,
        # The source terminal pose is the sole joint target throughout this
        # short hold.  A default-q handoff needs more than two seconds and is
        # intentionally disabled rather than compressed into this window.
        terminal_default_pose_enabled=False,
        terminal_default_pose_platform_support_params=ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
        terminal_default_pose_base_size=ELF3_CLIMB_PLATFORM_SIZE,
        terminal_default_pose_sole_height_tolerance=ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE,
        terminal_default_pose_min_upward_force=ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N,
        terminal_default_pose_min_total_load_fraction=ELF3_CLIMB_TERMINAL_DEFAULT_POSE_MIN_TOTAL_LOAD_FRACTION,
    )


@configclass
class ELF3ClimbActionsCfg:
    """ELF3 joint-position action configuration."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ELF3_CLIMB_JOINT_NAMES,
        scale=ELF3_CLIMB_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ELF3ClimbObservationsCfg:
    """Actor and critic observation configuration."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Noisy observations available to the actor policy."""

        # Observation term order is preserved in the concatenated tensor.
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        terminal_default_pose_alpha = ObsTerm(
            func=mdp.terminal_default_pose_alpha,
            params={"command_name": "motion"},
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        terminal_default_pose_active = ObsTerm(
            func=mdp.terminal_default_pose_active,
            params={"command_name": "motion"},
            noise=Unoise(n_min=0.0, n_max=0.0),
        )
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=0.0, n_max=0.0))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=0.0, n_max=0.0))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=0.0, n_max=0.0))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=0.0, n_max=0.0))
        actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.box_obstacle_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("platform"),
                "base_size": ELF3_CLIMB_PLATFORM_SIZE,
                "offset": ELF3_CLIMB_HEIGHT_SCAN_VALUE_OFFSET,
            },
            noise=Unoise(n_min=0.0, n_max=0.0),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        """Noise-free privileged observations available to the critic."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        terminal_default_pose_alpha = ObsTerm(func=mdp.terminal_default_pose_alpha, params={"command_name": "motion"})
        terminal_default_pose_active = ObsTerm(func=mdp.terminal_default_pose_active, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
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

    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class ELF3ClimbEventCfg:
    """Startup and interval domain-randomization events."""

    # Each cloned platform receives its own collider/visual dimensions before
    # PhysX parses the scene. This does not touch the ELF3 articulation asset.
    platform_geometry = EventTerm(
        func=mdp.randomize_climb_box_geometry,
        mode="prestartup",
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "length_range": ELF3_CLIMB_PLATFORM_LENGTH_RANGE,
            "width_range": ELF3_CLIMB_PLATFORM_WIDTH_RANGE,
            "height_range": ELF3_CLIMB_PLATFORM_HEIGHT_RANGE,
            "nominal_size_fraction": ELF3_CLIMB_RANDOM_PHASE_ENV_FRACTION,
        },
    )

    # On reset, move and yaw the kinematic platform. Its center height follows
    # the sampled geometry so its lower face remains exactly on the ground.
    platform_pose = EventTerm(
        func=mdp.reset_climb_box_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("platform"),
            "base_center_xy": ELF3_CLIMB_PLATFORM_CENTER[:2],
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "position_range": {
                "x": ELF3_CLIMB_PLATFORM_X_OFFSET_RANGE,
                "y": ELF3_CLIMB_PLATFORM_Y_OFFSET_RANGE,
            },
            "yaw_range": ELF3_CLIMB_PLATFORM_YAW_RANGE,
        },
    )

    # Randomize rigid-body contact material once at startup.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # Add a small fixed per-environment offset to default joint positions.
    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (0.0, 0.0),
            "operation": "add",
        },
    )

    # Randomize the ELF3 torso center of mass once at startup.
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        },
    )

    # Apply periodic velocity pushes during an episode.
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 2.0),
        params={"velocity_range": VELOCITY_RANGE},
    )


@configclass
class ELF3ClimbRewardsCfg:
    """Whole-body motion-tracking rewards and regularization penalties."""

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
        func=mdp.climb_motion_relative_body_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.3,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
            "terminal_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
            "terminal_body_weight": ELF3_CLIMB_TERMINAL_FOOT_TRACKING_WEIGHT,
            "platform_cfg": SceneEntityCfg("platform"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["l_ankle_x_link", "r_ankle_x_link"]
            ),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "foot_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
            "footprint_inset": ELF3_CLIMB_FOOTPRINT_INSET,
            "foot_height_std": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE,
            "min_contact_force": ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N,
            "contact_time_scale": ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S,
            "terminal_window_time_s": ELF3_CLIMB_TERMINAL_REWARD_WINDOW_S,
            "first_foothold_params": ELF3_CLIMB_FIRST_FOOTHOLD_PARAMS,
            "first_foothold_body_weights": ELF3_CLIMB_FIRST_FOOTHOLD_POSITION_TRACKING_WEIGHTS,
            "platform_support_params": ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
        },
    )
    motion_body_ori = RewTerm(
        func=mdp.climb_motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.4,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
            "terminal_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
            "terminal_body_weight": ELF3_CLIMB_TERMINAL_FOOT_TRACKING_WEIGHT,
            "platform_cfg": SceneEntityCfg("platform"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["l_ankle_x_link", "r_ankle_x_link"]
            ),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "foot_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
            "footprint_inset": ELF3_CLIMB_FOOTPRINT_INSET,
            "foot_height_std": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE,
            "min_contact_force": ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N,
            "contact_time_scale": ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S,
            "terminal_window_time_s": ELF3_CLIMB_TERMINAL_REWARD_WINDOW_S,
            "first_foothold_params": ELF3_CLIMB_FIRST_FOOTHOLD_PARAMS,
            "first_foothold_body_weights": ELF3_CLIMB_FIRST_FOOTHOLD_ORIENTATION_TRACKING_WEIGHTS,
            "platform_support_params": ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
        },
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.climb_motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 1.0,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
            "platform_cfg": SceneEntityCfg("platform"),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "first_foothold_params": ELF3_CLIMB_FIRST_FOOTHOLD_PARAMS,
            "first_foothold_body_weights": ELF3_CLIMB_FIRST_FOOTHOLD_LINEAR_VELOCITY_TRACKING_WEIGHTS,
        },
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.climb_motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 3.14,
            "body_names": ELF3_CLIMB_TRACKED_BODY_NAMES,
            "platform_cfg": SceneEntityCfg("platform"),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "first_foothold_params": ELF3_CLIMB_FIRST_FOOTHOLD_PARAMS,
            "first_foothold_body_weights": ELF3_CLIMB_FIRST_FOOTHOLD_ANGULAR_VELOCITY_TRACKING_WEIGHTS,
        },
    )
    platform_foot_contact = RewTerm(
        func=mdp.platform_foot_contact,
        weight=5.0,
        params={
            "command_name": "motion",
            "platform_cfg": SceneEntityCfg("platform"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_ankle_x_link", "r_ankle_x_link"],
            ),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "foot_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
            "footprint_inset": ELF3_CLIMB_FOOTPRINT_INSET,
            "foot_height_std": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE,
            "min_contact_force": ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N,
            "contact_time_scale": ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S,
            "terminal_window_time_s": ELF3_CLIMB_TERMINAL_REWARD_WINDOW_S,
            "platform_support_params": ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
        },
    )
    final_standing_stability = RewTerm(
        func=mdp.final_standing_stability,
        weight=12.0,
        params={
            "command_name": "motion",
            "platform_cfg": SceneEntityCfg("platform"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_ankle_x_link", "r_ankle_x_link"],
            ),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "foot_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
            "footprint_inset": ELF3_CLIMB_FOOTPRINT_INSET,
            "foot_height_std": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE,
            "min_contact_force": ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N,
            "contact_time_scale": ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S,
            "terminal_window_time_s": ELF3_CLIMB_TERMINAL_REWARD_WINDOW_S,
            "root_linear_speed_std": ELF3_CLIMB_ROOT_LINEAR_SPEED_REWARD_STD,
            "root_angular_speed_std": ELF3_CLIMB_ROOT_ANGULAR_SPEED_REWARD_STD,
            "joint_speed_std": ELF3_CLIMB_JOINT_SPEED_REWARD_STD,
            "torso_tilt_std": ELF3_CLIMB_TORSO_TILT_REWARD_STD,
            "stability_weights": ELF3_CLIMB_STABILITY_REWARD_WEIGHTS,
            "platform_support_params": ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
            "min_total_load_fraction": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_MIN_TOTAL_LOAD_FRACTION,
            "expert_joint_pose_std": ELF3_CLIMB_FINAL_EXPERT_JOINT_POSE_STD,
        },
    )
    final_joint_settling = RewTerm(
        func=mdp.final_joint_settling,
        weight=ELF3_CLIMB_FINAL_JOINT_SETTLING_REWARD_WEIGHT,
        params={
            "command_name": "motion",
            "platform_cfg": SceneEntityCfg("platform"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_ankle_x_link", "r_ankle_x_link"],
            ),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "foot_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
            "footprint_inset": ELF3_CLIMB_FOOTPRINT_INSET,
            "foot_height_std": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE,
            "min_contact_force": ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N,
            "contact_time_scale": ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S,
            "reference_max_joint_speed": ELF3_CLIMB_REFERENCE_STATIC_MAX_JOINT_SPEED,
            "rms_speed_tolerance": ELF3_CLIMB_SETTLING_RMS_SPEED_TOLERANCE,
            "max_speed_tolerance": ELF3_CLIMB_SETTLING_MAX_SPEED_TOLERANCE,
            "rms_speed_scale": ELF3_CLIMB_SETTLING_RMS_SPEED_SCALE,
            "max_speed_scale": ELF3_CLIMB_SETTLING_MAX_SPEED_SCALE,
            "fine_max_speed_scale": ELF3_CLIMB_SETTLING_FINE_MAX_SPEED_SCALE,
            "score_weights": ELF3_CLIMB_SETTLING_SCORE_WEIGHTS,
            "platform_support_params": ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
            "min_total_load_fraction": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_MIN_TOTAL_LOAD_FRACTION,
            "expert_joint_pose_std": ELF3_CLIMB_FINAL_EXPERT_JOINT_POSE_STD,
        },
    )
    final_expert_joint_pose = RewTerm(
        func=mdp.final_expert_joint_position_error_exp,
        weight=ELF3_CLIMB_FINAL_EXPERT_JOINT_POSE_REWARD_WEIGHT,
        params={
            "command_name": "motion",
            "platform_cfg": SceneEntityCfg("platform"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_ankle_x_link", "r_ankle_x_link"],
            ),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "foot_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
            "footprint_inset": ELF3_CLIMB_FOOTPRINT_INSET,
            "foot_height_std": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE,
            "min_contact_force": ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N,
            "contact_time_scale": ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S,
            "reference_max_joint_speed": ELF3_CLIMB_REFERENCE_STATIC_MAX_JOINT_SPEED,
            "static_window_time_s": ELF3_CLIMB_FINAL_EXPERT_JOINT_POSE_WINDOW_S,
            "std": ELF3_CLIMB_FINAL_EXPERT_JOINT_POSE_STD,
            "platform_support_params": ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
            "min_total_load_fraction": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_MIN_TOTAL_LOAD_FRACTION,
        },
    )
    first_foothold_support_quality = RewTerm(
        func=mdp.first_foothold_support_quality,
        weight=ELF3_CLIMB_FIRST_FOOTHOLD_REWARD_WEIGHT,
        params={
            "command_name": "motion",
            "platform_cfg": SceneEntityCfg("platform"),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "first_foothold_params": ELF3_CLIMB_FIRST_FOOTHOLD_PARAMS,
        },
    )
    climb_platform_progress = RewTerm(
        func=mdp.climb_platform_progress,
        weight=ELF3_CLIMB_PROGRESS_REWARD_WEIGHT,
        params={
            "command_name": "motion",
            "platform_cfg": SceneEntityCfg("platform"),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "support_body_names": ELF3_CLIMB_PROGRESS_SUPPORT_BODY_NAMES,
            "lift_body_names": ELF3_CLIMB_PROGRESS_SUPPORT_BODY_NAMES,
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=ELF3_CLIMB_PROGRESS_SUPPORT_BODY_NAMES, preserve_order=True
            ),
            "approach_distance": ELF3_CLIMB_PROGRESS_APPROACH_DISTANCE,
            "approach_lateral_margin": ELF3_CLIMB_PROGRESS_APPROACH_LATERAL_MARGIN,
            "support_xy_margin": ELF3_CLIMB_PROGRESS_SUPPORT_XY_MARGIN,
            "support_height_std": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE,
            "lift_height_window": ELF3_CLIMB_PROGRESS_LIFT_HEIGHT_WINDOW,
            "approach_side": ELF3_CLIMB_PROGRESS_APPROACH_SIDE,
            "approach_phase_end": ELF3_CLIMB_PROGRESS_APPROACH_PHASE_END,
            "lift_phase_start": ELF3_CLIMB_PROGRESS_LIFT_PHASE_START,
            "lift_phase_end": ELF3_CLIMB_PROGRESS_LIFT_PHASE_END,
            "approach_weight": ELF3_CLIMB_PROGRESS_APPROACH_WEIGHT,
            "lift_weight": ELF3_CLIMB_PROGRESS_LIFT_WEIGHT,
            "min_contact_force": ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N,
            "contact_time_scale": ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S,
            "max_delta_per_step": ELF3_CLIMB_PROGRESS_MAX_DELTA_PER_STEP,
            "first_foothold_params": ELF3_CLIMB_FIRST_FOOTHOLD_PARAMS,
            "platform_support_params": ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!l_ankle_x_link$)(?!r_ankle_x_link$)(?!l_wrist_z_link$)(?!r_wrist_z_link$).+$"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class ELF3ClimbTerminationsCfg:
    """Episode timeout and early motion-tracking termination conditions."""

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
            "body_names": ELF3_CLIMB_END_EFFECTOR_NAMES,
        },
    )
    # Keep all physical failures above the clip-boundary terms.  The timeout
    # functions inspect the accumulated terminated mask so a transition can
    # never be both a physical termination and a bootstrapped timeout.
    # A reference clip ending is an external data boundary, not a physical
    # failure.  Reset the environment and let PPO bootstrap the value target.
    motion_clip_end = DoneTerm(
        func=mdp.motion_clip_end,
        time_out=True,
        params={"command_name": "motion"},
    )
    # The following two terms partition completed clips for deterministic
    # evaluation. Both remain timeouts so the final standing classification
    # cannot alter the training termination semantics established above.
    motion_end_success = DoneTerm(
        func=mdp.motion_end_success,
        time_out=True,
        params={
            "command_name": "motion",
            "platform_cfg": SceneEntityCfg("platform"),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["l_ankle_x_link", "r_ankle_x_link"],
            ),
            "base_size": ELF3_CLIMB_PLATFORM_SIZE,
            "foot_body_names": ["l_ankle_x_link", "r_ankle_x_link"],
            "footprint_inset": ELF3_CLIMB_FOOTPRINT_INSET,
            "foot_height_range": ELF3_CLIMB_FOOT_HEIGHT_RANGE,
            "min_foot_contact_force": ELF3_CLIMB_MIN_FOOT_CONTACT_FORCE_N,
            "min_foot_contact_time": ELF3_CLIMB_MIN_FOOT_CONTACT_TIME_S,
            "max_root_height_error": ELF3_CLIMB_MAX_ROOT_HEIGHT_ERROR,
            "max_root_linear_speed": ELF3_CLIMB_MAX_ROOT_LINEAR_SPEED,
            "max_root_angular_speed": ELF3_CLIMB_MAX_ROOT_ANGULAR_SPEED,
            "max_joint_speed": ELF3_CLIMB_MAX_JOINT_SPEED,
            "max_torso_tilt": ELF3_CLIMB_MAX_TORSO_TILT,
            "min_stable_time": ELF3_CLIMB_MIN_STABLE_TIME_S,
            "platform_support_params": ELF3_CLIMB_PLATFORM_FOOT_SUPPORT_PARAMS,
            "sole_height_tolerance": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_SOLE_HEIGHT_TOLERANCE,
            "min_total_load_fraction": ELF3_CLIMB_TERMINAL_DEFAULT_POSE_MIN_TOTAL_LOAD_FRACTION,
            "max_expert_joint_pos_rms": ELF3_CLIMB_MAX_EXPERT_JOINT_POS_RMS,
        },
    )
    motion_end_failure = DoneTerm(
        func=mdp.motion_end_failure,
        time_out=True,
        params={
            "command_name": "motion",
            "success_term_name": "motion_end_success",
        },
    )


@configclass
class ELF3ClimbCurriculumCfg:
    """Performance-gated terrain randomization curriculum."""

    platform_pose = CurrTerm(
        func=mdp.climb_box_pose_curriculum,
        params={
            "event_term_name": "platform_pose",
            "success_term_name": "motion_end_success",
            "command_name": "motion",
            "full_position_range": {
                "x": ELF3_CLIMB_PLATFORM_X_OFFSET_RANGE,
                "y": ELF3_CLIMB_PLATFORM_Y_OFFSET_RANGE,
            },
            "full_yaw_range": ELF3_CLIMB_PLATFORM_YAW_RANGE,
            "stage_scales": ELF3_CLIMB_TERRAIN_CURRICULUM_STAGE_SCALES,
            "advance_success_rate": ELF3_CLIMB_TERRAIN_CURRICULUM_ADVANCE_SUCCESS_RATE,
            "regress_success_rate": ELF3_CLIMB_TERRAIN_CURRICULUM_REGRESS_SUCCESS_RATE,
            "min_evaluated_episodes": ELF3_CLIMB_TERRAIN_CURRICULUM_MIN_EPISODES,
        },
    )


##
# Environment configuration
##


@configclass
class ELF3ClimbEnvCfg(TrackingEnvCfg):
    """Complete ELF3 climb expert configuration, directly inheriting the tracking base."""

    # Scene and environment parallelism.
    scene: ELF3ClimbSceneCfg = ELF3ClimbSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
        replicate_physics=False,
    )

    # Actor/critic observations, actions, and reference-motion commands.
    observations: ELF3ClimbObservationsCfg = ELF3ClimbObservationsCfg()
    actions: ELF3ClimbActionsCfg = ELF3ClimbActionsCfg()
    commands: ELF3ClimbCommandsCfg = ELF3ClimbCommandsCfg()

    # Rewards, terminations, randomization events, and curriculum.
    rewards: ELF3ClimbRewardsCfg = ELF3ClimbRewardsCfg()
    terminations: ELF3ClimbTerminationsCfg = ELF3ClimbTerminationsCfg()
    events: ELF3ClimbEventCfg = ELF3ClimbEventCfg()
    curriculum: ELF3ClimbCurriculumCfg = ELF3ClimbCurriculumCfg()

    def __post_init__(self):
        """Set the control rate, simulation timing, physics capacity, and viewer."""

        # Keep the direct base initialization for compatibility, then restate
        # every inherited value explicitly below so this file is self-contained.
        super().__post_init__()

        # General environment settings: 0.005 s simulation step with four-step
        # action decimation gives a 0.02 s / 50 Hz policy control period.
        self.decimation = 4
        self.episode_length_s = 10.0

        # Simulation settings.
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Refresh the height map once per 50 Hz policy step.
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # Viewer settings.
        self.viewer.eye = (1.5, 1.5, 1.5)
        # self.viewer.origin_type = "asset_root"
        self.viewer.origin_type = "world"
        # self.viewer.asset_name = "robot"
        self.viewer.asset_name = None
