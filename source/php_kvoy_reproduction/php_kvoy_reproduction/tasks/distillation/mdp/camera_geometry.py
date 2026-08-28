"""Deployment-aligned ELF3 head D435i mounting geometry."""

from __future__ import annotations

import math
from typing import Final


# These transforms are copied from the unmodified ELF3 URDF chain:
# torso_link -> head_z_link -> head_y_link -> d435i_link.  The USD asset was
# converted with merge_fixed_joints=True, so d435i_link is not a runtime rigid
# body.  The simulated sensor must therefore use the equivalent torso-relative
# transform rather than modifying the URDF or USD robot asset.
ELF3_HEAD_PITCH_LIMIT_RAD: Final[tuple[float, float]] = (-0.785, 0.785)
ELF3_HEAD_D435I_FIXED_PITCH_RAD: Final[float] = math.radians(42.0)

# Reference D435i depth profile used by the verified deployment stack's torso
# camera.  Resolution, frame rate and fy below come from that stack's selected
# SDK profile and measured intrinsics.  The head sensor uses the same hardware
# profile; only its kinematic mounting pose differs.  The student consumes a
# lower-resolution image with this field of view, not a different physical
# camera mode.
ELF3_D435I_REFERENCE_DEPTH_WIDTH: Final[int] = 848
ELF3_D435I_REFERENCE_DEPTH_HEIGHT: Final[int] = 480
ELF3_D435I_REFERENCE_DEPTH_FPS: Final[float] = 30.0
ELF3_D435I_REFERENCE_DEPTH_FY_PX: Final[float] = 427.622711
ELF3_D435I_REFERENCE_DEPTH_VERTICAL_FOV_RAD: Final[float] = 2.0 * math.atan(
    ELF3_D435I_REFERENCE_DEPTH_HEIGHT / (2.0 * ELF3_D435I_REFERENCE_DEPTH_FY_PX)
)
# This is the operating range configured by the verified deployment simulator;
# it is not asserted to be an absolute physical range limit of every D435i.
ELF3_D435I_DEPLOYMENT_DEPTH_RANGE_M: Final[tuple[float, float]] = (0.1, 10.0)

_HEAD_PITCH_PIVOT_IN_TORSO: Final[tuple[float, float, float]] = (0.0, 0.0, 0.2995)
_D435I_ORIGIN_IN_HEAD_PITCH: Final[tuple[float, float, float]] = (0.0628, 0.0175, -0.048)


def elf3_head_d435i_camera_pose(
    head_pitch_rad: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return the torso-relative D435i pose for a fixed head pitch.

    Positive pitch rotates the head camera downward.  The returned quaternion
    is expressed in Isaac Lab's ``world`` camera convention: +X looks forward
    and +Z points upward.  The URDF's fixed d435i roll/yaw rotates into the ROS
    optical basis (+Z forward, +Y down); selecting the world camera convention
    performs that basis conversion without duplicating it in the physical
    mounting transform.
    """

    if isinstance(head_pitch_rad, bool) or not isinstance(head_pitch_rad, (int, float)):
        raise TypeError("head_pitch_rad must be a finite real number")
    pitch = float(head_pitch_rad)
    if not math.isfinite(pitch):
        raise ValueError("head_pitch_rad must be finite")
    lower, upper = ELF3_HEAD_PITCH_LIMIT_RAD
    if pitch < lower or pitch > upper:
        raise ValueError(
            f"head_pitch_rad={pitch} lies outside the ELF3 URDF limit [{lower}, {upper}]"
        )

    pivot_x, pivot_y, pivot_z = _HEAD_PITCH_PIVOT_IN_TORSO
    sensor_x, sensor_y, sensor_z = _D435I_ORIGIN_IN_HEAD_PITCH
    cosine = math.cos(pitch)
    sine = math.sin(pitch)
    position = (
        pivot_x + cosine * sensor_x + sine * sensor_z,
        pivot_y + sensor_y,
        pivot_z - sine * sensor_x + cosine * sensor_z,
    )
    orientation = (math.cos(0.5 * pitch), 0.0, math.sin(0.5 * pitch), 0.0)
    return position, orientation


ELF3_HEAD_D435I_CAMERA_POS, ELF3_HEAD_D435I_CAMERA_ROT = elf3_head_d435i_camera_pose(
    ELF3_HEAD_D435I_FIXED_PITCH_RAD
)


__all__ = [
    "ELF3_D435I_REFERENCE_DEPTH_FPS",
    "ELF3_D435I_REFERENCE_DEPTH_FY_PX",
    "ELF3_D435I_REFERENCE_DEPTH_HEIGHT",
    "ELF3_D435I_REFERENCE_DEPTH_VERTICAL_FOV_RAD",
    "ELF3_D435I_REFERENCE_DEPTH_WIDTH",
    "ELF3_D435I_DEPLOYMENT_DEPTH_RANGE_M",
    "ELF3_HEAD_D435I_CAMERA_POS",
    "ELF3_HEAD_D435I_CAMERA_ROT",
    "ELF3_HEAD_D435I_FIXED_PITCH_RAD",
    "ELF3_HEAD_PITCH_LIMIT_RAD",
    "elf3_head_d435i_camera_pose",
]
