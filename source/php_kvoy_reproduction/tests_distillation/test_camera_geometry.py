import math

import pytest

from php_kvoy_reproduction.tasks.distillation.mdp.camera_geometry import (
    ELF3_D435I_DEPLOYMENT_DEPTH_RANGE_M,
    ELF3_D435I_REFERENCE_DEPTH_FPS,
    ELF3_D435I_REFERENCE_DEPTH_FY_PX,
    ELF3_D435I_REFERENCE_DEPTH_HEIGHT,
    ELF3_D435I_REFERENCE_DEPTH_VERTICAL_FOV_RAD,
    ELF3_D435I_REFERENCE_DEPTH_WIDTH,
    ELF3_HEAD_D435I_CAMERA_POS,
    ELF3_HEAD_D435I_CAMERA_ROT,
    ELF3_HEAD_D435I_FIXED_PITCH_RAD,
    ELF3_HEAD_PITCH_LIMIT_RAD,
    elf3_head_d435i_camera_pose,
)


def test_reference_depth_profile_matches_verified_deployment_camera() -> None:
    assert (ELF3_D435I_REFERENCE_DEPTH_WIDTH, ELF3_D435I_REFERENCE_DEPTH_HEIGHT) == (
        848,
        480,
    )
    assert ELF3_D435I_REFERENCE_DEPTH_FPS == pytest.approx(30.0)
    assert ELF3_D435I_REFERENCE_DEPTH_FY_PX == pytest.approx(427.622711)
    assert ELF3_D435I_DEPLOYMENT_DEPTH_RANGE_M == pytest.approx((0.1, 10.0))
    expected_fov = 2.0 * math.atan(
        ELF3_D435I_REFERENCE_DEPTH_HEIGHT / (2.0 * ELF3_D435I_REFERENCE_DEPTH_FY_PX)
    )
    assert ELF3_D435I_REFERENCE_DEPTH_VERTICAL_FOV_RAD == pytest.approx(expected_fov)
    assert math.degrees(expected_fov) == pytest.approx(58.60597640272646)


def test_neutral_pose_matches_unmodified_urdf_chain() -> None:
    position, orientation = elf3_head_d435i_camera_pose(0.0)
    assert position == pytest.approx((0.0628, 0.0175, 0.2515), abs=1.0e-12)
    assert orientation == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1.0e-12)


def test_deployment_pitch_is_physical_and_has_required_vertical_coverage() -> None:
    lower, upper = ELF3_HEAD_PITCH_LIMIT_RAD
    assert lower < ELF3_HEAD_D435I_FIXED_PITCH_RAD < upper
    assert math.degrees(upper - ELF3_HEAD_D435I_FIXED_PITCH_RAD) > 2.5

    # The measured D435i vertical field of view produces a nominal
    # 12.697--71.303 degree downward view at the selected 42-degree head pitch.
    half_vertical_fov = 0.5 * ELF3_D435I_REFERENCE_DEPTH_VERTICAL_FOV_RAD
    assert math.degrees(ELF3_HEAD_D435I_FIXED_PITCH_RAD - half_vertical_fov) == pytest.approx(
        12.69701179863677
    )
    assert math.degrees(ELF3_HEAD_D435I_FIXED_PITCH_RAD + half_vertical_fov) == pytest.approx(
        71.30298820136323
    )

    position, orientation = elf3_head_d435i_camera_pose(ELF3_HEAD_D435I_FIXED_PITCH_RAD)
    assert position == pytest.approx(
        (0.014551225934755163, 0.0175, 0.22180764629774877), abs=1.0e-12
    )
    assert orientation == pytest.approx(
        (0.9335804264972017, 0.0, 0.35836794954530027, 0.0), abs=1.0e-12
    )
    assert position == pytest.approx(ELF3_HEAD_D435I_CAMERA_POS, abs=1.0e-12)
    assert orientation == pytest.approx(ELF3_HEAD_D435I_CAMERA_ROT, abs=1.0e-12)
    assert sum(value * value for value in orientation) == pytest.approx(1.0, abs=1.0e-12)


@pytest.mark.parametrize("value", [math.inf, -math.inf, math.nan])
def test_camera_pose_rejects_non_finite_pitch(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        elf3_head_d435i_camera_pose(value)


@pytest.mark.parametrize("value", [-0.786, 0.786])
def test_camera_pose_rejects_pitch_outside_urdf_limits(value: float) -> None:
    with pytest.raises(ValueError, match="outside"):
        elf3_head_d435i_camera_pose(value)
