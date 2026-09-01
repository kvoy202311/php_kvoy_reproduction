from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from php_kvoy_reproduction.distillation.motion_boundary import (
    detect_motion_execution_starts,
    motion_reference_advance_mask,
)

_REPOSITORY = Path(__file__).resolve().parents[3]


def _arrays(frame_count: int = 8) -> dict[str, torch.Tensor]:
    body_quat = torch.zeros(frame_count, 2, 4)
    body_quat[..., 0] = 1.0
    return {
        "joint_pos": torch.zeros(frame_count, 3),
        "joint_vel": torch.zeros(frame_count, 3),
        "body_pos_w": torch.zeros(frame_count, 2, 3),
        "body_quat_w": body_quat,
        "body_lin_vel_w": torch.zeros(frame_count, 2, 3),
        "body_ang_vel_w": torch.zeros(frame_count, 2, 3),
    }


def test_detects_first_active_frame_after_static_prefix() -> None:
    arrays = _arrays()
    arrays["joint_vel"][3, 1] = 0.02
    arrays["joint_pos"][4, 1] = 0.001
    starts = detect_motion_execution_starts(
        arrays,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([8], dtype=torch.long),
    )
    assert starts.tolist() == [3]


def test_ignores_conversion_noise_and_quaternion_sign() -> None:
    arrays = _arrays()
    arrays["joint_vel"][1:3] = 1.0e-8
    arrays["body_quat_w"][2] *= -1.0
    arrays["body_ang_vel_w"][5, 0, 2] = 1.0e-4
    starts = detect_motion_execution_starts(
        arrays,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([8], dtype=torch.long),
    )
    assert starts.tolist() == [5]


def test_detects_each_clip_in_concatenated_dataset() -> None:
    arrays = _arrays(frame_count=12)
    arrays["body_lin_vel_w"][2, 0, 0] = 0.1
    arrays["joint_vel"][9, 2] = 0.2
    starts = detect_motion_execution_starts(
        arrays,
        torch.tensor([0, 6], dtype=torch.long),
        torch.tensor([6, 12], dtype=torch.long),
    )
    assert starts.tolist() == [2, 9]


def test_keeps_raw_start_when_clip_is_already_active() -> None:
    arrays = _arrays()
    arrays["joint_vel"][0, 0] = 0.1
    starts = detect_motion_execution_starts(
        arrays,
        torch.tensor([0], dtype=torch.long),
        torch.tensor([8], dtype=torch.long),
    )
    assert starts.tolist() == [0]


def test_rejects_completely_static_clip() -> None:
    with pytest.raises(ValueError, match="contains no physical activity"):
        detect_motion_execution_starts(
            _arrays(),
            torch.tensor([0], dtype=torch.long),
            torch.tensor([8], dtype=torch.long),
        )


def test_motion_reference_advances_only_under_the_matching_applied_head() -> None:
    motion = torch.tensor([True, True, True, False])
    finished = torch.tensor([False, False, False, False])
    reference = torch.tensor([1, 1, 2, 0], dtype=torch.long)
    active_student = torch.tensor([1, 0, 2, 0], dtype=torch.long)
    reset = torch.tensor([False, False, True, False])

    advance = motion_reference_advance_mask(
        motion,
        finished,
        reference,
        active_student,
        reset,
    )

    # Matching climb advances.  Option-confirming locomotion pauses climb,
    # and a just-reset down-roll cannot consume its first active frame.
    assert advance.tolist() == [True, False, False, False]


def test_finished_motion_reference_never_advances() -> None:
    advance = motion_reference_advance_mask(
        torch.tensor([True]),
        torch.tensor([True]),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([1], dtype=torch.long),
        torch.tensor([False]),
    )
    assert advance.tolist() == [False]


def test_motion_reference_advance_mask_rejects_non_integer_routes() -> None:
    with pytest.raises(TypeError, match="must use an integer dtype"):
        motion_reference_advance_mask(
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([1.0]),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([False]),
        )


@pytest.mark.parametrize(
    "relative_directory",
    (
        "data/processed_motions/elf3/climb_50hz_default_start_v1",
        "data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1",
    ),
)
def test_elf3_distillation_clips_share_verified_frame_23_boundary(
    relative_directory: str,
) -> None:
    motion_files = sorted((_REPOSITORY / relative_directory).glob("*.npz"))
    assert len(motion_files) == 4
    for motion_file in motion_files:
        with np.load(motion_file, allow_pickle=False) as data:
            arrays = {
                name: torch.from_numpy(np.asarray(data[name]))
                for name in (
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                )
            }
        frame_count = arrays["joint_pos"].shape[0]
        execution_start = detect_motion_execution_starts(
            arrays,
            torch.tensor([0], dtype=torch.long),
            torch.tensor([frame_count], dtype=torch.long),
        )
        assert execution_start.item() == 23, motion_file.name
        assert torch.max(torch.abs(arrays["joint_pos"][23] - arrays["joint_pos"][0])) <= 1.0e-6
        assert torch.max(torch.abs(arrays["body_pos_w"][23] - arrays["body_pos_w"][0])) <= 1.0e-6
        assert torch.max(torch.abs(arrays["body_quat_w"][23] - arrays["body_quat_w"][0])) <= 1.0e-6
        assert torch.max(torch.abs(arrays["joint_vel"][:23])) <= 1.0e-6
        assert torch.max(torch.abs(arrays["body_lin_vel_w"][:23])) <= 1.0e-6
        assert torch.max(torch.abs(arrays["body_ang_vel_w"][:23])) <= 1.0e-6
