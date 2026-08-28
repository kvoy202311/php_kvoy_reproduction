from __future__ import annotations

import math

import pytest
import torch

from php_kvoy_reproduction.distillation.skill_routing import (
    NUM_SKILLS,
    approach_transition_status,
    balanced_skill_ids,
    climb_geometry_progress_score,
    climb_settle_geometry_ready,
    down_roll_geometry_progress_score,
    down_roll_settle_geometry_ready,
    down_roll_transition_ready,
    filtered_contact_upward_forces,
    lower_ground_contact_support_score,
    motion_boundary_alignment_ready,
    planar_command_speed_valid,
    platform_height_teacher_confidence,
    platform_reference_center_offsets,
)


def test_approach_transition_accepts_target_crossing_inside_lateral_corridor() -> None:
    ready, failed = approach_transition_status(
        longitudinal_error=torch.tensor([0.10, 0.05, -0.10, -0.25, 0.02]),
        lateral_error=torch.tensor([0.00, 0.10, 0.15, 0.00, 0.25]),
        elapsed_time=torch.tensor([1.0, 1.0, 1.0, 1.0, 4.0]),
        switch_distance=0.08,
        lateral_tolerance=0.20,
        maximum_overshoot=0.20,
        timeout=3.0,
    )
    assert ready.tolist() == [False, True, True, False, False]
    assert failed.tolist() == [False, False, False, True, True]


def test_planar_command_speed_contract_uses_vector_magnitude() -> None:
    valid = planar_command_speed_valid(
        torch.tensor([[0.0, 0.0], [0.6, 0.8], [1.01, 0.0], [0.8, 0.8]]),
        maximum_speed=1.0,
    )
    assert valid.tolist() == [True, True, False, False]


def test_motion_boundary_alignment_requires_pose_speed_and_upright_state() -> None:
    ready = motion_boundary_alignment_ready(
        joint_position_rms=torch.tensor([0.10, 0.40, 0.10, 0.10]),
        joint_speed_rms=torch.tensor([0.20, 0.20, 1.20, 0.20]),
        gravity_xy_norm=torch.tensor([0.05, 0.05, 0.05, 0.50]),
        maximum_joint_position_rms=0.35,
        maximum_joint_speed_rms=1.0,
        maximum_gravity_xy_norm=0.35,
    )
    assert ready.tolist() == [True, False, False, False]


def test_climb_settle_rechecks_position_and_heading_after_waiting() -> None:
    ready = climb_settle_geometry_ready(
        longitudinal_error=torch.tensor([0.05, 0.09, -0.21, 0.05, 0.05]),
        lateral_error=torch.tensor([0.0, 0.0, 0.0, 0.21, 0.0]),
        heading_error=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.36]),
        switch_distance=0.08,
        maximum_overshoot=0.20,
        lateral_tolerance=0.20,
        maximum_heading_error=0.35,
    )
    assert ready.tolist() == [True, False, False, False, False]


def test_down_roll_settle_rechecks_edge_without_requiring_forward_speed() -> None:
    ready = down_roll_settle_geometry_ready(
        forward_edge_distance=torch.tensor([0.40, 0.49, -0.06, 0.40, 0.40]),
        lateral_offset=torch.tensor([0.0, 0.0, 0.0, 0.33, 0.0]),
        half_width=torch.full((5,), 0.40),
        heading_error=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.36]),
        minimum_edge_distance=-0.05,
        maximum_edge_distance=0.48,
        lateral_margin=0.08,
        maximum_heading_error=0.35,
    )
    assert ready.tolist() == [True, False, False, False, False]


def test_balanced_stream_remains_exact_across_uneven_reset_batches() -> None:
    cursor = 0
    pieces = []
    generator = torch.Generator().manual_seed(7)
    for count in (4, 1, 8, 2, 6):
        routes, cursor = balanced_skill_ids(
            count,
            cursor=cursor,
            device="cpu",
            generator=generator,
        )
        assert routes.shape == (count,)
        assert routes.dtype == torch.long
        assert torch.all((routes >= 0) & (routes < NUM_SKILLS))
        pieces.append(routes)

    stream = torch.cat(pieces)
    counts = torch.bincount(stream, minlength=NUM_SKILLS)
    assert counts.max().item() - counts.min().item() <= 1
    assert counts.tolist() == [7, 7, 7]
    assert cursor == 0


def test_balanced_stream_cursor_tracks_zero_and_partial_batches() -> None:
    empty, cursor = balanced_skill_ids(0, cursor=2, device="cpu")
    assert empty.numel() == 0
    assert cursor == 2

    routes, cursor = balanced_skill_ids(2, cursor=2, device="cpu")
    assert sorted(routes.tolist()) == [0, 2]
    assert cursor == 1


@pytest.mark.parametrize(
    ("count", "cursor"),
    [(-1, 0), (True, 0), (1, -1), (1, NUM_SKILLS), (1, True)],
)
def test_balanced_stream_rejects_invalid_contract(count, cursor) -> None:
    with pytest.raises(ValueError):
        balanced_skill_ids(count, cursor=cursor, device="cpu")


def test_platform_height_teacher_confidence_has_smooth_verified_scope() -> None:
    heights = torch.tensor([0.60, 0.62, 0.64, 0.655, 0.66, 0.665, 0.68, 0.70, 0.72])
    confidence = platform_height_teacher_confidence(
        heights,
        nominal_height=0.66,
        full_tolerance=0.005,
        zero_tolerance=0.04,
    )
    assert confidence[0].item() == 0.0
    assert confidence[-1].item() == 0.0
    assert confidence[4].item() == 1.0
    assert confidence[3].item() == pytest.approx(1.0)
    assert confidence[5].item() == pytest.approx(1.0)
    torch.testing.assert_close(confidence[2], confidence[6])
    torch.testing.assert_close(confidence[1], confidence[7])
    assert torch.all((confidence >= 0.0) & (confidence <= 1.0))


@pytest.mark.parametrize(
    ("nominal", "full", "zero"),
    [(float("nan"), 0.005, 0.04), (0.66, float("nan"), 0.04), (0.66, 0.005, float("inf"))],
)
def test_platform_height_teacher_confidence_rejects_non_finite_configuration(
    nominal: float,
    full: float,
    zero: float,
) -> None:
    with pytest.raises(ValueError):
        platform_height_teacher_confidence(
            torch.tensor([0.66]),
            nominal_height=nominal,
            full_tolerance=full,
            zero_tolerance=zero,
        )


def test_geometry_progress_requires_physical_completion_for_majority_score() -> None:
    one = torch.ones(1)
    zero = torch.zeros(1)
    # Reaching the platform height without foot support cannot dominate climb.
    assert climb_geometry_progress_score(one, one, zero).item() == pytest.approx(0.4)
    # Crossing and descending without upright landing cannot dominate down-roll.
    assert down_roll_geometry_progress_score(one, one, zero).item() == pytest.approx(0.4)
    assert climb_geometry_progress_score(one, one, one).item() == pytest.approx(1.0)
    assert down_roll_geometry_progress_score(one, one, one).item() == pytest.approx(1.0)


def test_lower_ground_support_requires_both_surface_proximity_and_real_force() -> None:
    support = lower_ground_contact_support_score(
        sole_height_errors=torch.tensor(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.60, 0.60],
                [0.01, 0.20],
            ]
        ),
        upward_forces=torch.tensor(
            [
                [0.0, 0.0],
                [19.9, 19.9],
                [200.0, 200.0],
                [200.0, 0.0],
            ]
        ),
        height_tolerance=0.04,
        minimum_upward_force=20.0,
    )
    assert support[0].item() == 0.0
    assert support[1].item() == 0.0
    assert support[2].item() == pytest.approx(0.0, abs=1.0e-12)
    assert support[3].item() > 0.9


def test_filtered_ground_forces_preserve_independent_one_foot_contacts() -> None:
    left = torch.zeros(2, 1, 1, 3)
    right = torch.zeros(2, 1, 1, 3)
    left[0, 0, 0, 2] = 120.0
    left[1, 0, 0, 2] = -30.0
    right[1, 0, 0, 2] = 80.0

    upward = filtered_contact_upward_forces([left, right], num_envs=2)
    torch.testing.assert_close(
        upward,
        torch.tensor([[120.0, 0.0], [0.0, 80.0]]),
    )

    # The first foot may land on lower ground while the other remains at the
    # platform height; no environment-level contact exclusivity is imposed.
    support = lower_ground_contact_support_score(
        sole_height_errors=torch.tensor([[0.0, 0.66]]),
        upward_forces=upward[:1],
        height_tolerance=0.04,
        minimum_upward_force=20.0,
    )
    assert support.item() > 0.9


@pytest.mark.parametrize(
    "force_matrices",
    [
        [],
        [torch.zeros(2, 1, 3)],
        [torch.zeros(2, 1, 2, 3)],
        [torch.zeros(2, 1, 1, 2)],
        [torch.zeros(2, 1, 1, 3, dtype=torch.int64)],
    ],
)
def test_filtered_ground_forces_reject_non_filtered_or_invalid_tensors(force_matrices) -> None:
    with pytest.raises(ValueError):
        filtered_contact_upward_forces(force_matrices, num_envs=2)


@pytest.mark.parametrize(
    ("errors", "forces"),
    [
        (torch.zeros(2), torch.zeros(2)),
        (torch.zeros(1, 2), torch.zeros(1, 3)),
        (torch.tensor([[float("nan")]]), torch.ones(1, 1)),
        (torch.zeros(1, 1), torch.tensor([[-1.0]])),
    ],
)
def test_lower_ground_support_rejects_invalid_inputs(errors, forces) -> None:
    with pytest.raises(ValueError):
        lower_ground_contact_support_score(
            errors,
            forces,
            height_tolerance=0.04,
            minimum_upward_force=20.0,
        )


def test_reference_centers_keep_climb_entry_and_down_roll_exit_edges_aligned() -> None:
    lengths = torch.tensor([0.51, 1.01, 3.0])
    climb_offset, down_offset = platform_reference_center_offsets(
        lengths,
        nominal_length=0.51,
    )
    physical_center = lengths / 2.0
    nominal_half = 0.51 / 2.0
    # Entry edge is fixed at zero for every sampled length.
    torch.testing.assert_close(physical_center + climb_offset - nominal_half, torch.zeros(3))
    # The canonical exit edge maps to the sampled far edge.
    torch.testing.assert_close(
        physical_center + down_offset + nominal_half,
        lengths,
    )


def test_down_roll_transition_requires_visible_edge_motion_alignment_and_upright_state() -> None:
    ready = down_roll_transition_ready(
        forward_edge_distance=torch.tensor([0.40, 0.70, 0.40, 0.40, 0.40, 0.40]),
        lateral_offset=torch.tensor([0.0, 0.0, 0.40, 0.0, 0.0, 0.0]),
        half_width=torch.full((6,), 0.40),
        command_forward_speed=torch.tensor([0.6, 0.6, 0.6, 0.0, 0.6, 0.6]),
        command_heading_error=torch.zeros(6),
        body_heading_error=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.0]),
        gravity_xy_norm=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
        minimum_edge_distance=-0.05,
        maximum_edge_distance=0.48,
        lateral_margin=0.08,
        minimum_forward_speed=0.1,
        maximum_heading_error=0.35,
        maximum_gravity_xy_norm=0.35,
    )
    assert ready.tolist() == [True, False, False, False, False, False]


def test_down_roll_transition_rejects_lateral_request_with_small_forward_projection() -> None:
    ready = down_roll_transition_ready(
        forward_edge_distance=torch.tensor([0.40]),
        lateral_offset=torch.tensor([0.0]),
        half_width=torch.tensor([0.40]),
        command_forward_speed=torch.tensor([0.10]),
        command_heading_error=torch.tensor([math.atan2(1.0, 0.1)]),
        body_heading_error=torch.tensor([0.0]),
        gravity_xy_norm=torch.tensor([0.0]),
        minimum_edge_distance=-0.05,
        maximum_edge_distance=0.48,
        lateral_margin=0.08,
        minimum_forward_speed=0.1,
        maximum_heading_error=0.35,
        maximum_gravity_xy_norm=0.35,
    )
    assert ready.tolist() == [False]
