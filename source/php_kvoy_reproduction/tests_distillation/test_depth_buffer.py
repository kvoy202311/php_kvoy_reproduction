from __future__ import annotations

import torch

from php_kvoy_reproduction.tasks.distillation.mdp.depth_buffer import (
    PeriodicCaptureSchedule,
    TimestampedDepthBuffer,
    preprocess_depth_image,
)


def test_depth_preprocessing_maps_invalid_to_far_then_normalizes() -> None:
    raw = torch.tensor([[[float("nan"), float("inf"), float("-inf"), 0.10, 0.15, 2.5]]])
    result = preprocess_depth_image(raw, near_clip=0.15, far_clip=2.0)
    torch.testing.assert_close(result, torch.tensor([[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]))


def test_depth_noise_is_metric_and_clamped_before_normalization() -> None:
    raw = torch.ones(2, 1, 1)
    offset = torch.tensor([0.1, -0.1])
    noise = torch.tensor([[[0.05]], [[-0.05]]])
    result = preprocess_depth_image(raw, image_offset=offset, pixel_noise=noise)
    expected_metric = torch.tensor([1.15, 0.85])
    torch.testing.assert_close(result[:, 0, 0], (expected_metric - 0.15) / 1.85)


def test_depth_preprocessing_is_always_finite_and_bounded() -> None:
    raw = torch.tensor(
        [
            [
                [float("nan"), float("inf"), float("-inf"), -10.0],
                [0.14, 0.15, 1.0, 100.0],
            ]
        ]
    )
    offset = torch.tensor([0.03])
    noise = torch.tensor([[[0.0, -100.0, 100.0, 0.0], [0.1, -0.1, 0.2, -0.2]]])
    result = preprocess_depth_image(raw, image_offset=offset, pixel_noise=noise)
    assert torch.isfinite(result).all()
    assert torch.all((result >= 0.0) & (result <= 1.0))


def test_timestamp_buffer_never_reuses_previous_episode_frame() -> None:
    buffer = TimestampedDepthBuffer(1, 1, 1, 4, device="cpu")
    buffer.append(torch.tensor([[[0.2]]]), torch.tensor([0.0], dtype=torch.float64))
    torch.testing.assert_close(
        buffer.select(0.10, torch.tensor([0.08]))[0, 0, 0],
        torch.tensor(0.2),
    )

    buffer.reset(torch.tensor([0]))
    # Before the reset render, the explicit far image is returned.
    assert buffer.select(0.11, torch.tensor([0.08]))[0, 0, 0].item() == 1.0
    buffer.append(torch.tensor([[[0.7]]]), torch.tensor([0.12], dtype=torch.float64))
    # Warm-up repeats the first *new* frame even though it is not old enough.
    torch.testing.assert_close(
        buffer.select(0.13, torch.tensor([0.08]))[0, 0, 0],
        torch.tensor(0.7),
    )


def test_timestamp_buffer_selects_newest_capture_not_newer_than_delay() -> None:
    buffer = TimestampedDepthBuffer(1, 1, 1, 5, device="cpu")
    for timestamp, value in ((0.00, 0.1), (0.04, 0.2), (0.08, 0.3), (0.12, 0.4)):
        buffer.append(
            torch.tensor([[[value]]]),
            torch.tensor([timestamp], dtype=torch.float64),
        )
    selected = buffer.select(0.15, torch.tensor([0.07]))
    assert selected[0, 0, 0].item() == torch.tensor(0.3).item()


def test_partial_reset_clears_only_reset_environment_history() -> None:
    buffer = TimestampedDepthBuffer(2, 1, 1, 4, device="cpu")
    buffer.append(
        torch.tensor([[[0.2]], [[0.8]]]),
        torch.tensor([0.0, 0.0], dtype=torch.float64),
    )
    buffer.reset(torch.tensor([0]))

    selected = buffer.select(0.10, torch.tensor([0.08, 0.08]))
    assert selected[0, 0, 0].item() == 1.0
    torch.testing.assert_close(selected[1, 0, 0], torch.tensor(0.8))

    buffer.append(
        torch.tensor([[[0.4]]]),
        torch.tensor([0.12], dtype=torch.float64),
        torch.tensor([0]),
    )
    selected = buffer.select(0.13, torch.tensor([0.08, 0.08]))
    torch.testing.assert_close(selected[0, 0, 0], torch.tensor(0.4))
    torch.testing.assert_close(selected[1, 0, 0], torch.tensor(0.8))


def test_30hz_capture_schedule_on_50hz_grid_has_no_frequency_drift() -> None:
    schedule = PeriodicCaptureSchedule(1, 30.0, device="cpu")
    capture_times = []
    for step in range(51):
        now = step * 0.02
        if schedule.pop_due(now).numel():
            capture_times.append(now)

    # Immediate reset frame plus exactly 30 new captures over one second.
    assert len(capture_times) == 31
    assert capture_times[0] == 0.0
    assert capture_times[-1] == 1.0
    intervals = torch.diff(torch.tensor(capture_times, dtype=torch.float64))
    assert set(torch.round(intervals * 100).to(torch.int64).tolist()) == {2, 4}


def test_capture_schedule_partial_reset_restarts_only_selected_phase() -> None:
    schedule = PeriodicCaptureSchedule(2, 30.0, device="cpu")
    torch.testing.assert_close(schedule.pop_due(0.0), torch.tensor([0, 1]))
    assert schedule.pop_due(0.02).numel() == 0

    schedule.reset(torch.tensor([0]), now=0.02)
    torch.testing.assert_close(schedule.pop_due(0.02), torch.tensor([0]))
    torch.testing.assert_close(schedule.pop_due(0.04), torch.tensor([1]))
