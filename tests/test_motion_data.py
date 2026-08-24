from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


_MODULE_PATH = (
    Path(__file__).parents[1] / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/motion_data.py"
)
_SPEC = importlib.util.spec_from_file_location("php_kvoy_reproduction_motion_data", _MODULE_PATH)
motion_data = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(motion_data)


def _write_motion(path: Path, frame_count: int, fps: float = 50.0, joint_count: int = 2, body_count: int = 3):
    joint_pos = np.arange(frame_count * joint_count, dtype=np.float32).reshape(frame_count, joint_count)
    body_pos = np.zeros((frame_count, body_count, 3), dtype=np.float32)
    body_quat = np.zeros((frame_count, body_count, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    np.savez(
        path,
        fps=np.asarray(fps, dtype=np.float32),
        joint_pos=joint_pos,
        joint_vel=np.zeros_like(joint_pos),
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=np.zeros_like(body_pos),
        body_ang_vel_w=np.zeros_like(body_pos),
        joint_names=np.asarray([f"joint_{index}" for index in range(joint_count)]),
        body_names=np.asarray([f"body_{index}" for index in range(body_count)]),
    )


class MultiMotionLoaderTest(unittest.TestCase):
    def test_single_file_source_keeps_one_motion_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            motion_file = Path(temp_dir) / "single.npz"
            _write_motion(motion_file, frame_count=6)

            loader = motion_data.load_motion_dataset(
                motion_file=motion_file,
                motion_dir=None,
                body_indexes=[1],
            )

            self.assertEqual(loader.num_motions, 1)
            self.assertEqual(loader.motion_start_idx.tolist(), [0])
            self.assertEqual(loader.motion_end_idx.tolist(), [6])

    def test_concatenates_sorted_clips_and_preserves_boundaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            motion_dir = Path(temp_dir)
            _write_motion(motion_dir / "b.npz", frame_count=3)
            _write_motion(motion_dir / "a.npz", frame_count=5)

            loader = motion_data.MultiMotionLoader(motion_dir, body_indexes=[0, 2])

            self.assertEqual(loader.num_motions, 2)
            self.assertEqual(loader.motion_lengths.tolist(), [5, 3])
            self.assertEqual(loader.motion_start_idx.tolist(), [0, 5])
            self.assertEqual(loader.motion_end_idx.tolist(), [5, 8])
            self.assertEqual(tuple(loader.joint_pos.shape), (8, 2))
            self.assertEqual(tuple(loader.body_pos_w.shape), (8, 2, 3))
            self.assertTrue(loader.motion_files[0].endswith("a.npz"))
            self.assertTrue(loader.motion_signatures[0].startswith("a.npz:"))
            self.assertTrue(loader.motion_signatures[1].startswith("b.npz:"))
            self.assertNotEqual(loader.motion_signatures[0], loader.motion_signatures[1])

    def test_rejects_incompatible_fps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            motion_dir = Path(temp_dir)
            _write_motion(motion_dir / "a.npz", frame_count=5, fps=50.0)
            _write_motion(motion_dir / "b.npz", frame_count=5, fps=60.0)

            with self.assertRaisesRegex(ValueError, "FPS mismatch"):
                motion_data.MultiMotionLoader(motion_dir, body_indexes=[0])


class MultiMotionAdaptiveSamplerTest(unittest.TestCase):
    def setUp(self):
        self.starts = torch.tensor([0, 5, 12], dtype=torch.long)
        self.ends = torch.tensor([5, 12, 16], dtype=torch.long)
        self.sampler = motion_data.MultiMotionAdaptiveSampler(
            self.starts,
            self.ends,
            env_fps=2.0,
            device="cpu",
            adaptive_alpha=1.0,
        )

    def test_samples_each_clip_without_crossing_boundaries(self):
        torch.manual_seed(7)
        motion_ids, time_steps = self.sampler.sample(20_000)

        self.assertTrue(torch.all(time_steps >= self.starts[motion_ids]))
        self.assertTrue(torch.all(time_steps < self.ends[motion_ids] - 1))
        counts = torch.bincount(motion_ids, minlength=3)
        self.assertTrue(torch.all(counts > 6_000))

    def test_records_failure_in_the_matching_motion_and_local_bin(self):
        self.sampler.record_failures(torch.tensor([1, 1]), torch.tensor([10, 10]))
        self.sampler.update()

        probabilities = self.sampler.phase_sampling_probabilities
        self.assertEqual(int(probabilities[1].argmax().item()), 2)
        self.assertTrue(torch.allclose(probabilities[0, :3], torch.full((3,), 1.0 / 3.0)))
        self.assertEqual(float(self.sampler._current_bin_failed.sum().item()), 0.0)

    def test_uniform_mixture_guarantees_a_probability_floor_under_concentrated_failures(self):
        rho = 0.1
        sampler = motion_data.MultiMotionAdaptiveSampler(
            self.starts,
            self.ends,
            env_fps=2.0,
            device="cpu",
            adaptive_uniform_ratio=rho,
            adaptive_alpha=1.0,
        )
        sampler.bin_failed_count[0, 1] = 1.0e9

        probabilities = sampler.phase_sampling_probabilities[0, : sampler.bin_counts[0]]
        bin_count = int(sampler.bin_counts[0].item())
        probability_floor = rho / float(bin_count)

        self.assertTrue(torch.all(probabilities >= probability_floor))
        self.assertTrue(torch.allclose(probabilities.sum(), torch.tensor(1.0)))
        self.assertTrue(
            torch.allclose(
                probabilities,
                torch.tensor([probability_floor, 1.0 - rho + probability_floor, probability_floor]),
            )
        )

    def test_uniform_ratio_must_be_a_finite_probability(self):
        for invalid_ratio in (0.0, -0.1, 1.1, float("inf"), float("nan")):
            with self.subTest(adaptive_uniform_ratio=invalid_ratio):
                with self.assertRaisesRegex(ValueError, r"\(0, 1\]"):
                    motion_data.MultiMotionAdaptiveSampler(
                        self.starts,
                        self.ends,
                        env_fps=2.0,
                        device="cpu",
                        adaptive_uniform_ratio=invalid_ratio,
                    )

    def test_uniform_sampling_never_selects_a_final_frame(self):
        torch.manual_seed(11)
        motion_ids, time_steps = self.sampler.sample_uniform(20_000)

        self.assertTrue(torch.all(time_steps >= self.starts[motion_ids]))
        self.assertTrue(torch.all(time_steps < self.ends[motion_ids] - 1))

    def test_checkpoint_round_trip_restores_adaptive_statistics(self):
        self.sampler.record_failures(torch.tensor([0, 1, 1]), torch.tensor([3, 10, 10]))
        self.sampler.update()
        expected_probabilities = self.sampler.phase_sampling_probabilities.clone()
        state_dict = self.sampler.state_dict()

        restored = motion_data.MultiMotionAdaptiveSampler(
            self.starts,
            self.ends,
            env_fps=2.0,
            device="cpu",
            adaptive_alpha=1.0,
        )
        restored.load_state_dict(state_dict)

        self.assertTrue(torch.equal(restored.bin_failed_count, self.sampler.bin_failed_count))
        self.assertTrue(torch.equal(restored._current_bin_failed, self.sampler._current_bin_failed))
        self.assertTrue(torch.equal(restored.phase_sampling_probabilities, expected_probabilities))

    def test_checkpoint_rejects_a_different_motion_topology(self):
        state_dict = self.sampler.state_dict()
        incompatible = motion_data.MultiMotionAdaptiveSampler(
            torch.tensor([0, 6, 13]),
            torch.tensor([6, 13, 17]),
            env_fps=2.0,
            device="cpu",
            adaptive_alpha=1.0,
        )

        with self.assertRaisesRegex(ValueError, "NPZ topology"):
            incompatible.load_state_dict(state_dict)

    def test_checkpoint_rejects_different_equal_length_motion_files(self):
        original = motion_data.MultiMotionAdaptiveSampler(
            self.starts,
            self.ends,
            env_fps=2.0,
            device="cpu",
            adaptive_alpha=1.0,
            motion_signatures=("a.npz:111", "b.npz:222", "c.npz:333"),
        )
        different_files = motion_data.MultiMotionAdaptiveSampler(
            self.starts,
            self.ends,
            env_fps=2.0,
            device="cpu",
            adaptive_alpha=1.0,
            motion_signatures=("a.npz:111", "b.npz:changed", "c.npz:333"),
        )

        with self.assertRaisesRegex(ValueError, "NPZ signatures"):
            different_files.load_state_dict(original.state_dict())


class MotionBoundaryContractTest(unittest.TestCase):
    def setUp(self):
        self.starts = torch.tensor([0, 5, 12], dtype=torch.long)
        self.ends = torch.tensor([5, 12, 16], dtype=torch.long)

    def test_advance_reaches_but_never_crosses_each_npz_final_frame(self):
        motion_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        penultimate = self.ends - 2
        next_frames, completed = motion_data.advance_motion_frames(motion_ids, penultimate, self.ends)

        self.assertEqual(next_frames.tolist(), (self.ends - 1).tolist())
        self.assertTrue(torch.all(completed))

        clamped_frames, completed_again = motion_data.advance_motion_frames(motion_ids, next_frames, self.ends)
        self.assertEqual(clamped_frames.tolist(), next_frames.tolist())
        self.assertTrue(torch.all(completed_again))

    def test_final_frame_hold_delays_clip_completion_without_crossing_boundaries(self):
        motion_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        time_steps = self.ends - 2
        hold_counts = torch.zeros(3, dtype=torch.long)

        time_steps, hold_counts, completed = motion_data.advance_motion_frames_with_final_hold(
            motion_ids,
            time_steps,
            self.ends,
            hold_counts,
            final_hold_steps=3,
        )
        self.assertEqual(time_steps.tolist(), (self.ends - 1).tolist())
        self.assertEqual(hold_counts.tolist(), [0, 0, 0])
        self.assertFalse(torch.any(completed))

        for expected_count in (1, 2):
            time_steps, hold_counts, completed = motion_data.advance_motion_frames_with_final_hold(
                motion_ids,
                time_steps,
                self.ends,
                hold_counts,
                final_hold_steps=3,
            )
            self.assertEqual(time_steps.tolist(), (self.ends - 1).tolist())
            self.assertEqual(hold_counts.tolist(), [expected_count] * 3)
            self.assertFalse(torch.any(completed))

        time_steps, hold_counts, completed = motion_data.advance_motion_frames_with_final_hold(
            motion_ids,
            time_steps,
            self.ends,
            hold_counts,
            final_hold_steps=3,
        )
        self.assertEqual(hold_counts.tolist(), [3, 3, 3])
        self.assertTrue(torch.all(completed))

    def test_zero_final_hold_preserves_original_completion_behavior(self):
        motion_ids = torch.tensor([0, 1, 2], dtype=torch.long)
        time_steps, hold_counts, completed = motion_data.advance_motion_frames_with_final_hold(
            motion_ids,
            self.ends - 2,
            self.ends,
            torch.zeros(3, dtype=torch.long),
            final_hold_steps=0,
        )

        self.assertEqual(time_steps.tolist(), (self.ends - 1).tolist())
        self.assertEqual(hold_counts.tolist(), [0, 0, 0])
        self.assertTrue(torch.all(completed))

    def test_round_robin_assignment_starts_at_each_npz_frame_zero(self):
        env_ids = torch.arange(8, dtype=torch.long)
        motion_ids, time_steps = motion_data.deterministic_motion_starts(
            env_ids,
            self.starts,
            mode="round_robin",
        )

        self.assertEqual(motion_ids.tolist(), [0, 1, 2, 0, 1, 2, 0, 1])
        self.assertEqual(time_steps.tolist(), self.starts[motion_ids].tolist())

    def test_fixed_assignment_validates_the_requested_npz(self):
        env_ids = torch.arange(4, dtype=torch.long)
        motion_ids, time_steps = motion_data.deterministic_motion_starts(
            env_ids,
            self.starts,
            mode="fixed",
            fixed_motion_id=2,
        )

        self.assertEqual(motion_ids.tolist(), [2, 2, 2, 2])
        self.assertEqual(time_steps.tolist(), [12, 12, 12, 12])
        with self.assertRaisesRegex(ValueError, "fixed_motion_id"):
            motion_data.deterministic_motion_starts(env_ids, self.starts, mode="fixed", fixed_motion_id=3)

    def test_geometry_restricted_envs_start_at_their_own_clip_beginning(self):
        motion_ids = torch.tensor([0, 1, 2, 1], dtype=torch.long)
        sampled_frames = torch.tensor([3, 9, 14, 6], dtype=torch.long)
        force_start = torch.tensor([False, True, True, False])

        selected_frames = motion_data.apply_forced_motion_starts(
            motion_ids,
            sampled_frames,
            self.starts,
            force_start,
        )

        self.assertEqual(selected_frames.tolist(), [3, 5, 12, 6])

    def test_clip_boundary_never_overlaps_an_earlier_termination(self):
        finished = torch.tensor([True, True, False, False])
        terminated = torch.tensor([False, True, True, False])

        boundary = motion_data.motion_clip_boundary_mask(finished, terminated)

        self.assertEqual(boundary.tolist(), [True, False, False, False])
        self.assertFalse(torch.any(boundary & terminated))
        self.assertIs(motion_data.motion_clip_timeout_mask, motion_data.motion_clip_boundary_mask)

    def test_adaptive_failure_includes_configured_end_state_failures(self):
        terminated = torch.tensor([False, True, False, False])
        end_state_failure = torch.tensor([True, False, False, True])

        combined = motion_data.adaptive_failure_mask(terminated, (end_state_failure,))

        self.assertEqual(combined.tolist(), [True, True, False, True])


if __name__ == "__main__":
    unittest.main()
