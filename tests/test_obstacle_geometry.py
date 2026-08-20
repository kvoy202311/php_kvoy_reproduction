from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/obstacle_geometry.py"
)
_SPEC = importlib.util.spec_from_file_location("php_kvoy_reproduction_obstacle", _MODULE_PATH)
obstacle = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(obstacle)


class ClimbBoxGeometryTest(unittest.TestCase):
    @staticmethod
    def _platform(length: float = 0.46, width: float = 0.80):
        return (
            torch.tensor([[0.0, 0.0, 0.325]], dtype=torch.float32),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
            torch.tensor([[length, width, 0.65]], dtype=torch.float32),
        )

    @staticmethod
    def _sole_corners(rear_x: float, front_x: float, half_width: float = 0.04):
        return torch.tensor(
            [[[[rear_x, -half_width, 0.65], [rear_x, half_width, 0.65],
               [front_x, -half_width, 0.65], [front_x, half_width, 0.65]]]],
            dtype=torch.float32,
        )

    @staticmethod
    def _foothold_score(corners, centers, orientations, sizes):
        return obstacle.foothold_safety_score(
            corners,
            centers,
            orientations,
            sizes,
            approach_side=-1.0,
            max_heel_overhang=0.05,
            min_forefoot_inside=0.04,
            far_edge_margin=0.04,
            lateral_margin=0.02,
        )

    def test_nominal_box_top_is_exactly_point_six_five_metres(self):
        hits = torch.tensor([[[-0.95, 0.0, 0.0], [-0.50, 0.0, 0.0]]], dtype=torch.float32)
        centers = torch.tensor([[-0.95, 0.0, 0.325]], dtype=torch.float32)
        orientations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        sizes = torch.tensor([[0.46, 0.80, 0.65]], dtype=torch.float32)

        heights, mask = obstacle.climb_box_top_height(hits, centers, orientations, sizes)

        self.assertEqual(mask.tolist(), [[True, False]])
        self.assertTrue(torch.equal(heights, torch.tensor([[0.65, 0.0]], dtype=torch.float32)))

    def test_yaw_rotates_the_footprint_without_changing_top_height(self):
        half_yaw = math.pi / 4.0
        orientations = torch.tensor([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]], dtype=torch.float32)
        centers = torch.tensor([[0.0, 0.0, 0.30]], dtype=torch.float32)
        sizes = torch.tensor([[0.40, 1.00, 0.60]], dtype=torch.float32)
        # At 90 degrees yaw, the local short x axis lies along world y.
        hits = torch.tensor([[[0.0, 0.19, 0.0], [0.0, 0.21, 0.0]]], dtype=torch.float32)

        heights, mask = obstacle.climb_box_top_height(hits, centers, orientations, sizes)

        self.assertEqual(mask.tolist(), [[True, False]])
        self.assertTrue(torch.equal(heights, torch.tensor([[0.60, 0.0]], dtype=torch.float32)))

    def test_dimension_sampling_stays_inside_all_configured_ranges(self):
        torch.manual_seed(5)
        sizes = obstacle.sample_climb_box_sizes(
            10_000,
            length_range=(0.41, 0.51),
            width_range=(0.80, 1.50),
            height_range=(0.60, 0.70),
        )

        self.assertTrue(torch.all((sizes[:, 0] >= 0.41) & (sizes[:, 0] <= 0.51)))
        self.assertTrue(torch.all((sizes[:, 1] >= 0.80) & (sizes[:, 1] <= 1.50)))
        self.assertTrue(torch.all((sizes[:, 2] >= 0.60) & (sizes[:, 2] <= 0.70)))

    def test_nominal_partition_has_an_exact_reproducible_size(self):
        env_ids = torch.arange(10, dtype=torch.long)
        mask = obstacle.nominal_environment_mask(env_ids, num_envs=10, nominal_fraction=0.5)

        self.assertEqual(mask.tolist(), [True, True, True, True, True, False, False, False, False, False])

    def test_heel_overhang_exactly_five_centimetres_is_safe_but_more_is_not(self):
        centers, orientations, sizes = self._platform()
        at_limit_corners = self._sole_corners(-0.28, -0.04)
        beyond_corners = self._sole_corners(-0.281, -0.04)
        at_limit_score, at_limit_valid = self._foothold_score(
            at_limit_corners, centers, orientations, sizes
        )
        beyond_score, beyond_valid = self._foothold_score(
            beyond_corners, centers, orientations, sizes
        )
        at_limit_violation = obstacle.foothold_safety_violation(
            at_limit_corners,
            centers,
            orientations,
            sizes,
            approach_side=-1.0,
            max_heel_overhang=0.05,
            min_forefoot_inside=0.04,
            far_edge_margin=0.04,
            lateral_margin=0.02,
        )
        beyond_violation = obstacle.foothold_safety_violation(
            beyond_corners,
            centers,
            orientations,
            sizes,
            approach_side=-1.0,
            max_heel_overhang=0.05,
            min_forefoot_inside=0.04,
            far_edge_margin=0.04,
            lateral_margin=0.02,
        )

        self.assertTrue(at_limit_valid.item())
        self.assertGreater(at_limit_score.item(), 0.0)
        self.assertLess(at_limit_violation.item(), 1.0e-5)
        self.assertFalse(beyond_valid.item())
        self.assertEqual(beyond_score.item(), 0.0)
        self.assertGreater(beyond_violation.item(), 0.0)

    def test_far_and_lateral_safe_boundaries_are_strict(self):
        centers, orientations, sizes = self._platform()
        far_limit_score, far_limit_valid = self._foothold_score(
            self._sole_corners(-0.05, 0.19), centers, orientations, sizes
        )
        far_beyond_score, far_beyond_valid = self._foothold_score(
            self._sole_corners(-0.05, 0.191), centers, orientations, sizes
        )
        lateral_limit_score, lateral_limit_valid = self._foothold_score(
            self._sole_corners(-0.28, -0.04, half_width=0.38), centers, orientations, sizes
        )
        lateral_beyond_score, lateral_beyond_valid = self._foothold_score(
            self._sole_corners(-0.28, -0.04, half_width=0.381), centers, orientations, sizes
        )

        self.assertTrue(far_limit_valid.item())
        self.assertGreater(far_limit_score.item(), 0.0)
        self.assertFalse(far_beyond_valid.item())
        self.assertEqual(far_beyond_score.item(), 0.0)
        self.assertTrue(lateral_limit_valid.item())
        self.assertGreater(lateral_limit_score.item(), 0.0)
        self.assertFalse(lateral_beyond_valid.item())
        self.assertEqual(lateral_beyond_score.item(), 0.0)

    def test_safe_region_adapts_to_each_sampled_platform_length(self):
        centers, orientations, short_sizes = self._platform(length=0.41)
        _, _, long_sizes = self._platform(length=0.51)
        fixed_world_sole = self._sole_corners(-0.28, -0.04)
        _, short_valid = self._foothold_score(fixed_world_sole, centers, orientations, short_sizes)
        _, long_valid = self._foothold_score(fixed_world_sole, centers, orientations, long_sizes)
        adjusted_short_sole = self._sole_corners(-0.255, -0.015)
        adjusted_short_score, adjusted_short_valid = self._foothold_score(
            adjusted_short_sole, centers, orientations, short_sizes
        )

        self.assertFalse(short_valid.item())
        self.assertTrue(long_valid.item())
        self.assertTrue(adjusted_short_valid.item())
        self.assertGreater(adjusted_short_score.item(), 0.0)

    def test_foothold_score_is_invariant_to_shared_platform_yaw(self):
        centers, _, sizes = self._platform()
        local_corners = self._sole_corners(-0.28, -0.04)
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        identity_score, identity_valid = self._foothold_score(local_corners, centers, identity, sizes)

        half_yaw = math.pi / 4.0
        quarter_turn = torch.tensor([[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]], dtype=torch.float32)
        yawed_corners = local_corners.clone()
        yawed_corners[..., 0] = -local_corners[..., 1]
        yawed_corners[..., 1] = local_corners[..., 0]
        yawed_score, yawed_valid = self._foothold_score(yawed_corners, centers, quarter_turn, sizes)

        self.assertEqual(identity_valid.tolist(), yawed_valid.tolist())
        torch.testing.assert_close(identity_score, yawed_score, rtol=0.0, atol=1.0e-6)

    def test_sole_corner_transform_respects_ankle_yaw(self):
        foot_positions = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
        half_yaw = math.pi / 4.0
        foot_orientations = torch.tensor(
            [[[math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)]]], dtype=torch.float32
        )
        corners = obstacle.foot_sole_corners_world(
            foot_positions,
            foot_orientations,
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        )

        torch.testing.assert_close(corners, torch.tensor([[[[1.0, 3.0, 3.0]]]]), rtol=0.0, atol=1.0e-6)

    def test_batched_left_and_right_feet_are_not_hard_coded(self):
        centers, orientations, sizes = self._platform()
        valid = self._sole_corners(-0.28, -0.04)
        invalid = self._sole_corners(-0.281, -0.04)
        corners = torch.cat(
            (
                torch.cat((valid, invalid), dim=1),
                torch.cat((invalid, valid), dim=1),
            ),
            dim=0,
        )
        batched_centers = centers.repeat(2, 1)
        batched_orientations = orientations.repeat(2, 1)
        batched_sizes = sizes.repeat(2, 1)
        scores, valid_mask = self._foothold_score(corners, batched_centers, batched_orientations, batched_sizes)

        self.assertEqual(valid_mask.tolist(), [[True, False], [False, True]])
        self.assertGreater(scores[0, 0].item(), 0.0)
        self.assertGreater(scores[1, 1].item(), 0.0)
        self.assertEqual(scores[0, 1].item(), 0.0)
        self.assertEqual(scores[1, 0].item(), 0.0)

    def test_reference_gate_selects_mirrored_leader_and_closes_after_second_foot_arrives(self):
        centers, orientations, sizes = self._platform()
        reference_feet = torch.tensor(
            [[[-0.10, 0.0, 0.65], [-0.50, 0.0, 0.0]],
             [[-0.50, 0.0, 0.0], [-0.10, 0.0, 0.65]]],
            dtype=torch.float32,
        )
        gate, lead_mask = obstacle.first_foothold_reference_gate(
            reference_feet,
            centers.repeat(2, 1),
            orientations.repeat(2, 1),
            sizes.repeat(2, 1),
            torch.tensor([0.5, 0.5]),
            approach_side=-1.0,
            reference_activation_distance=0.20,
            reference_activation_inside=0.03,
            reference_release_distance=0.08,
            reference_release_inside=0.03,
            phase_start=0.28,
            phase_ramp=0.08,
            phase_end=0.72,
            phase_fade=0.10,
        )
        second_foot_caught_up = torch.tensor([[[-0.10, 0.0, 0.65], [-0.12, 0.0, 0.65]]], dtype=torch.float32)
        closed_gate, _ = obstacle.first_foothold_reference_gate(
            second_foot_caught_up,
            centers,
            orientations,
            sizes,
            torch.tensor([0.5]),
            approach_side=-1.0,
            reference_activation_distance=0.20,
            reference_activation_inside=0.03,
            reference_release_distance=0.08,
            reference_release_inside=0.03,
            phase_start=0.28,
            phase_ramp=0.08,
            phase_end=0.72,
            phase_fade=0.10,
        )

        self.assertGreater(gate[0].item(), 0.0)
        self.assertGreater(gate[1].item(), 0.0)
        self.assertEqual(lead_mask.tolist(), [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(closed_gate.item(), 0.0)

    def test_only_filtered_upward_platform_force_counts_as_support(self):
        forces = torch.tensor([[[0.0, 0.0, 20.0], [20.0, 0.0, 0.0]]], dtype=torch.float32)
        contact_times = torch.tensor([[0.06, 0.06]], dtype=torch.float32)
        scores = obstacle.filtered_platform_contact_score(
            forces,
            contact_times,
            min_upward_force=10.0,
            contact_time_scale=0.06,
        )

        self.assertGreater(scores[0, 0].item(), 0.0)
        self.assertEqual(scores[0, 1].item(), 0.0)

    def test_complete_sole_alignment_rejects_a_persistent_toe_stand(self):
        centers, orientations, sizes = self._platform()
        flat = self._sole_corners(-0.20, 0.04)
        toe_stand = flat.clone()
        toe_stand[..., 0, 2] += 0.05
        toe_stand[..., 1, 2] += 0.05
        corners = torch.cat((flat, toe_stand), dim=1)

        score, valid, maximum_error = obstacle.sole_surface_alignment_score(
            corners,
            centers,
            sizes,
            height_std=0.025,
            height_tolerance=0.03,
        )

        self.assertEqual(valid.tolist(), [[True, False]])
        torch.testing.assert_close(maximum_error, torch.tensor([[0.0, 0.05]]))
        self.assertGreater(score[0, 0].item(), score[0, 1].item())
        self.assertEqual(score[0, 0].item(), 1.0)

    def test_large_toe_tilt_keeps_dense_surface_gradient_but_fails_strict_support(self):
        centers, _, sizes = self._platform()
        toe_stand = self._sole_corners(-0.20, 0.04)
        toe_stand[..., 0, 2] += 0.25
        toe_stand[..., 1, 2] += 0.25
        toe_stand.requires_grad_()

        dense_score, height_spread, closest_error = obstacle.sole_surface_shaping_score(
            toe_stand,
            centers,
            sizes,
            tilt_scale=0.05,
            height_scale=0.06,
        )
        _, strict_valid, maximum_error = obstacle.sole_surface_alignment_score(
            toe_stand,
            centers,
            sizes,
            height_std=0.025,
            height_tolerance=0.03,
        )
        dense_score.sum().backward()

        self.assertAlmostEqual(dense_score.item(), 1.0 / math.sqrt(26.0), places=6)
        self.assertAlmostEqual(height_spread.item(), 0.25, places=6)
        self.assertEqual(closest_error.item(), 0.0)
        self.assertAlmostEqual(maximum_error.item(), 0.25, places=6)
        self.assertFalse(strict_valid.item())
        self.assertGreater(torch.linalg.vector_norm(toe_stand.grad).item(), 0.0)

    def test_filtered_platform_contact_time_needs_continuous_filtered_support(self):
        previous = torch.zeros((1, 2), dtype=torch.float32)
        left_only = torch.tensor([[True, False]])
        first = obstacle.advance_filtered_platform_contact_time(previous, left_only, step_dt=0.02)
        second = obstacle.advance_filtered_platform_contact_time(first, left_only, step_dt=0.02)
        third = obstacle.advance_filtered_platform_contact_time(second, left_only, step_dt=0.02)
        interrupted = obstacle.advance_filtered_platform_contact_time(
            third,
            torch.tensor([[False, True]]),
            step_dt=0.02,
        )

        torch.testing.assert_close(first, torch.tensor([[0.02, 0.00]]))
        torch.testing.assert_close(third, torch.tensor([[0.06, 0.00]]))
        torch.testing.assert_close(interrupted, torch.tensor([[0.00, 0.02]]))

    def test_terminal_alignment_uses_the_lowest_physical_sole_and_has_quiet_endpoints(self):
        foot_positions = torch.tensor(
            [
                [[0.0, 0.0, 0.70], [0.0, 0.0, 0.72]],
                [[0.0, 0.0, 0.68], [0.0, 0.0, 0.74]],
            ],
            dtype=torch.float32,
        )
        foot_orientations = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32
        ).expand(2, -1, -1)
        corners_b = torch.tensor(
            [[-0.09, -0.04, -0.04], [-0.09, 0.04, -0.04], [0.15, -0.04, -0.04], [0.15, 0.04, -0.04]],
            dtype=torch.float32,
        )
        support_z = obstacle.terminal_sole_support_plane_z(foot_positions, foot_orientations, corners_b)
        torch.testing.assert_close(support_z, torch.tensor([0.66, 0.64]))

        position_offset, velocity_offset, complete = obstacle.terminal_platform_z_alignment(
            support_z,
            torch.tensor([0.60, 0.70]),
            torch.tensor([0, 25]),
            ramp_steps=25,
            step_dt=0.02,
        )
        torch.testing.assert_close(position_offset, torch.tensor([0.0, 0.06]))
        torch.testing.assert_close(velocity_offset, torch.zeros(2))
        self.assertEqual(complete.tolist(), [False, True])

        # A low sampled box needs the terminal source reference to move down,
        # not merely a high-box upward correction.  At the end of the 0.5 s
        # ramp its lowest physical sole lies exactly on the actual top.
        low_offset, low_velocity, low_complete = obstacle.terminal_platform_z_alignment(
            support_z[:1],
            torch.tensor([0.60]),
            torch.tensor([25]),
            ramp_steps=25,
            step_dt=0.02,
        )
        torch.testing.assert_close(low_offset, torch.tensor([-0.06]))
        torch.testing.assert_close(low_velocity, torch.zeros(1))
        self.assertTrue(low_complete.item())
        torch.testing.assert_close(support_z[:1] + low_offset, torch.tensor([0.60]))

        midpoint_offset, midpoint_velocity, midpoint_complete = obstacle.terminal_platform_z_alignment(
            support_z[:1],
            torch.tensor([0.60]),
            torch.tensor([12]),
            ramp_steps=25,
            step_dt=0.02,
        )
        self.assertLess(midpoint_offset.abs().item(), 0.06)
        self.assertGreater(midpoint_velocity.abs().item(), 0.0)
        self.assertFalse(midpoint_complete.item())

if __name__ == "__main__":
    unittest.main()
