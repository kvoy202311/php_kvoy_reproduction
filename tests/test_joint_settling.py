from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch


_MODULE_PATH = (
    Path(__file__).parents[1]
    / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/mdp/joint_settling.py"
)
_SPEC = importlib.util.spec_from_file_location("php_kvoy_reproduction_joint_settling", _MODULE_PATH)
joint_settling = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(joint_settling)


class JointSettlingScoreTest(unittest.TestCase):
    def test_score_is_one_at_rest_and_decreases_with_speed(self):
        velocities = torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.25, 0.25, 0.25],
                [0.5, 0.5, 0.5],
                [1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0],
            ]
        )
        score = joint_settling.joint_settling_score(
            velocities,
            rms_speed_scale=1.0,
            max_speed_scale=2.0,
            fine_max_speed_scale=0.5,
            score_weights=(0.4, 0.3, 0.3),
        )

        self.assertAlmostEqual(score[0].item(), 1.0)
        self.assertTrue(torch.all(score[:-1] > score[1:]))
        self.assertGreater(score[-1].item(), 0.05)

    def test_maximum_component_exposes_one_fast_joint(self):
        distributed = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        one_fast_joint = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
        distributed_rms, distributed_max = joint_settling.joint_speed_statistics(distributed)
        sparse_rms, sparse_max = joint_settling.joint_speed_statistics(one_fast_joint)

        torch.testing.assert_close(distributed_rms, sparse_rms)
        self.assertGreater(sparse_max.item(), distributed_max.item())
        distributed_score = joint_settling.joint_settling_score(
            distributed,
            rms_speed_scale=1.0,
            max_speed_scale=2.0,
            fine_max_speed_scale=0.5,
            score_weights=(0.4, 0.3, 0.3),
        )
        sparse_score = joint_settling.joint_settling_score(
            one_fast_joint,
            rms_speed_scale=1.0,
            max_speed_scale=2.0,
            fine_max_speed_scale=0.5,
            score_weights=(0.4, 0.3, 0.3),
        )
        self.assertLess(sparse_score.item(), distributed_score.item())

    def test_rejects_invalid_shapes_and_parameters(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            joint_settling.joint_speed_statistics(torch.zeros(3))
        with self.assertRaisesRegex(ValueError, "positive"):
            joint_settling.joint_settling_score(
                torch.zeros(1, 2),
                rms_speed_scale=0.0,
                max_speed_scale=2.0,
                fine_max_speed_scale=0.5,
                score_weights=(0.4, 0.3, 0.3),
            )


if __name__ == "__main__":
    unittest.main()
