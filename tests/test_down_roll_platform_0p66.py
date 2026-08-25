from __future__ import annotations

import ast
import json
from pathlib import Path
import unittest

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPOSITORY_ROOT / "data/processed_motions/elf3/down_roll_50hz"
PROCESSED_DIR = REPOSITORY_ROOT / "data/processed_motions/elf3/down_roll_50hz_platform_0p66_v1"
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "source/php_kvoy_reproduction/php_kvoy_reproduction/tasks/tracking/config/elf3/down_roll_env_cfg.py"
)
REGISTRATION_PATH = CONFIG_PATH.with_name("__init__.py")
RUNNER_PATH = CONFIG_PATH.parent / "agents/rsl_rl_ppo_cfg.py"


def _metadata(archive: np.lib.npyio.NpzFile) -> dict[str, object]:
    return json.loads(str(np.asarray(archive["metadata_json"]).reshape(-1)[0]))


def _rotation_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    matrices = np.empty(quaternions.shape[:-1] + (3, 3), dtype=np.float64)
    matrices[..., 0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrices[..., 0, 1] = 2.0 * (x * y - z * w)
    matrices[..., 0, 2] = 2.0 * (x * z + y * w)
    matrices[..., 1, 0] = 2.0 * (x * y + z * w)
    matrices[..., 1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrices[..., 1, 2] = 2.0 * (y * z - x * w)
    matrices[..., 2, 0] = 2.0 * (x * z - y * w)
    matrices[..., 2, 1] = 2.0 * (y * z + x * w)
    matrices[..., 2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrices


class DownRollRetargetedDataTest(unittest.TestCase):
    def test_four_training_archives_are_present(self) -> None:
        expected = {
            "down_roll__jump_and_roll_pass1.npz",
            "down_roll__jump_and_roll_pass1__mirrored.npz",
            "down_roll__jump_and_roll_pass2.npz",
            "down_roll__jump_and_roll_pass2__mirrored.npz",
        }
        self.assertEqual({path.name for path in PROCESSED_DIR.glob("*.npz")}, expected)

    def test_source_joint_motion_and_ground_suffix_are_immutable(self) -> None:
        for rebuilt_path in sorted(PROCESSED_DIR.glob("*.npz")):
            source_path = SOURCE_DIR / rebuilt_path.name
            with np.load(source_path, allow_pickle=False) as source, np.load(
                rebuilt_path, allow_pickle=False
            ) as rebuilt:
                metadata = _metadata(rebuilt)
                patch = metadata["platform_height_patch"]
                landing = int(patch["ground_contact_frame"])
                offset = rebuilt["platform_height_retarget_offset_z"]

                self.assertEqual(patch["schema"], "elf3_down_roll_platform_height_v1")
                self.assertAlmostEqual(float(metadata["terrain_center_xyz"][2]), 0.33, places=12)
                self.assertAlmostEqual(float(metadata["terrain_size_xyz"][2]), 0.66, places=12)
                np.testing.assert_array_equal(rebuilt["joint_pos"], source["joint_pos"])
                np.testing.assert_array_equal(rebuilt["holosoma_root_qpos"][:, :2], source["holosoma_root_qpos"][:, :2])
                np.testing.assert_array_equal(rebuilt["holosoma_root_qpos"][:, 3:], source["holosoma_root_qpos"][:, 3:])
                np.testing.assert_allclose(
                    rebuilt["holosoma_root_qpos"][:, 2],
                    source["holosoma_root_qpos"][:, 2] + offset,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                self.assertTrue(np.all(offset >= 0.0))
                self.assertTrue(np.all(offset[landing:] == 0.0))
                self.assertLessEqual(float(np.max(np.abs(np.diff(offset) * 50.0))), 0.30)
                self.assertLessEqual(float(np.max(np.abs(np.diff(offset, n=2) * 50.0**2))), 3.0)
                for key in (
                    "joint_pos",
                    "joint_vel",
                    "body_pos_w",
                    "body_quat_w",
                    "body_lin_vel_w",
                    "body_ang_vel_w",
                    "holosoma_root_qpos",
                    "holosoma_root_qvel",
                ):
                    np.testing.assert_array_equal(rebuilt[key][landing:], source[key][landing:])

    def test_mirror_pairs_have_identical_vertical_correction(self) -> None:
        for base in ("down_roll__jump_and_roll_pass1", "down_roll__jump_and_roll_pass2"):
            with np.load(PROCESSED_DIR / f"{base}.npz", allow_pickle=False) as original, np.load(
                PROCESSED_DIR / f"{base}__mirrored.npz", allow_pickle=False
            ) as mirrored:
                np.testing.assert_allclose(
                    original["platform_height_retarget_offset_z"],
                    mirrored["platform_height_retarget_offset_z"],
                    rtol=0.0,
                    atol=1.0e-10,
                )

    def test_processed_archives_preserve_patch_and_runtime_schema(self) -> None:
        for path in sorted(PROCESSED_DIR.glob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                self.assertEqual(archive["joint_pos"].shape[1], 29)
                self.assertEqual(archive["joint_vel"].shape[1], 29)
                self.assertEqual(archive["holosoma_root_qpos"].shape[1], 7)
                self.assertEqual(archive["holosoma_root_qvel"].shape[1], 6)
                self.assertIn("platform_height_retarget_offset_z", archive.files)
                metadata = _metadata(archive)
                self.assertAlmostEqual(float(metadata["terrain_size_xyz"][2]), 0.66, places=12)

    def test_sole_corners_do_not_penetrate_the_actual_training_platform(self) -> None:
        corners = np.asarray(
            ((-0.09, -0.04, -0.041), (-0.09, 0.04, -0.041), (0.15, -0.04, -0.041), (0.15, 0.04, -0.041))
        )
        lower_xy = np.asarray((-0.95, 0.0)) - 0.5 * np.asarray((0.51, 0.80))
        upper_xy = np.asarray((-0.95, 0.0)) + 0.5 * np.asarray((0.51, 0.80))
        for path in sorted(PROCESSED_DIR.glob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                names = tuple(str(name) for name in archive["body_names"].tolist())
                foot_ids = [names.index("l_ankle_x_link"), names.index("r_ankle_x_link")]
                positions = archive["body_pos_w"][:, foot_ids]
                rotations = _rotation_matrices_wxyz(archive["body_quat_w"][:, foot_ids])
                points = positions[:, :, None, :] + np.einsum("tfij,kj->tfki", rotations, corners)
                inside = np.all((points[..., :2] >= lower_xy) & (points[..., :2] <= upper_xy), axis=-1)
                landing = int(_metadata(archive)["platform_height_patch"]["ground_contact_frame"])
                relevant = points[:landing, ..., 2][inside[:landing]]
                self.assertGreater(relevant.size, 0)
                self.assertGreaterEqual(float(np.min(relevant)), 0.66 - 1.0e-8)


class DownRollTaskConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tree = ast.parse(CONFIG_PATH.read_text())

    def _class(self, name: str) -> ast.ClassDef:
        return next(node for node in self.tree.body if isinstance(node, ast.ClassDef) and node.name == name)

    def test_task_and_runner_are_registered_separately(self) -> None:
        registration = REGISTRATION_PATH.read_text()
        runner = RUNNER_PATH.read_text()
        self.assertIn('id="Tracking-DownRoll-ELF3-v0"', registration)
        self.assertIn("down_roll_env_cfg.ELF3DownRollEnvCfg", registration)
        self.assertIn("ELF3DownRollPPORunnerCfg", registration)
        self.assertIn('experiment_name = "elf3_down_roll"', runner)

    def test_command_uses_fixed_source_and_clip_boundary(self) -> None:
        command_class = self._class("ELF3DownRollCommandsCfg")
        assignment = next(node for node in command_class.body if isinstance(node, ast.Assign))
        keywords = {keyword.arg: keyword.value for keyword in assignment.value.keywords}
        self.assertTrue(ast.literal_eval(keywords["terminate_on_motion_end"]))
        self.assertFalse(ast.literal_eval(keywords["use_adaptive_sampling"]))
        self.assertTrue(ast.literal_eval(keywords["exclude_repeated_terminal_frames_from_random_starts"]))
        self.assertFalse(ast.literal_eval(keywords["terminal_default_pose_enabled"]))

    def test_rewards_contain_no_climb_or_undesired_contact_objective(self) -> None:
        rewards = self._class("ELF3DownRollRewardsCfg")
        names = {
            target.id
            for node in rewards.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertTrue({"motion_joint_pos", "motion_joint_vel"}.issubset(names))
        self.assertFalse(
            {
                "platform_foot_contact",
                "platform_foot_surface_alignment",
                "first_foothold_support_quality",
                "first_foothold_surface_alignment",
                "climb_platform_progress",
                "undesired_contacts",
            }
            & names
        )

    def test_every_joint_group_has_pose_velocity_and_aggregation_parameters(self) -> None:
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id
            in {
                "ELF3_DOWN_ROLL_EXPERT_JOINT_POSITION_GROUP_STDS",
                "ELF3_DOWN_ROLL_EXPERT_JOINT_VELOCITY_GROUP_STDS",
                "ELF3_DOWN_ROLL_EXPERT_JOINT_GROUP_WEIGHTS",
            }
        }
        expected = {
            "waist",
            "left_arm",
            "right_arm",
            "left_leg",
            "left_ankle",
            "right_leg",
            "right_ankle",
        }
        self.assertEqual(set(assignments), {
            "ELF3_DOWN_ROLL_EXPERT_JOINT_POSITION_GROUP_STDS",
            "ELF3_DOWN_ROLL_EXPERT_JOINT_VELOCITY_GROUP_STDS",
            "ELF3_DOWN_ROLL_EXPERT_JOINT_GROUP_WEIGHTS",
        })
        for values in assignments.values():
            self.assertEqual(set(values), expected)


if __name__ == "__main__":
    unittest.main()
