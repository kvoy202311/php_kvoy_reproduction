#!/usr/bin/env python3
"""Convert 50 Hz ELF3 Holosoma WBT NPZ files to this project's NPZ layout.

The conversion is deliberately limited to slicing and permutation:

* ``joint_pos`` changes from ``[root xyz, root quat wxyz, 29 joints]`` to
  the 29 joints in Isaac/PhysX runtime order.
* ``joint_vel`` changes from ``[root linear velocity, root angular velocity,
  29 joints]`` to the 29 joints in Isaac/PhysX runtime order.
* all rigid-body arrays are permuted from Holosoma/MuJoCo body order to the
  actual Isaac/PhysX runtime body order of the ELF3 USD.

No trajectory value is interpolated, smoothed, differentiated, normalized,
clipped, or cast to another dtype. The source root qpos/qvel are retained in
``holosoma_root_qpos`` and ``holosoma_root_qvel``. After writing, the script
performs an inverse permutation and requires byte-for-byte recovery of every
source trajectory array.

The input is never overwritten. Existing output files are also never
overwritten.

Examples:

.. code-block:: bash

    # Validate one directory without writing anything.
    python scripts/convert_elf3_holosoma2wbt_npz.py \
        data/motions/elf3/climb_50hz \
        data/processed_motions/elf3/climb_50hz \
        --dry-run

    # Convert every NPZ below data/motions/elf3 while preserving subdirectories.
    python scripts/convert_elf3_holosoma2wbt_npz.py \
        data/motions/elf3 \
        data/processed_motions/elf3 \
        --recursive
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np


EXPECTED_FPS = 50.0
ROOT_QPOS_WIDTH = 7
ROOT_QVEL_WIDTH = 6

REQUIRED_KEYS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_names",
    "joint_names",
)

# Read from robot.joint_names after loading the current ELF3 USD in Isaac Lab
# 4.5. This is also the policy order used by TienKung-ELF3 and the deployment
# remapping in bxi_rl_controller_ros2_example.
ISAAC_JOINT_NAMES = (
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
)

# Read from robot.body_names after loading the current ELF3 USD in Isaac Lab
# 4.5. The current MotionLoader indexes motion body tensors with runtime body
# IDs, so the output arrays must use this exact order.
ISAAC_BODY_NAMES = (
    "torso_link",
    "l_shoulder_y_link",
    "r_shoulder_y_link",
    "waist_y_link",
    "l_shoulder_x_link",
    "r_shoulder_x_link",
    "waist_x_link",
    "l_shoulder_z_link",
    "r_shoulder_z_link",
    "waist_z_link",
    "l_elbow_y_link",
    "r_elbow_y_link",
    "l_hip_y_link",
    "r_hip_y_link",
    "l_wrist_x_link",
    "r_wrist_x_link",
    "l_hip_x_link",
    "r_hip_x_link",
    "l_wrist_y_link",
    "r_wrist_y_link",
    "l_hip_z_link",
    "r_hip_z_link",
    "l_wrist_z_link",
    "r_wrist_z_link",
    "l_knee_y_link",
    "r_knee_y_link",
    "l_ankle_y_link",
    "r_ankle_y_link",
    "l_ankle_x_link",
    "r_ankle_x_link",
)

TRAJECTORY_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


class ConversionError(ValueError):
    """Raised when an input or converted NPZ violates the expected contract."""


@dataclass(frozen=True)
class ConversionPlan:
    source_path: Path
    destination_path: Path
    source_sha256: str
    source_arrays: dict[str, np.ndarray]
    source_joint_names: tuple[str, ...]
    source_body_names: tuple[str, ...]
    joint_permutation: np.ndarray
    body_permutation: np.ndarray
    frames: int
    fps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="A Holosoma NPZ file or a directory containing NPZ files.")
    parser.add_argument("output_dir", type=Path, help="Separate directory for converted NPZ files.")
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively discover NPZ files when INPUT is a directory and preserve relative subdirectories.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and show planned outputs without creating directories or files.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bitwise_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Return True only when shape, dtype, and every stored bit are equal."""

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and left.tobytes(order="C") == right.tobytes(order="C")
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ConversionError(message)


def parse_names(array: np.ndarray, key: str, expected_count: int, path: Path) -> tuple[str, ...]:
    require(array.ndim == 1, f"{path}: {key} must be one-dimensional, got {array.shape}")
    require(array.dtype.kind in {"U", "S"}, f"{path}: {key} must be a string array, got {array.dtype}")
    try:
        names = tuple(
            value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
            for value in array.tolist()
        )
    except UnicodeDecodeError as error:
        raise ConversionError(f"{path}: {key} contains a non-UTF-8 byte string") from error
    require(len(names) == expected_count, f"{path}: {key} must contain {expected_count} names, got {len(names)}")
    require(len(set(names)) == len(names), f"{path}: {key} contains duplicate names")
    return names


def require_numeric_float(array: np.ndarray, key: str, path: Path) -> None:
    require(array.dtype.kind == "f", f"{path}: {key} must be floating-point, got {array.dtype}")
    require(np.isfinite(array).all(), f"{path}: {key} contains NaN or infinity")


def permutation_for(target_names: tuple[str, ...], source_names: tuple[str, ...], kind: str, path: Path) -> np.ndarray:
    missing = sorted(set(target_names) - set(source_names))
    unexpected = sorted(set(source_names) - set(target_names))
    require(not missing and not unexpected, f"{path}: {kind} name mismatch; missing={missing}, unexpected={unexpected}")
    source_index = {name: index for index, name in enumerate(source_names)}
    permutation = np.asarray([source_index[name] for name in target_names], dtype=np.int64)
    require(
        sorted(permutation.tolist()) == list(range(len(source_names))),
        f"{path}: {kind} permutation is not bijective",
    )
    return permutation


def validate_input(source_path: Path, destination_path: Path) -> ConversionPlan:
    source_sha256 = sha256_file(source_path)
    try:
        with np.load(source_path, allow_pickle=False) as archive:
            keys = tuple(archive.files)
            missing_keys = sorted(set(REQUIRED_KEYS) - set(keys))
            require(not missing_keys, f"{source_path}: missing required keys: {missing_keys}")
            source_arrays = {key: np.array(archive[key], copy=True) for key in keys}
    except (OSError, ValueError) as error:
        if isinstance(error, ConversionError):
            raise
        raise ConversionError(f"{source_path}: cannot safely read NPZ: {error}") from error

    for key, array in source_arrays.items():
        require(array.dtype.kind != "O", f"{source_path}: object/pickle array is forbidden: {key}")

    fps_array = source_arrays["fps"]
    require(fps_array.size == 1, f"{source_path}: fps must be scalar or length one, got {fps_array.shape}")
    require(np.issubdtype(fps_array.dtype, np.number), f"{source_path}: fps must be numeric, got {fps_array.dtype}")
    fps = float(fps_array.reshape(-1)[0])
    require(np.isfinite(fps) and fps > 0.0, f"{source_path}: invalid fps: {fps}")
    require(abs(fps - EXPECTED_FPS) <= 1.0e-12, f"{source_path}: expected exactly 50 Hz, got {fps}")

    source_joint_names = parse_names(source_arrays["joint_names"], "joint_names", len(ISAAC_JOINT_NAMES), source_path)
    source_body_names = parse_names(source_arrays["body_names"], "body_names", len(ISAAC_BODY_NAMES), source_path)
    joint_permutation = permutation_for(ISAAC_JOINT_NAMES, source_joint_names, "joint", source_path)
    body_permutation = permutation_for(ISAAC_BODY_NAMES, source_body_names, "body", source_path)

    joint_pos = source_arrays["joint_pos"]
    joint_vel = source_arrays["joint_vel"]
    body_pos = source_arrays["body_pos_w"]
    body_quat = source_arrays["body_quat_w"]
    body_lin_vel = source_arrays["body_lin_vel_w"]
    body_ang_vel = source_arrays["body_ang_vel_w"]

    for key in TRAJECTORY_KEYS:
        require_numeric_float(source_arrays[key], key, source_path)

    require(
        joint_pos.ndim == 2 and joint_pos.shape[1] == ROOT_QPOS_WIDTH + len(source_joint_names),
        f"{source_path}: joint_pos must have shape (T, 36), got {joint_pos.shape}",
    )
    require(
        joint_vel.ndim == 2 and joint_vel.shape[1] == ROOT_QVEL_WIDTH + len(source_joint_names),
        f"{source_path}: joint_vel must have shape (T, 35), got {joint_vel.shape}",
    )
    frames = joint_pos.shape[0]
    require(frames >= 2, f"{source_path}: at least two frames are required, got {frames}")
    require(joint_vel.shape[0] == frames, f"{source_path}: joint_pos/joint_vel frame counts differ")

    expected_body_shapes = {
        "body_pos_w": (frames, len(source_body_names), 3),
        "body_quat_w": (frames, len(source_body_names), 4),
        "body_lin_vel_w": (frames, len(source_body_names), 3),
        "body_ang_vel_w": (frames, len(source_body_names), 3),
    }
    for key, expected_shape in expected_body_shapes.items():
        require(
            source_arrays[key].shape == expected_shape,
            f"{source_path}: {key} must have shape {expected_shape}, got {source_arrays[key].shape}",
        )

    # Every non-scalar auxiliary array in the reviewed Holosoma expert schema
    # is either name metadata or has one entry per frame. Reject a different
    # leading dimension instead of silently copying a potentially misaligned
    # per-frame signal.
    for key, array in source_arrays.items():
        if key in {"fps", "joint_names", "body_names", "metadata_json"} or array.ndim == 0:
            continue
        if key not in TRAJECTORY_KEYS:
            require(
                array.shape[0] == frames,
                f"{source_path}: auxiliary array {key} has {array.shape[0]} rows, expected {frames}",
            )

    root_body_index = source_body_names.index("torso_link")
    root_pos = joint_pos[:, :3]
    root_quat = joint_pos[:, 3:7]
    torso_pos = body_pos[:, root_body_index]
    torso_quat = body_quat[:, root_body_index]
    require(
        np.allclose(root_pos, torso_pos, rtol=0.0, atol=1.0e-9),
        f"{source_path}: qpos root position does not match torso_link body position",
    )
    quat_direct = np.max(np.abs(root_quat - torso_quat), axis=-1)
    quat_negated = np.max(np.abs(root_quat + torso_quat), axis=-1)
    require(
        np.max(np.minimum(quat_direct, quat_negated)) <= 1.0e-9,
        f"{source_path}: qpos root quaternion does not match torso_link body quaternion",
    )
    require(
        np.max(np.abs(np.linalg.norm(root_quat, axis=-1) - 1.0)) <= 1.0e-6,
        f"{source_path}: root quaternion is not normalized",
    )
    require(
        np.max(np.abs(np.linalg.norm(body_quat, axis=-1) - 1.0)) <= 1.0e-6,
        f"{source_path}: a body quaternion is not normalized",
    )

    return ConversionPlan(
        source_path=source_path,
        destination_path=destination_path,
        source_sha256=source_sha256,
        source_arrays=source_arrays,
        source_joint_names=source_joint_names,
        source_body_names=source_body_names,
        joint_permutation=joint_permutation,
        body_permutation=body_permutation,
        frames=frames,
        fps=fps,
    )


def build_output_arrays(plan: ConversionPlan) -> dict[str, np.ndarray]:
    arrays = {key: np.array(value, copy=True) for key, value in plan.source_arrays.items()}
    source_joint_pos = plan.source_arrays["joint_pos"]
    source_joint_vel = plan.source_arrays["joint_vel"]

    arrays["holosoma_root_qpos"] = np.array(source_joint_pos[:, :ROOT_QPOS_WIDTH], copy=True)
    arrays["holosoma_root_qvel"] = np.array(source_joint_vel[:, :ROOT_QVEL_WIDTH], copy=True)
    arrays["joint_pos"] = np.array(
        source_joint_pos[:, ROOT_QPOS_WIDTH:][:, plan.joint_permutation], copy=True
    )
    arrays["joint_vel"] = np.array(
        source_joint_vel[:, ROOT_QVEL_WIDTH:][:, plan.joint_permutation], copy=True
    )
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        arrays[key] = np.array(plan.source_arrays[key][:, plan.body_permutation], copy=True)

    arrays["joint_names"] = np.asarray(ISAAC_JOINT_NAMES, dtype=plan.source_arrays["joint_names"].dtype)
    arrays["body_names"] = np.asarray(ISAAC_BODY_NAMES, dtype=plan.source_arrays["body_names"].dtype)
    conversion_metadata = {
        "schema": "elf3_whole_body_tracking_npz_v1",
        "source_filename": plan.source_path.name,
        "source_sha256": plan.source_sha256,
        "source_layout": "holosoma_root_qpos_qvel_plus_29_joints",
        "target_layout": "isaac_runtime_order_29_joints_30_bodies",
        "numeric_transform": "slice_and_permutation_only",
        "quaternion_order": "wxyz_unchanged",
        "fps": plan.fps,
        "root_qpos_key": "holosoma_root_qpos",
        "root_qvel_key": "holosoma_root_qvel",
    }
    arrays["conversion_metadata_json"] = np.asarray(
        json.dumps(conversion_metadata, sort_keys=True, separators=(",", ":")), dtype=np.str_
    )
    return arrays


def deterministic_npz_write(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write an allow_pickle=False-compatible NPZ with deterministic ZIP metadata."""

    with ZipFile(path, mode="w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for key, value in arrays.items():
            require("/" not in key and "\\" not in key, f"Invalid NPZ key: {key}")
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asarray(value), allow_pickle=False)
            info = ZipInfo(filename=f"{key}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue(), compress_type=ZIP_DEFLATED, compresslevel=6)


def verify_converted_file(path: Path, plan: ConversionPlan) -> str:
    with np.load(path, allow_pickle=False) as archive:
        output = {key: np.array(archive[key], copy=True) for key in archive.files}

    require(tuple(output["joint_names"].tolist()) == ISAAC_JOINT_NAMES, f"{path}: joint order verification failed")
    require(tuple(output["body_names"].tolist()) == ISAAC_BODY_NAMES, f"{path}: body order verification failed")
    require(output["joint_pos"].shape == (plan.frames, len(ISAAC_JOINT_NAMES)), f"{path}: wrong joint_pos shape")
    require(output["joint_vel"].shape == (plan.frames, len(ISAAC_JOINT_NAMES)), f"{path}: wrong joint_vel shape")

    reconstructed_joint_pos_tail = np.empty_like(plan.source_arrays["joint_pos"][:, ROOT_QPOS_WIDTH:])
    reconstructed_joint_pos_tail[:, plan.joint_permutation] = output["joint_pos"]
    reconstructed_joint_pos = np.concatenate(
        [output["holosoma_root_qpos"], reconstructed_joint_pos_tail], axis=1
    )
    reconstructed_joint_vel_tail = np.empty_like(plan.source_arrays["joint_vel"][:, ROOT_QVEL_WIDTH:])
    reconstructed_joint_vel_tail[:, plan.joint_permutation] = output["joint_vel"]
    reconstructed_joint_vel = np.concatenate(
        [output["holosoma_root_qvel"], reconstructed_joint_vel_tail], axis=1
    )
    require(
        bitwise_equal(reconstructed_joint_pos, plan.source_arrays["joint_pos"]),
        f"{path}: inverse joint_pos conversion is not byte-for-byte lossless",
    )
    require(
        bitwise_equal(reconstructed_joint_vel, plan.source_arrays["joint_vel"]),
        f"{path}: inverse joint_vel conversion is not byte-for-byte lossless",
    )

    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        reconstructed = np.empty_like(plan.source_arrays[key])
        reconstructed[:, plan.body_permutation] = output[key]
        require(
            bitwise_equal(reconstructed, plan.source_arrays[key]),
            f"{path}: inverse {key} conversion is not byte-for-byte lossless",
        )

    changed_keys = {
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
    }
    for key, source_value in plan.source_arrays.items():
        if key not in changed_keys:
            require(bitwise_equal(output[key], source_value), f"{path}: auxiliary field changed: {key}")

    return sha256_file(path)


def convert(plan: ConversionPlan) -> str:
    destination = plan.destination_path
    require(not destination.exists(), f"Refusing to overwrite existing output: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.stem}_", suffix=".tmp.npz", delete=False
        ) as stream:
            temporary_path = Path(stream.name)
        deterministic_npz_write(temporary_path, build_output_arrays(plan))
        output_sha256 = verify_converted_file(temporary_path, plan)
        # Atomic and non-overwriting because the temporary file is on the same
        # filesystem and os.link fails if destination already exists.
        os.link(temporary_path, destination)
        return output_sha256
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def discover_inputs(input_path: Path, recursive: bool) -> tuple[Path, list[Path]]:
    input_path = input_path.resolve()
    require(input_path.exists(), f"Input does not exist: {input_path}")
    if input_path.is_file():
        require(input_path.suffix.lower() == ".npz", f"Input file must have .npz suffix: {input_path}")
        return input_path.parent, [input_path]
    require(input_path.is_dir(), f"Input is neither a file nor a directory: {input_path}")
    pattern = "**/*.npz" if recursive else "*.npz"
    inputs = sorted(path.resolve() for path in input_path.glob(pattern) if path.is_file())
    require(bool(inputs), f"No NPZ files found in {input_path} (recursive={recursive})")
    return input_path, inputs


def main() -> int:
    args = parse_args()
    input_root, source_paths = discover_inputs(args.input, args.recursive)
    output_dir = args.output_dir.resolve()

    if args.input.resolve().is_dir():
        try:
            output_dir.relative_to(args.input.resolve())
        except ValueError:
            pass
        else:
            raise ConversionError("Output directory must not be inside the input directory")

    plans = []
    for source_path in source_paths:
        relative_path = source_path.relative_to(input_root)
        destination_path = output_dir / relative_path
        require(source_path != destination_path, f"Input and output are the same file: {source_path}")
        require(not destination_path.exists(), f"Refusing to overwrite existing output: {destination_path}")
        plans.append(validate_input(source_path, destination_path))

    for plan in plans:
        duration = (plan.frames - 1) / plan.fps
        if args.dry_run:
            print(
                f"[VALID] {plan.source_path} -> {plan.destination_path} | "
                f"frames={plan.frames}, fps={plan.fps:g}, duration={duration:.6f}s, "
                f"source_sha256={plan.source_sha256}"
            )
        else:
            output_sha256 = convert(plan)
            print(
                f"[CONVERTED] {plan.source_path} -> {plan.destination_path} | "
                f"frames={plan.frames}, fps={plan.fps:g}, duration={duration:.6f}s, "
                f"source_sha256={plan.source_sha256}, output_sha256={output_sha256}"
            )

    action = "validated" if args.dry_run else "converted and byte-for-byte inverse-verified"
    print(f"[DONE] {len(plans)} file(s) {action}; source files were not modified.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ConversionError as error:
        raise SystemExit(f"[ERROR] {error}") from error
