#!/usr/bin/env python3
"""Retarget ELF3 down-roll experts to a higher, ground-mounted platform.

This is an offline data repair, not a runtime reference warp.  The source
joint motion, root XY/orientation, and the complete ground-roll/recovery suffix
are immutable.  Only root Z is raised where the source sole geometry would
intersect the requested platform top.  A constrained smooth fit makes that
correction continuous, keeps it above the exact non-penetration lower bound,
and returns it to zero before the first ground contact.

The complete WBT state is regenerated with MuJoCo FK/differentiation.  Source
files are never overwritten, and the output directory is committed only after
all clips and sidecars pass verification.

Example::

    conda run -p /home/kvoy/.holosoma_deps/miniconda3/envs/hsretargeting \
      python scripts/rebuild_elf3_down_roll_platform_height.py \
        data/motions/elf3/down_roll_50hz \
        data/motions/elf3/down_roll_50hz_platform_0p66_v1 \
        --holosoma-root /home/kvoy/Desktop/PHP-kvoy/holosoma \
        --platform-height 0.66
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


PATCH_SCHEMA = "elf3_down_roll_platform_height_v1"
EXPECTED_FPS = 50.0
EXPECTED_QPOS_WIDTH = 36
EXPECTED_QVEL_WIDTH = 35
EXPECTED_BODY_COUNT = 30
FOOT_BODY_NAMES = ("l_ankle_x_link", "r_ankle_x_link")
# Exact sole geometry used by the Isaac task's physical foot-surface checks.
SOLE_CORNERS_B = np.asarray(
    (
        (-0.09, -0.04, -0.041),
        (-0.09, 0.04, -0.041),
        (0.15, -0.04, -0.041),
        (0.15, 0.04, -0.041),
    ),
    dtype=np.float64,
)
MAX_ROOT_Z_CORRECTION_SPEED_MPS = 0.30
MAX_ROOT_Z_CORRECTION_ACCELERATION_MPS2 = 3.0
TRAJECTORY_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
WBT_BASE_KEYS = {
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_names",
    "joint_names",
    "metadata_json",
}
REQUIRED_KEYS = WBT_BASE_KEYS | {"source_frame", "source_generated"}


class RebuildError(ValueError):
    """Raised when safe contact-consistent rebuilding cannot be proven."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RebuildError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", type=Path, help="Original Holosoma-layout down-roll directory.")
    parser.add_argument("output_dir", type=Path, help="New, currently non-existent output directory.")
    parser.add_argument("--holosoma-root", type=Path, required=True, help="Checked Holosoma repository root.")
    parser.add_argument("--platform-height", type=float, default=0.66, help="Target platform top in metres.")
    parser.add_argument(
        "--sole-clearance",
        type=float,
        default=0.002,
        help="Minimum sole-corner clearance above the platform top (default: 2 mm).",
    )
    parser.add_argument(
        "--ground-contact-height",
        type=float,
        default=0.01,
        help="Sole height identifying the first post-platform ground contact (default: 1 cm).",
    )
    parser.add_argument(
        "--smoothness-weight",
        type=float,
        default=1000.0,
        help="Second-difference regularization for the constrained Z correction.",
    )
    parser.add_argument(
        "--max-extra-correction",
        type=float,
        default=0.01,
        help="Maximum correction beyond platform-height delta (default: 1 cm).",
    )
    parser.add_argument("--model-xml", type=Path, default=None, help="Override the ELF3 MuJoCo model XML.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and solve without writing outputs.")
    return parser.parse_args()


def load_archive(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise RebuildError(f"Cannot safely read {path}: {error}") from error
    missing = sorted(REQUIRED_KEYS - set(arrays))
    require(not missing, f"{path}: missing WBT fields: {missing}")
    require(all(value.dtype.kind != "O" for value in arrays.values()), f"{path}: object arrays are forbidden")
    fps = np.asarray(arrays["fps"])
    require(fps.size == 1 and abs(float(fps.reshape(-1)[0]) - EXPECTED_FPS) <= 1.0e-12, f"{path}: expected 50 Hz")
    qpos = arrays["joint_pos"]
    require(
        qpos.ndim == 2 and qpos.shape[1] == EXPECTED_QPOS_WIDTH and qpos.shape[0] >= 3,
        f"{path}: invalid qpos shape {qpos.shape}",
    )
    require(arrays["joint_vel"].shape == (qpos.shape[0], EXPECTED_QVEL_WIDTH), f"{path}: invalid qvel shape")
    for name, width in (("body_pos_w", 3), ("body_quat_w", 4), ("body_lin_vel_w", 3), ("body_ang_vel_w", 3)):
        require(
            arrays[name].shape == (qpos.shape[0], EXPECTED_BODY_COUNT, width),
            f"{path}: invalid {name} shape {arrays[name].shape}",
        )
    for name in TRAJECTORY_KEYS:
        require(
            arrays[name].dtype.kind == "f" and np.isfinite(arrays[name]).all(),
            f"{path}: invalid trajectory {name}",
        )
    require(arrays["joint_names"].shape == (29,), f"{path}: expected 29 joint names")
    require(arrays["body_names"].shape == (EXPECTED_BODY_COUNT,), f"{path}: expected 30 body names")
    require(arrays["source_frame"].shape == (qpos.shape[0],), f"{path}: invalid source_frame")
    require(arrays["source_generated"].shape == (qpos.shape[0],), f"{path}: invalid source_generated")
    return arrays


def parse_metadata(path: Path, arrays: dict[str, np.ndarray], *, allow_patch: bool = False) -> dict[str, Any]:
    raw = np.asarray(arrays["metadata_json"])
    require(raw.size == 1, f"{path}: metadata_json must be scalar")
    try:
        metadata = json.loads(str(raw.reshape(-1)[0]))
    except (TypeError, json.JSONDecodeError) as error:
        raise RebuildError(f"{path}: invalid metadata_json: {error}") from error
    require(isinstance(metadata, dict), f"{path}: metadata must be an object")
    require(metadata.get("expert_role") == "down_roll", f"{path}: not a down-roll expert")
    if not allow_patch:
        require("platform_height_patch" not in metadata, f"{path}: already contains a platform-height patch")
    try:
        center = np.asarray(metadata["terrain_center_xyz"], dtype=np.float64)
        size = np.asarray(metadata["terrain_size_xyz"], dtype=np.float64)
        yaw = float(metadata["terrain_yaw_degrees"])
    except (KeyError, TypeError, ValueError) as error:
        raise RebuildError(f"{path}: invalid terrain metadata: {error}") from error
    require(center.shape == (3,) and size.shape == (3,), f"{path}: terrain vectors must have three elements")
    require(
        np.isfinite(center).all() and np.isfinite(size).all() and np.all(size > 0.0),
        f"{path}: invalid terrain geometry",
    )
    require(abs(yaw) <= 1.0e-12, f"{path}: v1 supports the reviewed zero-yaw down-roll set only")
    require(abs(center[2] - 0.5 * size[2]) <= 1.0e-9, f"{path}: platform must be ground-mounted")
    return metadata


def import_build_dependencies(holosoma_root: Path):
    source_root = holosoma_root.expanduser().resolve() / "src" / "motionmatching"
    require(source_root.is_dir(), f"Holosoma motionmatching source is missing: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        import mujoco
        from motionmatching.terrain import write_platform_terrain_obj, write_platform_urdf
        from motionmatching.wbt import export_qpos_wbt
        from scipy.optimize import minimize
    except ImportError as error:
        raise RebuildError("Run with the hsretargeting environment containing MuJoCo and SciPy.") from error
    return mujoco, minimize, export_qpos_wbt, write_platform_terrain_obj, write_platform_urdf


def load_local_converter():
    path = Path(__file__).with_name("convert_elf3_holosoma2wbt_npz.py")
    spec = importlib.util.spec_from_file_location("elf3_wbt_converter", path)
    require(spec is not None and spec.loader is not None, f"Cannot load converter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def sole_points_world(model: Any, qpos: np.ndarray) -> np.ndarray:
    """Return [frames, feet, corners, xyz] sole points from MuJoCo FK."""

    body_ids = tuple(int(model.body(name).id) for name in FOOT_BODY_NAMES)
    require(all(body_id > 0 for body_id in body_ids), "ELF3 foot bodies are absent from the model")
    import mujoco

    mj_data = mujoco.MjData(model)
    points = np.empty((qpos.shape[0], len(body_ids), SOLE_CORNERS_B.shape[0], 3), dtype=np.float64)
    for frame, pose in enumerate(qpos):
        mj_data.qpos[:] = pose
        mujoco.mj_forward(model, mj_data)
        for foot, body_id in enumerate(body_ids):
            rotation = mj_data.xmat[body_id].reshape(3, 3)
            points[frame, foot] = mj_data.xpos[body_id] + SOLE_CORNERS_B @ rotation.T
    return points


def penetration_lower_bound(
    sole_points: np.ndarray,
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    target_top: float,
    clearance: float,
) -> np.ndarray:
    lower_xy = center_xy - 0.5 * size_xy
    upper_xy = center_xy + 0.5 * size_xy
    inside = np.all((sole_points[..., :2] >= lower_xy) & (sole_points[..., :2] <= upper_xy), axis=-1)
    required = np.zeros(sole_points.shape[0], dtype=np.float64)
    for frame in range(sole_points.shape[0]):
        if np.any(inside[frame]):
            minimum = float(np.min(sole_points[frame, ..., 2][inside[frame]]))
            required[frame] = max(0.0, target_top + clearance - minimum)
    return required


def first_ground_contact_frame(
    sole_points: np.ndarray,
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    after_frame: int,
    ground_contact_height: float,
) -> int:
    lower_xy = center_xy - 0.5 * size_xy
    upper_xy = center_xy + 0.5 * size_xy
    inside = np.all((sole_points[..., :2] >= lower_xy) & (sole_points[..., :2] <= upper_xy), axis=-1)
    for frame in range(after_frame + 1, sole_points.shape[0]):
        entirely_off_platform = not np.any(inside[frame])
        if entirely_off_platform and float(np.min(sole_points[frame, ..., 2])) <= ground_contact_height:
            return frame
    raise RebuildError("No post-platform ground-contact frame was found")


def solve_smooth_correction(
    lower_bound: np.ndarray,
    landing_frame: int,
    max_correction: float,
    smoothness_weight: float,
    minimize: Any,
) -> np.ndarray:
    """Solve a bounded convex fit that majorizes the penetration correction."""

    require(lower_bound.ndim == 1 and lower_bound.size >= 3, "Invalid correction lower bound")
    require(0 < landing_frame < lower_bound.size, "Invalid ground-contact frame")
    require(smoothness_weight > 0.0 and max_correction > 0.0, "Invalid smoothing parameters")
    require(float(np.max(lower_bound)) <= max_correction + 1.0e-12, "Required correction exceeds the safety limit")
    lower = lower_bound.copy()
    upper = np.full(lower.shape, max_correction, dtype=np.float64)
    # The source start and complete ground-contact suffix remain exact.
    require(lower[0] <= 1.0e-12, "The first frame already penetrates the requested platform")
    require(float(np.max(lower[landing_frame:])) <= 1.0e-12, "Platform correction overlaps ground contact")
    lower[0] = upper[0] = 0.0
    lower[landing_frame:] = 0.0
    upper[landing_frame:] = 0.0

    def objective(values: np.ndarray) -> float:
        curvature = np.diff(values, n=2)
        return 0.5 * float(np.sum(np.square(values - lower_bound))) + 0.5 * smoothness_weight * float(
            np.sum(np.square(curvature))
        )

    def gradient(values: np.ndarray) -> np.ndarray:
        result = values - lower_bound
        curvature = np.diff(values, n=2)
        result[:-2] += smoothness_weight * curvature
        result[1:-1] -= 2.0 * smoothness_weight * curvature
        result[2:] += smoothness_weight * curvature
        return result

    solved = minimize(
        objective,
        lower.copy(),
        jac=gradient,
        bounds=list(zip(lower, upper, strict=True)),
        method="L-BFGS-B",
        options={"ftol": 1.0e-15, "gtol": 1.0e-10, "maxiter": 10000, "maxls": 50},
    )
    require(bool(solved.success), f"Constrained Z smoothing failed: {solved.message}")
    correction = np.asarray(solved.x, dtype=np.float64)
    require(np.isfinite(correction).all(), "Z correction contains NaN or Inf")
    require(float(np.min(correction - lower_bound)) >= -1.0e-9, "Smoothed correction violates non-penetration")
    require(np.count_nonzero(correction[landing_frame:]) == 0, "Ground suffix correction is not exactly zero")
    speed = np.diff(correction) * EXPECTED_FPS
    acceleration = np.diff(correction, n=2) * EXPECTED_FPS**2
    require(
        float(np.max(np.abs(speed))) <= MAX_ROOT_Z_CORRECTION_SPEED_MPS,
        "Z correction is too fast for a physically credible retarget",
    )
    require(
        float(np.max(np.abs(acceleration))) <= MAX_ROOT_Z_CORRECTION_ACCELERATION_MPS2,
        "Z correction acceleration exceeds the physical continuity limit",
    )
    return correction


def retarget_qpos(
    source: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    model: Any,
    target_height: float,
    sole_clearance: float,
    ground_contact_height: float,
    smoothness_weight: float,
    max_extra_correction: float,
    minimize: Any,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    source_qpos = np.asarray(arrays["joint_pos"], dtype=np.float64)
    center = np.asarray(metadata["terrain_center_xyz"], dtype=np.float64)
    size = np.asarray(metadata["terrain_size_xyz"], dtype=np.float64)
    source_height = float(size[2])
    require(target_height > source_height, f"{source}: target height must exceed source height {source_height}")
    require(0.0 <= sole_clearance <= 0.01, "sole-clearance must be between 0 and 1 cm")
    require(0.0 <= ground_contact_height <= 0.03, "ground-contact-height must be between 0 and 3 cm")
    points = sole_points_world(model, source_qpos)
    lower = penetration_lower_bound(points, center[:2], size[:2], target_height, sole_clearance)
    active = np.flatnonzero(lower > 0.0)
    require(active.size > 0, f"{source}: no platform-height correction is required")
    landing = first_ground_contact_frame(points, center[:2], size[:2], int(active[-1]), ground_contact_height)
    require(landing - int(active[-1]) >= 3, f"{source}: insufficient airborne release before ground contact")
    maximum = target_height - source_height + max_extra_correction
    correction = solve_smooth_correction(lower, landing, maximum, smoothness_weight, minimize)
    rebuilt = source_qpos.copy()
    rebuilt[:, 2] += correction
    require(np.array_equal(rebuilt[:, :2], source_qpos[:, :2]), f"{source}: root XY changed")
    require(np.array_equal(rebuilt[:, 3:], source_qpos[:, 3:]), f"{source}: root orientation or joints changed")
    require(np.array_equal(rebuilt[landing:], source_qpos[landing:]), f"{source}: ground suffix changed")
    return rebuilt, correction, landing, source_height


def patched_metadata(
    metadata: dict[str, Any],
    source: Path,
    final_output: Path,
    model_xml: Path,
    source_height: float,
    target_height: float,
    correction: np.ndarray,
    landing_frame: int,
    sole_clearance: float,
    smoothness_weight: float,
    mujoco_version: str,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    result = dict(metadata)
    center = np.asarray(result["terrain_center_xyz"], dtype=np.float64)
    size = np.asarray(result["terrain_size_xyz"], dtype=np.float64)
    center[2] = 0.5 * target_height
    size[2] = target_height
    result["terrain_center_xyz"] = center.tolist()
    result["terrain_size_xyz"] = size.tolist()
    result["terrain_obj"] = str(final_output.with_name(f"{final_output.stem}_terrain.obj").resolve())
    result["platform_urdf"] = str(final_output.with_name(f"{final_output.stem}_platform.urdf").resolve())
    active = np.flatnonzero(correction > 1.0e-8)
    result["platform_height_patch"] = {
        "schema": PATCH_SCHEMA,
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "model_xml": str(model_xml.resolve()),
        "model_xml_sha256": sha256_file(model_xml),
        "mujoco_version": mujoco_version,
        "source_platform_height_m": source_height,
        "target_platform_height_m": target_height,
        "sole_clearance_m": sole_clearance,
        "sole_corners_body_m": SOLE_CORNERS_B.tolist(),
        "smoothness_weight": smoothness_weight,
        "first_corrected_frame": int(active[0]),
        "last_corrected_frame": int(active[-1]),
        "ground_contact_frame": landing_frame,
        "maximum_root_z_correction_m": float(np.max(correction)),
    }
    return result, center, size


def verify_output(
    source: Path,
    output: Path,
    source_arrays: dict[str, np.ndarray],
    correction: np.ndarray,
    landing_frame: int,
    target_height: float,
    model: Any,
    converter: Any,
) -> None:
    converter.validate_input(output, output.with_suffix(".processed.npz"))
    rebuilt = load_archive(output)
    np.testing.assert_array_equal(rebuilt["joint_pos"][:, :2], source_arrays["joint_pos"][:, :2])
    np.testing.assert_array_equal(rebuilt["joint_pos"][:, 3:], source_arrays["joint_pos"][:, 3:])
    np.testing.assert_allclose(
        rebuilt["joint_pos"][:, 2], source_arrays["joint_pos"][:, 2] + correction, rtol=0.0, atol=1.0e-12
    )
    for key in TRAJECTORY_KEYS:
        np.testing.assert_array_equal(rebuilt[key][landing_frame:], source_arrays[key][landing_frame:])
    for key, source_values in source_arrays.items():
        if key not in WBT_BASE_KEYS:
            np.testing.assert_array_equal(rebuilt[key], source_values)
    metadata = parse_metadata(output, rebuilt, allow_patch=True)
    require(abs(float(metadata["terrain_size_xyz"][2]) - target_height) <= 1.0e-12, f"{output}: wrong target height")
    points = sole_points_world(model, rebuilt["joint_pos"])
    center = np.asarray(metadata["terrain_center_xyz"], dtype=np.float64)
    size = np.asarray(metadata["terrain_size_xyz"], dtype=np.float64)
    residual = penetration_lower_bound(points, center[:2], size[:2], target_height, 0.0)
    require(float(np.max(residual[:landing_frame])) <= 1.0e-8, f"{output}: sole still penetrates the 0.66 m platform")


def rebuild_one(
    source: Path,
    destination: Path,
    final_destination: Path,
    model: Any,
    model_xml: Path,
    mujoco_version: str,
    target_height: float,
    sole_clearance: float,
    ground_contact_height: float,
    smoothness_weight: float,
    max_extra_correction: float,
    minimize: Any,
    export_qpos_wbt: Any,
    write_platform_terrain_obj: Any,
    write_platform_urdf: Any,
    converter: Any,
    write: bool,
) -> tuple[np.ndarray, int]:
    arrays = load_archive(source)
    metadata = parse_metadata(source, arrays)
    rebuilt_qpos, correction, landing, source_height = retarget_qpos(
        source,
        arrays,
        metadata,
        model,
        target_height,
        sole_clearance,
        ground_contact_height,
        smoothness_weight,
        max_extra_correction,
        minimize,
    )
    if not write:
        return correction, landing
    output_metadata, center, size = patched_metadata(
        metadata,
        source,
        final_destination,
        model_xml,
        source_height,
        target_height,
        correction,
        landing,
        sole_clearance,
        smoothness_weight,
        mujoco_version,
    )
    write_platform_terrain_obj(
        destination.with_name(f"{destination.stem}_terrain.obj"),
        center,
        size,
        float(metadata["terrain_yaw_degrees"]),
        10.0,
    )
    write_platform_urdf(
        destination.with_name(f"{destination.stem}_platform.urdf"), center, size, float(metadata["terrain_yaw_degrees"])
    )
    extras = {key: np.array(value, copy=True) for key, value in arrays.items() if key not in WBT_BASE_KEYS}
    extras["platform_height_retarget_offset_z"] = correction
    extras["platform_height_retarget_active"] = correction > 1.0e-8
    export_qpos_wbt(
        rebuilt_qpos,
        float(arrays["fps"].reshape(-1)[0]),
        model_xml,
        tuple(str(name) for name in arrays["joint_names"].tolist()),
        destination,
        metadata=output_metadata,
        extra_arrays=extras,
    )
    verify_output(source, destination, arrays, correction, landing, target_height, model, converter)
    return correction, landing


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require(input_dir.is_dir(), f"Input directory does not exist: {input_dir}")
    require(output_dir != input_dir and input_dir not in output_dir.parents, "Output must be separate from input")
    require(not output_dir.exists(), f"Refusing to overwrite output directory: {output_dir}")
    require(np.isfinite(args.platform_height) and args.platform_height > 0.0, "Invalid target platform height")
    sources = sorted(path.resolve() for path in input_dir.glob("*.npz") if path.is_file())
    require(sources, f"No NPZ files found in {input_dir}")

    holosoma_root = args.holosoma_root.expanduser().resolve()
    default_model = holosoma_root / "src/holosoma_retargeting/holosoma_retargeting/models/elf3/elf3_29dof.xml"
    model_xml = (args.model_xml or default_model).expanduser().resolve()
    require(model_xml.is_file(), f"MuJoCo model XML does not exist: {model_xml}")
    mujoco, minimize, export_qpos_wbt, write_platform_terrain_obj, write_platform_urdf = import_build_dependencies(
        holosoma_root
    )
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    converter = load_local_converter()

    if args.dry_run:
        reports: dict[str, tuple[np.ndarray, int]] = {}
        for source in sources:
            correction, landing = rebuild_one(
                source, output_dir / source.name, output_dir / source.name, model, model_xml, str(mujoco.__version__),
                args.platform_height, args.sole_clearance, args.ground_contact_height, args.smoothness_weight,
                args.max_extra_correction, minimize, export_qpos_wbt, write_platform_terrain_obj, write_platform_urdf,
                converter, False,
            )
            reports[source.name] = (correction, landing)
            active = np.flatnonzero(correction > 1.0e-8)
            print(
                f"[DRY RUN] {source.name}: correction={active[0]}..{active[-1]}, "
                f"landing={landing}, max={np.max(correction):.6f} m"
            )
        verify_mirror_pairs(reports)
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    committed = False
    reports: dict[str, tuple[np.ndarray, int]] = {}
    try:
        for source in sources:
            destination = staging / source.name
            correction, landing = rebuild_one(
                source, destination, output_dir / source.name, model, model_xml, str(mujoco.__version__),
                args.platform_height, args.sole_clearance, args.ground_contact_height, args.smoothness_weight,
                args.max_extra_correction, minimize, export_qpos_wbt, write_platform_terrain_obj, write_platform_urdf,
                converter, True,
            )
            reports[source.name] = (correction, landing)
            print(f"[OK] {source.name}: landing={landing}, max_correction={np.max(correction):.6f} m")
        verify_mirror_pairs(reports)
        require(not output_dir.exists(), f"Output directory appeared during generation: {output_dir}")
        os.rename(staging, output_dir)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(staging, ignore_errors=True)
    print(f"[DONE] Rebuilt {len(sources)} clips in {output_dir}")
    return 0


def verify_mirror_pairs(reports: dict[str, tuple[np.ndarray, int]]) -> None:
    for name, (correction, landing) in reports.items():
        if "__mirrored" in name:
            continue
        mirrored = name.replace(".npz", "__mirrored.npz")
        require(mirrored in reports, f"Missing mirrored partner for {name}")
        mirror_correction, mirror_landing = reports[mirrored]
        require(landing == mirror_landing, f"Mirror landing-frame mismatch for {name}")
        require(
            np.allclose(correction, mirror_correction, rtol=0.0, atol=1.0e-10),
            f"Mirror correction mismatch for {name}",
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RebuildError, AssertionError) as error:
        raise SystemExit(f"ERROR: {error}") from error
