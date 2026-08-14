#!/usr/bin/env python3
"""Build a separate ELF3 climb expert set with a natural default start.

The reviewed 50 Hz climb experts have a generated 63-frame prefix.  That
prefix starts in a pose that is not the natural standing pose configured by
``ELF3_CFG.init_state``.  Replacing only joint positions in a processed Isaac
Lab NPZ would make its body pose and velocity references inconsistent, so this
tool operates on the original Holosoma WBT layout instead:

1. It keeps every frame from the first real source frame onward unchanged.
2. It replaces the generated prefix with a default stand and a single-foot
   contact-constrained transition into that first real frame.
3. It recomputes the complete WBT state through MuJoCo FK and velocity
   differentiation, then verifies that every trajectory array after the
   prefix is byte-for-byte equal to the input.

The input is never modified.  Output is written to a new directory only after
all clips have been generated and verified successfully.  Convert that new
directory with ``convert_elf3_holosoma2wbt_npz.py`` before training.

Example:

    conda run -p /home/kvoy/.holosoma_deps/miniconda3/envs/hsretargeting \\
      python scripts/rebuild_elf3_climb_default_start.py \\
        data/motions/elf3/climb_50hz \\
        data/motions/elf3/climb_50hz_default_start_v1 \\
        --holosoma-root /home/kvoy/Desktop/PHP-kvoy/holosoma
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


PATCH_SCHEMA = "elf3_climb_default_start_v1"
EXPECTED_FPS = 50.0
ROOT_QPOS_WIDTH = 7
DEFAULT_ROOT_QPOS = np.asarray((0.0, 0.0, 1.05, 1.0, 0.0, 0.0, 0.0), dtype=np.float64)

# This is exactly ELF3_CFG.init_state.joint_pos, expressed in the original
# Holosoma/MuJoCo joint-name layout.  The root is later translated and yawed
# to retain the raw clip's support-foot location and approach direction.
DEFAULT_JOINT_POSITIONS = {
    "waist_y_joint": 0.0,
    "waist_x_joint": 0.0,
    "waist_z_joint": 0.0,
    "l_hip_y_joint": -0.3,
    "l_hip_x_joint": 0.0,
    "l_hip_z_joint": 0.0,
    "l_knee_y_joint": 0.6,
    "l_ankle_y_joint": -0.3,
    "l_ankle_x_joint": 0.0,
    "r_hip_y_joint": -0.3,
    "r_hip_x_joint": 0.0,
    "r_hip_z_joint": 0.0,
    "r_knee_y_joint": 0.6,
    "r_ankle_y_joint": -0.3,
    "r_ankle_x_joint": 0.0,
    "l_shoulder_y_joint": 0.2,
    "l_shoulder_x_joint": 0.2,
    "l_shoulder_z_joint": 0.0,
    "l_elbow_y_joint": 0.6,
    "l_wrist_x_joint": 0.0,
    "l_wrist_y_joint": 0.0,
    "l_wrist_z_joint": 0.0,
    "r_shoulder_y_joint": 0.2,
    "r_shoulder_x_joint": -0.2,
    "r_shoulder_z_joint": 0.0,
    "r_elbow_y_joint": 0.6,
    "r_wrist_x_joint": 0.0,
    "r_wrist_y_joint": 0.0,
    "r_wrist_z_joint": 0.0,
}

_REQUIRED_WBT_KEYS = {
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
    "source_frame",
    "source_generated",
}
_TRAJECTORY_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)
_WBT_BASE_KEYS = {
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
_SIDECAR_SUFFIXES = ("_terrain.obj", "_platform.urdf")


class RebuildError(ValueError):
    """Raised when a source clip cannot be rebuilt without changing its contract."""


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
    parser.add_argument("input_dir", type=Path, help="Directory containing original Holosoma-layout climb NPZ files.")
    parser.add_argument("output_dir", type=Path, help="New, currently non-existent output directory.")
    parser.add_argument(
        "--holosoma-root",
        type=Path,
        required=True,
        help="Root of the checked Holosoma checkout that provides MuJoCo contact-transition utilities.",
    )
    parser.add_argument(
        "--model-xml",
        type=Path,
        default=None,
        help="ELF3 MuJoCo model; defaults to the checked Holosoma ELF3 29-DoF model.",
    )
    parser.add_argument(
        "--transition-config",
        type=Path,
        default=None,
        help="Holosoma motion-matching config; defaults to configs/elf3_php.json in --holosoma-root.",
    )
    parser.add_argument(
        "--transition-duration-s",
        type=float,
        default=0.8,
        help="Duration of the single-support start transition in seconds (default: 0.8).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all inputs and synthesize the prefix in memory without writing files.",
    )
    return parser.parse_args()


def load_archive(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except (OSError, ValueError) as error:
        raise RebuildError(f"Cannot read {path}: {error}") from error

    missing = sorted(_REQUIRED_WBT_KEYS - set(arrays))
    require(not missing, f"{path}: missing required WBT fields: {missing}")
    for name, array in arrays.items():
        require(array.dtype.kind != "O", f"{path}: object array is forbidden: {name}")

    fps = np.asarray(arrays["fps"])
    require(fps.size == 1, f"{path}: fps must be a scalar")
    require(np.issubdtype(fps.dtype, np.number), f"{path}: fps must be numeric")
    require(abs(float(fps.reshape(-1)[0]) - EXPECTED_FPS) <= 1.0e-12, f"{path}: expected exactly 50 Hz")

    qpos = arrays["joint_pos"]
    qvel = arrays["joint_vel"]
    require(qpos.ndim == 2 and qpos.shape[1] == 36 and qpos.shape[0] >= 2, f"{path}: invalid qpos shape {qpos.shape}")
    require(qvel.shape == (qpos.shape[0], 35), f"{path}: invalid qvel shape {qvel.shape}")
    for name in _TRAJECTORY_KEYS:
        values = arrays[name]
        require(values.dtype.kind == "f" and np.all(np.isfinite(values)), f"{path}: invalid numeric trajectory {name}")
    for name, width in (
        ("body_pos_w", 3),
        ("body_quat_w", 4),
        ("body_lin_vel_w", 3),
        ("body_ang_vel_w", 3),
    ):
        require(
            arrays[name].shape == (qpos.shape[0], 30, width),
            f"{path}: invalid {name} shape {arrays[name].shape}",
        )
    require(arrays["joint_names"].shape == (29,), f"{path}: expected 29 joint names")
    require(arrays["body_names"].shape == (30,), f"{path}: expected 30 body names")
    require(arrays["source_frame"].shape == (qpos.shape[0],), f"{path}: invalid source_frame shape")
    require(arrays["source_generated"].shape == (qpos.shape[0],), f"{path}: invalid source_generated shape")
    return arrays


def parse_metadata(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    metadata_raw = np.asarray(arrays["metadata_json"])
    require(metadata_raw.size == 1, f"{path}: metadata_json must be a scalar")
    try:
        metadata = json.loads(str(metadata_raw.reshape(-1)[0]))
    except (TypeError, json.JSONDecodeError) as error:
        raise RebuildError(f"{path}: invalid metadata_json: {error}") from error
    require(isinstance(metadata, dict), f"{path}: metadata_json must encode an object")
    require("default_start_patch" not in metadata, f"{path}: already contains a default-start patch")
    return metadata


def first_real_frame(path: Path, arrays: dict[str, np.ndarray]) -> int:
    source_frame = np.asarray(arrays["source_frame"])
    source_generated = np.asarray(arrays["source_generated"], dtype=bool)
    # The provenance flag is authoritative.  A resampled real trajectory can
    # repeat source-frame IDs, and generated tails can occur after it, so only
    # the first non-generated sample identifies the start of the raw climb.
    real_indices = np.flatnonzero(~source_generated)
    source_indices = np.flatnonzero(source_frame >= 0)
    require(real_indices.size > 0, f"{path}: no non-generated source frame was found")
    require(source_indices.size > 0, f"{path}: no non-negative source-frame ID was found")
    first_real = int(real_indices[0])
    require(
        first_real == int(source_indices[0]) and source_frame[first_real] >= 0,
        f"{path}: source_generated and source_frame disagree about the first real frame",
    )
    require(first_real >= 2, f"{path}: generated prefix is too short: {first_real} frames")
    require(np.all(source_frame[:first_real] < 0), f"{path}: generated prefix has a real source-frame entry")
    require(
        np.all(source_generated[:first_real]),
        f"{path}: generated prefix is not consistently marked source_generated",
    )
    return first_real


def default_qpos(joint_names: tuple[str, ...]) -> np.ndarray:
    missing = sorted(set(joint_names) - set(DEFAULT_JOINT_POSITIONS))
    unexpected = sorted(set(DEFAULT_JOINT_POSITIONS) - set(joint_names))
    require(not missing and not unexpected, f"ELF3 default joint-name mismatch: missing={missing}, unexpected={unexpected}")
    qpos = np.empty(ROOT_QPOS_WIDTH + len(joint_names), dtype=np.float64)
    qpos[:ROOT_QPOS_WIDTH] = DEFAULT_ROOT_QPOS
    qpos[ROOT_QPOS_WIDTH:] = [DEFAULT_JOINT_POSITIONS[name] for name in joint_names]
    return qpos


def terrain_entry_point(metadata: dict[str, Any], movement_axis_xy: tuple[float, float]) -> np.ndarray:
    try:
        center = np.asarray(metadata["terrain_center_xyz"], dtype=np.float64)
        size = np.asarray(metadata["terrain_size_xyz"], dtype=np.float64)
        yaw_degrees = float(metadata["terrain_yaw_degrees"])
    except (KeyError, TypeError, ValueError) as error:
        raise RebuildError(f"Missing or invalid terrain metadata: {error}") from error
    require(center.shape == (3,) and size.shape == (3,), "Terrain metadata must contain three-element center and size")
    require(np.all(np.isfinite(center)) and np.all(np.isfinite(size)) and np.all(size > 0.0), "Invalid terrain geometry")
    axis = np.asarray(movement_axis_xy, dtype=np.float64)
    require(axis.shape == (2,) and np.linalg.norm(axis) > 0.0, "Invalid traversal movement axis")
    yaw = np.deg2rad(yaw_degrees)
    world_axis = np.asarray(
        (np.cos(yaw) * axis[0] - np.sin(yaw) * axis[1], np.sin(yaw) * axis[0] + np.cos(yaw) * axis[1], 0.0)
    )
    source_length = float(abs(axis[0]) * size[0] + abs(axis[1]) * size[1])
    return center - 0.5 * source_length * world_axis


def import_holosoma_modules(holosoma_root: Path):
    holosoma_root = holosoma_root.expanduser().resolve()
    source_root = holosoma_root / "src" / "motionmatching"
    require(source_root.is_dir(), f"Holosoma motionmatching source is missing: {source_root}")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        import mujoco
        from motionmatching.config import load_config
        from motionmatching.contact_ik import generate_support_transition
        from motionmatching.skill_preprocessing import _align_facing_standing_pose
        from motionmatching.wbt import export_qpos_wbt
    except ImportError as error:
        raise RebuildError(
            "Could not import MuJoCo/Holosoma modules. Run this script with the "
            "hsretargeting environment and pass the correct --holosoma-root."
        ) from error
    return mujoco, load_config, generate_support_transition, _align_facing_standing_pose, export_qpos_wbt


def load_local_converter():
    converter_path = Path(__file__).with_name("convert_elf3_holosoma2wbt_npz.py")
    spec = importlib.util.spec_from_file_location("elf3_wbt_converter", converter_path)
    require(spec is not None and spec.loader is not None, f"Cannot load local converter: {converter_path}")
    module = importlib.util.module_from_spec(spec)
    # ``dataclass`` resolves postponed annotations through ``sys.modules``
    # while the converter module is executing, so register the dynamic module
    # before evaluating it just as Python's normal importer would do.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _find_traversal(config: Any, metadata: dict[str, Any], path: Path) -> Any:
    traversal_name = metadata.get("traversal")
    require(isinstance(traversal_name, str), f"{path}: metadata has no traversal name")
    matches = [traversal for traversal in config.traversals if traversal.name == traversal_name]
    require(len(matches) == 1, f"{path}: no unique Holosoma traversal for {traversal_name!r}")
    return matches[0]


def _copy_sidecars(source: Path, destination: Path) -> list[str]:
    copied = []
    for suffix in _SIDECAR_SUFFIXES:
        source_sidecar = source.with_name(f"{source.stem}{suffix}")
        if not source_sidecar.is_file():
            continue
        destination_sidecar = destination.with_name(f"{destination.stem}{suffix}")
        shutil.copy2(source_sidecar, destination_sidecar)
        require(
            source_sidecar.read_bytes() == destination_sidecar.read_bytes(),
            f"Failed to verify copied sidecar: {destination_sidecar}",
        )
        copied.append(suffix)
    return copied


def _patched_metadata(
    metadata: dict[str, Any],
    *,
    source: Path,
    output: Path,
    model_xml: Path,
    first_real: int,
    hold_frames: int,
    transition_frames: int,
    support_foot_index: int,
    transition_duration_s: float,
    mujoco_version: str,
    copied_sidecars: list[str],
) -> dict[str, Any]:
    result = dict(metadata)
    if "_terrain.obj" in copied_sidecars:
        result["terrain_obj"] = str(output.with_name(f"{output.stem}_terrain.obj").resolve())
    if "_platform.urdf" in copied_sidecars:
        result["platform_urdf"] = str(output.with_name(f"{output.stem}_platform.urdf").resolve())
    result["default_start_patch"] = {
        "schema": PATCH_SCHEMA,
        "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source),
        "model_xml": str(model_xml.resolve()),
        "model_xml_sha256": sha256_file(model_xml),
        "mujoco_version": mujoco_version,
        "first_real_frame": first_real,
        "default_hold_frames": hold_frames,
        "transition_internal_frames": transition_frames,
        "transition_duration_seconds": transition_duration_s,
        "support_foot_index": support_foot_index,
        "source_mirrored": bool(metadata.get("source_mirrored", False)),
    }
    return result


def build_replacement_qpos(
    *,
    source: Path,
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    model: Any,
    config: Any,
    align_facing_standing_pose: Any,
    generate_support_transition: Any,
    transition_duration_s: float,
) -> tuple[np.ndarray, int, int, int, Any]:
    require(transition_duration_s > 0.0, "transition-duration-s must be positive")
    qpos = np.asarray(arrays["joint_pos"], dtype=np.float64)
    joint_names = tuple(str(name) for name in arrays["joint_names"].tolist())
    require(qpos.shape[1] == model.nq, f"{source}: qpos width {qpos.shape[1]} does not match model.nq={model.nq}")
    first_real = first_real_frame(source, arrays)
    require(first_real + 1 < qpos.shape[0], f"{source}: first real frame has no successor")
    traversal = _find_traversal(config, metadata, source)
    source_mirrored = bool(metadata.get("source_mirrored", False))
    support_foot_index = int(traversal.climb_support_foot_index)
    if source_mirrored:
        support_foot_index = 1 - support_foot_index
    require(support_foot_index in (0, 1), f"{source}: invalid support-foot index")

    contact_site_ids = tuple(
        int(model.site(name).id) for name in config.transition.foot_contact_sites
    )
    require(all(site_id >= 0 for site_id in contact_site_ids), f"{source}: configured contact sites are absent")
    aligned_default = align_facing_standing_pose(
        model,
        default_qpos(joint_names),
        qpos[first_real],
        contact_site_ids,
        support_foot_index,
        terrain_entry_point(metadata, traversal.terrain.movement_axis_xy),
    )
    transition = generate_support_transition(
        model,
        aligned_default,
        qpos[first_real],
        qpos[first_real + 1],
        tuple(config.foot_bodies),
        float(arrays["fps"].reshape(-1)[0]),
        config.transition,
        support_foot_index,
        transition_duration_s,
    )
    require(
        np.array_equal(transition.qpos[0], aligned_default) and np.array_equal(transition.qpos[-1], qpos[first_real]),
        f"{source}: transition endpoints do not match the requested start and raw first frame",
    )
    transition_internal = np.asarray(transition.qpos[1:-1], dtype=np.float64)
    hold_frames = first_real - transition_internal.shape[0]
    require(
        hold_frames >= 2,
        f"{source}: prefix has only {first_real} frames, but transition needs {transition_internal.shape[0]}; "
        "choose a shorter --transition-duration-s",
    )
    prefix = np.concatenate(
        (np.repeat(aligned_default[None, :], hold_frames, axis=0), transition_internal), axis=0
    )
    require(prefix.shape == (first_real, model.nq), f"{source}: rebuilt prefix has wrong shape {prefix.shape}")
    rebuilt = np.concatenate((prefix, qpos[first_real:]), axis=0)
    require(
        np.array_equal(rebuilt[first_real:], qpos[first_real:]),
        f"{source}: the real reference trajectory would change",
    )
    return rebuilt, first_real, hold_frames, support_foot_index, transition


def verify_rebuilt_file(
    *,
    source: Path,
    output: Path,
    source_arrays: dict[str, np.ndarray],
    first_real: int,
    converter: Any,
) -> None:
    # This performs structural, body-root consistency, quaternion, name-order,
    # and finite-value checks against the exact contract used by the existing
    # source-to-processed converter.
    converter.validate_input(output, output.with_suffix(".processed.npz"))
    rebuilt = load_archive(output)
    for key in _TRAJECTORY_KEYS:
        require(
            np.array_equal(rebuilt[key][first_real:], source_arrays[key][first_real:]),
            f"{output}: {key} changed after first real frame {first_real}",
        )
    for key, source_values in source_arrays.items():
        if key not in _WBT_BASE_KEYS:
            require(np.array_equal(rebuilt[key], source_values), f"{output}: auxiliary field changed: {key}")


def rebuild_one(
    *,
    source: Path,
    destination: Path,
    final_destination: Path,
    model: Any,
    config: Any,
    mujoco_version: str,
    align_facing_standing_pose: Any,
    generate_support_transition: Any,
    export_qpos_wbt: Any,
    converter: Any,
    model_xml: Path,
    transition_duration_s: float,
    write: bool,
) -> tuple[int, int, int]:
    arrays = load_archive(source)
    metadata = parse_metadata(source, arrays)
    rebuilt_qpos, first_real, hold_frames, support_foot_index, transition = build_replacement_qpos(
        source=source,
        arrays=arrays,
        metadata=metadata,
        model=model,
        config=config,
        align_facing_standing_pose=align_facing_standing_pose,
        generate_support_transition=generate_support_transition,
        transition_duration_s=transition_duration_s,
    )
    if not write:
        return first_real, hold_frames, support_foot_index

    copied_sidecars = _copy_sidecars(source, destination)
    output_metadata = _patched_metadata(
        metadata,
        source=source,
        # Files are first written in a staging directory, but metadata must
        # record their eventual committed location rather than that temporary
        # path.
        output=final_destination,
        model_xml=model_xml,
        first_real=first_real,
        hold_frames=hold_frames,
        transition_frames=int(transition.qpos.shape[0] - 2),
        support_foot_index=support_foot_index,
        transition_duration_s=transition_duration_s,
        mujoco_version=mujoco_version,
        copied_sidecars=copied_sidecars,
    )
    extras = {key: np.array(value, copy=True) for key, value in arrays.items() if key not in _WBT_BASE_KEYS}
    export_qpos_wbt(
        rebuilt_qpos,
        float(arrays["fps"].reshape(-1)[0]),
        model_xml,
        tuple(str(name) for name in arrays["joint_names"].tolist()),
        destination,
        metadata=output_metadata,
        extra_arrays=extras,
    )
    verify_rebuilt_file(
        source=source,
        output=destination,
        source_arrays=arrays,
        first_real=first_real,
        converter=converter,
    )
    return first_real, hold_frames, support_foot_index


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    require(input_dir.is_dir(), f"Input directory does not exist: {input_dir}")
    require(output_dir != input_dir and input_dir not in output_dir.parents, "Output must not be inside input")
    require(not output_dir.exists(), f"Refusing to overwrite existing output directory: {output_dir}")
    source_paths = sorted(path.resolve() for path in input_dir.glob("*.npz") if path.is_file())
    require(source_paths, f"No NPZ files found directly under {input_dir}")

    holosoma_root = args.holosoma_root.expanduser().resolve()
    default_model_xml = (
        holosoma_root
        / "src/holosoma_retargeting/holosoma_retargeting/models/elf3/elf3_29dof.xml"
    )
    default_config = holosoma_root / "src/motionmatching/configs/elf3_php.json"
    model_xml = (args.model_xml or default_model_xml).expanduser().resolve()
    config_path = (args.transition_config or default_config).expanduser().resolve()
    require(model_xml.is_file(), f"MuJoCo model XML does not exist: {model_xml}")
    require(config_path.is_file(), f"Holosoma transition config does not exist: {config_path}")
    require(args.transition_duration_s > 0.0, "transition-duration-s must be positive")

    mujoco, load_config, generate_support_transition, align_facing_standing_pose, export_qpos_wbt = import_holosoma_modules(
        holosoma_root
    )
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    config = load_config(config_path, repository_root=holosoma_root)
    converter = load_local_converter()

    if args.dry_run:
        for source in source_paths:
            arrays = load_archive(source)
            metadata = parse_metadata(source, arrays)
            _, first_real, hold_frames, support_foot_index, _ = build_replacement_qpos(
                source=source,
                arrays=arrays,
                metadata=metadata,
                model=model,
                config=config,
                align_facing_standing_pose=align_facing_standing_pose,
                generate_support_transition=generate_support_transition,
                transition_duration_s=args.transition_duration_s,
            )
            print(
                f"[DRY RUN] {source.name}: first_real={first_real}, default_hold={hold_frames}, "
                f"support_foot={support_foot_index}"
            )
        return 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    committed = False
    try:
        for source in source_paths:
            destination = staging_dir / source.name
            first_real, hold_frames, support_foot_index = rebuild_one(
                source=source,
                destination=destination,
                final_destination=output_dir / source.name,
                model=model,
                config=config,
                mujoco_version=str(mujoco.__version__),
                align_facing_standing_pose=align_facing_standing_pose,
                generate_support_transition=generate_support_transition,
                export_qpos_wbt=export_qpos_wbt,
                converter=converter,
                model_xml=model_xml,
                transition_duration_s=args.transition_duration_s,
                write=True,
            )
            print(
                f"[OK] {source.name}: first_real={first_real}, default_hold={hold_frames}, "
                f"support_foot={support_foot_index}"
            )
        require(not output_dir.exists(), f"Output directory appeared during generation: {output_dir}")
        os.rename(staging_dir, output_dir)
        committed = True
    finally:
        if not committed:
            shutil.rmtree(staging_dir, ignore_errors=True)
    print(f"[DONE] Rebuilt {len(source_paths)} clips in {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RebuildError as error:
        raise SystemExit(f"ERROR: {error}") from error
