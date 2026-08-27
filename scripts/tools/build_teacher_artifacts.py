"""Build hash-pinned actor-only artifacts for three-skill distillation.

This utility deliberately does not start Isaac Sim.  It parses the current
ELF3 Python/URDF contracts as data, converts only audited teacher actors, writes
strict manifests, and compares each converted policy against its source before
publishing the bundle atomically.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import types
from typing import Any, Mapping, Sequence
import xml.etree.ElementTree as ET

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = (
    REPO_ROOT
    / "source"
    / "php_kvoy_reproduction"
    / "php_kvoy_reproduction"
)
CLIMB_CFG = PACKAGE_ROOT / "tasks" / "tracking" / "config" / "elf3" / "climb_env_cfg.py"
ELF3_ASSET_CFG = PACKAGE_ROOT / "assets" / "elf3" / "elf3.py"
DEFAULT_URDF = PACKAGE_ROOT / "assets" / "elf3" / "urdf" / "elf3.urdf"
DEFAULT_USD = PACKAGE_ROOT / "assets" / "elf3" / "usd" / "elf3.usd"

AUDITED_SOURCE_HASHES = {
    "locomotion": "7a5bb55e35de342796f832bec045baa9fc167bcdc66a1e4421704e3642b67ea5",
    "climb": "91bd94eb681add9fe4bff89f624dde0c46da88e8c6707e9859a7bad457dec480",
    "down_roll": "3bd0420f420a0ab863620067210c6d3fd2cc40ef9d68cff845ecb5d8b8692833",
}


def _install_dependency_light_namespace() -> None:
    """Import distillation helpers without executing task registration."""

    if "php_kvoy_reproduction" not in sys.modules:
        package = types.ModuleType("php_kvoy_reproduction")
        package.__path__ = [str(PACKAGE_ROOT)]
        sys.modules["php_kvoy_reproduction"] = package
    if "php_kvoy_reproduction.distillation" not in sys.modules:
        package = types.ModuleType("php_kvoy_reproduction.distillation")
        package.__path__ = [str(PACKAGE_ROOT / "distillation")]
        sys.modules["php_kvoy_reproduction.distillation"] = package


_install_dependency_light_namespace()

from php_kvoy_reproduction.distillation.teacher_artifact import (  # noqa: E402
    build_locomotion_teacher_artifact,
    build_tracking_teacher_artifact,
)
from php_kvoy_reproduction.distillation.teacher_manifest import (  # noqa: E402
    MANIFEST_SCHEMA_VERSION,
    NormalizerSpec,
    ObservationSchema,
    ObservationTerm,
    TeacherManifest,
    sha256_file,
)
from php_kvoy_reproduction.distillation.teacher_policy import (  # noqa: E402
    TeacherPolicy,
    build_actor_mlp,
)


def _parse_python(path: Path) -> ast.Module:
    if not path.is_file():
        raise FileNotFoundError(path)
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assignment_value(module: ast.Module, name: str) -> ast.expr:
    matches = [
        node.value
        for node in module.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and (
            (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            )
            or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name)
        )
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one assignment to {name!r} in {module.__dict__.get('filename', 'module')}")
    return matches[0]


def _literal_assignment(module: ast.Module, name: str) -> Any:
    try:
        return ast.literal_eval(_assignment_value(module, name))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name!r} must remain a literal contract") from exc


def _call_keyword(call: ast.expr, name: str) -> ast.expr:
    if not isinstance(call, ast.Call):
        raise ValueError(f"expected a constructor call while resolving {name!r}")
    matches = [keyword.value for keyword in call.keywords if keyword.arg == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {name!r} keyword")
    return matches[0]


def _current_action_contract() -> tuple[tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    climb_module = _parse_python(CLIMB_CFG)
    joint_order = tuple(_literal_assignment(climb_module, "ELF3_CLIMB_JOINT_NAMES"))
    scale_patterns = _literal_assignment(climb_module, "ELF3_CLIMB_ACTION_SCALE")
    if len(joint_order) != 29 or len(set(joint_order)) != 29:
        raise ValueError("current ELF3 action order must contain 29 unique joints")
    if not isinstance(scale_patterns, dict):
        raise ValueError("ELF3_CLIMB_ACTION_SCALE must remain a literal mapping")

    action_scale: list[float] = []
    for joint_name in joint_order:
        matches = [float(value) for pattern, value in scale_patterns.items() if re.fullmatch(pattern, joint_name)]
        if len(matches) != 1:
            raise ValueError(
                f"joint {joint_name!r} must match exactly one current action-scale pattern, got {len(matches)}"
            )
        if matches[0] <= 0.0:
            raise ValueError(f"joint {joint_name!r} has a non-positive action scale")
        action_scale.append(matches[0])

    asset_module = _parse_python(ELF3_ASSET_CFG)
    elf3_cfg = _assignment_value(asset_module, "ELF3_CFG")
    init_state = _call_keyword(elf3_cfg, "init_state")
    joint_pos_node = _call_keyword(init_state, "joint_pos")
    try:
        default_mapping = ast.literal_eval(joint_pos_node)
    except (TypeError, ValueError) as exc:
        raise ValueError("ELF3_CFG default joint positions must remain a literal mapping") from exc
    if not isinstance(default_mapping, dict) or set(default_mapping) != set(joint_order):
        raise ValueError("ELF3_CFG default joint positions do not exactly match the policy joint order")
    default_q = tuple(float(default_mapping[name]) for name in joint_order)
    return joint_order, tuple(action_scale), default_q


def _urdf_limits(
    urdf_path: Path,
    joint_order: Sequence[str],
) -> tuple[tuple[float, ...], tuple[float, ...], tuple[float, ...]]:
    root = ET.parse(urdf_path).getroot()
    joints: dict[str, tuple[float, float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in joint_order:
            continue
        if joint.get("type") in {"fixed", "continuous"}:
            raise ValueError(f"policy joint {name!r} must be a bounded non-fixed URDF joint")
        limit = joint.find("limit")
        if limit is None:
            raise ValueError(f"URDF joint {name!r} has no limit element")
        try:
            lower = float(limit.attrib["lower"])
            upper = float(limit.attrib["upper"])
            effort = float(limit.attrib["effort"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"URDF joint {name!r} has an invalid lower/upper/effort limit") from exc
        if lower > upper or effort <= 0.0:
            raise ValueError(f"URDF joint {name!r} has inconsistent hard limits")
        joints[name] = (lower, upper, effort)
    missing = set(joint_order) - set(joints)
    if missing:
        raise ValueError(f"URDF is missing policy joint limits: {sorted(missing)}")
    return (
        tuple(joints[name][0] for name in joint_order),
        tuple(joints[name][1] for name in joint_order),
        tuple(joints[name][2] for name in joint_order),
    )


def _locomotion_schema() -> ObservationSchema:
    return ObservationSchema(
        terms=(
            ObservationTerm("locomotion_proprio_frame", 102, history_length=10),
            ObservationTerm("height_scan", 187),
        ),
        history_order="oldest_to_newest",
        previous_action_semantics="previous_raw_teacher_action",
        clip_low=-100.0,
        clip_high=100.0,
    )


def _motion_schema() -> ObservationSchema:
    return ObservationSchema(
        terms=(
            ObservationTerm("command", 58),
            ObservationTerm("motion_anchor_pos_b", 3),
            ObservationTerm("motion_anchor_ori_b", 6),
            ObservationTerm("base_lin_vel", 3),
            ObservationTerm("base_ang_vel", 3),
            ObservationTerm("joint_pos", 29),
            ObservationTerm("joint_vel", 29),
            ObservationTerm("previous_raw_teacher_action", 29),
            ObservationTerm("height_scan", 187),
        ),
        history_order="none",
        previous_action_semantics="previous_raw_teacher_action",
        clip_low=None,
        clip_high=None,
    )


def _manifest(
    *,
    skill: str,
    artifact_name: str,
    artifact_sha256: str,
    actor_input_dim: int,
    observation_schema: ObservationSchema,
    joint_order: tuple[str, ...],
    default_q: tuple[float, ...],
    action_scale: tuple[float, ...],
    lower: tuple[float, ...],
    upper: tuple[float, ...],
    effort: tuple[float, ...],
    urdf_sha256: str,
    usd_sha256: str,
    source_sha256: str,
) -> TeacherManifest:
    normalizer = (
        NormalizerSpec(
            kind="none",
            dimension=None,
            artifact_path=None,
            artifact_sha256=None,
            state_key=None,
            epsilon=None,
        )
        if skill == "locomotion"
        else NormalizerSpec(
            kind="empirical_std_plus_eps",
            dimension=actor_input_dim,
            artifact_path=artifact_name,
            artifact_sha256=artifact_sha256,
            state_key="actor_normalizer",
            epsilon=0.01,
        )
    )
    return TeacherManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        skill_name=skill,
        checkpoint_path=artifact_name,
        checkpoint_sha256=artifact_sha256,
        state_dict_key="actor_state_dict",
        state_prefix="",
        actor_input_dim=actor_input_dim,
        action_dim=29,
        hidden_dims=(512, 256, 128),
        activation="elu",
        joint_order=joint_order,
        default_q=default_q,
        action_scale=action_scale,
        observation_schema=observation_schema,
        normalizer=normalizer,
        urdf_path="elf3.urdf",
        urdf_sha256=urdf_sha256,
        usd_path="elf3.usd",
        usd_sha256=usd_sha256,
        control_dt=0.02,
        hard_lower_limits=lower,
        hard_upper_limits=upper,
        effort_limits=effort,
        artifact_metadata={
            "source_sha256": source_sha256,
            "asset_contract": "current_php_kvoy_reproduction_elf3",
            "effort_limit_source": "elf3.urdf",
        },
    )


def _golden_observations(dimension: int) -> torch.Tensor:
    values = torch.linspace(-0.25, 0.25, steps=3 * dimension, dtype=torch.float32)
    return values.reshape(3, dimension)


def _tracking_source_output(
    source_checkpoint: Path,
    manifest: TeacherManifest,
    observations: torch.Tensor,
) -> torch.Tensor:
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("tracking source checkpoint root must be a mapping")
    actor = build_actor_mlp(manifest)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise ValueError("tracking source checkpoint has no model_state_dict mapping")
    actor_state = {
        key[len("actor.") :]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith("actor.")
    }
    actor.load_state_dict(actor_state, strict=True)
    normalizer = checkpoint.get("obs_norm_state_dict")
    if not isinstance(normalizer, Mapping):
        raise ValueError("tracking source checkpoint has no observation normalizer")
    mean = normalizer["_mean"].reshape(-1)
    std = normalizer["_std"].reshape(-1)
    with torch.inference_mode():
        return actor((observations - mean) / (std + 0.01))


def _assert_golden_output(
    *,
    skill: str,
    source_path: Path,
    manifest_path: Path,
) -> None:
    manifest = TeacherManifest.load(manifest_path)
    manifest.verify_all_files()
    converted = TeacherPolicy.from_manifest(manifest, device="cpu")
    observations = _golden_observations(manifest.actor_input_dim)
    if skill == "locomotion":
        source_policy = torch.jit.load(str(source_path), map_location="cpu")
        source_policy.eval()
        with torch.inference_mode():
            expected = source_policy(observations)
    else:
        expected = _tracking_source_output(source_path, manifest, observations)
    with torch.inference_mode():
        actual = converted(observations)
    torch.testing.assert_close(actual, expected, rtol=1.0e-6, atol=1.0e-6)


def _write_bundle(args: argparse.Namespace, staging: Path) -> dict[str, Path]:
    joint_order, action_scale, default_q = _current_action_contract()
    lower, upper, effort = _urdf_limits(args.urdf, joint_order)

    copied_urdf = staging / "elf3.urdf"
    copied_usd = staging / "elf3.usd"
    shutil.copyfile(args.urdf, copied_urdf)
    shutil.copyfile(args.usd, copied_usd)
    urdf_sha256 = sha256_file(copied_urdf)
    usd_sha256 = sha256_file(copied_usd)

    specifications = {
        "locomotion": {
            "source": args.locomotion_source,
            "source_sha256": args.locomotion_sha256,
            "artifact": "locomotion_actor.pt",
            "input_dim": 1207,
            "schema": _locomotion_schema(),
        },
        "climb": {
            "source": args.climb_checkpoint,
            "source_sha256": args.climb_sha256,
            "artifact": "climb_actor.pt",
            "input_dim": 347,
            "schema": _motion_schema(),
        },
        "down_roll": {
            "source": args.down_roll_checkpoint,
            "source_sha256": args.down_roll_sha256,
            "artifact": "down_roll_actor.pt",
            "input_dim": 347,
            "schema": _motion_schema(),
        },
    }
    manifests: dict[str, Path] = {}
    for skill, spec in specifications.items():
        artifact_path = staging / str(spec["artifact"])
        if skill == "locomotion":
            result = build_locomotion_teacher_artifact(
                spec["source"],
                artifact_path,
                expected_source_sha256=str(spec["source_sha256"]),
                source_format="trusted_torchscript",
                state_prefix="actor.",
                trusted_source=True,
            )
        else:
            result = build_tracking_teacher_artifact(
                spec["source"],
                artifact_path,
                expected_source_sha256=str(spec["source_sha256"]),
                skill_name=skill,
            )
        manifest = _manifest(
            skill=skill,
            artifact_name=artifact_path.name,
            artifact_sha256=result.sha256,
            actor_input_dim=int(spec["input_dim"]),
            observation_schema=spec["schema"],
            joint_order=joint_order,
            default_q=default_q,
            action_scale=action_scale,
            lower=lower,
            upper=upper,
            effort=effort,
            urdf_sha256=urdf_sha256,
            usd_sha256=usd_sha256,
            source_sha256=str(spec["source_sha256"]),
        )
        manifest_path = staging / f"{skill}_manifest.json"
        manifest.save(manifest_path)
        _assert_golden_output(
            skill=skill,
            source_path=Path(spec["source"]),
            manifest_path=manifest_path,
        )
        manifests[skill] = manifest_path

    contract = {
        "joint_order": list(joint_order),
        "default_q": list(default_q),
        "action_scale": list(action_scale),
        "hard_lower_limits": list(lower),
        "hard_upper_limits": list(upper),
        "effort_limits_from_urdf": list(effort),
        "urdf_sha256": urdf_sha256,
        "usd_sha256": usd_sha256,
    }
    (staging / "elf3_action_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifests


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _sha256(value: str) -> str:
    normalized = value.lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise argparse.ArgumentTypeError("expected a 64-character SHA256 digest")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--locomotion_source",
        type=_path,
        default=_path(
            "/home/kvoy/Desktop/TienKung/logs/elf3_walk_expert/"
            "2026-08-18_11-03-08/exported/policy.pt"
        ),
    )
    parser.add_argument(
        "--climb_checkpoint",
        type=_path,
        default=_path(
            "/home/kvoy/Desktop/expert/expertmodel/climb/"
            "2026-08-24_19-47-32_elf3_climb/model_36000.pt"
        ),
    )
    parser.add_argument(
        "--down_roll_checkpoint",
        type=_path,
        default=_path(
            "/home/kvoy/Desktop/expert/expertmodel/down_roll/"
            "2026-08-25_13-29-40_elf3_down_roll/model_12000.pt"
        ),
    )
    parser.add_argument("--locomotion_sha256", type=_sha256, default=AUDITED_SOURCE_HASHES["locomotion"])
    parser.add_argument("--climb_sha256", type=_sha256, default=AUDITED_SOURCE_HASHES["climb"])
    parser.add_argument("--down_roll_sha256", type=_sha256, default=AUDITED_SOURCE_HASHES["down_roll"])
    parser.add_argument("--urdf", type=_path, default=DEFAULT_URDF.resolve())
    parser.add_argument("--usd", type=_path, default=DEFAULT_USD.resolve())
    parser.add_argument("--output_dir", type=_path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    for name in ("locomotion_source", "climb_checkpoint", "down_roll_checkpoint", "urdf", "usd"):
        path = getattr(args, name)
        if not path.is_file():
            raise FileNotFoundError(f"--{name} is not a regular file: {path}")
    output_dir: Path = args.output_dir
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite to replace it: {output_dir}")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"output path exists and is not a directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    temporary_root = Path(tempfile.mkdtemp(prefix=".teacher-artifacts-", dir=output_dir.parent))
    staging = temporary_root / "bundle"
    staging.mkdir()
    try:
        manifests = _write_bundle(args, staging)
        if output_dir.exists():
            shutil.rmtree(output_dir)
        os.replace(staging, output_dir)
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)

    print(f"[INFO] Published verified teacher bundle: {output_dir}")
    for skill in ("locomotion", "climb", "down_roll"):
        final_manifest = output_dir / manifests[skill].name
        print(f"[INFO] {skill}: {final_manifest}")


if __name__ == "__main__":
    main()
