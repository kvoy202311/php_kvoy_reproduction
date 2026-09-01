from __future__ import annotations

import ast
from pathlib import Path


_PACKAGE = Path(__file__).parents[1] / "php_kvoy_reproduction" / "tasks" / "distillation" / "mdp"
_REPOSITORY = Path(__file__).parents[3]


def _function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(matches) == 1
    return matches[0]


def _attribute_names(function: ast.FunctionDef) -> set[str]:
    return {node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute)}


def test_actor_reads_only_persistent_requested_command() -> None:
    function = _function(_PACKAGE / "observations.py", "body_planar_command")
    attributes = _attribute_names(function)
    assert "requested_world_command" in attributes
    assert "world_command" not in attributes


def test_internal_settle_command_cannot_modify_actor_request() -> None:
    function = _function(_PACKAGE / "commands.py", "_set_fixed_world_command")
    attributes = _attribute_names(function)
    assert "world_command" in attributes
    assert "requested_world_command" not in attributes


def test_motion_skill_switch_cannot_modify_actor_request() -> None:
    function = _function(_PACKAGE / "commands.py", "_reset_motion_skill")
    assert "requested_world_command" not in _attribute_names(function)


def test_live_request_updates_only_command_responsive_locomotion() -> None:
    function = _function(_PACKAGE / "commands.py", "set_requested_world_command")
    attributes = _attribute_names(function)
    assert "requested_world_command" in attributes
    assert "motion_control_locked" in attributes
    calls = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_sync_requested_locomotion_command" in calls


def test_climb_approach_applies_the_actor_visible_request() -> None:
    function = _function(_PACKAGE / "commands.py", "_begin_climb_approach")
    attributes = _attribute_names(function)
    assert "requested_world_command" in attributes
    assert "composed_locomotion_speed" not in attributes


def test_locked_request_randomization_cannot_change_active_teacher_command() -> None:
    function = _function(_PACKAGE / "commands.py", "_update_locked_requested_commands")
    attributes = _attribute_names(function)
    assert "requested_world_command" in attributes
    assert "world_command" not in attributes


def test_post_motion_release_applies_the_latest_request() -> None:
    function = _function(_PACKAGE / "commands.py", "_update_command")
    calls = {
        node.func.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "_sync_requested_locomotion_command" in calls
    assert "_begin_down_roll_from_edge_settle" in calls


def test_playback_configures_checkpoint_stage_before_scene_creation() -> None:
    path = _REPOSITORY / "scripts" / "rsl_rl" / "play_distillation.py"
    source = path.read_text(encoding="utf-8")
    assert source.index("configure_training_stage(env_cfg") < source.index("gym.make(args_cli.task")


def test_fixed_skill_playback_routes_environment_and_student_to_the_same_head() -> None:
    path = _REPOSITORY / "scripts" / "rsl_rl" / "play_distillation.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assignments = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == "forced_skill_id"
            for target in node.targets
        )
    ]
    assert len(assignments) == 2
    assert any(
        isinstance(node.value, ast.Name)
        and node.value.id == "fixed_student_skill_id"
        for node in assignments
    )
    inference_calls = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_inference_policy"
    ]
    assert len(inference_calls) == 1
    fixed_keywords = [
        keyword
        for keyword in inference_calls[0].keywords
        if keyword.arg == "fixed_skill_id"
    ]
    assert len(fixed_keywords) == 1
    assert isinstance(fixed_keywords[0].value, ast.Name)
    assert fixed_keywords[0].value.id == "fixed_student_skill_id"


def test_playback_rejects_composed_evaluation_of_atomic_checkpoint() -> None:
    path = _REPOSITORY / "scripts" / "rsl_rl" / "play_distillation.py"
    source = path.read_text(encoding="utf-8")
    assert 'args_cli.playback_mode == "composed" and checkpoint_stage == "atomic"' in source
    guard = source.index(
        'args_cli.playback_mode == "composed" and checkpoint_stage == "atomic"'
    )
    configure = source.index("configure_training_stage(env_cfg")
    scene = source.index("gym.make(args_cli.task")
    assert guard < configure < scene


def test_motion_reset_uses_detected_execution_start_and_stationary_boundary() -> None:
    path = _PACKAGE / "commands.py"
    reset = _function(path, "_reset_motion_skill")
    attributes = _attribute_names(reset)
    assert "motion_execution_start_idx" in attributes
    stationary_branches = [
        node
        for node in ast.walk(reset)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "start_at_beginning"
    ]
    zeroed_targets = {
        target.id
        for branch in stationary_branches
        for assignment in branch.body
        if isinstance(assignment, ast.Assign)
        and isinstance(assignment.value, ast.Call)
        and isinstance(assignment.value.func, ast.Attribute)
        and assignment.value.func.attr == "zeros_like"
        for target in assignment.targets
        if isinstance(target, ast.Name)
    }
    assert zeroed_targets == {"root_lin_vel", "root_ang_vel", "joint_vel"}


def test_fixed_playback_requests_the_same_detected_motion_start() -> None:
    path = _REPOSITORY / "scripts" / "rsl_rl" / "play_distillation.py"
    source = path.read_text(encoding="utf-8")
    start = source.index("command.start_at_motion_beginning = True")
    scene = source.index("gym.make(args_cli.task")
    assert start < scene


def test_motion_clock_uses_applied_student_route_and_termination_mask() -> None:
    path = _PACKAGE / "commands.py"
    update = _function(path, "_update_command")
    attributes = _attribute_names(update)
    assert "student_active_skill_ids" in attributes
    assert "termination_manager" in attributes
    calls = {
        node.func.id
        for node in ast.walk(update)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "motion_reference_advance_mask" in calls


def test_command_accepts_one_explicit_student_route_per_environment() -> None:
    setter = _function(_PACKAGE / "commands.py", "set_student_active_skill_ids")
    attributes = _attribute_names(setter)
    assert "student_active_skill_ids" in attributes
    assert "student_route_published" in attributes
    assert "num_envs" in attributes
