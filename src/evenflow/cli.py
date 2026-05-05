from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable
import inspect

from .evaluation import evaluate_plan
from .io import load_layout, load_plan, load_robot, load_scene, load_task, save_plan
from .planners import GeometryPlanner
from .render import (
    save_layout_figure,
    save_scene_figure,
    save_scene_task_figure,
    save_scene_task_plan_figure,
)
from .validation import (
    validate_layout,
    validate_plan_result,
    validate_robot,
    validate_scene,
    validate_task,
)


def _resolve_relative(base_file: str | Path, relative_path: str | Path) -> Path:
    """Resolve a path stored inside an EvenFlow JSON artifact."""
    base = Path(base_file).resolve().parent
    return (base / relative_path).resolve()


def _ensure_parent_dir(path: str | Path) -> None:
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def _load_scene_from_task(task_json: str | Path):
    task = load_task(task_json)
    scene_json_path = _resolve_relative(task_json, task.scene.path)
    scene = load_scene(scene_json_path)
    return task, scene, scene_json_path


def _load_layout_from_scene(scene_json: str | Path, scene=None):
    if scene is None:
        scene = load_scene(scene_json)
    layout_json_path = _resolve_relative(scene_json, scene.layout.path)
    layout = load_layout(layout_json_path)
    return layout, layout_json_path


def _load_task_bundle(task_json: str | Path):
    """Load task, scene, layout, and resolved paths from a task JSON."""
    task, scene, scene_json_path = _load_scene_from_task(task_json)
    layout, layout_json_path = _load_layout_from_scene(scene_json_path, scene)
    return layout, scene, task, layout_json_path, scene_json_path


def _call_render(fn, *args, **kwargs):
    sig = inspect.signature(fn)
    valid_kwargs = {
        k: v for k, v in kwargs.items()
        if k in sig.parameters
    }
    return fn(*args, **valid_kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evenflow",
        description=(
            "EvenFlow tools for validating, rendering, running, and evaluating "
            "shared-space navigation benchmark tasks."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Rendering ---------------------------------------------------------------

    render_scene_parser = subparsers.add_parser(
        "render-scene",
        help="Render a scene. Layout is inferred from the scene JSON.",
    )
    render_scene_parser.add_argument("scene_json")
    render_scene_parser.add_argument("output_image")
    render_scene_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_scene_parser.add_argument("--no-exit-labels", action="store_true")
    render_scene_parser.add_argument("--no-scene-annotations", action="store_true")
    render_scene_parser.add_argument("--no-p-star", action="store_true")
    render_scene_parser.add_argument("--no-u-hat", action="store_true")
    render_scene_parser.add_argument("--u-hat-scale", type=float, default=1.0)
    render_scene_parser.add_argument("--show-tracks", action="store_true")
    render_scene_parser.add_argument("--max-tracks", type=int, default=None)
    render_scene_parser.add_argument("--title", default=None)

    render_task_parser = subparsers.add_parser(
        "render-task",
        help="Render a task. Scene and layout are inferred from the task JSON.",
    )
    render_task_parser.add_argument("task_json")
    render_task_parser.add_argument("output_image")
    render_task_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_task_parser.add_argument("--no-exit-labels", action="store_true")
    render_task_parser.add_argument("--no-scene-annotations", action="store_true")
    render_task_parser.add_argument("--no-task-annotations", action="store_true")
    render_task_parser.add_argument("--no-p-star", action="store_true")
    render_task_parser.add_argument("--no-u-hat", action="store_true")
    render_task_parser.add_argument("--u-hat-scale", type=float, default=1.0)
    render_task_parser.add_argument("--no-straight-line", action="store_true")
    render_task_parser.add_argument("--show-tracks", action="store_true")
    render_task_parser.add_argument("--max-tracks", type=int, default=None)
    render_task_parser.add_argument("--title", default=None)

    render_plan_parser = subparsers.add_parser(
        "render-plan",
        help="Render a plan in task context. Scene and layout are inferred from the task JSON.",
    )
    render_plan_parser.add_argument("task_json")
    render_plan_parser.add_argument("plan_json")
    render_plan_parser.add_argument("output_image")
    render_plan_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_plan_parser.add_argument("--no-exit-labels", action="store_true")
    render_plan_parser.add_argument("--no-scene-annotations", action="store_true")
    render_plan_parser.add_argument("--no-task-annotations", action="store_true")
    render_plan_parser.add_argument("--no-plan-annotations", action="store_true")
    render_plan_parser.add_argument("--no-p-star", action="store_true")
    render_plan_parser.add_argument("--no-u-hat", action="store_true")
    render_plan_parser.add_argument("--u-hat-scale", type=float, default=1.0)
    render_plan_parser.add_argument("--no-straight-line", action="store_true")
    render_plan_parser.add_argument("--show-tracks", action="store_true")
    render_plan_parser.add_argument("--show-samples", action="store_true")
    render_plan_parser.add_argument("--max-tracks", type=int, default=None)
    render_plan_parser.add_argument("--title", default=None)

    render_layout_parser = subparsers.add_parser(
        "render-layout",
        help="Render a layout JSON to an image.",
    )
    render_layout_parser.add_argument("layout_json")
    render_layout_parser.add_argument("output_image")
    render_layout_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_layout_parser.add_argument("--no-exit-labels", action="store_true")
    render_layout_parser.add_argument("--title", default=None)

    # Validation --------------------------------------------------------------

    validate_layout_parser = subparsers.add_parser("validate-layout", help="Validate a layout JSON.")
    validate_layout_parser.add_argument("layout_json")

    validate_scene_parser = subparsers.add_parser(
        "validate-scene",
        help="Validate a scene JSON and its referenced layout.",
    )
    validate_scene_parser.add_argument("scene_json")

    validate_task_parser = subparsers.add_parser(
        "validate-task",
        help="Validate a task JSON and its referenced scene/layout.",
    )
    validate_task_parser.add_argument("task_json")

    validate_robot_parser = subparsers.add_parser("validate-robot", help="Validate a robot JSON.")
    validate_robot_parser.add_argument("robot_json")

    validate_plan_parser = subparsers.add_parser(
        "validate-plan",
        help=(
            "Validate a plan JSON. Use `validate-plan plan.json` for a schema "
            "check or `validate-plan task.json robot.json plan.json` for a "
            "context-aware check."
        ),
    )
    validate_plan_parser.add_argument("paths", nargs="+")

    # Execution / evaluation --------------------------------------------------

    evaluate_plan_parser = subparsers.add_parser(
        "evaluate-plan",
        help="Evaluate a plan against a task. Scene and layout are inferred from the task JSON.",
    )
    evaluate_plan_parser.add_argument("task_json")
    evaluate_plan_parser.add_argument("robot_json")
    evaluate_plan_parser.add_argument("plan_json")

    run_geometry_parser = subparsers.add_parser(
        "run-geometry",
        help="Run the baseline geometry planner from task + robot and write a plan JSON.",
    )
    run_geometry_parser.add_argument("task_json")
    run_geometry_parser.add_argument("robot_json")
    run_geometry_parser.add_argument("plan_json")

    return parser


# Rendering ------------------------------------------------------------------


def cmd_render_layout(args):
    layout = load_layout(args.layout_json)
    _ensure_parent_dir(args.output_image)
    _call_render(
        save_layout_figure,
        layout,
        args.output_image,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        title=args.title,
    )
    print(f"✓ Wrote {args.output_image}")
    return 0


def cmd_render_scene(args):
    scene = load_scene(args.scene_json)
    layout, _ = _load_layout_from_scene(args.scene_json, scene)
    _ensure_parent_dir(args.output_image)

    _call_render(
        save_scene_figure,
        layout,
        scene,
        args.output_image,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        show_scene_annotations=not args.no_scene_annotations,
        show_p_star=not args.no_p_star,
        show_u_hat=not args.no_u_hat,
        u_hat_scale=args.u_hat_scale,
        show_tracks=args.show_tracks,
        max_tracks=args.max_tracks,
        title=args.title,
        scene_json_path=args.scene_json,
    )
    print(f"✓ Wrote {args.output_image}")
    return 0


def cmd_render_task(args):
    layout, scene, task, _, scene_json_path = _load_task_bundle(args.task_json)
    _ensure_parent_dir(args.output_image)

    _call_render(
        save_scene_task_figure,
        layout,
        scene,
        task,
        args.output_image,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        show_scene_annotations=not args.no_scene_annotations,
        show_task_annotations=not args.no_task_annotations,
        show_p_star=not args.no_p_star,
        show_u_hat=not args.no_u_hat,
        u_hat_scale=args.u_hat_scale,
        show_straight_line=not args.no_straight_line,
        show_tracks=args.show_tracks,
        max_tracks=args.max_tracks,
        title=args.title,
        scene_json_path=scene_json_path,
        task_json_path=args.task_json,
    )
    print(f"✓ Wrote {args.output_image}")
    return 0


def cmd_render_plan(args):
    layout, scene, task, _, scene_json_path = _load_task_bundle(args.task_json)
    plan = load_plan(args.plan_json)
    _ensure_parent_dir(args.output_image)

    _call_render(
        save_scene_task_plan_figure,
        layout,
        scene,
        task,
        plan,
        args.output_image,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        show_scene_annotations=not args.no_scene_annotations,
        show_task_annotations=not args.no_task_annotations,
        show_plan_annotations=not args.no_plan_annotations,
        show_p_star=not args.no_p_star,
        show_u_hat=not args.no_u_hat,
        u_hat_scale=args.u_hat_scale,
        show_straight_line=not args.no_straight_line,
        show_tracks=args.show_tracks,
        show_samples=args.show_samples,
        max_tracks=args.max_tracks,
        title=args.title,
        scene_json_path=scene_json_path,
        task_json_path=args.task_json,
        plan_json_path=args.plan_json,
    )
    print(f"✓ Wrote {args.output_image}")
    return 0


# Validation -----------------------------------------------------------------


def cmd_validate_layout(args):
    layout = load_layout(args.layout_json, validate=False)
    validate_layout(layout)
    print(f"✓ Layout OK: {layout.layout_id}")
    return 0


def cmd_validate_scene(args):
    scene = load_scene(args.scene_json, validate=False)
    validate_scene(scene)
    layout, layout_json_path = _load_layout_from_scene(args.scene_json, scene)
    validate_layout(layout)

    print(f"✓ Scene OK: {scene.scene_id}")
    print(f"  layout: {layout_json_path}")
    return 0


def cmd_validate_task(args):
    layout, scene, task, layout_json_path, scene_json_path = _load_task_bundle(args.task_json)
    validate_layout(layout)
    validate_scene(scene)
    validate_task(task)

    print(f"✓ Task OK: {task.task_id}")
    print(f"  scene: {scene_json_path}")
    print(f"  layout: {layout_json_path}")
    return 0


def cmd_validate_robot(args):
    robot = load_robot(args.robot_json, validate=False)
    validate_robot(robot)
    print(f"✓ Robot OK: {robot.robot_id}")
    return 0


def _print_plan_summary(plan) -> None:
    print(f"  planner: {plan.planner_name}")
    print(f"  success: {plan.success}")
    if plan.path_length_m is not None:
        print(f"  path_length_m: {plan.path_length_m}")
    if plan.runtime_s is not None:
        print(f"  runtime_s: {plan.runtime_s}")
    if plan.track is not None:
        try:
            n = len(plan.track.timestamps)
            print(f"  samples: {n}")
            if n:
                print(f"  t_start: {plan.track.timestamps[0]}")
                print(f"  t_end: {plan.track.timestamps[-1]}")
        except Exception:
            pass
    if plan.message:
        print(f"  message: {plan.message}")


def cmd_validate_plan(args):
    if len(args.paths) == 1:
        plan_json = args.paths[0]
        plan = load_plan(plan_json, validate=False)
        validate_plan_result(plan)
        print("✓ Plan schema OK")
        _print_plan_summary(plan)
        return 0

    if len(args.paths) == 3:
        task_json, robot_json, plan_json = args.paths

        layout, scene, task, _, scene_json_path = _load_task_bundle(task_json)
        robot = load_robot(robot_json)
        plan = load_plan(plan_json, validate=False)

        validate_layout(layout)
        validate_scene(scene)
        validate_task(task)
        validate_robot(robot)
        validate_plan_result(plan)

        result = evaluate_plan(
            layout,
            scene,
            task,
            robot,
            plan,
            scene_json_path=scene_json_path,
        )

        print("✓ Plan context check OK")
        _print_plan_summary(plan)

        if result.metadata and result.metadata.get("evaluation_error"):
            print("  warning: evaluator reported an issue")
            print(f"  evaluation_error: {result.metadata['evaluation_error']}")
            return 1

        if result.overall_score is not None:
            print(f"  overall_score: {result.overall_score}")
        return 0

    raise SystemExit(
        "validate-plan expects either:\n"
        "  evenflow validate-plan <plan.json>\n"
        "or:\n"
        "  evenflow validate-plan <task.json> <robot.json> <plan.json>"
    )


# Execution / evaluation ------------------------------------------------------


def cmd_evaluate_plan(args):
    layout, scene, task, _, scene_json_path = _load_task_bundle(args.task_json)
    robot = load_robot(args.robot_json)
    plan = load_plan(args.plan_json)

    result = evaluate_plan(
        layout,
        scene,
        task,
        robot,
        plan,
        scene_json_path=scene_json_path,
    )

    eval_version = (
        result.metadata.get("evaluation_version", "canonical")
        if result.metadata else "canonical"
    )
    print(f"✓ Evaluation OK ({eval_version})")
    print(f"  success: {result.success}")
    print(f"  path_length_m: {result.path_length_m}")
    print(f"  runtime_s: {result.runtime_s}")
    print(f"  min_human_distance_m: {result.min_human_distance_m}")

    if result.human_likeness_score is not None:
        print(f"  human_likeness_score: {result.human_likeness_score}")
    if result.social_compatibility_score is not None:
        print(f"  social_compatibility_score: {result.social_compatibility_score}")
    if result.task_efficiency_score is not None:
        print(f"  task_efficiency_score: {result.task_efficiency_score}")
    if result.overall_score is not None:
        print(f"  overall_score: {result.overall_score}")
    if result.message:
        print(f"  message: {result.message}")

    if result.metadata and result.metadata.get("evaluation_error"):
        print(f"  evaluation_error: {result.metadata['evaluation_error']}")
        return 1

    return 0


def cmd_run_geometry(args):
    layout, scene, task, _, _ = _load_task_bundle(args.task_json)
    robot = load_robot(args.robot_json)

    planner = GeometryPlanner()
    plan = planner.plan(layout, scene, task, robot)

    _ensure_parent_dir(args.plan_json)
    save_plan(args.plan_json, plan)
    print(f"✓ Wrote {args.plan_json}")
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    command_handlers = {
        "render-layout": cmd_render_layout,
        "render-scene": cmd_render_scene,
        "render-task": cmd_render_task,
        "render-plan": cmd_render_plan,
        "validate-layout": cmd_validate_layout,
        "validate-scene": cmd_validate_scene,
        "validate-task": cmd_validate_task,
        "validate-robot": cmd_validate_robot,
        "validate-plan": cmd_validate_plan,
        "evaluate-plan": cmd_evaluate_plan,
        "run-geometry": cmd_run_geometry,
    }

    handler = command_handlers.get(args.command)
    if handler is None:
        parser.error(f"Unknown command: {args.command}")
        return 2

    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
