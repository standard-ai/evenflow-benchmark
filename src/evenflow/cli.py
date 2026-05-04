from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import evaluate_plan
from .io import load_layout, load_plan, load_robot, load_scene, load_task, save_plan
from .planners import GeometryPlanner
from .render import (
    save_layout_figure,
    save_plan_figure,
    save_scene_figure,
    save_scene_task_figure,
    save_scene_task_plan_figure,
    save_task_figure,
)
from .validation import (
    validate_layout,
    validate_plan_result,
    validate_robot,
    validate_scene,
    validate_task,
)


def _resolve_relative(base_file: str | Path, relative_path: str) -> Path:
    base = Path(base_file).resolve().parent
    return (base / relative_path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evenflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_layout_parser = subparsers.add_parser(
        "render-layout",
        help="Render a layout JSON to an image",
    )
    render_layout_parser.add_argument("layout_json")
    render_layout_parser.add_argument("output_image")
    render_layout_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_layout_parser.add_argument("--no-exit-labels", action="store_true")
    render_layout_parser.add_argument("--title", default=None)

    render_scene_parser = subparsers.add_parser(
        "render-scene",
        help="Render a scene on top of a layout",
    )
    render_scene_parser.add_argument("layout_json")
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
        help="Render a task on top of a layout",
    )
    render_task_parser.add_argument("layout_json")
    render_task_parser.add_argument("task_json")
    render_task_parser.add_argument("output_image")
    render_task_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_task_parser.add_argument("--no-exit-labels", action="store_true")
    render_task_parser.add_argument("--no-task-annotations", action="store_true")
    render_task_parser.add_argument("--no-straight-line", action="store_true")
    render_task_parser.add_argument("--title", default=None)

    render_scene_task_parser = subparsers.add_parser(
        "render-scene-task",
        help="Render a scene and task together on top of a layout",
    )
    render_scene_task_parser.add_argument("layout_json")
    render_scene_task_parser.add_argument("scene_json")
    render_scene_task_parser.add_argument("task_json")
    render_scene_task_parser.add_argument("output_image")
    render_scene_task_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_scene_task_parser.add_argument("--no-exit-labels", action="store_true")
    render_scene_task_parser.add_argument("--no-scene-annotations", action="store_true")
    render_scene_task_parser.add_argument("--no-task-annotations", action="store_true")
    render_scene_task_parser.add_argument("--no-p-star", action="store_true")
    render_scene_task_parser.add_argument("--no-u-hat", action="store_true")
    render_scene_task_parser.add_argument("--u-hat-scale", type=float, default=1.0)
    render_scene_task_parser.add_argument("--no-straight-line", action="store_true")
    render_scene_task_parser.add_argument("--show-tracks", action="store_true")
    render_scene_task_parser.add_argument("--max-tracks", type=int, default=None)
    render_scene_task_parser.add_argument("--title", default=None)

    render_plan_parser = subparsers.add_parser(
        "render-plan",
        help="Render a plan on top of a layout",
    )
    render_plan_parser.add_argument("layout_json")
    render_plan_parser.add_argument("plan_json")
    render_plan_parser.add_argument("output_image")
    render_plan_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_plan_parser.add_argument("--no-exit-labels", action="store_true")
    render_plan_parser.add_argument("--no-plan-annotations", action="store_true")
    render_plan_parser.add_argument("--show-samples", action="store_true")
    render_plan_parser.add_argument("--title", default=None)

    render_scene_task_plan_parser = subparsers.add_parser(
        "render-scene-task-plan",
        help="Render a scene, task, and plan together on top of a layout",
    )
    render_scene_task_plan_parser.add_argument("layout_json")
    render_scene_task_plan_parser.add_argument("scene_json")
    render_scene_task_plan_parser.add_argument("task_json")
    render_scene_task_plan_parser.add_argument("plan_json")
    render_scene_task_plan_parser.add_argument("output_image")
    render_scene_task_plan_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_scene_task_plan_parser.add_argument("--no-exit-labels", action="store_true")
    render_scene_task_plan_parser.add_argument("--no-scene-annotations", action="store_true")
    render_scene_task_plan_parser.add_argument("--no-task-annotations", action="store_true")
    render_scene_task_plan_parser.add_argument("--no-plan-annotations", action="store_true")
    render_scene_task_plan_parser.add_argument("--no-p-star", action="store_true")
    render_scene_task_plan_parser.add_argument("--no-u-hat", action="store_true")
    render_scene_task_plan_parser.add_argument("--u-hat-scale", type=float, default=1.0)
    render_scene_task_plan_parser.add_argument("--no-straight-line", action="store_true")
    render_scene_task_plan_parser.add_argument("--show-tracks", action="store_true")
    render_scene_task_plan_parser.add_argument("--show-samples", action="store_true")
    render_scene_task_plan_parser.add_argument("--max-tracks", type=int, default=None)
    render_scene_task_plan_parser.add_argument("--title", default=None)

    validate_layout_parser = subparsers.add_parser("validate-layout")
    validate_layout_parser.add_argument("layout_json")

    validate_scene_parser = subparsers.add_parser("validate-scene")
    validate_scene_parser.add_argument("scene_json")

    validate_task_parser = subparsers.add_parser("validate-task")
    validate_task_parser.add_argument("task_json")

    validate_robot_parser = subparsers.add_parser("validate-robot")
    validate_robot_parser.add_argument("robot_json")

    validate_plan_parser = subparsers.add_parser("validate-plan")
    validate_plan_parser.add_argument("plan_json")

    evaluate_plan_parser = subparsers.add_parser(
        "evaluate-plan",
        help="Evaluate a plan against a layout / scene / task / robot bundle",
    )
    evaluate_plan_parser.add_argument("layout_json")
    evaluate_plan_parser.add_argument("scene_json")
    evaluate_plan_parser.add_argument("task_json")
    evaluate_plan_parser.add_argument("robot_json")
    evaluate_plan_parser.add_argument("plan_json")
    evaluate_plan_parser.add_argument(
        "--evaluation-version",
        choices=["v1", "v2", "v3","v4"],
        default="v1",
        help="Evaluation metric version (default: v1)",
    )

    run_geometry_parser = subparsers.add_parser(
        "run-geometry",
        help="Run the baseline geometry planner from task + robot and write a plan JSON",
    )
    run_geometry_parser.add_argument("task_json")
    run_geometry_parser.add_argument("robot_json")
    run_geometry_parser.add_argument("plan_json")

    return parser


def cmd_validate_layout(args):
    layout = load_layout(args.layout_json, validate=False)
    validate_layout(layout)
    print(f"Layout OK: {layout.layout_id}")
    return 0


def cmd_validate_scene(args):
    scene = load_scene(args.scene_json, validate=False)
    validate_scene(scene)
    print(f"Scene OK: {scene.scene_id}")
    return 0


def cmd_validate_task(args):
    task = load_task(args.task_json, validate=False)
    validate_task(task)
    print(f"Task OK: {task.task_id}")
    return 0


def cmd_validate_robot(args):
    robot = load_robot(args.robot_json, validate=False)
    validate_robot(robot)
    print(f"Robot OK: {robot.robot_id}")
    return 0


def cmd_validate_plan(args):
    plan = load_plan(args.plan_json, validate=False)
    validate_plan_result(plan)
    print(f"Plan OK: {plan.planner_name}")
    return 0


def cmd_evaluate_plan(args):
    layout = load_layout(args.layout_json)
    scene = load_scene(args.scene_json)
    task = load_task(args.task_json)
    robot = load_robot(args.robot_json)
    plan = load_plan(args.plan_json)

    result = evaluate_plan(
        layout,
        scene,
        task,
        robot,
        plan,
        scene_json_path=args.scene_json,
        evaluation_version=args.evaluation_version,
    )

    print(f"Evaluation OK ({args.evaluation_version})")
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

    return 0


def cmd_run_geometry(args):
    task = load_task(args.task_json)
    robot = load_robot(args.robot_json)

    scene_json_path = _resolve_relative(args.task_json, task.scene.path)
    scene = load_scene(scene_json_path)

    layout_json_path = _resolve_relative(scene_json_path, scene.layout.path)
    layout = load_layout(layout_json_path)

    planner = GeometryPlanner()
    plan = planner.plan(layout, scene, task, robot)

    save_plan(args.plan_json, plan)
    print(f"Wrote {args.plan_json}")
    return 0


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "validate-layout":
        return cmd_validate_layout(args)
    if args.command == "validate-scene":
        return cmd_validate_scene(args)
    if args.command == "validate-task":
        return cmd_validate_task(args)
    if args.command == "validate-robot":
        return cmd_validate_robot(args)
    if args.command == "validate-plan":
        return cmd_validate_plan(args)
    if args.command == "evaluate-plan":
        return cmd_evaluate_plan(args)
    if args.command == "run-geometry":
        return cmd_run_geometry(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
