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
    render_scene_parser.add_argument("--tracks-x-field", default="bkg_x")
    render_scene_parser.add_argument("--tracks-y-field", default="bkg_y")
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
    render_scene_task_parser.add_argument("--tracks-x-field", default="bkg_x")
    render_scene_task_parser.add_argument("--tracks-y-field", default="bkg_y")
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
    render_plan_parser.add_argument("--show-waypoints", action="store_true")
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
    render_scene_task_plan_parser.add_argument("--show-waypoints", action="store_true")
    render_scene_task_plan_parser.add_argument("--tracks-x-field", default="bkg_x")
    render_scene_task_plan_parser.add_argument("--tracks-y-field", default="bkg_y")
    render_scene_task_plan_parser.add_argument("--max-tracks", type=int, default=None)
    render_scene_task_plan_parser.add_argument("--title", default=None)

    validate_layout_parser = subparsers.add_parser(
        "validate-layout",
        help="Validate a layout JSON file",
    )
    validate_layout_parser.add_argument("layout_json")

    validate_scene_parser = subparsers.add_parser(
        "validate-scene",
        help="Validate a scene JSON file",
    )
    validate_scene_parser.add_argument("scene_json")

    validate_task_parser = subparsers.add_parser(
        "validate-task",
        help="Validate a task JSON file",
    )
    validate_task_parser.add_argument("task_json")

    validate_robot_parser = subparsers.add_parser(
        "validate-robot",
        help="Validate a robot JSON file",
    )
    validate_robot_parser.add_argument("robot_json")

    validate_plan_parser = subparsers.add_parser(
        "validate-plan",
        help="Validate a plan JSON file",
    )
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

    run_geometry_parser = subparsers.add_parser(
        "run-geometry",
        help="Run the baseline geometry planner from task + robot and write a plan JSON",
    )
    run_geometry_parser.add_argument("task_json")
    run_geometry_parser.add_argument("robot_json")
    run_geometry_parser.add_argument("plan_json")

    return parser


def cmd_render_layout(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout_json)
    save_layout_figure(
        layout,
        args.output_image,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        title=args.title,
    )
    print(f"Wrote {args.output_image}")
    return 0


def cmd_render_scene(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout_json)
    scene = load_scene(args.scene_json)
    save_scene_figure(
        layout,
        scene,
        args.output_image,
        scene_json_path=args.scene_json,
        show_tracks=args.show_tracks,
        tracks_x_field=args.tracks_x_field,
        tracks_y_field=args.tracks_y_field,
        max_tracks=args.max_tracks,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        annotate_scene=not args.no_scene_annotations,
        show_p_star=not args.no_p_star,
        show_u_hat=not args.no_u_hat,
        u_hat_scale=args.u_hat_scale,
        title=args.title,
    )
    print(f"Wrote {args.output_image}")
    return 0


def cmd_render_task(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout_json)
    task = load_task(args.task_json)
    save_task_figure(
        layout,
        task,
        args.output_image,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        annotate_task=not args.no_task_annotations,
        show_straight_line=not args.no_straight_line,
        title=args.title,
    )
    print(f"Wrote {args.output_image}")
    return 0


def cmd_render_scene_task(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout_json)
    scene = load_scene(args.scene_json)
    task = load_task(args.task_json)
    save_scene_task_figure(
        layout,
        scene,
        task,
        args.output_image,
        scene_json_path=args.scene_json,
        show_tracks=args.show_tracks,
        tracks_x_field=args.tracks_x_field,
        tracks_y_field=args.tracks_y_field,
        max_tracks=args.max_tracks,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        annotate_scene=not args.no_scene_annotations,
        annotate_task=not args.no_task_annotations,
        show_p_star=not args.no_p_star,
        show_u_hat=not args.no_u_hat,
        u_hat_scale=args.u_hat_scale,
        show_straight_line=not args.no_straight_line,
        title=args.title,
    )
    print(f"Wrote {args.output_image}")
    return 0


def cmd_render_plan(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout_json)
    plan = load_plan(args.plan_json)
    save_plan_figure(
        layout,
        plan,
        args.output_image,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        annotate_plan=not args.no_plan_annotations,
        show_waypoints=args.show_waypoints,
        title=args.title,
    )
    print(f"Wrote {args.output_image}")
    return 0


def cmd_render_scene_task_plan(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout_json)
    scene = load_scene(args.scene_json)
    task = load_task(args.task_json)
    plan = load_plan(args.plan_json)
    save_scene_task_plan_figure(
        layout,
        scene,
        task,
        plan,
        args.output_image,
        scene_json_path=args.scene_json,
        show_tracks=args.show_tracks,
        tracks_x_field=args.tracks_x_field,
        tracks_y_field=args.tracks_y_field,
        max_tracks=args.max_tracks,
        show_obstacle_labels=not args.no_obstacle_labels,
        show_exit_labels=not args.no_exit_labels,
        annotate_scene=not args.no_scene_annotations,
        annotate_task=not args.no_task_annotations,
        annotate_plan=not args.no_plan_annotations,
        show_p_star=not args.no_p_star,
        show_u_hat=not args.no_u_hat,
        u_hat_scale=args.u_hat_scale,
        show_straight_line=not args.no_straight_line,
        show_waypoints=args.show_waypoints,
        title=args.title,
    )
    print(f"Wrote {args.output_image}")
    return 0


def cmd_validate_layout(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout_json, validate=False)
    validate_layout(layout)
    print(f"Layout OK: {layout.layout_id}")
    return 0


def cmd_validate_scene(args: argparse.Namespace) -> int:
    scene = load_scene(args.scene_json, validate=False)
    validate_scene(scene)
    print(f"Scene OK: {scene.scene_id}")
    return 0


def cmd_validate_task(args: argparse.Namespace) -> int:
    task = load_task(args.task_json, validate=False)
    validate_task(task)
    print(f"Task OK: {task.task_id}")
    return 0


def cmd_validate_robot(args: argparse.Namespace) -> int:
    robot = load_robot(args.robot_json, validate=False)
    validate_robot(robot)
    print(f"Robot OK: {robot.robot_id}")
    return 0


def cmd_validate_plan(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan_json, validate=False)
    validate_plan_result(plan)
    print(f"Plan OK: {plan.planner_name}")
    return 0


def cmd_evaluate_plan(args: argparse.Namespace) -> int:
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
    )

    print("Evaluation OK")
    print(f"  success: {result.success}")
    print(f"  path_length_m: {result.path_length_m}")
    print(f"  runtime_s: {result.runtime_s}")
    print(f"  num_waypoints: {result.num_waypoints}")
    print(f"  min_human_distance_m: {result.min_human_distance_m}")
    if result.message:
        print(f"  message: {result.message}")

    return 0


def cmd_run_geometry(args: argparse.Namespace) -> int:
    task = load_task(args.task_json)
    robot = load_robot(args.robot_json)

    # load scene from task reference
    scene_json_path = _resolve_relative(args.task_json, task.scene.path)
    scene = load_scene(scene_json_path)

    # sanity check IDs match
    if scene.scene_id != task.scene.scene_id:
        raise ValueError(
            f"Scene ID mismatch: task expects {task.scene.scene_id!r} "
            f"but loaded {scene.scene_id!r}"
        )

    # load layout from scene reference
    layout_json_path = _resolve_relative(scene_json_path, scene.layout.path)
    layout = load_layout(layout_json_path)

    # sanity check layout IDs
    if layout.layout_id != scene.layout.layout_id:
        raise ValueError(
            f"Layout ID mismatch: scene expects {scene.layout.layout_id!r} "
            f"but loaded {layout.layout_id!r}"
        )

    planner = GeometryPlanner()
    plan = planner.plan(layout, scene, task, robot)

    save_plan(args.plan_json, plan)

    print(f"Wrote {args.plan_json}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "render-layout":
        return cmd_render_layout(args)
    if args.command == "render-scene":
        return cmd_render_scene(args)
    if args.command == "render-task":
        return cmd_render_task(args)
    if args.command == "render-scene-task":
        return cmd_render_scene_task(args)
    if args.command == "render-plan":
        return cmd_render_plan(args)
    if args.command == "render-scene-task-plan":
        return cmd_render_scene_task_plan(args)
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
