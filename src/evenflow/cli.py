from __future__ import annotations

import argparse

from .io import load_layout, load_scene, load_task, load_robot
from .render import (
    save_layout_figure,
    save_scene_figure,
    save_scene_task_figure,
    save_task_figure,
)
from .validation import (
    validate_layout,
    validate_scene,
    validate_task,
    validate_robot,
)

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
    if args.command == "validate-layout":
        return cmd_validate_layout(args)
    if args.command == "validate-scene":
        return cmd_validate_scene(args)
    if args.command == "validate-task":
        return cmd_validate_task(args)
    if args.command == "validate-robot":
        return cmd_validate_robot(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
