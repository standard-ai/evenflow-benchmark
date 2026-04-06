from __future__ import annotations

import argparse

from .io import load_layout, load_scenario
from .render import render_layout, save_layout_figure
from .scenarios import draw_scenario
from .validation import validate_layout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evenflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_layout_parser = subparsers.add_parser("render-layout", help="Render a layout JSON to an image")
    render_layout_parser.add_argument("layout_json")
    render_layout_parser.add_argument("output_image")
    render_layout_parser.add_argument("--no-obstacle-labels", action="store_true")
    render_layout_parser.add_argument("--no-exit-labels", action="store_true")
    render_layout_parser.add_argument("--title", default=None)

    validate_parser = subparsers.add_parser("validate-layout", help="Validate a layout JSON file")
    validate_parser.add_argument("layout_json")

    render_scenario_parser = subparsers.add_parser(
        "render-scenario",
        help="Render a scenario on top of a layout",
    )
    render_scenario_parser.add_argument("layout_json")
    render_scenario_parser.add_argument("scenario_json")
    render_scenario_parser.add_argument("output_image")
    render_scenario_parser.add_argument("--title", default=None)

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


def cmd_validate_layout(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout_json, validate=False)
    validate_layout(layout)
    print(f"Layout OK: {layout.layout_id}")
    return 0


def cmd_render_scenario(args: argparse.Namespace) -> int:
    layout = load_layout(args.layout_json)
    scenario = load_scenario(args.scenario_json)
    fig, ax = render_layout(layout, title=args.title or f"Scenario: {scenario.scenario_id}")
    draw_scenario(ax, scenario)
    fig.tight_layout()
    fig.savefig(args.output_image, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output_image}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "render-layout":
        return cmd_render_layout(args)
    if args.command == "validate-layout":
        return cmd_validate_layout(args)
    if args.command == "render-scenario":
        return cmd_render_scenario(args)

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
