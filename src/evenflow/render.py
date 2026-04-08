from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

from .geometry import bounds, centroid
from .models import Layout, PlanResult, Scene, Task


def render_layout(
    layout: Layout,
    *,
    ax: Axes | None = None,
    show_obstacle_labels: bool = True,
    show_exit_labels: bool = True,
    padding: float = 1.0,
    title: str | None = None,
    grid: bool = True,
) -> tuple[Figure, Axes]:
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    ax.add_patch(MplPolygon(layout.boundary, closed=True, fill=False, linewidth=2))

    for obs in layout.obstacles:
        ax.add_patch(MplPolygon(obs.polygon, closed=True, alpha=0.35))
        if show_obstacle_labels:
            cx, cy = centroid(obs.polygon)
            ax.text(cx, cy, obs.id, fontsize=6, ha="center", va="center")

    for ex in layout.exits:
        ax.add_patch(
            MplPolygon(ex.polygon, closed=True, fill=False, linestyle="--", linewidth=1.5)
        )
        if show_exit_labels:
            cx, cy = centroid(ex.polygon)
            ax.text(cx, cy, ex.id, fontsize=7, ha="center", va="center")

    min_x, min_y, max_x, max_y = bounds(layout.boundary)
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"EvenFlow Layout: {layout.layout_id}")
    ax.grid(grid, alpha=0.3)

    return fig, ax


def _legend_outside(ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)


def _scene_tracking_csv_path(scene_json_path: str | Path, tracking_rel_path: str) -> Path:
    return Path(scene_json_path).resolve().parent / tracking_rel_path


def draw_scene_tracks(
    ax: Axes,
    scene: Scene,
    *,
    scene_json_path: str | Path,
    x_field: str = "bkg_x",
    y_field: str = "bkg_y",
    max_tracks: int | None = None,
    alpha: float = 0.8,
    linewidth: float = 1.5,
) -> None:
    if scene.tracking.format.lower() != "csv":
        raise ValueError(f"Unsupported tracking format: {scene.tracking.format}")

    csv_path = _scene_tracking_csv_path(scene_json_path, scene.tracking.path)

    tracks: dict[str, list[tuple[float, float]]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = row[scene.tracking.track_id_field]
            try:
                x = float(row[x_field])
                y = float(row[y_field])
            except (KeyError, TypeError, ValueError):
                continue
            tracks.setdefault(track_id, []).append((x, y))

    track_items = list(tracks.items())
    if max_tracks is not None:
        track_items = track_items[:max_tracks]

    for i, (_, pts) in enumerate(track_items):
        if len(pts) < 2:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        label = "tracks" if i == 0 else None
        ax.plot(xs, ys, alpha=alpha, linewidth=linewidth, zorder=2, label=label)


def draw_scene(
    ax: Axes,
    scene: Scene,
    *,
    annotate: bool = True,
    show_p_star: bool = True,
    show_u_hat: bool = True,
    u_hat_scale: float = 1.0,
) -> None:
    if scene.flow is None:
        return

    if show_p_star and scene.flow.p_star is not None:
        px, py = scene.flow.p_star
        ax.plot(
            [px],
            [py],
            marker="*",
            linestyle="None",
            markersize=14,
            zorder=4,
            label="p*",
        )
        if annotate:
            ax.text(px, py, "p*", fontsize=8, ha="left", va="bottom")

    if (
        show_u_hat
        and scene.flow.p_star is not None
        and scene.flow.u_hat is not None
    ):
        px, py = scene.flow.p_star
        ux, uy = scene.flow.u_hat

        mag = math.hypot(ux, uy)
        if mag > 0:
            ux /= mag
            uy /= mag

            ax.arrow(
                px,
                py,
                ux * u_hat_scale,
                uy * u_hat_scale,
                length_includes_head=True,
                head_width=0.15,
                head_length=0.25,
                linewidth=2,
                alpha=0.8,
                zorder=3,
            )

            if annotate:
                ax.text(
                    px + ux * u_hat_scale,
                    py + uy * u_hat_scale,
                    "u_hat",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                )


def draw_task(
    ax: Axes,
    task: Task,
    *,
    annotate: bool = False,
    show_straight_line: bool = True,
) -> None:
    sx, sy = task.robot.start
    gx, gy = task.robot.goal

    if show_straight_line:
        ax.plot(
            [sx, gx],
            [sy, gy],
            linestyle="-",
            linewidth=1.8,
            alpha=0.6,
            zorder=1,
            label="straight-line",
        )

    ax.plot(
        [sx],
        [sy],
        marker="o",
        linestyle="None",
        markersize=8,
        zorder=5,
        label="start",
    )

    ax.plot(
        [gx],
        [gy],
        marker="x",
        linestyle="None",
        markersize=9,
        mew=2,
        zorder=5,
        label="goal",
    )

    if annotate:
        ax.text(sx, sy, "start", fontsize=8, ha="left", va="bottom")
        ax.text(gx, gy, "goal", fontsize=8, ha="left", va="bottom")


def draw_plan(
    ax: Axes,
    plan: PlanResult,
    *,
    annotate: bool = False,
    show_waypoints: bool = False,
    linewidth: float = 2.5,
    alpha: float = 0.9,
) -> None:
    """
    Draw a planner output on an existing axes.

    If the plan is unsuccessful or has no waypoints, nothing is drawn.
    """
    if not plan.success or not plan.waypoints:
        return

    xs = [wp.x for wp in plan.waypoints]
    ys = [wp.y for wp in plan.waypoints]

    ax.plot(
        xs,
        ys,
        linewidth=linewidth,
        alpha=alpha,
        zorder=6,
        label=f"plan:{plan.planner_name}",
    )

    if show_waypoints:
        ax.plot(
            xs,
            ys,
            marker=".",
            linestyle="None",
            markersize=6,
            alpha=alpha,
            zorder=7,
            label="plan-waypoints",
        )

    if annotate and plan.waypoints:
        start = plan.waypoints[0]
        end = plan.waypoints[-1]
        ax.text(start.x, start.y, f"{plan.planner_name}:start", fontsize=8, ha="left", va="bottom")
        ax.text(end.x, end.y, f"{plan.planner_name}:end", fontsize=8, ha="left", va="bottom")


def save_layout_figure(layout: Layout, out_path: str | Path, **kwargs) -> None:
    fig, _ = render_layout(layout, **kwargs)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scene_figure(
    layout: Layout,
    scene: Scene,
    out_path: str | Path,
    *,
    scene_json_path: str | Path | None = None,
    show_tracks: bool = False,
    tracks_x_field: str = "bkg_x",
    tracks_y_field: str = "bkg_y",
    max_tracks: int | None = None,
    show_obstacle_labels: bool = True,
    show_exit_labels: bool = True,
    annotate_scene: bool = True,
    show_p_star: bool = True,
    show_u_hat: bool = True,
    u_hat_scale: float = 1.0,
    title: str | None = None,
) -> None:
    fig, ax = render_layout(
        layout,
        show_obstacle_labels=show_obstacle_labels,
        show_exit_labels=show_exit_labels,
        title=title or f"Scene: {scene.scene_id}",
    )

    if show_tracks:
        if scene_json_path is None:
            raise ValueError("scene_json_path is required when show_tracks=True")
        draw_scene_tracks(
            ax,
            scene,
            scene_json_path=scene_json_path,
            x_field=tracks_x_field,
            y_field=tracks_y_field,
            max_tracks=max_tracks,
        )

    draw_scene(
        ax,
        scene,
        annotate=annotate_scene,
        show_p_star=show_p_star,
        show_u_hat=show_u_hat,
        u_hat_scale=u_hat_scale,
    )

    _legend_outside(ax)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_task_figure(
    layout: Layout,
    task: Task,
    out_path: str | Path,
    *,
    show_obstacle_labels: bool = True,
    show_exit_labels: bool = True,
    annotate_task: bool = True,
    show_straight_line: bool = True,
    title: str | None = None,
) -> None:
    fig, ax = render_layout(
        layout,
        show_obstacle_labels=show_obstacle_labels,
        show_exit_labels=show_exit_labels,
        title=title or f"Task: {task.task_id} | {task.task_type}",
    )
    draw_task(
        ax,
        task,
        annotate=annotate_task,
        show_straight_line=show_straight_line,
    )
    _legend_outside(ax)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scene_task_figure(
    layout: Layout,
    scene: Scene,
    task: Task,
    out_path: str | Path,
    *,
    scene_json_path: str | Path | None = None,
    show_tracks: bool = False,
    tracks_x_field: str = "bkg_x",
    tracks_y_field: str = "bkg_y",
    max_tracks: int | None = None,
    show_obstacle_labels: bool = True,
    show_exit_labels: bool = True,
    annotate_scene: bool = True,
    annotate_task: bool = True,
    show_p_star: bool = True,
    show_u_hat: bool = True,
    u_hat_scale: float = 1.0,
    show_straight_line: bool = True,
    title: str | None = None,
) -> None:
    fig, ax = render_layout(
        layout,
        show_obstacle_labels=show_obstacle_labels,
        show_exit_labels=show_exit_labels,
        title=title or f"{scene.scene_id} | {task.task_id}",
    )

    if show_tracks:
        if scene_json_path is None:
            raise ValueError("scene_json_path is required when show_tracks=True")
        draw_scene_tracks(
            ax,
            scene,
            scene_json_path=scene_json_path,
            x_field=tracks_x_field,
            y_field=tracks_y_field,
            max_tracks=max_tracks,
        )

    draw_scene(
        ax,
        scene,
        annotate=annotate_scene,
        show_p_star=show_p_star,
        show_u_hat=show_u_hat,
        u_hat_scale=u_hat_scale,
    )
    draw_task(
        ax,
        task,
        annotate=annotate_task,
        show_straight_line=show_straight_line,
    )

    _legend_outside(ax)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_plan_figure(
    layout: Layout,
    plan: PlanResult,
    out_path: str | Path,
    *,
    show_obstacle_labels: bool = True,
    show_exit_labels: bool = True,
    annotate_plan: bool = True,
    show_waypoints: bool = False,
    title: str | None = None,
) -> None:
    fig, ax = render_layout(
        layout,
        show_obstacle_labels=show_obstacle_labels,
        show_exit_labels=show_exit_labels,
        title=title or f"Plan: {plan.planner_name}",
    )
    draw_plan(
        ax,
        plan,
        annotate=annotate_plan,
        show_waypoints=show_waypoints,
    )
    _legend_outside(ax)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_scene_task_plan_figure(
    layout: Layout,
    scene: Scene,
    task: Task,
    plan: PlanResult,
    out_path: str | Path,
    *,
    scene_json_path: str | Path | None = None,
    show_tracks: bool = False,
    tracks_x_field: str = "bkg_x",
    tracks_y_field: str = "bkg_y",
    max_tracks: int | None = None,
    show_obstacle_labels: bool = True,
    show_exit_labels: bool = True,
    annotate_scene: bool = True,
    annotate_task: bool = True,
    annotate_plan: bool = True,
    show_p_star: bool = True,
    show_u_hat: bool = True,
    u_hat_scale: float = 1.0,
    show_straight_line: bool = True,
    show_waypoints: bool = False,
    title: str | None = None,
) -> None:
    fig, ax = render_layout(
        layout,
        show_obstacle_labels=show_obstacle_labels,
        show_exit_labels=show_exit_labels,
        title=title or f"{scene.scene_id} | {task.task_id} | {plan.planner_name}",
    )

    if show_tracks:
        if scene_json_path is None:
            raise ValueError("scene_json_path is required when show_tracks=True")
        draw_scene_tracks(
            ax,
            scene,
            scene_json_path=scene_json_path,
            x_field=tracks_x_field,
            y_field=tracks_y_field,
            max_tracks=max_tracks,
        )

    draw_scene(
        ax,
        scene,
        annotate=annotate_scene,
        show_p_star=show_p_star,
        show_u_hat=show_u_hat,
        u_hat_scale=u_hat_scale,
    )
    draw_task(
        ax,
        task,
        annotate=annotate_task,
        show_straight_line=show_straight_line,
    )
    draw_plan(
        ax,
        plan,
        annotate=annotate_plan,
        show_waypoints=show_waypoints,
    )

    _legend_outside(ax)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
