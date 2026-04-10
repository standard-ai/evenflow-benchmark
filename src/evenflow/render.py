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

    ax.add_patch(
        MplPolygon(
            layout.boundary,
            closed=True,
            fill=False,
            linewidth=2,
            edgecolor="black",
            zorder=1,
        )
    )

    for obs in layout.obstacles:
        ax.add_patch(
            MplPolygon(
                obs.polygon,
                closed=True,
                facecolor="#4C78A8",
                edgecolor="none",
                alpha=0.25,
                zorder=1,
            )
        )
        if show_obstacle_labels:
            cx, cy = centroid(obs.polygon)
            ax.text(cx, cy, obs.id, fontsize=6, ha="center", va="center", zorder=2)

    for ex in layout.exits:
        ax.add_patch(
            MplPolygon(
                ex.polygon,
                closed=True,
                fill=False,
                linestyle="--",
                linewidth=1.5,
                edgecolor="black",
                zorder=1,
            )
        )
        if show_exit_labels:
            cx, cy = centroid(ex.polygon)
            ax.text(cx, cy, ex.id, fontsize=7, ha="center", va="center", zorder=2)

    min_x, min_y, max_x, max_y = bounds(layout.boundary)
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title or f"EvenFlow Layout: {layout.layout_id}")
    ax.grid(grid, alpha=0.25)

    return fig, ax


def _legend_outside(ax: Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
        )


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
    alpha: float = 0.45,
    linewidth: float = 1.0,
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
        ax.scatter(
            [px],
            [py],
            marker="*",
            s=260,
            edgecolors="black",
            linewidths=1.0,
            zorder=20,
            label="p*",
        )
        if annotate:
            ax.text(px + 0.05, py + 0.05, "p*", fontsize=9, ha="left", va="bottom", zorder=21)

    if show_u_hat and scene.flow.p_star is not None and scene.flow.u_hat is not None:
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
                head_width=0.14,
                head_length=0.22,
                linewidth=2,
                alpha=0.9,
                zorder=19,
            )

            if annotate:
                ax.text(
                    px + ux * u_hat_scale,
                    py + uy * u_hat_scale,
                    "u_hat",
                    fontsize=8,
                    ha="left",
                    va="bottom",
                    zorder=20,
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
            linestyle="--",
            color="gray",
            linewidth=1.5,
            alpha=0.7,
            zorder=3,
            label="straight-line",
        )

    ax.plot(
        [sx],
        [sy],
        marker="o",
        linestyle="None",
        markersize=9,
        zorder=8,
        label="start",
    )

    ax.plot(
        [gx],
        [gy],
        marker="x",
        linestyle="None",
        markersize=10,
        mew=2.2,
        zorder=8,
        label="goal",
    )

    if annotate:
        ax.text(sx, sy, "start", fontsize=8, ha="left", va="bottom", zorder=9)
        ax.text(gx, gy, "goal", fontsize=8, ha="left", va="bottom", zorder=9)


def draw_plan(
    ax: Axes,
    plan: PlanResult,
    *,
    annotate: bool = False,
    show_waypoints: bool = False,
    linewidth: float = 3.0,
    alpha: float = 0.95,
) -> None:
    if not plan.success or not plan.waypoints:
        return

    xs = [wp.x for wp in plan.waypoints]
    ys = [wp.y for wp in plan.waypoints]

    ax.plot(
        xs,
        ys,
        linewidth=linewidth,
        alpha=alpha,
        zorder=10,
        solid_joinstyle="round",
        solid_capstyle="round",
        label=f"plan:{plan.planner_name}",
    )

    if show_waypoints:
        ax.plot(
            xs,
            ys,
            marker=".",
            linestyle="None",
            markersize=5,
            alpha=alpha,
            zorder=11,
            label="plan-waypoints",
        )

    if annotate:
        ax.text(xs[0], ys[0] + 0.08, "start", fontsize=8, ha="left", va="bottom", zorder=12)
        ax.text(xs[-1], ys[-1] + 0.08, "goal", fontsize=8, ha="left", va="bottom", zorder=12)


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
