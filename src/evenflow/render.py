from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon

from .geometry import bounds, centroid
from .models import Layout


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


def save_layout_figure(layout: Layout, out_path: str | Path, **kwargs) -> None:
    fig, _ = render_layout(layout, **kwargs)
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
