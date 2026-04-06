from __future__ import annotations

from matplotlib.axes import Axes

from .models import Scenario


def draw_scenario(ax: Axes, scenario: Scenario, *, annotate: bool = True) -> None:
    sx, sy = scenario.start
    gx, gy = scenario.goal

    ax.plot([sx], [sy], marker="o")
    ax.plot([gx], [gy], marker="*")
    ax.plot([sx, gx], [sy, gy], linestyle=":")

    if annotate:
        ax.text(sx, sy, f"start: {scenario.scenario_id}", fontsize=8, ha="left", va="bottom")
        ax.text(gx, gy, "goal", fontsize=8, ha="left", va="bottom")
