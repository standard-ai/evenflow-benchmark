from __future__ import annotations

from pathlib import Path

from evenflow.io import load_plan


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_plan_reads_basic_fields() -> None:
    plan = load_plan(FIXTURES / "minimal_plan.json")

    assert plan.planner_name == "geometry"
    assert plan.success is True
    assert len(plan.waypoints) == 3
    assert plan.waypoints[0].x == 1.0
    assert plan.waypoints[0].y == 8.5
    assert plan.waypoints[-1].x == 8.5
    assert plan.waypoints[-1].y == 2.0
