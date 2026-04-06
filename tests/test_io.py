from __future__ import annotations

from pathlib import Path

from evenflow.io import load_layout, load_scenario


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_layout_reads_basic_fields() -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json")

    assert layout.layout_id == "test.minimal_layout"
    assert len(layout.boundary) == 4
    assert len(layout.obstacles) == 2
    assert len(layout.exits) == 1
    assert layout.obstacles[0].id == "obs_1"
    assert layout.exits[0].id == "exit_1"


def test_load_scenario_reads_basic_fields() -> None:
    scenario = load_scenario(FIXTURES / "minimal_scenario.json")

    assert scenario.scenario_id == "test.goal_to_exit"
    assert scenario.layout_id == "test.minimal_layout"
    assert scenario.start == (1.0, 9.0)
    assert scenario.goal == (9.5, 0.5)
