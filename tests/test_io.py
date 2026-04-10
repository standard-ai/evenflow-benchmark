from __future__ import annotations

from pathlib import Path

from evenflow.io import load_layout, load_scene, load_task


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_layout_reads_basic_fields() -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json")

    assert layout.layout_id == "test.minimal_layout"
    assert len(layout.boundary) == 4
    assert len(layout.obstacles) == 2
    assert len(layout.exits) == 1
    assert layout.obstacles[0].id == "obs_1"
    assert layout.exits[0].id == "exit_1"


def test_load_scene_reads_basic_fields() -> None:
    scene = load_scene(FIXTURES / "minimal_scene.json")

    assert scene.scene_id == "test.simple"
    assert scene.layout.layout_id == "test.minimal_layout"
    assert scene.layout.path == "../fixtures/minimal_layout.json"
    assert scene.window.duration_s == 10.0
    assert scene.flow.p_star == (5.0, 5.0)


def test_load_task_reads_basic_fields() -> None:
    task = load_task(FIXTURES / "minimal_task.json")

    assert task.scene.scene_id == "test.simple"
    assert task.scene.path == "../fixtures/minimal_scene.json"
    assert task.task_type == "cross_flow"
    assert task.robot.start == (1.0, 8.5)
    assert task.robot.goal == (8.5, 2.0)
