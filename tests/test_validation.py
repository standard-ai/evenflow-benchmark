from __future__ import annotations

from pathlib import Path

import pytest

from evenflow.io import load_layout
from evenflow.models import (
    Layout,
    Scene,
    SceneFlow,
    SceneWindow,
    Task,
    TaskRobot,
)
from evenflow.validation import (
    ValidationError,
    validate_layout,
    validate_scene,
    validate_task,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_validate_layout_accepts_valid_fixture() -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json", validate=False)
    validate_layout(layout)


def test_validate_layout_requires_layout_id() -> None:
    layout = Layout(layout_id="", boundary=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])

    with pytest.raises(ValidationError, match="layout_id is required"):
        validate_layout(layout)


def test_validate_layout_rejects_too_small_boundary() -> None:
    layout = Layout(layout_id="bad", boundary=[(0.0, 0.0), (1.0, 0.0)])

    with pytest.raises(ValidationError, match="boundary must have at least 3 points"):
        validate_layout(layout)


def test_validate_scene_accepts_valid_scene() -> None:
    scene = Scene(
        scene_id="scene_1",
        layout_id="layout_1",
        tracks_file="tracks/scene_1.json",
        window=SceneWindow(
            start="1971-01-01T00:00:00Z",
            end="1971-01-01T00:00:10Z",
            duration_s=10.0,
        ),
        flow=SceneFlow(
            p_star=(5.0, 5.0),
            u_hat=(0.0, -1.0),
        ),
    )

    validate_scene(scene)


def test_validate_scene_requires_tracks_file() -> None:
    scene = Scene(
        scene_id="scene_1",
        layout_id="layout_1",
        tracks_file="",
        window=SceneWindow(
            start="1971-01-01T00:00:00Z",
            end="1971-01-01T00:00:10Z",
        ),
    )

    with pytest.raises(ValidationError, match="tracks_file is required"):
        validate_scene(scene)


def test_validate_task_accepts_valid_task() -> None:
    task = Task(
        task_id="task_1",
        scene_id="scene_1",
        task_type="cross_flow",
        robot=TaskRobot(start=(1.0, 8.0), goal=(8.0, 2.0)),
    )

    validate_task(task)


def test_validate_task_requires_scene_id() -> None:
    task = Task(
        task_id="task_1",
        scene_id="",
        task_type="cross_flow",
        robot=TaskRobot(start=(1.0, 8.0), goal=(8.0, 2.0)),
    )

    with pytest.raises(ValidationError, match="scene_id is required"):
        validate_task(task)
