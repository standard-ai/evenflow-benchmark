from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from evenflow.io import load_layout
from evenflow.models import (
    Layout,
    PlanResult,
    Robot,
    Scene,
    SceneFlow,
    SceneLayoutRef,
    SceneTracking,
    SceneWindow,
    Task,
    TaskRobot,
    TaskSceneRef,
    TrackSimple,
)
from evenflow.validation import (
    ValidationError,
    validate_layout,
    validate_plan_result,
    validate_robot,
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
        layout=SceneLayoutRef(
            layout_id="layout_1",
            path="../fixtures/layout_1.json",
            coordinate_frame="layout_xy_meters",
        ),
        tracking=SceneTracking(
            tracking_id="scene_1",
            format="csv",
            path="tracks/scene_1.csv",
            timestamp_field="timestamp",
            track_id_field="person_track_id",
            coordinate_frame="layout_xy_meters",
        ),
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


def test_validate_scene_requires_tracking_path() -> None:
    scene = Scene(
        scene_id="scene_1",
        layout=SceneLayoutRef(
            layout_id="layout_1",
            path="../fixtures/layout_1.json",
        ),
        tracking=SceneTracking(
            tracking_id="scene_1",
            format="csv",
            path="",
            timestamp_field="timestamp",
            track_id_field="person_track_id",
        ),
        window=SceneWindow(
            start="1971-01-01T00:00:00Z",
            end="1971-01-01T00:00:10Z",
        ),
    )

    with pytest.raises(ValidationError, match="scene.tracking.path is required"):
        validate_scene(scene)


def test_validate_task_accepts_valid_task() -> None:
    task = Task(
        task_id="task_1",
        scene=TaskSceneRef(
            scene_id="scene_1",
            path="../fixtures/scene_1.json",
        ),
        task_type="cross_flow",
        robot=TaskRobot(start=(1.0, 8.0), goal=(8.0, 2.0)),
    )

    validate_task(task)


def test_validate_task_requires_scene_id() -> None:
    task = Task(
        task_id="task_1",
        scene=TaskSceneRef(
            scene_id="",
            path="../fixtures/scene_1.json",
        ),
        task_type="cross_flow",
        robot=TaskRobot(start=(1.0, 8.0), goal=(8.0, 2.0)),
    )

    with pytest.raises(ValidationError, match="task.scene.scene_id is required"):
        validate_task(task)


def test_validate_robot_accepts_valid_robot() -> None:
    robot = Robot(
        robot_id="test.simple_disk",
        kinematics="holonomic",
        radius_m=0.25,
        max_speed_mps=1.2,
    )

    validate_robot(robot)


def test_validate_robot_requires_positive_radius() -> None:
    robot = Robot(
        robot_id="test.simple_disk",
        kinematics="holonomic",
        radius_m=0.0,
        max_speed_mps=1.2,
    )

    with pytest.raises(ValidationError, match="radius_m must be positive"):
        validate_robot(robot)


def test_validate_plan_result_accepts_valid_plan() -> None:
    track = TrackSimple(
        track_id="robot_plan",
        timestamps=np.array([0.0, 1.0, 2.0], dtype=float),
        x=np.array([1.0, 8.5, 8.5], dtype=float),
        y=np.array([8.5, 8.5, 2.0], dtype=float),
        vx=np.array([1.2, 0.0, 0.0], dtype=float),
        vy=np.array([0.0, -1.2, -1.2], dtype=float),
        position_valid=np.array([True, True, True], dtype=bool),
        velocity_valid=np.array([True, True, True], dtype=bool),
    )

    plan = PlanResult(
        planner_name="geometry",
        success=True,
        track=track,
        path_length_m=14.0,
        runtime_s=0.01,
        message="ok",
    )

    validate_plan_result(plan)


def test_validate_plan_result_rejects_successful_empty_plan() -> None:
    plan = PlanResult(
        planner_name="geometry",
        success=True,
        track=None,
    )

    with pytest.raises(
        ValidationError,
        match="successful plans must contain a track trajectory",
    ):
        validate_plan_result(plan)


def test_validate_plan_result_accepts_track_only_plan() -> None:
    track = TrackSimple(
        track_id="robot_plan",
        timestamps=np.array([0.0, 1.0], dtype=float),
        x=np.array([0.0, 1.0], dtype=float),
        y=np.array([0.0, 1.0], dtype=float),
    )

    plan = PlanResult(
        planner_name="geometry",
        success=True,
        track=track,
    )

    validate_plan_result(plan)
