from __future__ import annotations

from pathlib import Path

from evenflow.io import load_layout, load_robot, load_scene, load_task
from evenflow.planners import GeometryPlanner


FIXTURES = Path(__file__).parent / "fixtures"


def test_geometry_planner_returns_successful_plan() -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json")
    scene = load_scene(FIXTURES / "minimal_scene.json")
    task = load_task(FIXTURES / "minimal_task.json")
    robot = load_robot(FIXTURES / "minimal_robot.json")

    planner = GeometryPlanner()
    result = planner.plan(layout, scene, task, robot)

    assert result.planner_name == "geometry"
    assert result.success is True
    assert result.path_length_m is not None
    assert result.path_length_m > 0.0
    assert result.track is not None
    assert result.track.num_samples() >= 2
    assert result.track.x[0] == task.robot.start[0]
    assert result.track.y[0] == task.robot.start[1]
    assert result.track.x[-1] == task.robot.goal[0]
    assert result.track.y[-1] == task.robot.goal[1]
    assert result.runtime_s is not None


def test_geometry_planner_can_fail_for_impossible_goal() -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json")
    scene = load_scene(FIXTURES / "minimal_scene.json")
    robot = load_robot(FIXTURES / "minimal_robot.json")

    from evenflow.models import Task, TaskRobot, TaskSceneRef

    bad_task = Task(
        task_id="bad_task",
        scene=TaskSceneRef(
            scene_id=scene.scene_id,
            path="../fixtures/minimal_scene.json",
        ),
        task_type="cross_flow",
        robot=TaskRobot(start=(1.0, 8.5), goal=(-999.0, -999.0)),
    )

    planner = GeometryPlanner()
    result = planner.plan(layout, scene, bad_task, robot)

    assert result.success is False
    assert result.track is None
