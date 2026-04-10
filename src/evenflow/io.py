from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    EvalResult,
    Exit,
    Layout,
    Obstacle,
    PlanResult,
    PlanWaypoint,
    Robot,
    Scene,
    SceneFlow,
    SceneLayoutRef,
    SceneTracking,
    SceneWindow,
    Task,
    TaskRobot,
    TaskSceneRef,
)
from .validation import validate_layout


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


def load_layout(path: str | Path, *, validate: bool = True) -> Layout:
    data = _read_json(path)
    known = {"layout_id", "boundary", "obstacles", "exits"}

    layout = Layout(
        layout_id=data.get("layout_id", Path(path).stem),
        boundary=[tuple(p) for p in data["boundary"]],
        obstacles=[
            Obstacle(
                id=o["id"],
                polygon=[tuple(p) for p in o["polygon"]],
                metadata={k: v for k, v in o.items() if k not in {"id", "polygon"}},
            )
            for o in data.get("obstacles", [])
        ],
        exits=[
            Exit(
                id=e["id"],
                polygon=[tuple(p) for p in e["polygon"]],
                metadata={k: v for k, v in e.items() if k not in {"id", "polygon"}},
            )
            for e in data.get("exits", [])
        ],
        metadata={k: v for k, v in data.items() if k not in known},
    )

    if validate:
        validate_layout(layout)
    return layout


def load_scene(path: str | Path, *, validate: bool = True) -> Scene:
    data = _read_json(path)

    window_data = data["window"]
    layout_data = data["layout"]
    tracking_data = data["tracking"]
    flow_data = data.get("flow")

    scene = Scene(
        scene_id=data.get("scene_id", Path(path).stem),
        layout=SceneLayoutRef(
            layout_id=layout_data["layout_id"],
            path=layout_data["path"],
            coordinate_frame=layout_data.get("coordinate_frame"),
            metadata={
                k: v
                for k, v in layout_data.items()
                if k not in {"layout_id", "path", "coordinate_frame"}
            },
        ),
        tracking=SceneTracking(
            tracking_id=tracking_data.get("tracking_id"),
            format=tracking_data["format"],
            path=tracking_data["path"],
            timestamp_field=tracking_data["timestamp_field"],
            track_id_field=tracking_data["track_id_field"],
            coordinate_frame=tracking_data.get("coordinate_frame"),
            metadata={
                k: v
                for k, v in tracking_data.items()
                if k
                not in {
                    "tracking_id",
                    "format",
                    "path",
                    "timestamp_field",
                    "track_id_field",
                    "coordinate_frame",
                }
            },
        ),
        window=SceneWindow(
            start=window_data["start"],
            end=window_data["end"],
            duration_s=window_data.get("duration_s"),
        ),
        flow=SceneFlow(
            p_star=tuple(flow_data["p_star"])
            if flow_data and flow_data.get("p_star")
            else None,
            u_hat=tuple(flow_data["u_hat"])
            if flow_data and flow_data.get("u_hat")
            else None,
            metadata={
                k: v
                for k, v in (flow_data or {}).items()
                if k not in {"p_star", "u_hat"}
            },
        )
        if flow_data
        else None,
        metadata=data.get("metadata", {}),
        provenance=data.get("provenance", {}),
    )

    if validate:
        from .validation import validate_scene
        validate_scene(scene)

    return scene


def load_task(path: str | Path, *, validate: bool = True) -> Task:
    data = _read_json(path)

    robot_data = data["robot"]
    scene_data = data["scene"]

    task = Task(
        task_id=data.get("task_id", Path(path).stem),
        scene=TaskSceneRef(
            scene_id=scene_data["scene_id"],
            path=scene_data["path"],
            metadata={
                k: v for k, v in scene_data.items()
                if k not in {"scene_id", "path"}
            },
        ),
        task_type=data["task_type"],
        robot=TaskRobot(
            start=tuple(robot_data["start"]),
            goal=tuple(robot_data["goal"]),
        ),
        metadata=data.get("metadata", {}),
        provenance=data.get("provenance", {}),
    )

    if validate:
        from .validation import validate_task
        validate_task(task)

    return task


def load_robot(path: str | Path, *, validate: bool = True) -> Robot:
    data = _read_json(path)

    footprint = data.get("footprint", {})
    dynamics = data.get("dynamics", {})

    robot = Robot(
        robot_id=data.get("robot_id", Path(path).stem),
        kinematics=data["kinematics"],
        radius_m=float(footprint["radius_m"]),
        max_speed_mps=float(dynamics["max_speed_mps"]),
        metadata=data.get("metadata", {}),
    )

    if validate:
        from .validation import validate_robot
        validate_robot(robot)

    return robot


def load_plan(path: str | Path, *, validate: bool = True) -> PlanResult:
    data = _read_json(path)

    plan = PlanResult(
        planner_name=data["planner_name"],
        success=bool(data["success"]),
        waypoints=tuple(
            PlanWaypoint(
                x=float(wp["x"]),
                y=float(wp["y"]),
                t=float(wp["t"]) if wp.get("t") is not None else None,
            )
            for wp in data.get("waypoints", [])
        ),
        path_length_m=(
            float(data["path_length_m"])
            if data.get("path_length_m") is not None
            else None
        ),
        runtime_s=(
            float(data["runtime_s"])
            if data.get("runtime_s") is not None
            else None
        ),
        message=data.get("message", ""),
        metadata=data.get("metadata", {}),
    )

    if validate:
        from .validation import validate_plan_result
        validate_plan_result(plan)

    return plan


def save_plan(path: str | Path, plan: PlanResult) -> None:
    data = {
        "planner_name": plan.planner_name,
        "success": plan.success,
        "waypoints": [
            {
                "x": wp.x,
                "y": wp.y,
                "t": wp.t,
            }
            for wp in plan.waypoints
        ],
        "path_length_m": plan.path_length_m,
        "runtime_s": plan.runtime_s,
        "message": plan.message,
        "metadata": plan.metadata,
    }
    _write_json(path, data)


def load_eval(path: str | Path, *, validate: bool = False) -> EvalResult:
    data = _read_json(path)

    result = EvalResult(
        success=bool(data["success"]),
        path_length_m=(
            float(data["path_length_m"])
            if data.get("path_length_m") is not None
            else None
        ),
        runtime_s=(
            float(data["runtime_s"])
            if data.get("runtime_s") is not None
            else None
        ),
        num_waypoints=int(data["num_waypoints"]),
        min_human_distance_m=(
            float(data["min_human_distance_m"])
            if data.get("min_human_distance_m") is not None
            else None
        ),
        message=data.get("message", ""),
        metadata=data.get("metadata", {}),
    )

    if validate:
        from .validation import validate_eval_result
        validate_eval_result(result)

    return result


def save_eval(path: str | Path, result: EvalResult) -> None:
    data = {
        "success": result.success,
        "path_length_m": result.path_length_m,
        "runtime_s": result.runtime_s,
        "num_waypoints": result.num_waypoints,
        "min_human_distance_m": result.min_human_distance_m,
        "message": result.message,
        "metadata": result.metadata,
    }
    _write_json(path, data)
