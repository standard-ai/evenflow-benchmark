from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import (
    Exit,
    Layout,
    Obstacle,
    Scene,
    SceneFlow,
    SceneTracking,
    SceneWindow,
    Task,
    TaskRobot,
)
from .validation import validate_layout


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


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
    tracking_data = data["tracking"]
    flow_data = data.get("flow")

    scene = Scene(
        scene_id=data.get("scene_id", Path(path).stem),
        layout_id=data["layout_id"],
        tracking=SceneTracking(
            format=tracking_data["format"],
            path=tracking_data["path"],
            timestamp_field=tracking_data["timestamp_field"],
            track_id_field=tracking_data["track_id_field"],
            metadata={
                k: v
                for k, v in tracking_data.items()
                if k not in {"format", "path", "timestamp_field", "track_id_field"}
            },
        ),
        window=SceneWindow(
            start=window_data["start"],
            end=window_data["end"],
            duration_s=window_data.get("duration_s"),
        ),
        flow=SceneFlow(
            p_star=tuple(flow_data["p_star"]) if flow_data and flow_data.get("p_star") else None,
            u_hat=tuple(flow_data["u_hat"]) if flow_data and flow_data.get("u_hat") else None,
            metadata={
                k: v for k, v in (flow_data or {}).items()
                if k not in {"p_star", "u_hat"}
            },
        ) if flow_data else None,
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

    task = Task(
        task_id=data.get("task_id", Path(path).stem),
        scene_id=data["scene_id"],
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
