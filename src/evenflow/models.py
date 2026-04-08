from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Point = tuple[float, float]
Polygon = list[Point]


@dataclass(slots=True)
class Obstacle:
    id: str
    polygon: Polygon
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Exit:
    id: str
    polygon: Polygon
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Layout:
    layout_id: str
    boundary: Polygon
    obstacles: list[Obstacle] = field(default_factory=list)
    exits: list[Exit] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneWindow:
    start: str
    end: str
    duration_s: float | None = None


@dataclass(slots=True)
class SceneTracking:
    format: str
    path: str
    timestamp_field: str
    track_id_field: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneFlow:
    p_star: Point | None = None
    u_hat: Point | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Scene:
    scene_id: str
    layout_id: str
    tracking: SceneTracking
    window: SceneWindow
    flow: SceneFlow | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskRobot:
    start: Point
    goal: Point


@dataclass(slots=True)
class Task:
    task_id: str
    scene_id: str
    task_type: str
    robot: TaskRobot
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
