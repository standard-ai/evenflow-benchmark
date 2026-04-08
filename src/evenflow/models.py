from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

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


@dataclass(slots=True)
class Robot:
    robot_id: str
    kinematics: str
    radius_m: float
    max_speed_mps: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlanWaypoint:
    """
    A single waypoint in a planner output.

    Semantics:
    - x, y are layout-frame coordinates in meters.
    - t is optional and may be omitted by untimed planners.
    - If provided, t should be interpreted consistently within a plan
      (for example, seconds from plan start).
    """
    x: float
    y: float
    t: Optional[float] = None


@dataclass(frozen=True, slots=True)
class PlanResult:
    """
    Canonical planner output.

    Semantics:
    - planner_name identifies the planner implementation.
    - success=True means the planner produced a usable plan.
    - success=False means planning failed; waypoints may be empty.
    - waypoints is the returned path/trajectory representation.
    - path_length_m and runtime_s are optional summary fields.
    - message is a human-readable status/debug string.
    - metadata may contain planner-specific diagnostics.
    """
    planner_name: str
    success: bool
    waypoints: Tuple[PlanWaypoint, ...]
    path_length_m: Optional[float] = None
    runtime_s: Optional[float] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    success: bool
    path_length_m: float | None
    runtime_s: float | None
    num_waypoints: int
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
