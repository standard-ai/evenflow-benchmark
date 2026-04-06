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
class Scenario:
    scenario_id: str
    layout_id: str
    start: Point
    goal: Point
    metadata: dict[str, Any] = field(default_factory=dict)
