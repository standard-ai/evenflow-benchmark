from __future__ import annotations

import time

from .geometry import astar_grid, build_occupancy_grid, path_length
from .models import Layout, PlanResult, PlanWaypoint, Robot, Scene, Task
from .planner import BasePlanner


class GeometryPlanner(BasePlanner):
    """
    Reference geometric baseline planner.

    Uses only:
    - layout boundary
    - static obstacle polygons
    - task robot start/goal
    - robot footprint radius

    Ignores scene flow and tracking data.
    """

    def __init__(
        self,
        *,
        resolution_m: float = 0.05,
        allow_diagonal: bool = True,
    ) -> None:
        self.resolution_m = resolution_m
        self.allow_diagonal = allow_diagonal

    @property
    def name(self) -> str:
        return "geometry"

    def plan(
        self,
        layout: Layout,
        scene: Scene,
        task: Task,
        robot: Robot,
    ) -> PlanResult:
        t0 = time.perf_counter()

        grid = build_occupancy_grid(
            layout.boundary,
            [obs.polygon for obs in layout.obstacles],
            resolution_m=self.resolution_m,
            robot_radius=robot.radius_m,
        )

        try:
            path = astar_grid(
                grid,
                start_xy=task.robot.start,
                goal_xy=task.robot.goal,
                allow_diagonal=self.allow_diagonal,
            )
        except ValueError as e:
            runtime_s = time.perf_counter() - t0
            return PlanResult(
                planner_name=self.name,
                success=False,
                waypoints=(),
                path_length_m=None,
                runtime_s=runtime_s,
                message=str(e),
            )

        runtime_s = time.perf_counter() - t0

        if path is None:
            return PlanResult(
                planner_name=self.name,
                success=False,
                waypoints=(),
                path_length_m=None,
                runtime_s=runtime_s,
                message="No feasible path found.",
            )

        return PlanResult(
            planner_name=self.name,
            success=True,
            waypoints=tuple(PlanWaypoint(x=x, y=y) for x, y in path),
            path_length_m=path_length(path),
            runtime_s=runtime_s,
            message="ok",
        )
