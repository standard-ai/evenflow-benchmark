from __future__ import annotations

import math
import time

import numpy as np

from .geometry import astar_grid, build_occupancy_grid, path_length
from .models import Layout, PlanResult, Robot, Scene, Task, TrackSimple
from .planner import BasePlanner


def _path_to_track_simple(
    path: list[tuple[float, float]],
    *,
    track_id: str,
    max_speed_mps: float,
) -> TrackSimple:
    """
    Convert a geometric path into a time-indexed TrackSimple.

    Timing policy:
    - start at t=0
    - traverse each segment at constant speed = max_speed_mps
    - no waiting
    - segment velocity is assigned to the segment start sample
    - final velocity sample is copied from the previous segment when possible
    """
    if len(path) < 2:
        raise ValueError("path must contain at least 2 points to build a track")

    if max_speed_mps <= 0:
        raise ValueError("max_speed_mps must be positive")

    xs = np.asarray([p[0] for p in path], dtype=float)
    ys = np.asarray([p[1] for p in path], dtype=float)

    timestamps = np.zeros(len(path), dtype=float)
    vx = np.zeros(len(path), dtype=float)
    vy = np.zeros(len(path), dtype=float)

    for i in range(1, len(path)):
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        seg_len = math.hypot(dx, dy)
        dt = seg_len / max_speed_mps

        timestamps[i] = timestamps[i - 1] + dt

        if dt > 0:
            vx[i - 1] = dx / dt
            vy[i - 1] = dy / dt
        else:
            vx[i - 1] = 0.0
            vy[i - 1] = 0.0

    if len(path) >= 2:
        vx[-1] = vx[-2]
        vy[-1] = vy[-2]

    position_valid = np.ones(len(path), dtype=bool)
    velocity_valid = np.ones(len(path), dtype=bool)

    return TrackSimple(
        track_id=track_id,
        timestamps=timestamps,
        x=xs,
        y=ys,
        vx=vx,
        vy=vy,
        position_valid=position_valid,
        velocity_valid=velocity_valid,
        metadata={
            "source": "geometry_planner",
            "timing_policy": "constant_speed_no_wait",
            "max_speed_mps": max_speed_mps,
        },
    )


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
                track=None,
                path_length_m=None,
                runtime_s=runtime_s,
                message=str(e),
            )

        runtime_s = time.perf_counter() - t0

        if path is None:
            return PlanResult(
                planner_name=self.name,
                success=False,
                track=None,
                path_length_m=None,
                runtime_s=runtime_s,
                message="No feasible path found.",
            )

        track = _path_to_track_simple(
            path,
            track_id=f"{self.name}_plan",
            max_speed_mps=robot.max_speed_mps,
        )

        return PlanResult(
            planner_name=self.name,
            success=True,
            track=track,
            path_length_m=path_length(path),
            runtime_s=runtime_s,
            message="ok",
        )
