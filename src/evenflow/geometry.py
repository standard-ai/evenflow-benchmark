from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Iterable

from .models import Point, Polygon


def centroid(poly: Polygon) -> Point:
    if not poly:
        raise ValueError("Cannot compute centroid of an empty polygon")
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def bounds(poly: Polygon) -> tuple[float, float, float, float]:
    if not poly:
        raise ValueError("Cannot compute bounds of an empty polygon")
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


GridIndex = tuple[int, int]


def point_in_polygon(point: Point, poly: Polygon) -> bool:
    """
    Ray-casting point-in-polygon test.
    Suitable for coarse occupancy-grid planning.
    """
    x, y = point
    inside = False

    n = len(poly)
    if n < 3:
        return False

    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
        )
        if intersects:
            inside = not inside
        j = i

    return inside


def distance_point_to_segment(point: Point, a: Point, b: Point) -> float:
    px, py = point
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay
    seg_len_sq = dx * dx + dy * dy

    if seg_len_sq == 0.0:
        return math.hypot(px - ax, py - ay)

    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    qx = ax + t * dx
    qy = ay + t * dy
    return math.hypot(px - qx, py - qy)


def distance_point_to_polygon(point: Point, poly: Polygon) -> float:
    if not poly:
        raise ValueError("Cannot compute distance to an empty polygon")

    best = float("inf")
    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        best = min(best, distance_point_to_segment(point, a, b))
    return best


def is_collision_free_point(
    point: Point,
    boundary: Polygon,
    obstacles: list[Polygon],
    robot_radius: float = 0.0,
) -> bool:
    """
    Conservative point feasibility check.

    A point is free iff:
    - it lies inside the boundary
    - it is not inside any obstacle
    - if robot_radius > 0, it remains at least that far from obstacle edges
    """
    if not point_in_polygon(point, boundary):
        return False

    for obs in obstacles:
        if point_in_polygon(point, obs):
            return False
        if robot_radius > 0.0 and distance_point_to_polygon(point, obs) < robot_radius:
            return False

    return True


@dataclass(slots=True)
class OccupancyGrid:
    resolution_m: float
    min_x: float
    min_y: float
    width: int
    height: int
    occupied: list[list[bool]]

    def in_bounds(self, ij: GridIndex) -> bool:
        i, j = ij
        return 0 <= i < self.height and 0 <= j < self.width

    def is_free(self, ij: GridIndex) -> bool:
        return self.in_bounds(ij) and (not self.occupied[ij[0]][ij[1]])

    def world_to_grid(self, point: Point) -> GridIndex:
        x, y = point
        j = int(math.floor((x - self.min_x) / self.resolution_m))
        i = int(math.floor((y - self.min_y) / self.resolution_m))
        return (i, j)

    def grid_to_world(self, ij: GridIndex) -> Point:
        i, j = ij
        x = self.min_x + (j + 0.5) * self.resolution_m
        y = self.min_y + (i + 0.5) * self.resolution_m
        return (x, y)


def build_occupancy_grid(
    boundary: Polygon,
    obstacles: list[Polygon],
    *,
    resolution_m: float = 0.05,
    robot_radius: float = 0.0,
    pad_cells: int = 1,
) -> OccupancyGrid:
    min_x, min_y, max_x, max_y = bounds(boundary)
    min_x -= pad_cells * resolution_m
    min_y -= pad_cells * resolution_m
    max_x += pad_cells * resolution_m
    max_y += pad_cells * resolution_m

    width = int(math.ceil((max_x - min_x) / resolution_m))
    height = int(math.ceil((max_y - min_y) / resolution_m))

    occupied: list[list[bool]] = []
    for i in range(height):
        row: list[bool] = []
        for j in range(width):
            x = min_x + (j + 0.5) * resolution_m
            y = min_y + (i + 0.5) * resolution_m
            free = is_collision_free_point(
                (x, y),
                boundary=boundary,
                obstacles=obstacles,
                robot_radius=robot_radius,
            )
            row.append(not free)
        occupied.append(row)

    return OccupancyGrid(
        resolution_m=resolution_m,
        min_x=min_x,
        min_y=min_y,
        width=width,
        height=height,
        occupied=occupied,
    )


def nearest_free_cell(
    grid: OccupancyGrid,
    start_ij: GridIndex,
    *,
    max_radius_cells: int = 20,
) -> GridIndex:
    if grid.is_free(start_ij):
        return start_ij

    si, sj = start_ij
    for r in range(1, max_radius_cells + 1):
        for i in range(si - r, si + r + 1):
            for j in range(sj - r, sj + r + 1):
                if grid.is_free((i, j)):
                    return (i, j)

    raise ValueError(f"No nearby free cell found for grid index {start_ij}")


def grid_neighbors(ij: GridIndex, *, allow_diagonal: bool = True) -> Iterable[GridIndex]:
    i, j = ij

    yield (i - 1, j)
    yield (i + 1, j)
    yield (i, j - 1)
    yield (i, j + 1)

    if allow_diagonal:
        yield (i - 1, j - 1)
        yield (i - 1, j + 1)
        yield (i + 1, j - 1)
        yield (i + 1, j + 1)


def grid_step_cost(a: GridIndex, b: GridIndex, resolution_m: float) -> float:
    di = abs(a[0] - b[0])
    dj = abs(a[1] - b[1])
    if di == 1 and dj == 1:
        return math.sqrt(2.0) * resolution_m
    return resolution_m


def grid_heuristic(grid: OccupancyGrid, a: GridIndex, b: GridIndex) -> float:
    ax, ay = grid.grid_to_world(a)
    bx, by = grid.grid_to_world(b)
    return math.hypot(ax - bx, ay - by)


def reconstruct_grid_path(
    came_from: dict[GridIndex, GridIndex],
    current: GridIndex,
) -> list[GridIndex]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar_grid(
    grid: OccupancyGrid,
    *,
    start_xy: Point,
    goal_xy: Point,
    allow_diagonal: bool = True,
) -> list[Point] | None:
    start_ij = nearest_free_cell(grid, grid.world_to_grid(start_xy))
    goal_ij = nearest_free_cell(grid, grid.world_to_grid(goal_xy))

    open_heap: list[tuple[float, int, GridIndex]] = []
    came_from: dict[GridIndex, GridIndex] = {}
    g_cost: dict[GridIndex, float] = {start_ij: 0.0}
    closed: set[GridIndex] = set()

    push_id = 0
    heapq.heappush(open_heap, (grid_heuristic(grid, start_ij, goal_ij), push_id, start_ij))

    while open_heap:
        _, _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal_ij:
            path_ij = reconstruct_grid_path(came_from, current)
            path_xy = [grid.grid_to_world(ij) for ij in path_ij]
            if path_xy:
                path_xy[0] = start_xy
                path_xy[-1] = goal_xy
            return path_xy

        for nbr in grid_neighbors(current, allow_diagonal=allow_diagonal):
            if not grid.is_free(nbr) or nbr in closed:
                continue

            tentative_g = g_cost[current] + grid_step_cost(current, nbr, grid.resolution_m)
            if tentative_g < g_cost.get(nbr, float("inf")):
                came_from[nbr] = current
                g_cost[nbr] = tentative_g
                push_id += 1
                f_cost = tentative_g + grid_heuristic(grid, nbr, goal_ij)
                heapq.heappush(open_heap, (f_cost, push_id, nbr))

    return None


def path_length(path: list[Point]) -> float:
    if len(path) < 2:
        return 0.0

    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        total += math.hypot(b[0] - a[0], b[1] - a[1])
    return total
