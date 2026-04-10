from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import Layout, PlanResult, Robot, Scene, Task, EvalResult


def _scene_tracking_csv_path(
    scene: Scene,
    *,
    scene_json_path: str | Path | None = None,
) -> Path:
    tracking_path = Path(scene.tracking.path)

    if tracking_path.is_absolute():
        return tracking_path

    if scene_json_path is not None:
        return Path(scene_json_path).resolve().parent / tracking_path

    return tracking_path


def _load_human_points(
    scene: Scene,
    *,
    scene_json_path: str | Path | None = None,
    x_field: str = "bkg_x",
    y_field: str = "bkg_y",
) -> list[tuple[float, float]]:
    if scene.tracking.format.lower() != "csv":
        raise ValueError(f"Unsupported tracking format: {scene.tracking.format}")

    csv_path = _scene_tracking_csv_path(scene, scene_json_path=scene_json_path)

    points: list[tuple[float, float]] = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x = float(row[x_field])
                y = float(row[y_field])
            except (KeyError, TypeError, ValueError):
                continue
            points.append((x, y))

    return points


def _compute_min_human_distance_m(
    plan: PlanResult,
    human_points: list[tuple[float, float]],
) -> float | None:
    if not plan.success or not plan.waypoints:
        return None

    if not human_points:
        return None

    best = float("inf")
    for wp in plan.waypoints:
        for hx, hy in human_points:
            d = math.hypot(wp.x - hx, wp.y - hy)
            if d < best:
                best = d

    return best if best != float("inf") else None


def evaluate_plan(
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    plan: PlanResult,
    *,
    scene_json_path: str | Path | None = None,
    tracks_x_field: str = "bkg_x",
    tracks_y_field: str = "bkg_y",
) -> EvalResult:
    min_human_distance_m: float | None = None

    try:
        human_points = _load_human_points(
            scene,
            scene_json_path=scene_json_path,
            x_field=tracks_x_field,
            y_field=tracks_y_field,
        )
        min_human_distance_m = _compute_min_human_distance_m(plan, human_points)
    except Exception as e:
        # Keep evaluation lightweight and non-fatal at this stage.
        return EvalResult(
            success=plan.success,
            path_length_m=plan.path_length_m,
            runtime_s=plan.runtime_s,
            num_waypoints=len(plan.waypoints),
            min_human_distance_m=None,
            message=plan.message,
            metadata={
                "planner": plan.planner_name,
                "min_human_distance_error": str(e),
            },
        )

    return EvalResult(
        success=plan.success,
        path_length_m=plan.path_length_m,
        runtime_s=plan.runtime_s,
        num_waypoints=len(plan.waypoints),
        min_human_distance_m=min_human_distance_m,
        message=plan.message,
        metadata={
            "planner": plan.planner_name,
        },
    )
