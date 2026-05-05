#!/usr/bin/env python3
"""
Minimal track-aware EvenFlow planner example.

This script demonstrates how to:
1. Load a benchmark task and robot config.
2. Resolve the referenced scene.
3. Parse the scene's tracks.csv using EvenFlow helpers.
4. Produce a valid PlanResult JSON.

Usage:
  python examples/planners/track_aware_demo.py \
    data/benchmark/aligned_flow/tasks/aligned_flow.af_0001.task.json \
    examples/robots/simple_disk.json \
    outputs/track_aware_plan.json
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from evenflow.io import load_robot, load_scene, load_task, load_track_store, save_plan
from evenflow.models import PlanResult, TrackSimple


def resolve_relative(base_file: str | Path, relative_path: str | Path) -> Path:
    """Resolve a path stored inside an EvenFlow JSON artifact."""
    return (Path(base_file).resolve().parent / relative_path).resolve()


def make_track_aware_demo_plan(task_json: str | Path, robot_json: str | Path) -> PlanResult:
    """
    Build a very simple track-aware demo plan.

    This is not intended to be a strong planner. It only demonstrates how to
    parse surrounding human tracks and use them in planner logic.

    Behavior:
    - load task / robot / scene
    - parse the scene's tracks.csv into a TrackStore
    - estimate the mean surrounding human flow direction
    - generate a simple start-to-goal trajectory
    - store track-derived metadata in the PlanResult
    """
    t_start_wall = time.perf_counter()

    task = load_task(task_json)
    robot = load_robot(robot_json)

    scene_json = resolve_relative(task_json, task.scene.path)
    scene = load_scene(scene_json)

    # Important part: this helper parses the scene's tracks.csv.
    track_store = load_track_store(scene, scene_json_path=scene_json)

    # Extract surrounding human tracks as canonical TrackSimple objects.
    human_tracks = list(track_store.iter_simple_tracks())

    # Optionally exclude the focal/target human if present.
    if task.target is not None:
        human_tracks = [
            tr for tr in human_tracks
            if str(tr.track_id) != str(task.target.track_id)
        ]

    # Estimate a crude local flow direction from observed human velocities.
    velocity_chunks: list[np.ndarray] = []
    for tr in human_tracks:
        if not tr.has_velocity():
            continue

        vxy = tr.vxy()
        if vxy is None or len(vxy) == 0:
            continue

        speeds = np.linalg.norm(vxy, axis=1)
        good = np.isfinite(speeds) & (speeds > 0.05)
        if np.any(good):
            velocity_chunks.append(vxy[good])

    if velocity_chunks:
        velocities = np.vstack(velocity_chunks)
        mean_flow = np.nanmean(velocities, axis=0)
        norm = float(np.linalg.norm(mean_flow))
        if norm > 1e-9:
            mean_flow = mean_flow / norm
        else:
            mean_flow = np.zeros(2, dtype=float)
    else:
        mean_flow = np.zeros(2, dtype=float)

    start = np.asarray(task.robot.start, dtype=float)
    goal = np.asarray(task.robot.goal, dtype=float)
    direct = goal - start
    distance = float(np.linalg.norm(direct))

    if distance <= 1e-9:
        direction = np.zeros(2, dtype=float)
    else:
        direction = direct / distance

    # Demonstrate using human motion: add a small flow bias.
    # This is intentionally simple; real planners should reason about collisions,
    # timing, geometry, and dynamic context.
    biased_direction = direction + 0.15 * mean_flow
    biased_norm = float(np.linalg.norm(biased_direction))
    if biased_norm > 1e-9:
        biased_direction = biased_direction / biased_norm
    else:
        biased_direction = direction

    # Preserve the task endpoint contract in this minimal example.
    # The flow estimate is stored in metadata rather than changing the endpoint.
    max_speed = float(robot.max_speed_mps)
    duration = distance / max(max_speed, 1e-6)

    n_samples = max(2, int(np.ceil(duration / 0.1)) + 1)
    timestamps = np.linspace(0.0, duration, n_samples)
    alpha = np.linspace(0.0, 1.0, n_samples)

    xy = start[None, :] * (1.0 - alpha[:, None]) + goal[None, :] * alpha[:, None]

    if n_samples > 1 and duration > 0:
        vx = np.gradient(xy[:, 0], timestamps)
        vy = np.gradient(xy[:, 1], timestamps)
    else:
        vx = np.zeros(n_samples)
        vy = np.zeros(n_samples)

    track = TrackSimple(
        track_id="track_aware_demo_plan",
        timestamps=timestamps,
        x=xy[:, 0],
        y=xy[:, 1],
        vx=vx,
        vy=vy,
        position_valid=np.ones(n_samples, dtype=bool),
        velocity_valid=np.ones(n_samples, dtype=bool),
        metadata={
            "source": "track_aware_demo",
            "mean_flow_x": float(mean_flow[0]),
            "mean_flow_y": float(mean_flow[1]),
            "biased_direction_x": float(biased_direction[0]),
            "biased_direction_y": float(biased_direction[1]),
            "n_human_tracks_used": len(human_tracks),
        },
    )

    return PlanResult(
        planner_name="track_aware_demo",
        success=True,
        track=track,
        path_length_m=track.path_length_m(),
        runtime_s=time.perf_counter() - t_start_wall,
        message="ok",
        metadata={
            "description": (
                "Minimal example planner demonstrating how to parse and inspect "
                "scene tracks. Not intended as a strong baseline."
            ),
            "scene_id": scene.scene_id,
            "task_id": task.task_id,
            "n_rows_in_track_store": int(track_store.num_rows()),
            "n_tracks_in_track_store": int(track_store.num_tracks()),
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a minimal track-aware EvenFlow demo planner."
    )
    parser.add_argument("task_json", help="Path to an EvenFlow task JSON.")
    parser.add_argument("robot_json", help="Path to an EvenFlow robot JSON.")
    parser.add_argument("plan_json", help="Output path for the generated plan JSON.")
    args = parser.parse_args()

    plan = make_track_aware_demo_plan(args.task_json, args.robot_json)

    Path(args.plan_json).resolve().parent.mkdir(parents=True, exist_ok=True)
    save_plan(args.plan_json, plan)

    print(f"✓ Wrote {args.plan_json}")
    print(f"  planner: {plan.planner_name}")
    print(f"  success: {plan.success}")
    print(f"  path_length_m: {plan.path_length_m}")
    print(f"  runtime_s: {plan.runtime_s}")
    print(f"  n_human_tracks_used: {plan.track.metadata.get('n_human_tracks_used') if plan.track else None}")
    print(f"  mean_flow: ({plan.track.metadata.get('mean_flow_x') if plan.track else None}, "
          f"{plan.track.metadata.get('mean_flow_y') if plan.track else None})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
