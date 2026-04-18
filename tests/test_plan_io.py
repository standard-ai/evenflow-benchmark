from __future__ import annotations

from pathlib import Path

import numpy as np

from evenflow.io import load_plan, save_plan
from evenflow.models import PlanResult, TrackSimple


def test_save_and_load_plan_with_track_round_trip(tmp_path: Path) -> None:
    track = TrackSimple(
        track_id="robot_plan",
        timestamps=np.array([0.0, 1.0, 2.0], dtype=float),
        x=np.array([1.0, 2.0, 3.0], dtype=float),
        y=np.array([1.5, 2.5, 3.5], dtype=float),
        vx=np.array([1.0, 1.0, 1.0], dtype=float),
        vy=np.array([1.0, 1.0, 1.0], dtype=float),
        position_valid=np.array([True, True, True], dtype=bool),
        velocity_valid=np.array([True, True, True], dtype=bool),
        metadata={"source": "test"},
    )

    plan = PlanResult(
        planner_name="test_track_plan",
        success=True,
        track=track,
        path_length_m=track.path_length_m(),
        runtime_s=0.01,
        message="ok",
        metadata={"planner_family": "test"},
    )

    out_path = tmp_path / "plan_with_track.json"
    save_plan(out_path, plan)

    loaded = load_plan(out_path)

    assert loaded.planner_name == "test_track_plan"
    assert loaded.success is True
    assert loaded.track is not None

    assert loaded.track.track_id == "robot_plan"
    assert np.array_equal(loaded.track.timestamps, track.timestamps)
    assert np.array_equal(loaded.track.x, track.x)
    assert np.array_equal(loaded.track.y, track.y)
    assert np.array_equal(loaded.track.vx, track.vx)
    assert np.array_equal(loaded.track.vy, track.vy)
    assert np.array_equal(loaded.track.position_valid, track.position_valid)
    assert np.array_equal(loaded.track.velocity_valid, track.velocity_valid)
    assert loaded.track.metadata == {"source": "test"}

    assert loaded.path_length_m == plan.path_length_m
    assert loaded.runtime_s == 0.01
    assert loaded.message == "ok"
    assert loaded.metadata == {"planner_family": "test"}
