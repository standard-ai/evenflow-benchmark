from __future__ import annotations

from pathlib import Path

import numpy as np

from evenflow.io import load_track_store
from evenflow.models import Scene, SceneLayoutRef, SceneTracking, SceneWindow


TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"

MINIMAL_CSV_PATH = FIXTURES_DIR / "minimal_tracks.csv"
POSE_CSV_PATH = FIXTURES_DIR / "minimal_tracks_with_pose.csv"


def _make_scene(csv_path: Path) -> Scene:
    return Scene(
        scene_id=csv_path.stem,
        layout=SceneLayoutRef(
            layout_id="dummy_layout",
            path="dummy_layout.json",
        ),
        tracking=SceneTracking(
            tracking_id=f"{csv_path.stem}.tracks",
            format="csv",
            path=str(csv_path),
            timestamp_field="timestamp",
            track_id_field="person_track_id",
            coordinate_frame="world",
            metadata={},
        ),
        window=SceneWindow(
            start="1971-01-01T00:00:00Z",
            end="1971-01-01T00:00:10Z",
            duration_s=10.0,
        ),
    )


def test_load_track_store_smoke() -> None:
    scene = _make_scene(MINIMAL_CSV_PATH)

    store = load_track_store(scene)

    assert store.num_rows() > 0
    assert store.num_tracks() > 0
    assert store.timestamps.shape == (store.num_rows(),)
    assert store.track_ids.shape == (store.num_rows(),)
    assert store.x.shape == (store.num_rows(),)
    assert store.y.shape == (store.num_rows(),)


def test_load_track_store_core_canonical_fields_present() -> None:
    scene = _make_scene(MINIMAL_CSV_PATH)

    store = load_track_store(scene)

    assert store.vx is not None
    assert store.vy is not None
    assert store.position_valid is not None
    assert store.velocity_valid is not None

    assert store.vx.shape == (store.num_rows(),)
    assert store.vy.shape == (store.num_rows(),)
    assert store.position_valid.shape == (store.num_rows(),)
    assert store.velocity_valid.shape == (store.num_rows(),)

    assert np.issubdtype(store.timestamps.dtype, np.floating)
    assert np.issubdtype(store.x.dtype, np.floating)
    assert np.issubdtype(store.y.dtype, np.floating)
    assert np.issubdtype(store.vx.dtype, np.floating)
    assert np.issubdtype(store.vy.dtype, np.floating)
    assert store.position_valid.dtype == np.bool_
    assert store.velocity_valid.dtype == np.bool_


def test_load_track_store_timestamps_are_relative_seconds() -> None:
    scene = _make_scene(MINIMAL_CSV_PATH)

    store = load_track_store(scene)

    assert "time_origin_iso" in store.metadata
    assert store.metadata["time_origin_iso"]
    assert float(np.min(store.timestamps)) == 0.0
    assert np.all(np.isfinite(store.timestamps))
    assert np.all(store.timestamps >= 0.0)


def test_load_track_store_builds_track_indexes() -> None:
    scene = _make_scene(MINIMAL_CSV_PATH)

    store = load_track_store(scene)

    assert len(store.unique_track_ids) == store.num_tracks()
    assert len(store.track_to_indices) == store.num_tracks()

    first_track_id = next(iter(store.iter_track_ids()))
    idx = store.get_track_indices(first_track_id)

    assert idx.ndim == 1
    assert len(idx) > 0
    assert np.all(np.diff(store.timestamps[idx]) >= 0.0)


def test_get_track_simple_returns_sorted_single_track_view() -> None:
    scene = _make_scene(MINIMAL_CSV_PATH)

    store = load_track_store(scene)
    first_track_id = next(iter(store.iter_track_ids()))

    track = store.get_track_simple(first_track_id)

    assert track.track_id == first_track_id
    assert track.num_samples() > 0
    assert track.timestamps.ndim == 1
    assert track.x.shape == track.timestamps.shape
    assert track.y.shape == track.timestamps.shape
    assert np.all(np.diff(track.timestamps) > 0)

    if track.has_velocity():
        assert track.vx is not None
        assert track.vy is not None
        assert track.vx.shape == track.timestamps.shape
        assert track.vy.shape == track.timestamps.shape


def test_load_track_store_pose_is_present_and_3d() -> None:
    scene = _make_scene(POSE_CSV_PATH)

    store = load_track_store(scene)

    assert store.has_pose()
    assert store.keypoints is not None
    assert store.keypoint_names is not None

    assert store.keypoints.ndim == 3
    assert store.keypoints.shape[0] == store.num_rows()
    assert store.keypoints.shape[2] == 3
    assert len(store.keypoint_names) == store.keypoints.shape[1]


def test_get_track_pose_and_to_simple_round_trip() -> None:
    scene = _make_scene(POSE_CSV_PATH)

    store = load_track_store(scene)
    first_track_id = next(iter(store.iter_track_ids()))

    pose_track = store.get_track_pose(first_track_id)
    simple_track = pose_track.to_simple()

    assert pose_track.track_id == first_track_id
    assert pose_track.num_samples() > 0
    assert pose_track.num_keypoints() > 0
    assert np.all(np.diff(pose_track.timestamps) > 0)

    assert simple_track.track_id == pose_track.track_id
    assert np.array_equal(simple_track.timestamps, pose_track.timestamps)
    assert np.array_equal(simple_track.x, pose_track.x)
    assert np.array_equal(simple_track.y, pose_track.y)

    if pose_track.vx is not None:
        assert np.array_equal(simple_track.vx, pose_track.vx)
    if pose_track.vy is not None:
        assert np.array_equal(simple_track.vy, pose_track.vy)


def test_track_store_time_queries_return_rows() -> None:
    scene = _make_scene(MINIMAL_CSV_PATH)

    store = load_track_store(scene)

    t0 = float(store.timestamps[0])
    rows_at_t0 = store.get_rows_at_time(t0, tolerance=0.0)
    rows_in_window = store.get_rows_in_window(t0, t0 + 0.5)

    assert rows_at_t0.ndim == 1
    assert rows_in_window.ndim == 1
    assert len(rows_at_t0) > 0
    assert len(rows_in_window) > 0

    assert np.all(np.abs(store.timestamps[rows_at_t0] - t0) <= 0.0)
    assert np.all(store.timestamps[rows_in_window] >= t0)
    assert np.all(store.timestamps[rows_in_window] <= t0 + 0.5)


def test_track_ids_at_time_is_consistent_with_rows_at_time() -> None:
    scene = _make_scene(MINIMAL_CSV_PATH)

    store = load_track_store(scene)

    t0 = float(store.timestamps[0])
    rows = store.get_rows_at_time(t0, tolerance=0.0)
    ids_from_rows = tuple(sorted({str(x) for x in store.track_ids[rows]}))
    ids_from_method = store.track_ids_at_time(t0, tolerance=0.0)

    assert ids_from_method == ids_from_rows
