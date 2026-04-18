from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional

import numpy as np

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
class SceneLayoutRef:
    layout_id: str
    path: str
    coordinate_frame: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneWindow:
    start: str
    end: str
    duration_s: float | None = None


@dataclass(slots=True)
class SceneTracking:
    tracking_id: str | None = None
    format: str = "csv"
    path: str = ""
    timestamp_field: str = "timestamp"
    track_id_field: str = "person_track_id"
    coordinate_frame: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SceneFlow:
    p_star: Point | None = None
    u_hat: Point | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Scene:
    scene_id: str
    layout: SceneLayoutRef
    tracking: SceneTracking
    window: SceneWindow
    flow: SceneFlow | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskSceneRef:
    scene_id: str
    path: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TaskRobot:
    start: Point
    goal: Point


@dataclass(slots=True)
class TaskTargetRef:
    track_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Task:
    task_id: str
    scene: TaskSceneRef
    task_type: str
    robot: TaskRobot
    target: TaskTargetRef | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Robot:
    robot_id: str
    kinematics: str
    radius_m: float
    max_speed_mps: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TrackSimple:
    """
    Canonical simplified time-indexed 2D trajectory.

    Used for:
    - extracted human/reference trajectories
    - canonical planner outputs
    """

    track_id: str
    timestamps: np.ndarray  # shape (T,)
    x: np.ndarray           # shape (T,)
    y: np.ndarray           # shape (T,)
    vx: np.ndarray | None = None  # shape (T,)
    vy: np.ndarray | None = None  # shape (T,)
    position_valid: np.ndarray | None = None  # shape (T,), bool
    velocity_valid: np.ndarray | None = None  # shape (T,), bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)

        if self.timestamps.ndim != 1:
            raise ValueError("TrackSimple.timestamps must be 1D")
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("TrackSimple.x and TrackSimple.y must be 1D")

        T = len(self.timestamps)
        if len(self.x) != T or len(self.y) != T:
            raise ValueError("TrackSimple timestamps/x/y must have the same length")

        if self.vx is not None:
            self.vx = np.asarray(self.vx, dtype=float)
            if self.vx.shape != (T,):
                raise ValueError("TrackSimple.vx must have shape (T,)")

        if self.vy is not None:
            self.vy = np.asarray(self.vy, dtype=float)
            if self.vy.shape != (T,):
                raise ValueError("TrackSimple.vy must have shape (T,)")

        if self.position_valid is not None:
            self.position_valid = np.asarray(self.position_valid, dtype=bool)
            if self.position_valid.shape != (T,):
                raise ValueError("TrackSimple.position_valid must have shape (T,)")

        if self.velocity_valid is not None:
            self.velocity_valid = np.asarray(self.velocity_valid, dtype=bool)
            if self.velocity_valid.shape != (T,):
                raise ValueError("TrackSimple.velocity_valid must have shape (T,)")

        if T > 1 and not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("TrackSimple.timestamps must be strictly increasing")

    def num_samples(self) -> int:
        return len(self.timestamps)

    def duration_s(self) -> float:
        if len(self.timestamps) == 0:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    def path_length_m(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        dx = np.diff(self.x)
        dy = np.diff(self.y)
        return float(np.sum(np.hypot(dx, dy)))

    def xy(self) -> np.ndarray:
        return np.stack((self.x, self.y), axis=1)

    def has_velocity(self) -> bool:
        return self.vx is not None and self.vy is not None

    def vxy(self) -> np.ndarray | None:
        if not self.has_velocity():
            return None
        return np.stack((self.vx, self.vy), axis=1)

    def speed(self) -> np.ndarray | None:
        if not self.has_velocity():
            return None
        return np.hypot(self.vx, self.vy)

    def sample_at(self, t: float) -> np.ndarray:
        """
        Linearly interpolate x,y at time t.
        Clamps to endpoints outside the sampled range.
        """
        if len(self.timestamps) == 0:
            raise ValueError("Cannot sample an empty TrackSimple")

        if t <= self.timestamps[0]:
            return np.array([self.x[0], self.y[0]], dtype=float)
        if t >= self.timestamps[-1]:
            return np.array([self.x[-1], self.y[-1]], dtype=float)

        x = np.interp(t, self.timestamps, self.x)
        y = np.interp(t, self.timestamps, self.y)
        return np.array([x, y], dtype=float)

    def crop(self, t_start: float, t_end: float) -> TrackSimple:
        if t_end < t_start:
            raise ValueError("crop end must be >= crop start")

        mask = (self.timestamps >= t_start) & (self.timestamps <= t_end)

        return TrackSimple(
            track_id=self.track_id,
            timestamps=self.timestamps[mask],
            x=self.x[mask],
            y=self.y[mask],
            vx=self.vx[mask] if self.vx is not None else None,
            vy=self.vy[mask] if self.vy is not None else None,
            position_valid=self.position_valid[mask] if self.position_valid is not None else None,
            velocity_valid=self.velocity_valid[mask] if self.velocity_valid is not None else None,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class TrackPose:
    """
    Full-pose per-track trajectory view.

    Carries pose and, when available, canonical simplified motion channels.
    """

    track_id: str
    timestamps: np.ndarray  # shape (T,)
    keypoints: np.ndarray   # shape (T, K, D)
    keypoints_valid: np.ndarray | None = None  # shape (T, K), bool
    keypoint_names: tuple[str, ...] | None = None

    x: np.ndarray | None = None  # shape (T,)
    y: np.ndarray | None = None  # shape (T,)
    vx: np.ndarray | None = None  # shape (T,)
    vy: np.ndarray | None = None  # shape (T,)
    position_valid: np.ndarray | None = None  # shape (T,), bool
    velocity_valid: np.ndarray | None = None  # shape (T,), bool

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.keypoints = np.asarray(self.keypoints, dtype=float)

        if self.timestamps.ndim != 1:
            raise ValueError("TrackPose.timestamps must be 1D")
        if self.keypoints.ndim != 3:
            raise ValueError("TrackPose.keypoints must have shape (T, K, D)")

        T = len(self.timestamps)
        if self.keypoints.shape[0] != T:
            raise ValueError("TrackPose.keypoints first dimension must match timestamps")

        if self.keypoints_valid is not None:
            self.keypoints_valid = np.asarray(self.keypoints_valid, dtype=bool)
            if self.keypoints_valid.shape != self.keypoints.shape[:2]:
                raise ValueError("TrackPose.keypoints_valid must have shape (T, K)")

        for name in ("x", "y", "vx", "vy"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value, dtype=float)
                if arr.shape != (T,):
                    raise ValueError(f"TrackPose.{name} must have shape (T,)")
                setattr(self, name, arr)

        for name in ("position_valid", "velocity_valid"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value, dtype=bool)
                if arr.shape != (T,):
                    raise ValueError(f"TrackPose.{name} must have shape (T,)")
                setattr(self, name, arr)

        if T > 1 and not np.all(np.diff(self.timestamps) > 0):
            raise ValueError("TrackPose.timestamps must be strictly increasing")

        if self.keypoint_names is not None and len(self.keypoint_names) != self.keypoints.shape[1]:
            raise ValueError("TrackPose.keypoint_names length must match number of keypoints")

    def num_samples(self) -> int:
        return len(self.timestamps)

    def num_keypoints(self) -> int:
        return int(self.keypoints.shape[1])

    def has_canonical_simple(self) -> bool:
        return self.x is not None and self.y is not None

    def xy(self) -> np.ndarray | None:
        if not self.has_canonical_simple():
            return None
        return np.stack((self.x, self.y), axis=1)

    def to_simple(self) -> TrackSimple:
        if self.x is None or self.y is None:
            raise ValueError(
                "TrackPose.to_simple() requires canonical x and y to be present"
            )

        return TrackSimple(
            track_id=self.track_id,
            timestamps=self.timestamps,
            x=self.x,
            y=self.y,
            vx=self.vx,
            vy=self.vy,
            position_valid=self.position_valid,
            velocity_valid=self.velocity_valid,
            metadata=dict(self.metadata),
        )

    def sample_keypoints_at(self, t: float) -> np.ndarray:
        """
        Linearly interpolate keypoints at time t.
        Clamps to endpoints outside the sampled range.
        """
        if len(self.timestamps) == 0:
            raise ValueError("Cannot sample an empty TrackPose")

        if t <= self.timestamps[0]:
            return self.keypoints[0]
        if t >= self.timestamps[-1]:
            return self.keypoints[-1]

        idx = int(np.searchsorted(self.timestamps, t, side="right"))
        i0 = idx - 1
        i1 = idx

        t0 = self.timestamps[i0]
        t1 = self.timestamps[i1]
        alpha = (t - t0) / (t1 - t0)

        return (1.0 - alpha) * self.keypoints[i0] + alpha * self.keypoints[i1]

    def crop(self, t_start: float, t_end: float) -> TrackPose:
        if t_end < t_start:
            raise ValueError("crop end must be >= crop start")

        mask = (self.timestamps >= t_start) & (self.timestamps <= t_end)

        return TrackPose(
            track_id=self.track_id,
            timestamps=self.timestamps[mask],
            keypoints=self.keypoints[mask],
            keypoints_valid=self.keypoints_valid[mask] if self.keypoints_valid is not None else None,
            keypoint_names=self.keypoint_names,
            x=self.x[mask] if self.x is not None else None,
            y=self.y[mask] if self.y is not None else None,
            vx=self.vx[mask] if self.vx is not None else None,
            vy=self.vy[mask] if self.vy is not None else None,
            position_valid=self.position_valid[mask] if self.position_valid is not None else None,
            velocity_valid=self.velocity_valid[mask] if self.velocity_valid is not None else None,
            metadata=dict(self.metadata),
        )


@dataclass(slots=True)
class TrackStore:
    """
    Scene-level owner of dense tracking data for a single scene.

    Stores canonical row-oriented arrays plus indexes for:
    - per-track extraction
    - time-window / timeslice access
    """

    timestamps: np.ndarray             # shape (N,)
    track_ids: np.ndarray              # shape (N,)
    x: np.ndarray                      # shape (N,)
    y: np.ndarray                      # shape (N,)
    vx: np.ndarray | None = None       # shape (N,)
    vy: np.ndarray | None = None       # shape (N,)
    position_valid: np.ndarray | None = None  # shape (N,), bool
    velocity_valid: np.ndarray | None = None  # shape (N,), bool

    keypoints: np.ndarray | None = None       # shape (N, K, D)
    keypoints_valid: np.ndarray | None = None  # shape (N, K), bool
    keypoint_names: tuple[str, ...] | None = None

    timestamp_field: str = "timestamp"
    track_id_field: str = "person_track_id"
    coordinate_frame: str | None = None
    source_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    track_to_indices: dict[str, np.ndarray] = field(default_factory=dict)
    time_order: np.ndarray | None = None
    unique_track_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.timestamps = np.asarray(self.timestamps, dtype=float)
        self.track_ids = np.asarray(self.track_ids)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)

        if self.timestamps.ndim != 1:
            raise ValueError("TrackStore.timestamps must be 1D")
        if self.track_ids.ndim != 1:
            raise ValueError("TrackStore.track_ids must be 1D")
        if self.x.ndim != 1 or self.y.ndim != 1:
            raise ValueError("TrackStore.x and TrackStore.y must be 1D")

        N = len(self.timestamps)
        if len(self.track_ids) != N or len(self.x) != N or len(self.y) != N:
            raise ValueError("TrackStore core row arrays must all have length N")

        if self.vx is not None:
            self.vx = np.asarray(self.vx, dtype=float)
            if self.vx.shape != (N,):
                raise ValueError("TrackStore.vx must have shape (N,)")

        if self.vy is not None:
            self.vy = np.asarray(self.vy, dtype=float)
            if self.vy.shape != (N,):
                raise ValueError("TrackStore.vy must have shape (N,)")

        if self.position_valid is not None:
            self.position_valid = np.asarray(self.position_valid, dtype=bool)
            if self.position_valid.shape != (N,):
                raise ValueError("TrackStore.position_valid must have shape (N,)")

        if self.velocity_valid is not None:
            self.velocity_valid = np.asarray(self.velocity_valid, dtype=bool)
            if self.velocity_valid.shape != (N,):
                raise ValueError("TrackStore.velocity_valid must have shape (N,)")

        if self.keypoints is not None:
            self.keypoints = np.asarray(self.keypoints, dtype=float)
            if self.keypoints.ndim != 3 or self.keypoints.shape[0] != N:
                raise ValueError("TrackStore.keypoints must have shape (N, K, D)")

        if self.keypoints_valid is not None:
            self.keypoints_valid = np.asarray(self.keypoints_valid, dtype=bool)
            if self.keypoints is None:
                raise ValueError("TrackStore.keypoints_valid requires keypoints")
            if self.keypoints_valid.shape != self.keypoints.shape[:2]:
                raise ValueError("TrackStore.keypoints_valid must have shape (N, K)")

        if self.keypoint_names is not None:
            if self.keypoints is None:
                raise ValueError("TrackStore.keypoint_names requires keypoints")
            if len(self.keypoint_names) != self.keypoints.shape[1]:
                raise ValueError("TrackStore.keypoint_names must match K")

        if self.time_order is None:
            self.time_order = np.argsort(self.timestamps, kind="stable")
        else:
            self.time_order = np.asarray(self.time_order, dtype=int)
            if self.time_order.shape != (N,):
                raise ValueError("TrackStore.time_order must have shape (N,)")

        if not self.track_to_indices:
            grouped: dict[str, list[int]] = {}
            for idx, raw_track_id in enumerate(self.track_ids):
                track_id = str(raw_track_id)
                grouped.setdefault(track_id, []).append(idx)

            self.track_to_indices = {
                track_id: np.array(
                    sorted(indices, key=lambda i: self.timestamps[i]),
                    dtype=int,
                )
                for track_id, indices in grouped.items()
            }
        else:
            self.track_to_indices = {
                str(track_id): np.asarray(indices, dtype=int)
                for track_id, indices in self.track_to_indices.items()
            }

        if not self.unique_track_ids:
            self.unique_track_ids = tuple(self.track_to_indices.keys())

    def num_rows(self) -> int:
        return len(self.timestamps)

    def num_tracks(self) -> int:
        return len(self.unique_track_ids)

    def has_pose(self) -> bool:
        return self.keypoints is not None

    def has_velocity(self) -> bool:
        return self.vx is not None and self.vy is not None

    def iter_track_ids(self) -> Iterable[str]:
        return iter(self.unique_track_ids)

    def get_track_indices(self, track_id: str) -> np.ndarray:
        try:
            return self.track_to_indices[str(track_id)]
        except KeyError as e:
            raise KeyError(f"Unknown track_id: {track_id}") from e

    def get_track_simple(self, track_id: str) -> TrackSimple:
        idx = self.get_track_indices(track_id)
        return TrackSimple(
            track_id=str(track_id),
            timestamps=self.timestamps[idx],
            x=self.x[idx],
            y=self.y[idx],
            vx=self.vx[idx] if self.vx is not None else None,
            vy=self.vy[idx] if self.vy is not None else None,
            position_valid=self.position_valid[idx] if self.position_valid is not None else None,
            velocity_valid=self.velocity_valid[idx] if self.velocity_valid is not None else None,
            metadata={"source_path": self.source_path, **self.metadata},
        )

    def get_track_pose(self, track_id: str) -> TrackPose:
        if self.keypoints is None:
            raise ValueError("TrackStore does not contain pose data")

        idx = self.get_track_indices(track_id)
        return TrackPose(
            track_id=str(track_id),
            timestamps=self.timestamps[idx],
            keypoints=self.keypoints[idx],
            keypoints_valid=self.keypoints_valid[idx] if self.keypoints_valid is not None else None,
            keypoint_names=self.keypoint_names,
            x=self.x[idx] if self.x is not None else None,
            y=self.y[idx] if self.y is not None else None,
            vx=self.vx[idx] if self.vx is not None else None,
            vy=self.vy[idx] if self.vy is not None else None,
            position_valid=self.position_valid[idx] if self.position_valid is not None else None,
            velocity_valid=self.velocity_valid[idx] if self.velocity_valid is not None else None,
            metadata={"source_path": self.source_path, **self.metadata},
        )

    def iter_simple_tracks(self) -> Iterable[TrackSimple]:
        for track_id in self.unique_track_ids:
            yield self.get_track_simple(track_id)

    def get_rows_in_window(self, t0: float, t1: float) -> np.ndarray:
        if t1 < t0:
            raise ValueError("window end must be >= window start")
        return np.nonzero((self.timestamps >= t0) & (self.timestamps <= t1))[0]

    def get_rows_at_time(self, t: float, tolerance: float = 0.0) -> np.ndarray:
        if tolerance < 0:
            raise ValueError("tolerance must be nonnegative")
        return np.nonzero(np.abs(self.timestamps - t) <= tolerance)[0]

    def track_ids_at_time(self, t: float, tolerance: float = 0.0) -> tuple[str, ...]:
        rows = self.get_rows_at_time(t, tolerance=tolerance)
        ids = {str(track_id) for track_id in self.track_ids[rows]}
        return tuple(sorted(ids))


@dataclass(frozen=True, slots=True)
class PlanResult:
    """
    Canonical planner output.

    A plan is a time-indexed trajectory. Waypoint-based plans have been retired
    from the public data model.
    """

    planner_name: str
    success: bool
    track: TrackSimple | None = None
    path_length_m: Optional[float] = None
    runtime_s: Optional[float] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalResult:
    success: bool
    path_length_m: float | None
    runtime_s: float | None

    # Backward-compatible headline fields
    min_human_distance_m: float | None = None
    message: str = ""

    # Evaluation context
    planner_name: str | None = None
    scene_type: str | None = None
    target_track_id: str | None = None
    scene_scale_m: float | None = None
    comfort_radius_m: float | None = None

    # Aggregate benchmark scores
    human_likeness_score: float | None = None
    social_compatibility_score: float | None = None
    task_efficiency_score: float | None = None
    overall_score: float | None = None

    # Human-likeness components
    path_deviation_m: float | None = None
    lateral_deviation_m: float | None = None
    direction_similarity: float | None = None

    # Social-compatibility components
    min_other_human_distance_m: float | None = None
    mean_other_human_min_distance_m: float | None = None
    flow_axis_alignment: float | None = None
    flow_coherence: float | None = None
    disruption_rate: float | None = None

    # Efficiency components
    robot_path_length_m: float | None = None
    human_path_length_m: float | None = None
    path_efficiency_ratio: float | None = None
    duration_ratio: float | None = None

    # REACT-style diagnostic
    react_proxy_robot: float | None = None
    react_proxy_human: float | None = None
    react_proxy_gap: float | None = None

    # Bookkeeping
    n_other_human_tracks: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
