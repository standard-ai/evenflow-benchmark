from __future__ import annotations

import numpy as np

from .models import (
    EvalResult,
    Layout,
    PlanResult,
    Polygon,
    Robot,
    Scene,
    Task,
    TrackPose,
    TrackSimple,
    TrackStore,
)


class ValidationError(ValueError):
    pass


def _validate_polygon(name: str, poly: Polygon) -> None:
    if len(poly) < 3:
        raise ValidationError(f"{name} must have at least 3 points")
    for i, point in enumerate(poly):
        if len(point) != 2:
            raise ValidationError(f"{name}[{i}] must be a 2D point")
        if not all(isinstance(v, (int, float)) for v in point):
            raise ValidationError(f"{name}[{i}] must contain numeric coordinates")


def _validate_point(name: str, point: tuple[float, float]) -> None:
    if len(point) != 2:
        raise ValidationError(f"{name} must be a 2D point")
    if not all(isinstance(v, (int, float)) for v in point):
        raise ValidationError(f"{name} must contain numeric coordinates")


def _validate_numeric_array(
    name: str,
    arr: np.ndarray | None,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> None:
    if arr is None:
        return
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"{name} must be a numpy array")
    if ndim is not None and arr.ndim != ndim:
        raise ValidationError(f"{name} must have ndim={ndim}")
    if shape is not None and arr.shape != shape:
        raise ValidationError(f"{name} must have shape {shape}")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValidationError(f"{name} must contain numeric values")


def _validate_bool_array(
    name: str,
    arr: np.ndarray | None,
    *,
    shape: tuple[int, ...] | None = None,
) -> None:
    if arr is None:
        return
    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"{name} must be a numpy array")
    if arr.dtype != np.bool_:
        raise ValidationError(f"{name} must have boolean dtype")
    if shape is not None and arr.shape != shape:
        raise ValidationError(f"{name} must have shape {shape}")


def _validate_optional_probability(name: str, value: float | None) -> None:
    if value is None:
        return
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric when provided")
    if value < 0.0 or value > 1.0:
        raise ValidationError(f"{name} must be in [0, 1]")


def validate_track_simple(track: TrackSimple) -> None:
    if not track.track_id:
        raise ValidationError("track_id is required")

    _validate_numeric_array("track.timestamps", track.timestamps, ndim=1)
    _validate_numeric_array("track.x", track.x, ndim=1)
    _validate_numeric_array("track.y", track.y, ndim=1)

    T = len(track.timestamps)
    if track.x.shape != (T,) or track.y.shape != (T,):
        raise ValidationError("track timestamps/x/y must all have shape (T,)")

    _validate_numeric_array("track.vx", track.vx, shape=(T,))
    _validate_numeric_array("track.vy", track.vy, shape=(T,))
    _validate_bool_array("track.position_valid", track.position_valid, shape=(T,))
    _validate_bool_array("track.velocity_valid", track.velocity_valid, shape=(T,))

    if T > 1 and not np.all(np.diff(track.timestamps) > 0):
        raise ValidationError("track.timestamps must be strictly increasing")


def validate_track_pose(track: TrackPose) -> None:
    if not track.track_id:
        raise ValidationError("track_id is required")

    _validate_numeric_array("track.timestamps", track.timestamps, ndim=1)
    _validate_numeric_array("track.keypoints", track.keypoints, ndim=3)

    T = len(track.timestamps)
    if track.keypoints.shape[0] != T:
        raise ValidationError("track.keypoints first dimension must match timestamps")

    _validate_bool_array(
        "track.keypoints_valid",
        track.keypoints_valid,
        shape=track.keypoints.shape[:2],
    )

    _validate_numeric_array("track.x", track.x, shape=(T,))
    _validate_numeric_array("track.y", track.y, shape=(T,))
    _validate_numeric_array("track.vx", track.vx, shape=(T,))
    _validate_numeric_array("track.vy", track.vy, shape=(T,))
    _validate_bool_array("track.position_valid", track.position_valid, shape=(T,))
    _validate_bool_array("track.velocity_valid", track.velocity_valid, shape=(T,))

    if T > 1 and not np.all(np.diff(track.timestamps) > 0):
        raise ValidationError("track.timestamps must be strictly increasing")

    if track.keypoint_names is not None:
        if len(track.keypoint_names) != track.keypoints.shape[1]:
            raise ValidationError(
                "track.keypoint_names length must match number of keypoints"
            )


def validate_track_store(store: TrackStore) -> None:
    _validate_numeric_array("store.timestamps", store.timestamps, ndim=1)
    _validate_numeric_array("store.x", store.x, ndim=1)
    _validate_numeric_array("store.y", store.y, ndim=1)

    if not isinstance(store.track_ids, np.ndarray):
        raise ValidationError("store.track_ids must be a numpy array")
    if store.track_ids.ndim != 1:
        raise ValidationError("store.track_ids must be 1D")

    N = len(store.timestamps)
    if len(store.track_ids) != N or len(store.x) != N or len(store.y) != N:
        raise ValidationError("store core row arrays must all have length N")

    _validate_numeric_array("store.vx", store.vx, shape=(N,))
    _validate_numeric_array("store.vy", store.vy, shape=(N,))
    _validate_bool_array("store.position_valid", store.position_valid, shape=(N,))
    _validate_bool_array("store.velocity_valid", store.velocity_valid, shape=(N,))

    if store.keypoints is not None:
        _validate_numeric_array("store.keypoints", store.keypoints, ndim=3)
        if store.keypoints.shape[0] != N:
            raise ValidationError("store.keypoints first dimension must match timestamps")

    if store.keypoints_valid is not None:
        if store.keypoints is None:
            raise ValidationError("store.keypoints_valid requires store.keypoints")
        _validate_bool_array(
            "store.keypoints_valid",
            store.keypoints_valid,
            shape=store.keypoints.shape[:2],
        )

    if store.keypoint_names is not None:
        if store.keypoints is None:
            raise ValidationError("store.keypoint_names requires store.keypoints")
        if len(store.keypoint_names) != store.keypoints.shape[1]:
            raise ValidationError(
                "store.keypoint_names length must match number of keypoints"
            )

    if store.time_order is not None:
        if not isinstance(store.time_order, np.ndarray):
            raise ValidationError("store.time_order must be a numpy array")
        if store.time_order.shape != (N,):
            raise ValidationError("store.time_order must have shape (N,)")

    if len(store.unique_track_ids) != len(store.track_to_indices):
        raise ValidationError(
            "store.unique_track_ids length must match store.track_to_indices"
        )

    for track_id, indices in store.track_to_indices.items():
        if not isinstance(track_id, str) or not track_id:
            raise ValidationError("store.track_to_indices keys must be non-empty strings")
        if not isinstance(indices, np.ndarray):
            raise ValidationError("store.track_to_indices values must be numpy arrays")
        if indices.ndim != 1:
            raise ValidationError("store.track_to_indices values must be 1D arrays")
        if len(indices) == 0:
            raise ValidationError("store.track_to_indices values must be non-empty")
        if np.any(indices < 0) or np.any(indices >= N):
            raise ValidationError("store.track_to_indices contains out-of-bounds indices")


def validate_layout(layout: Layout) -> None:
    if not layout.layout_id:
        raise ValidationError("layout_id is required")

    _validate_polygon("boundary", layout.boundary)

    for obs in layout.obstacles:
        if not obs.id:
            raise ValidationError("Each obstacle must have an id")
        _validate_polygon(f"obstacle:{obs.id}", obs.polygon)

    for ex in layout.exits:
        if not ex.id:
            raise ValidationError("Each exit must have an id")
        _validate_polygon(f"exit:{ex.id}", ex.polygon)


def validate_scene(scene: Scene) -> None:
    if not scene.scene_id:
        raise ValidationError("scene_id is required")

    if not scene.layout.layout_id:
        raise ValidationError("scene.layout.layout_id is required")

    if not scene.layout.path:
        raise ValidationError("scene.layout.path is required")

    if not scene.tracking.format:
        raise ValidationError("scene.tracking.format is required")

    if not scene.tracking.path:
        raise ValidationError("scene.tracking.path is required")

    if not scene.tracking.timestamp_field:
        raise ValidationError("scene.tracking.timestamp_field is required")

    if not scene.tracking.track_id_field:
        raise ValidationError("scene.tracking.track_id_field is required")

    if not scene.window.start:
        raise ValidationError("scene.window.start is required")

    if not scene.window.end:
        raise ValidationError("scene.window.end is required")

    if scene.window.duration_s is not None and scene.window.duration_s <= 0:
        raise ValidationError("scene.window.duration_s must be positive")

    if scene.flow:
        if scene.flow.p_star:
            _validate_point("scene.flow.p_star", scene.flow.p_star)

        if scene.flow.u_hat:
            _validate_point("scene.flow.u_hat", scene.flow.u_hat)


def validate_task(task: Task) -> None:
    if not task.task_id:
        raise ValidationError("task_id is required")

    if not task.scene.scene_id:
        raise ValidationError("task.scene.scene_id is required")

    if not task.scene.path:
        raise ValidationError("task.scene.path is required")

    if not task.task_type:
        raise ValidationError("task_type is required")

    _validate_point("task.robot.start", task.robot.start)
    _validate_point("task.robot.goal", task.robot.goal)

    if task.target is not None and not task.target.track_id:
        raise ValidationError("task.target.track_id is required when target is present")


def validate_robot(robot: Robot) -> None:
    if not robot.robot_id:
        raise ValidationError("robot_id is required")

    if not robot.kinematics:
        raise ValidationError("kinematics is required")

    if robot.kinematics not in {"holonomic", "differential_drive", "ackermann"}:
        raise ValidationError(
            "kinematics must be one of: holonomic, differential_drive, ackermann"
        )

    if robot.radius_m <= 0:
        raise ValidationError("radius_m must be positive")

    if robot.max_speed_mps <= 0:
        raise ValidationError("max_speed_mps must be positive")


def validate_plan_result(plan: PlanResult) -> None:
    if not plan.planner_name:
        raise ValidationError("planner_name is required")

    if not isinstance(plan.success, bool):
        raise ValidationError("success must be a bool")

    if plan.track is not None:
        validate_track_simple(plan.track)

    if plan.success and plan.track is None:
        raise ValidationError("successful plans must contain a track trajectory")

    if plan.path_length_m is not None and plan.path_length_m < 0:
        raise ValidationError("path_length_m must be nonnegative")

    if plan.runtime_s is not None and plan.runtime_s < 0:
        raise ValidationError("runtime_s must be nonnegative")


def validate_eval_result(result: EvalResult) -> None:
    if not isinstance(result.success, bool):
        raise ValidationError("success must be a bool")

    for name in (
        "path_length_m",
        "runtime_s",
        "min_human_distance_m",
        "scene_scale_m",
        "comfort_radius_m",
        "human_likeness_score",
        "social_compatibility_score",
        "task_efficiency_score",
        "overall_score",
        "path_deviation_m",
        "lateral_deviation_m",
        "min_other_human_distance_m",
        "mean_other_human_min_distance_m",
        "flow_coherence",
        "disruption_rate",
        "robot_path_length_m",
        "human_path_length_m",
        "path_efficiency_ratio",
        "duration_ratio",
        "react_proxy_robot",
        "react_proxy_human",
        "react_proxy_gap",
    ):
        value = getattr(result, name)
        if value is not None and value < 0:
            raise ValidationError(f"{name} must be nonnegative")

    _validate_optional_probability("direction_similarity", result.direction_similarity)
    _validate_optional_probability("flow_axis_alignment", result.flow_axis_alignment)
    _validate_optional_probability("flow_coherence", result.flow_coherence)
    _validate_optional_probability("disruption_rate", result.disruption_rate)
    _validate_optional_probability("path_efficiency_ratio", result.path_efficiency_ratio)
    _validate_optional_probability("duration_ratio", result.duration_ratio)
    _validate_optional_probability("human_likeness_score", result.human_likeness_score)
    _validate_optional_probability(
        "social_compatibility_score", result.social_compatibility_score
    )
    _validate_optional_probability("task_efficiency_score", result.task_efficiency_score)
    _validate_optional_probability("overall_score", result.overall_score)

    if result.n_other_human_tracks < 0:
        raise ValidationError("n_other_human_tracks must be nonnegative")
