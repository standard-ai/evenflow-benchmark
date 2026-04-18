from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    EvalResult,
    Exit,
    Layout,
    Obstacle,
    PlanResult,
    Robot,
    Scene,
    SceneFlow,
    SceneLayoutRef,
    SceneTracking,
    SceneWindow,
    Task,
    TaskRobot,
    TaskSceneRef,
    TaskTargetRef,
    TrackSimple,
    TrackStore,
)
from .validation import validate_layout


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(data, indent=2))


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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False

    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _optional_float(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value is None or value == "":
        return None
    return float(value)


def _required_float_or_nan(row, field):
    value = row.get(field)
    if value is None or value == "":
        return np.nan
    return float(value)


def _optional_bool(row: dict[str, Any], field: str) -> bool | None:
    if field not in row:
        return None
    value = row.get(field)
    if value is None or value == "":
        return None
    return _parse_bool(value)


def _parse_timestamp_iso(value: str) -> datetime:
    """
    Parse timestamp strings like:
      1971-03-24 00:43:20.091391+00:00

    This is the expected tracking CSV timestamp format.
    """
    text = value.strip()
    if not text:
        raise ValueError("Empty timestamp string")
    return datetime.fromisoformat(text)


def _infer_keypoint_names(fieldnames: list[str]) -> tuple[str, ...]:
    """
    Infer keypoint names from strict pose columns like:
      left_wrist_x, left_wrist_y, left_wrist_z
    Requiring at least x and y to consider a keypoint present.
    """
    suffixes = ("_x", "_y", "_z")
    candidates: dict[str, set[str]] = {}

    for name in fieldnames:
        for suffix in suffixes:
            if name.endswith(suffix):
                base = name[: -len(suffix)]
                candidates.setdefault(base, set()).add(suffix[1:])  # x / y / z
                break

    keypoints: list[str] = []
    for base, dims in candidates.items():
        if "x" in dims and "y" in dims:
            keypoints.append(base)

    keypoints.sort()
    return tuple(keypoints)


def _pose_schema_from_tracking(
    tracking: SceneTracking,
    fieldnames: list[str],
) -> tuple[tuple[str, ...] | None, int | None]:
    """
    Determine pose schema from tracking metadata or infer from CSV fieldnames.

    Supported optional metadata:
      scene.tracking.metadata["keypoint_names"] = ["nose", "left_wrist", ...]
      scene.tracking.metadata["keypoint_dimensions"] = 2 or 3

    If metadata is absent, infer from strict <keypoint>_x/_y/_z columns.
    """
    metadata = tracking.metadata or {}

    keypoint_names_meta = metadata.get("keypoint_names")
    keypoint_dims_meta = metadata.get("keypoint_dimensions")

    if keypoint_names_meta:
        keypoint_names = tuple(str(x) for x in keypoint_names_meta)
        dims = int(keypoint_dims_meta) if keypoint_dims_meta is not None else 2

        if dims == 3:
            missing_z = [kp for kp in keypoint_names if f"{kp}_z" not in fieldnames]
            if missing_z:
                dims = 2

        return keypoint_names, dims

    inferred = _infer_keypoint_names(fieldnames)
    if not inferred:
        return None, None

    dims = 3 if all(f"{kp}_z" in fieldnames for kp in inferred) else 2
    return inferred, dims


def load_layout(path: str | Path, *, validate: bool = True) -> Layout:
    data = _read_json(path)
    known = {"layout_id", "boundary", "obstacles", "exits"}

    layout = Layout(
        layout_id=data.get("layout_id", Path(path).stem),
        boundary=[tuple(p) for p in data["boundary"]],
        obstacles=[
            Obstacle(
                id=o["id"],
                polygon=[tuple(p) for p in o["polygon"]],
                metadata={k: v for k, v in o.items() if k not in {"id", "polygon"}},
            )
            for o in data.get("obstacles", [])
        ],
        exits=[
            Exit(
                id=e["id"],
                polygon=[tuple(p) for p in e["polygon"]],
                metadata={k: v for k, v in e.items() if k not in {"id", "polygon"}},
            )
            for e in data.get("exits", [])
        ],
        metadata={k: v for k, v in data.items() if k not in known},
    )

    if validate:
        validate_layout(layout)
    return layout


def load_scene(path: str | Path, *, validate: bool = True) -> Scene:
    data = _read_json(path)

    window_data = data["window"]
    layout_data = data["layout"]
    tracking_data = data["tracking"]
    flow_data = data.get("flow")

    scene = Scene(
        scene_id=data.get("scene_id", Path(path).stem),
        layout=SceneLayoutRef(
            layout_id=layout_data["layout_id"],
            path=layout_data["path"],
            coordinate_frame=layout_data.get("coordinate_frame"),
            metadata={
                k: v
                for k, v in layout_data.items()
                if k not in {"layout_id", "path", "coordinate_frame"}
            },
        ),
        tracking=SceneTracking(
            tracking_id=tracking_data.get("tracking_id"),
            format=tracking_data["format"],
            path=tracking_data["path"],
            timestamp_field=tracking_data["timestamp_field"],
            track_id_field=tracking_data["track_id_field"],
            coordinate_frame=tracking_data.get("coordinate_frame"),
            metadata={
                k: v
                for k, v in tracking_data.items()
                if k
                not in {
                    "tracking_id",
                    "format",
                    "path",
                    "timestamp_field",
                    "track_id_field",
                    "coordinate_frame",
                }
            },
        ),
        window=SceneWindow(
            start=window_data["start"],
            end=window_data["end"],
            duration_s=window_data.get("duration_s"),
        ),
        flow=SceneFlow(
            p_star=tuple(flow_data["p_star"])
            if flow_data and flow_data.get("p_star")
            else None,
            u_hat=tuple(flow_data["u_hat"])
            if flow_data and flow_data.get("u_hat")
            else None,
            metadata={
                k: v
                for k, v in (flow_data or {}).items()
                if k not in {"p_star", "u_hat"}
            },
        )
        if flow_data
        else None,
        metadata=data.get("metadata", {}),
        provenance=data.get("provenance", {}),
    )

    if validate:
        from .validation import validate_scene
        validate_scene(scene)

    return scene


def load_task(path: str | Path, *, validate: bool = True) -> Task:
    data = _read_json(path)

    robot_data = data["robot"]
    scene_data = data["scene"]
    target_data = data.get("target")

    target = None
    if target_data is not None:
        target = TaskTargetRef(
            track_id=target_data["track_id"],
            metadata={k: v for k, v in target_data.items() if k != "track_id"},
        )

    task = Task(
        task_id=data.get("task_id", Path(path).stem),
        scene=TaskSceneRef(
            scene_id=scene_data["scene_id"],
            path=scene_data["path"],
            metadata={
                k: v for k, v in scene_data.items() if k not in {"scene_id", "path"}
            },
        ),
        task_type=data["task_type"],
        robot=TaskRobot(
            start=tuple(robot_data["start"]),
            goal=tuple(robot_data["goal"]),
        ),
        target=target,
        metadata=data.get("metadata", {}),
        provenance=data.get("provenance", {}),
    )

    if validate:
        from .validation import validate_task
        validate_task(task)

    return task


def load_robot(path: str | Path, *, validate: bool = True) -> Robot:
    data = _read_json(path)

    footprint = data.get("footprint", {})
    dynamics = data.get("dynamics", {})

    robot = Robot(
        robot_id=data.get("robot_id", Path(path).stem),
        kinematics=data["kinematics"],
        radius_m=float(footprint["radius_m"]),
        max_speed_mps=float(dynamics["max_speed_mps"]),
        metadata=data.get("metadata", {}),
    )

    if validate:
        from .validation import validate_robot
        validate_robot(robot)

    return robot


def load_track_store(
    scene: Scene,
    *,
    scene_json_path: str | Path | None = None,
) -> TrackStore:
    """
    Load a scene's tracking artifact into a canonical numpy-backed TrackStore.

    Expected CSV schema:
      Required:
        - scene.tracking.timestamp_field (ISO datetime string)
        - scene.tracking.track_id_field
        - x
        - y

      Optional canonical fields:
        - vx
        - vy
        - position_valid
        - velocity_valid

      Optional pose fields:
        - <keypoint>_x
        - <keypoint>_y
        - <keypoint>_z   (if 3D pose is present)

    In-memory timestamp representation:
      - float seconds relative to the earliest timestamp in the file
    """
    if scene.tracking.format.lower() != "csv":
        raise ValueError(f"Unsupported tracking format: {scene.tracking.format}")

    csv_path = _scene_tracking_csv_path(scene, scene_json_path=scene_json_path)

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"No CSV header found in {csv_path}")

        timestamp_field = scene.tracking.timestamp_field
        track_id_field = scene.tracking.track_id_field

        missing_required = [
            name
            for name in (timestamp_field, track_id_field, "x", "y")
            if name not in fieldnames
        ]
        if missing_required:
            raise ValueError(
                f"Missing required tracking columns {missing_required} in {csv_path}"
            )

        keypoint_names, keypoint_dims = _pose_schema_from_tracking(
            scene.tracking,
            fieldnames,
        )

        has_vx = "vx" in fieldnames
        has_vy = "vy" in fieldnames
        has_position_valid = "position_valid" in fieldnames
        has_velocity_valid = "velocity_valid" in fieldnames
        has_pose = keypoint_names is not None and keypoint_dims is not None

        has_per_keypoint_valid = False
        if has_pose:
            has_per_keypoint_valid = all(
                f"{kp}_valid" in fieldnames for kp in keypoint_names
            )

        timestamp_datetimes: list[datetime] = []
        track_ids: list[str] = []
        xs: list[float] = []
        ys: list[float] = []

        vxs: list[float | None] = []
        vys: list[float | None] = []
        position_valids: list[bool | None] = []
        velocity_valids: list[bool | None] = []

        keypoints_rows: list[np.ndarray] = []
        keypoints_valid_rows: list[np.ndarray] = []

        for row in reader:
            timestamp_datetimes.append(_parse_timestamp_iso(row[timestamp_field]))
            track_ids.append(str(row[track_id_field]))
            xs.append(_required_float_or_nan(row, "x"))
            ys.append(_required_float_or_nan(row, "y"))

            vxs.append(_optional_float(row, "vx") if has_vx else None)
            vys.append(_optional_float(row, "vy") if has_vy else None)
            position_valids.append(
                _optional_bool(row, "position_valid") if has_position_valid else None
            )
            velocity_valids.append(
                _optional_bool(row, "velocity_valid") if has_velocity_valid else None
            )

            if has_pose:
                coords = np.zeros((len(keypoint_names), keypoint_dims), dtype=float)

                for i, kp in enumerate(keypoint_names):
                    x_val = _optional_float(row, f"{kp}_x")
                    y_val = _optional_float(row, f"{kp}_y")

                    coords[i, 0] = np.nan if x_val is None else x_val
                    coords[i, 1] = np.nan if y_val is None else y_val

                    if keypoint_dims == 3:
                        z_val = _optional_float(row, f"{kp}_z")
                        coords[i, 2] = np.nan if z_val is None else z_val

                keypoints_rows.append(coords)

                if has_per_keypoint_valid:
                    valid = np.zeros((len(keypoint_names),), dtype=bool)
                    for i, kp in enumerate(keypoint_names):
                        valid[i] = _parse_bool(row[f"{kp}_valid"])
                    keypoints_valid_rows.append(valid)

    if not timestamp_datetimes:
        return TrackStore(
            timestamps=np.asarray([], dtype=float),
            track_ids=np.asarray([], dtype=str),
            x=np.asarray([], dtype=float),
            y=np.asarray([], dtype=float),
            vx=None,
            vy=None,
            position_valid=None,
            velocity_valid=None,
            keypoints=None,
            keypoints_valid=None,
            keypoint_names=keypoint_names,
            timestamp_field=scene.tracking.timestamp_field,
            track_id_field=scene.tracking.track_id_field,
            coordinate_frame=scene.tracking.coordinate_frame,
            source_path=str(csv_path),
            metadata=dict(scene.tracking.metadata),
        )

    t0 = min(timestamp_datetimes)
    time_origin_iso = t0.isoformat()

    timestamps_arr = np.asarray(
        [(dt - t0).total_seconds() for dt in timestamp_datetimes],
        dtype=float,
    )

    track_ids_arr = np.asarray(track_ids)
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)

    vx_arr = None
    if has_vx:
        vx_arr = np.asarray(
            [np.nan if v is None else float(v) for v in vxs],
            dtype=float,
        )

    vy_arr = None
    if has_vy:
        vy_arr = np.asarray(
            [np.nan if v is None else float(v) for v in vys],
            dtype=float,
        )

    position_valid_arr = None
    if has_position_valid:
        position_valid_arr = np.asarray(
            [False if v is None else bool(v) for v in position_valids],
            dtype=bool,
        )

    velocity_valid_arr = None
    if has_velocity_valid:
        velocity_valid_arr = np.asarray(
            [False if v is None else bool(v) for v in velocity_valids],
            dtype=bool,
        )

    keypoints_arr = None
    keypoints_valid_arr = None
    if has_pose:
        keypoints_arr = np.asarray(keypoints_rows, dtype=float)
        if has_per_keypoint_valid:
            keypoints_valid_arr = np.asarray(keypoints_valid_rows, dtype=bool)

    return TrackStore(
        timestamps=timestamps_arr,
        track_ids=track_ids_arr,
        x=x_arr,
        y=y_arr,
        vx=vx_arr,
        vy=vy_arr,
        position_valid=position_valid_arr,
        velocity_valid=velocity_valid_arr,
        keypoints=keypoints_arr,
        keypoints_valid=keypoints_valid_arr,
        keypoint_names=keypoint_names,
        timestamp_field=scene.tracking.timestamp_field,
        track_id_field=scene.tracking.track_id_field,
        coordinate_frame=scene.tracking.coordinate_frame,
        source_path=str(csv_path),
        metadata={
            **scene.tracking.metadata,
            "time_origin_iso": time_origin_iso,
        },
    )


def load_plan(path: str | Path, *, validate: bool = True) -> PlanResult:
    data = _read_json(path)

    track_data = data.get("track")
    track = None
    if track_data is not None:
        track = TrackSimple(
            track_id=track_data["track_id"],
            timestamps=np.asarray(track_data["timestamps"], dtype=float),
            x=np.asarray(track_data["x"], dtype=float),
            y=np.asarray(track_data["y"], dtype=float),
            vx=(
                np.asarray(track_data["vx"], dtype=float)
                if track_data.get("vx") is not None
                else None
            ),
            vy=(
                np.asarray(track_data["vy"], dtype=float)
                if track_data.get("vy") is not None
                else None
            ),
            position_valid=(
                np.asarray(track_data["position_valid"], dtype=bool)
                if track_data.get("position_valid") is not None
                else None
            ),
            velocity_valid=(
                np.asarray(track_data["velocity_valid"], dtype=bool)
                if track_data.get("velocity_valid") is not None
                else None
            ),
            metadata=track_data.get("metadata", {}),
        )

    plan = PlanResult(
        planner_name=data["planner_name"],
        success=bool(data["success"]),
        track=track,
        path_length_m=(
            float(data["path_length_m"])
            if data.get("path_length_m") is not None
            else None
        ),
        runtime_s=(
            float(data["runtime_s"])
            if data.get("runtime_s") is not None
            else None
        ),
        message=data.get("message", ""),
        metadata=data.get("metadata", {}),
    )

    if validate:
        from .validation import validate_plan_result
        validate_plan_result(plan)

    return plan


def save_plan(path: str | Path, plan: PlanResult) -> None:
    track_data = None
    if plan.track is not None:
        track_data = {
            "track_id": plan.track.track_id,
            "timestamps": plan.track.timestamps.tolist(),
            "x": plan.track.x.tolist(),
            "y": plan.track.y.tolist(),
            "vx": plan.track.vx.tolist() if plan.track.vx is not None else None,
            "vy": plan.track.vy.tolist() if plan.track.vy is not None else None,
            "position_valid": (
                plan.track.position_valid.tolist()
                if plan.track.position_valid is not None
                else None
            ),
            "velocity_valid": (
                plan.track.velocity_valid.tolist()
                if plan.track.velocity_valid is not None
                else None
            ),
            "metadata": plan.track.metadata,
        }

    data = {
        "planner_name": plan.planner_name,
        "success": plan.success,
        "track": track_data,
        "path_length_m": plan.path_length_m,
        "runtime_s": plan.runtime_s,
        "message": plan.message,
        "metadata": plan.metadata,
    }
    _write_json(path, data)


def load_eval(path: str | Path, *, validate: bool = False) -> EvalResult:
    data = _read_json(path)

    result = EvalResult(
        success=bool(data["success"]),
        path_length_m=(
            float(data["path_length_m"])
            if data.get("path_length_m") is not None
            else None
        ),
        runtime_s=(
            float(data["runtime_s"])
            if data.get("runtime_s") is not None
            else None
        ),
        min_human_distance_m=(
            float(data["min_human_distance_m"])
            if data.get("min_human_distance_m") is not None
            else None
        ),
        message=data.get("message", ""),
        planner_name=data.get("planner_name"),
        scene_type=data.get("scene_type"),
        target_track_id=data.get("target_track_id"),
        scene_scale_m=(
            float(data["scene_scale_m"])
            if data.get("scene_scale_m") is not None
            else None
        ),
        comfort_radius_m=(
            float(data["comfort_radius_m"])
            if data.get("comfort_radius_m") is not None
            else None
        ),
        human_likeness_score=(
            float(data["human_likeness_score"])
            if data.get("human_likeness_score") is not None
            else None
        ),
        social_compatibility_score=(
            float(data["social_compatibility_score"])
            if data.get("social_compatibility_score") is not None
            else None
        ),
        task_efficiency_score=(
            float(data["task_efficiency_score"])
            if data.get("task_efficiency_score") is not None
            else None
        ),
        overall_score=(
            float(data["overall_score"])
            if data.get("overall_score") is not None
            else None
        ),
        path_deviation_m=(
            float(data["path_deviation_m"])
            if data.get("path_deviation_m") is not None
            else None
        ),
        lateral_deviation_m=(
            float(data["lateral_deviation_m"])
            if data.get("lateral_deviation_m") is not None
            else None
        ),
        direction_similarity=(
            float(data["direction_similarity"])
            if data.get("direction_similarity") is not None
            else None
        ),
        min_other_human_distance_m=(
            float(data["min_other_human_distance_m"])
            if data.get("min_other_human_distance_m") is not None
            else None
        ),
        mean_other_human_min_distance_m=(
            float(data["mean_other_human_min_distance_m"])
            if data.get("mean_other_human_min_distance_m") is not None
            else None
        ),
        flow_axis_alignment=(
            float(data["flow_axis_alignment"])
            if data.get("flow_axis_alignment") is not None
            else None
        ),
        flow_coherence=(
            float(data["flow_coherence"])
            if data.get("flow_coherence") is not None
            else None
        ),
        disruption_rate=(
            float(data["disruption_rate"])
            if data.get("disruption_rate") is not None
            else None
        ),
        robot_path_length_m=(
            float(data["robot_path_length_m"])
            if data.get("robot_path_length_m") is not None
            else None
        ),
        human_path_length_m=(
            float(data["human_path_length_m"])
            if data.get("human_path_length_m") is not None
            else None
        ),
        path_efficiency_ratio=(
            float(data["path_efficiency_ratio"])
            if data.get("path_efficiency_ratio") is not None
            else None
        ),
        duration_ratio=(
            float(data["duration_ratio"])
            if data.get("duration_ratio") is not None
            else None
        ),
        react_proxy_robot=(
            float(data["react_proxy_robot"])
            if data.get("react_proxy_robot") is not None
            else None
        ),
        react_proxy_human=(
            float(data["react_proxy_human"])
            if data.get("react_proxy_human") is not None
            else None
        ),
        react_proxy_gap=(
            float(data["react_proxy_gap"])
            if data.get("react_proxy_gap") is not None
            else None
        ),
        n_other_human_tracks=int(data.get("n_other_human_tracks", 0)),
        metadata=data.get("metadata", {}),
    )

    if validate:
        from .validation import validate_eval_result
        validate_eval_result(result)

    return result


def save_eval(path: str | Path, result: EvalResult) -> None:
    data = {
        "success": result.success,
        "path_length_m": result.path_length_m,
        "runtime_s": result.runtime_s,
        "min_human_distance_m": result.min_human_distance_m,
        "message": result.message,
        "planner_name": result.planner_name,
        "scene_type": result.scene_type,
        "target_track_id": result.target_track_id,
        "scene_scale_m": result.scene_scale_m,
        "comfort_radius_m": result.comfort_radius_m,
        "human_likeness_score": result.human_likeness_score,
        "social_compatibility_score": result.social_compatibility_score,
        "task_efficiency_score": result.task_efficiency_score,
        "overall_score": result.overall_score,
        "path_deviation_m": result.path_deviation_m,
        "lateral_deviation_m": result.lateral_deviation_m,
        "direction_similarity": result.direction_similarity,
        "min_other_human_distance_m": result.min_other_human_distance_m,
        "mean_other_human_min_distance_m": result.mean_other_human_min_distance_m,
        "flow_axis_alignment": result.flow_axis_alignment,
        "flow_coherence": result.flow_coherence,
        "disruption_rate": result.disruption_rate,
        "robot_path_length_m": result.robot_path_length_m,
        "human_path_length_m": result.human_path_length_m,
        "path_efficiency_ratio": result.path_efficiency_ratio,
        "duration_ratio": result.duration_ratio,
        "react_proxy_robot": result.react_proxy_robot,
        "react_proxy_human": result.react_proxy_human,
        "react_proxy_gap": result.react_proxy_gap,
        "n_other_human_tracks": result.n_other_human_tracks,
        "metadata": result.metadata,
    }
    _write_json(path, data)
