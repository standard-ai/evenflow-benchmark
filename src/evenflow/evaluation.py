from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import load_track_store
from .models import EvalResult, Layout, PlanResult, Robot, Scene, Task, TrackSimple


@dataclass(frozen=True)
class EvalConfig:
    version: str
    weights_by_scene_type: dict[str | None, tuple[float, float, float, float]]
    task_efficiency_mode: str  # "legacy", "path_only", "progress_smoothed", "path_only" for v4 gate input
    overall_mode: str = "linear"  # "linear" or "behavior_gated"
    efficiency_gate_threshold: float = 0.75
    efficiency_gate_alpha: float = 4.0


EVAL_CONFIGS: dict[str, EvalConfig] = {
    "v1": EvalConfig(
        version="v1",
        weights_by_scene_type={
            "aligned_flow": (0.25, 0.20, 0.20, 0.35),  # goal, social, efficiency, human
            "cross_flow": (0.25, 0.25, 0.20, 0.30),
            "icn": (0.20, 0.35, 0.15, 0.30),
            "generic": (0.25, 0.25, 0.20, 0.30),
            None: (0.25, 0.25, 0.20, 0.30),
        },
        task_efficiency_mode="legacy",
        overall_mode="linear",
    ),
    "v2": EvalConfig(
        version="v2",
        weights_by_scene_type={
            "aligned_flow": (0.20, 0.15, 0.20, 0.45),
            "cross_flow": (0.25, 0.25, 0.15, 0.35),
            "icn": (0.20, 0.40, 0.10, 0.30),
            "generic": (0.25, 0.25, 0.15, 0.35),
            None: (0.25, 0.25, 0.15, 0.35),
        },
        task_efficiency_mode="path_only",
        overall_mode="linear",
    ),
    "v3": EvalConfig(
        version="v3",
        weights_by_scene_type={
            "aligned_flow": (0.20, 0.15, 0.15, 0.50),
            "cross_flow": (0.25, 0.25, 0.10, 0.40),
            "icn": (0.20, 0.45, 0.05, 0.30),
            "generic": (0.25, 0.25, 0.10, 0.40),
            None: (0.25, 0.25, 0.10, 0.40),
        },
        task_efficiency_mode="progress_smoothed",
        overall_mode="linear",
    ),
    "v4": EvalConfig(
        version="v4",
        # for v4, the efficiency weight is ignored in final aggregation because
        # efficiency enters through a nonlinear gate. Keep the tuple shape for compatibility.
        weights_by_scene_type={
            "aligned_flow": (0.25, 0.25, 0.00, 0.50),  # goal, social, efficiency_unused, human
            "cross_flow": (0.25, 0.30, 0.00, 0.45),
            "icn": (0.20, 0.50, 0.00, 0.30),
            "generic": (0.25, 0.30, 0.00, 0.45),
            None: (0.25, 0.30, 0.00, 0.45),
        },
        task_efficiency_mode="path_only",
        overall_mode="behavior_gated",
        efficiency_gate_threshold=0.78,
        efficiency_gate_alpha=5.0,
    ),
}


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _infer_scene_type(scene: Scene, task: Task) -> str | None:
    candidates: list[str] = []
    for value in (
        getattr(scene, "scene_id", None),
        getattr(task, "task_type", None),
        getattr(scene, "metadata", {}).get("scene_type"),
        getattr(task, "metadata", {}).get("scene_type"),
    ):
        if value is not None:
            candidates.append(str(value).lower())

    joined = " ".join(candidates)
    if "aligned" in joined:
        return "aligned_flow"
    if "cross" in joined:
        return "cross_flow"
    if "icn" in joined or "corridor" in joined or "interaction" in joined:
        return "icn"
    if joined:
        return "generic"
    return None


def _get_robot_track(plan: PlanResult) -> TrackSimple | None:
    if not plan.success:
        return None
    return plan.track


def _valid_xy(track: TrackSimple) -> np.ndarray:
    xy = np.asarray(track.xy(), dtype=float)
    if track.position_valid is not None:
        xy = xy[np.asarray(track.position_valid, dtype=bool)]
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"Expected xy to have shape (T, 2), got {xy.shape}")
    return xy


def _valid_txy(track: TrackSimple) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(track.timestamps, dtype=float)
    xy = np.asarray(track.xy(), dtype=float)

    if track.position_valid is not None:
        mask = np.asarray(track.position_valid, dtype=bool)
    else:
        mask = np.ones(len(t), dtype=bool)

    mask &= np.isfinite(t)
    mask &= np.isfinite(xy).all(axis=1)

    return t[mask], xy[mask]


def _path_length_xy(xy: np.ndarray) -> float:
    if len(xy) <= 1:
        return 0.0
    dxy = np.diff(xy, axis=0)
    return float(np.sum(np.linalg.norm(dxy, axis=1)))


def _step_dirs(xy: np.ndarray) -> np.ndarray:
    if len(xy) <= 1:
        return np.zeros((0, 2), dtype=float)
    dxy = np.diff(xy, axis=0)
    step_norms = np.linalg.norm(dxy, axis=1)
    mask = step_norms > 1e-9
    if not np.any(mask):
        return np.zeros((0, 2), dtype=float)
    return dxy[mask] / step_norms[mask][:, None]


def _resample_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0, 2), dtype=float)
    if len(xy) == 1:
        return np.repeat(xy, n, axis=0)

    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 1e-9:
        return np.repeat(xy[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=float)
    j = 0
    for i, target in enumerate(targets):
        while j + 1 < len(s) and s[j + 1] < target:
            j += 1
        if j + 1 >= len(s):
            out[i] = xy[-1]
            continue
        denom = s[j + 1] - s[j]
        alpha = 0.0 if denom <= 1e-9 else (target - s[j]) / denom
        out[i] = (1.0 - alpha) * xy[j] + alpha * xy[j + 1]
    return out


def _pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    diff = a[:, None, :] - b[None, :, :]
    return np.linalg.norm(diff, axis=2)


def _compute_min_distance_between_tracks(
    robot_track: TrackSimple,
    human_track: TrackSimple,
) -> float | None:
    robot_xy = _valid_xy(robot_track)
    human_xy = _valid_xy(human_track)
    if len(robot_xy) == 0 or len(human_xy) == 0:
        return None
    d = _pairwise_distances(robot_xy, human_xy)
    return float(np.min(d)) if d.size else None


def _scene_scale_m(layout: Layout, task: Task, human_xy: np.ndarray) -> float:
    try:
        start = np.asarray(task.robot.start, dtype=float)
        goal = np.asarray(task.robot.goal, dtype=float)
        if start.shape == (2,) and goal.shape == (2,):
            sg = float(np.linalg.norm(goal - start))
            if sg > 1e-6:
                return sg
    except Exception:
        pass

    hp = _path_length_xy(human_xy)
    if hp > 1e-6:
        return hp

    try:
        boundary = np.asarray(layout.boundary, dtype=float)
        if boundary.ndim == 2 and boundary.shape[1] >= 2 and len(boundary) >= 2:
            mins = np.min(boundary[:, :2], axis=0)
            maxs = np.max(boundary[:, :2], axis=0)
            diag = float(np.linalg.norm(maxs - mins))
            if diag > 1e-6:
                return diag
    except Exception:
        pass

    return 10.0


def _goal_completion_metrics(
    robot_xy: np.ndarray,
    task: Task,
    *,
    robot_radius_m: float,
    plan_success: bool,
) -> dict[str, float | None]:
    if len(robot_xy) == 0:
        return {
            "goal_distance_m": None,
            "goal_completion_score": 0.0 if not plan_success else None,
        }

    goal = np.asarray(task.robot.goal, dtype=float)
    goal_dist = float(np.linalg.norm(robot_xy[-1] - goal))
    goal_tol = max(0.5, 2.0 * robot_radius_m)
    goal_score = _clip01(1.0 - goal_dist / goal_tol)

    if not plan_success:
        goal_score = 0.0

    return {
        "goal_distance_m": goal_dist,
        "goal_completion_score": goal_score,
    }


def _goal_progress_score(robot_xy: np.ndarray, task: Task) -> float | None:
    if len(robot_xy) == 0:
        return None

    start = np.asarray(task.robot.start, dtype=float)
    goal = np.asarray(task.robot.goal, dtype=float)
    sg = float(np.linalg.norm(goal - start))
    if sg <= 1e-9:
        return 1.0

    final_goal_dist = float(np.linalg.norm(robot_xy[-1] - goal))
    return _clip01(1.0 - final_goal_dist / sg)


def _human_likeness_metrics(
    robot_xy: np.ndarray,
    human_xy: np.ndarray,
    *,
    scene_scale_m: float,
) -> dict[str, float | None]:
    if len(robot_xy) == 0 or len(human_xy) == 0:
        return {
            "path_deviation_m": None,
            "lateral_deviation_m": None,
            "direction_similarity": None,
            "human_likeness_score": None,
        }

    n = max(32, min(128, max(len(robot_xy), len(human_xy))))
    r_rs = _resample_polyline(robot_xy, n)
    h_rs = _resample_polyline(human_xy, n)

    path_dev = float(np.mean(np.linalg.norm(r_rs - h_rs, axis=1)))
    d = _pairwise_distances(robot_xy, human_xy)
    lateral_dev = float(np.mean(np.min(d, axis=1))) if d.size else None

    r_dirs = _step_dirs(r_rs)
    h_dirs = _step_dirs(h_rs)
    direction_similarity = None
    if len(r_dirs) > 0 and len(h_dirs) > 0:
        m = min(len(r_dirs), len(h_dirs))
        dots = np.sum(r_dirs[:m] * h_dirs[:m], axis=1)
        direction_similarity = float(np.mean((dots + 1.0) / 2.0))

    path_score = _clip01(1.0 - path_dev / max(scene_scale_m, 1e-6))
    lateral_score = _clip01(1.0 - (lateral_dev or 0.0) / max(scene_scale_m, 1e-6))
    dir_score = 0.5 if direction_similarity is None else _clip01(direction_similarity)

    hls = float(0.50 * path_score + 0.30 * lateral_score + 0.20 * dir_score)

    return {
        "path_deviation_m": path_dev,
        "lateral_deviation_m": lateral_dev,
        "direction_similarity": direction_similarity,
        "human_likeness_score": hls,
    }


def _get_other_human_tracks(store, target_track_id: str) -> list[TrackSimple]:
    return [store.get_track_simple(tid) for tid in store.iter_track_ids() if tid != target_track_id]


def _compute_clearance_to_other_humans(
    robot_xy: np.ndarray,
    other_human_tracks: list[TrackSimple],
) -> tuple[float | None, float | None]:
    if len(robot_xy) == 0 or not other_human_tracks:
        return None, None

    per_track_min: list[float] = []
    for ht in other_human_tracks:
        hxy = _valid_xy(ht)
        if len(hxy) == 0:
            continue
        d = _pairwise_distances(robot_xy, hxy)
        if d.size:
            per_track_min.append(float(np.min(d)))

    if not per_track_min:
        return None, None
    return float(np.min(per_track_min)), float(np.mean(per_track_min))


def _local_flow_axis(humans_xy: list[np.ndarray]) -> tuple[np.ndarray | None, float | None]:
    dirs: list[np.ndarray] = []
    for xy in humans_xy:
        step_dirs = _step_dirs(xy)
        if len(step_dirs):
            dirs.extend(step_dirs)

    if len(dirs) < 2:
        return None, None

    arr = np.asarray(dirs, dtype=float)
    cov = arr.T @ arr
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    denom = float(evals[0] + evals[1]) if len(evals) >= 2 else 0.0
    if denom <= 1e-12:
        return None, None

    axis = evecs[:, 0]
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    coherence = float((evals[0] - evals[1]) / denom)
    return axis, _clip01(coherence)


def _social_compatibility_metrics(
    robot_xy: np.ndarray,
    target_human_xy: np.ndarray,
    other_human_tracks: list[TrackSimple],
    *,
    comfort_radius_m: float,
) -> dict[str, float | None]:
    if len(robot_xy) == 0:
        return {
            "min_other_human_distance_m": None,
            "mean_other_human_min_distance_m": None,
            "flow_axis_alignment": None,
            "flow_coherence": None,
            "disruption_rate": None,
            "social_compatibility_score": None,
        }

    other_human_xys = []
    for ht in other_human_tracks:
        hxy = _valid_xy(ht)
        if len(hxy):
            other_human_xys.append(hxy)

    min_clearance, mean_clearance = _compute_clearance_to_other_humans(robot_xy, other_human_tracks)

    axis, coherence = _local_flow_axis(([target_human_xy] if len(target_human_xy) else []) + other_human_xys)
    robot_dirs = _step_dirs(robot_xy)

    flow_alignment = None
    if axis is not None and len(robot_dirs):
        dots = np.abs(robot_dirs @ axis)
        flow_alignment = float(np.mean(dots))

    disruption_rate = None
    if other_human_xys:
        close_flags = []
        for pt in robot_xy:
            d_min = float("inf")
            for hxy in other_human_xys:
                d = np.linalg.norm(hxy - pt[None, :], axis=1)
                if len(d):
                    d_min = min(d_min, float(np.min(d)))
            if d_min != float("inf"):
                close_flags.append(1.0 if d_min < comfort_radius_m else 0.0)
        if close_flags:
            disruption_rate = float(np.mean(close_flags))

    clearance_score = 0.5 if min_clearance is None else _clip01(min_clearance / max(comfort_radius_m, 1e-6))
    align_score = 0.5 if flow_alignment is None else _clip01(flow_alignment)
    disruption_penalty = 0.5 if disruption_rate is None else _clip01(disruption_rate)

    scs = float(
        0.50 * clearance_score
        + 0.25 * align_score
        + 0.25 * (1.0 - disruption_penalty)
    )

    return {
        "min_other_human_distance_m": min_clearance,
        "mean_other_human_min_distance_m": mean_clearance,
        "flow_axis_alignment": flow_alignment,
        "flow_coherence": coherence,
        "disruption_rate": disruption_rate,
        "social_compatibility_score": scs,
    }


def _resolve_eval_config(
    evaluation_version: str | None = None,
    eval_config: EvalConfig | None = None,
) -> EvalConfig:
    if eval_config is not None:
        return eval_config

    version = evaluation_version or "v1"
    if version not in EVAL_CONFIGS:
        raise ValueError(
            f"Unknown evaluation_version={version!r}. "
            f"Expected one of {sorted(EVAL_CONFIGS.keys())}"
        )
    return EVAL_CONFIGS[version]


def _task_efficiency_metrics(
    robot_xy: np.ndarray,
    human_xy: np.ndarray,
    task: Task,
    *,
    plan_success: bool,
    mode: str,
) -> dict[str, float | None]:
    if len(robot_xy) == 0 or len(human_xy) == 0:
        return {
            "robot_path_length_m": None,
            "human_path_length_m": None,
            "path_efficiency_ratio": None,
            "duration_ratio": None,
            "goal_progress_score": None,
            "task_efficiency_score": 0.0 if not plan_success else None,
        }

    robot_len = _path_length_xy(robot_xy)
    human_len = _path_length_xy(human_xy)

    path_ratio = None
    if robot_len > 1e-9 and human_len > 1e-9:
        path_ratio = float(min(human_len / robot_len, robot_len / human_len))

    duration_ratio = float(min(len(human_xy), len(robot_xy)) / max(len(human_xy), len(robot_xy)))
    path_score = 0.5 if path_ratio is None else _clip01(path_ratio)
    dur_score = _clip01(duration_ratio)
    progress_score = _goal_progress_score(robot_xy, task)

    if mode == "legacy":
        success_score = 1.0 if plan_success else 0.0
        tes = float(0.55 * path_score + 0.15 * dur_score + 0.30 * success_score)

    elif mode == "path_only":
        tes = float(0.70 * path_score + 0.30 * dur_score)
        if not plan_success:
            tes = 0.0

    elif mode == "progress_smoothed":
        progress_term = 0.0 if progress_score is None else progress_score
        tes = float(0.45 * path_score + 0.20 * dur_score + 0.35 * progress_term)

    else:
        raise ValueError(f"Unknown task_efficiency mode: {mode!r}")

    return {
        "robot_path_length_m": robot_len,
        "human_path_length_m": human_len,
        "path_efficiency_ratio": path_ratio,
        "duration_ratio": duration_ratio,
        "goal_progress_score": progress_score,
        "task_efficiency_score": tes,
    }


def _efficiency_gate(efficiency_score: float | None, threshold: float, alpha: float) -> float | None:
    if efficiency_score is None:
        return None
    e = float(_clip01(efficiency_score))
    if e >= threshold:
        return 1.0
    return float(np.exp(-alpha * (threshold - e)))


def _aggregate_overall_score(
    *,
    cfg: EvalConfig,
    goal_score: float | None,
    social_score: float | None,
    efficiency_score: float | None,
    human_score: float | None,
    scene_type: str | None,
) -> tuple[float | None, dict[str, float | None]]:
    wg, ws, we, wb = cfg.weights_by_scene_type.get(
        scene_type,
        cfg.weights_by_scene_type["generic"],
    )

    if goal_score is None or social_score is None or human_score is None:
        return None, {
            "goal_completion": wg,
            "social_compatibility": ws,
            "task_efficiency": we,
            "human_likeness": wb,
            "base_score": None,
            "efficiency_gate": None,
        }

    if cfg.overall_mode == "linear":
        if efficiency_score is None:
            return None, {
                "goal_completion": wg,
                "social_compatibility": ws,
                "task_efficiency": we,
                "human_likeness": wb,
                "base_score": None,
                "efficiency_gate": None,
            }
        overall = float(
            wg * float(goal_score)
            + ws * float(social_score)
            + we * float(efficiency_score)
            + wb * float(human_score)
        )
        return overall, {
            "goal_completion": wg,
            "social_compatibility": ws,
            "task_efficiency": we,
            "human_likeness": wb,
            "base_score": overall,
            "efficiency_gate": 1.0,
        }

    if cfg.overall_mode == "behavior_gated":
        # efficiency weight is intentionally not used linearly here
        weight_sum = wg + ws + wb
        if weight_sum <= 1e-9:
            return None, {
                "goal_completion": wg,
                "social_compatibility": ws,
                "task_efficiency": we,
                "human_likeness": wb,
                "base_score": None,
                "efficiency_gate": None,
            }

        base_score = float(
            (wg * float(goal_score) + ws * float(social_score) + wb * float(human_score))
            / weight_sum
        )
        gate = _efficiency_gate(
            efficiency_score,
            threshold=cfg.efficiency_gate_threshold,
            alpha=cfg.efficiency_gate_alpha,
        )
        if gate is None:
            return None, {
                "goal_completion": wg,
                "social_compatibility": ws,
                "task_efficiency": we,
                "human_likeness": wb,
                "base_score": base_score,
                "efficiency_gate": None,
            }
        overall = float(base_score * gate)
        return overall, {
            "goal_completion": wg,
            "social_compatibility": ws,
            "task_efficiency": we,
            "human_likeness": wb,
            "base_score": base_score,
            "efficiency_gate": gate,
        }

    raise ValueError(f"Unknown overall_mode: {cfg.overall_mode!r}")


def _turning_angles(xy: np.ndarray) -> np.ndarray:
    dirs = _step_dirs(xy)
    if len(dirs) <= 1:
        return np.zeros((0,), dtype=float)
    dots = np.sum(dirs[:-1] * dirs[1:], axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.arccos(dots)


def _react_proxy(xy: np.ndarray) -> float | None:
    ang = _turning_angles(xy)
    if len(ang) == 0:
        return None
    return float(np.mean(ang))


def _react_metrics(robot_xy: np.ndarray, human_xy: np.ndarray) -> dict[str, float | None]:
    robot_react = _react_proxy(robot_xy)
    human_react = _react_proxy(human_xy)
    react_gap = None
    if robot_react is not None and human_react is not None:
        react_gap = float(abs(robot_react - human_react))
    return {
        "react_proxy_robot": robot_react,
        "react_proxy_human": human_react,
        "react_proxy_gap": react_gap,
    }


def evaluate_plan(
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    plan: PlanResult,
    *,
    scene_json_path: str | Path | None = None,
    evaluation_version: str = "v1",
    eval_config: EvalConfig | None = None,
) -> EvalResult:
    """
    EvenFlow evaluation.

    Supports multiple evaluation configurations via `evaluation_version`
    or an explicit `eval_config`.

    Metric families:
      1. Goal completion
      2. Social safety / human clearance
      3. Efficiency
      4. Behavioral alignment to reference human
    """

    scene_type = _infer_scene_type(scene, task)
    cfg = _resolve_eval_config(evaluation_version=evaluation_version, eval_config=eval_config)

    try:
        if task.target is None:
            raise ValueError("Task has no target track reference")

        store = load_track_store(scene, scene_json_path=scene_json_path)
        human_track = store.get_track_simple(task.target.track_id)

        robot_track = _get_robot_track(plan)
        if robot_track is None:
            raise ValueError("Plan has no track trajectory")

        robot_xy = _valid_xy(robot_track)
        human_xy = _valid_xy(human_track)
        if len(robot_xy) == 0:
            raise ValueError("Robot track has no valid samples")
        if len(human_xy) == 0:
            raise ValueError("Human target track has no valid samples")

        other_human_tracks = _get_other_human_tracks(store, task.target.track_id)

        path_length_m = (
            plan.path_length_m if plan.path_length_m is not None else robot_track.path_length_m()
        )
        min_human_distance_m = _compute_min_distance_between_tracks(robot_track, human_track)

        scene_scale_m = _scene_scale_m(layout, task, human_xy)
        robot_radius_m = float(getattr(robot, "radius_m", 0.3) or 0.3)
        comfort_radius_m = max(0.8, 2.0 * robot_radius_m)

        goal_metrics = _goal_completion_metrics(
            robot_xy,
            task,
            robot_radius_m=robot_radius_m,
            plan_success=bool(plan.success),
        )
        hls = _human_likeness_metrics(robot_xy, human_xy, scene_scale_m=scene_scale_m)
        scs = _social_compatibility_metrics(
            robot_xy,
            human_xy,
            other_human_tracks,
            comfort_radius_m=comfort_radius_m,
        )
        tes = _task_efficiency_metrics(
            robot_xy,
            human_xy,
            task,
            plan_success=bool(plan.success),
            mode=cfg.task_efficiency_mode,
        )
        react = _react_metrics(robot_xy, human_xy)

        overall_score, agg_meta = _aggregate_overall_score(
            cfg=cfg,
            goal_score=goal_metrics["goal_completion_score"],
            social_score=scs["social_compatibility_score"],
            efficiency_score=tes["task_efficiency_score"],
            human_score=hls["human_likeness_score"],
            scene_type=scene_type,
        )

        return EvalResult(
            success=plan.success,
            path_length_m=path_length_m,
            runtime_s=plan.runtime_s,
            min_human_distance_m=min_human_distance_m,
            message=plan.message,
            planner_name=plan.planner_name,
            scene_type=scene_type,
            target_track_id=task.target.track_id,
            scene_scale_m=scene_scale_m,
            comfort_radius_m=comfort_radius_m,
            human_likeness_score=hls["human_likeness_score"],
            social_compatibility_score=scs["social_compatibility_score"],
            task_efficiency_score=tes["task_efficiency_score"],
            overall_score=overall_score,
            path_deviation_m=hls["path_deviation_m"],
            lateral_deviation_m=hls["lateral_deviation_m"],
            direction_similarity=hls["direction_similarity"],
            min_other_human_distance_m=scs["min_other_human_distance_m"],
            mean_other_human_min_distance_m=scs["mean_other_human_min_distance_m"],
            flow_axis_alignment=scs["flow_axis_alignment"],
            flow_coherence=scs["flow_coherence"],
            disruption_rate=scs["disruption_rate"],
            robot_path_length_m=tes["robot_path_length_m"],
            human_path_length_m=tes["human_path_length_m"],
            path_efficiency_ratio=tes["path_efficiency_ratio"],
            duration_ratio=tes["duration_ratio"],
            react_proxy_robot=react["react_proxy_robot"],
            react_proxy_human=react["react_proxy_human"],
            react_proxy_gap=react["react_proxy_gap"],
            n_other_human_tracks=len(other_human_tracks),
            metadata={
                "evaluation_version": cfg.version,
                "goal_completion_score": goal_metrics["goal_completion_score"],
                "goal_distance_m": goal_metrics["goal_distance_m"],
                "hls": hls.get("human_likeness_score"),
                "scs": scs.get("social_compatibility_score"),
                "tes": tes.get("task_efficiency_score"),
                "weights": {
                    "goal_completion": agg_meta["goal_completion"],
                    "social_compatibility": agg_meta["social_compatibility"],
                    "task_efficiency": agg_meta["task_efficiency"],
                    "human_likeness": agg_meta["human_likeness"],
                },
                "overall_mode": cfg.overall_mode,
                "task_efficiency_mode": cfg.task_efficiency_mode,
                "goal_progress_score": tes.get("goal_progress_score"),
                "base_score": agg_meta.get("base_score"),
                "efficiency_gate": agg_meta.get("efficiency_gate"),
                "efficiency_gate_threshold": cfg.efficiency_gate_threshold,
                "efficiency_gate_alpha": cfg.efficiency_gate_alpha,
            },
        )

    except Exception as e:
        return EvalResult(
            success=plan.success,
            path_length_m=plan.path_length_m,
            runtime_s=plan.runtime_s,
            min_human_distance_m=None,
            message=plan.message,
            planner_name=plan.planner_name,
            scene_type=scene_type,
            target_track_id=task.target.track_id if task.target is not None else None,
            task_efficiency_score=0.0 if not plan.success else None,
            overall_score=0.0 if not plan.success else None,
            n_other_human_tracks=0,
            metadata={
                "evaluation_version": cfg.version,
                "overall_mode": cfg.overall_mode,
                "task_efficiency_mode": cfg.task_efficiency_mode,
                "evaluation_error": str(e),
            },
        )


def evaluate_plan_v1(
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    plan: PlanResult,
    *,
    scene_json_path: str | Path | None = None,
) -> EvalResult:
    return evaluate_plan(
        layout,
        scene,
        task,
        robot,
        plan,
        scene_json_path=scene_json_path,
        evaluation_version="v1",
    )


def evaluate_plan_v2(
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    plan: PlanResult,
    *,
    scene_json_path: str | Path | None = None,
) -> EvalResult:
    return evaluate_plan(
        layout,
        scene,
        task,
        robot,
        plan,
        scene_json_path=scene_json_path,
        evaluation_version="v2",
    )


def evaluate_plan_v3(
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    plan: PlanResult,
    *,
    scene_json_path: str | Path | None = None,
) -> EvalResult:
    return evaluate_plan(
        layout,
        scene,
        task,
        robot,
        plan,
        scene_json_path=scene_json_path,
        evaluation_version="v3",
    )


def evaluate_plan_v4(
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    plan: PlanResult,
    *,
    scene_json_path: str | Path | None = None,
) -> EvalResult:
    return evaluate_plan(
        layout,
        scene,
        task,
        robot,
        plan,
        scene_json_path=scene_json_path,
        evaluation_version="v4",
    )
