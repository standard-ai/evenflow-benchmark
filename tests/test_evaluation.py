import importlib
import os

import numpy as np
import pytest


def _resolve_package_name() -> str:
    candidates = []
    env_name = os.environ.get("BENCHMARK_PACKAGE")
    if env_name:
        candidates.append(env_name)

    candidates.extend(
        [
            "scenebenchmarks",
            "evenflow",
        ]
    )

    for name in candidates:
        try:
            importlib.import_module(f"{name}.models")
            importlib.import_module(f"{name}.evaluation")
            return name
        except ImportError:
            continue

    raise ImportError(
        "Could not resolve the benchmark package. "
        "Set BENCHMARK_PACKAGE to your package name before running pytest."
    )


PKG = _resolve_package_name()

evaluation_module = importlib.import_module(f"{PKG}.evaluation")
models_module = importlib.import_module(f"{PKG}.models")

evaluate_plan = evaluation_module.evaluate_plan
Layout = models_module.Layout
Scene = models_module.Scene
SceneFlow = models_module.SceneFlow
SceneLayoutRef = models_module.SceneLayoutRef
SceneTracking = models_module.SceneTracking
SceneWindow = models_module.SceneWindow
Task = models_module.Task
TaskRobot = models_module.TaskRobot
TaskSceneRef = models_module.TaskSceneRef
TaskTargetRef = models_module.TaskTargetRef
Robot = models_module.Robot
TrackSimple = models_module.TrackSimple
TrackStore = models_module.TrackStore
PlanResult = models_module.PlanResult


@pytest.fixture
def layout() -> Layout:
    return Layout(
        layout_id="test.layout",
        boundary=[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
        obstacles=[],
        exits=[],
        metadata={},
    )


@pytest.fixture
def scene() -> Scene:
    return Scene(
        scene_id="test.cross_flow.scene",
        layout=SceneLayoutRef(
            layout_id="test.layout",
            path="minimal_layout.json",
            coordinate_frame="layout_xy_meters",
        ),
        tracking=SceneTracking(
            tracking_id="test.scene",
            format="csv",
            path="minimal_tracks.csv",
            timestamp_field="timestamp",
            track_id_field="person_track_id",
            coordinate_frame="layout_xy_meters",
        ),
        window=SceneWindow(
            start="1971-01-01T00:00:00Z",
            end="1971-01-01T00:00:04Z",
            duration_s=4.0,
        ),
        flow=SceneFlow(p_star=(5.0, 5.0), u_hat=(1.0, 0.0)),
        metadata={},
        provenance={},
    )


@pytest.fixture
def task() -> Task:
    return Task(
        task_id="test.cross_flow.001",
        scene=TaskSceneRef(scene_id="test.cross_flow.scene", path="minimal_scene.json"),
        task_type="cross_flow",
        robot=TaskRobot(start=(1.0, 8.5), goal=(8.0, 2.0)),
        target=TaskTargetRef(track_id="t1"),
        metadata={},
        provenance={},
    )


@pytest.fixture
def robot() -> Robot:
    return Robot(
        robot_id="test.disk",
        kinematics="holonomic",
        radius_m=0.25,
        max_speed_mps=1.2,
        metadata={},
    )


@pytest.fixture
def target_track() -> TrackSimple:
    # Ends exactly at the task goal to make goal completion checks crisp.
    return TrackSimple(
        track_id="t1",
        timestamps=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        x=np.array([1.0, 3.0, 5.5, 8.0], dtype=float),
        y=np.array([8.5, 6.5, 4.25, 2.0], dtype=float),
        vx=np.array([2.0, 2.5, 2.5, 2.5], dtype=float),
        vy=np.array([-2.0, -2.25, -2.25, -2.25], dtype=float),
        position_valid=np.array([True, True, True, True]),
        velocity_valid=np.array([True, True, True, True]),
        metadata={},
    )


@pytest.fixture
def other_track_a() -> TrackSimple:
    return TrackSimple(
        track_id="t2",
        timestamps=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        x=np.array([2.0, 2.0, 2.0, 2.0], dtype=float),
        y=np.array([9.0, 7.0, 5.0, 3.0], dtype=float),
        vx=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        vy=np.array([-2.0, -2.0, -2.0, -2.0], dtype=float),
        position_valid=np.array([True, True, True, True]),
        velocity_valid=np.array([True, True, True, True]),
        metadata={},
    )


@pytest.fixture
def other_track_b() -> TrackSimple:
    return TrackSimple(
        track_id="t3",
        timestamps=np.array([0.0, 1.0, 2.0, 3.0], dtype=float),
        x=np.array([7.5, 7.5, 7.5, 7.5], dtype=float),
        y=np.array([8.0, 6.0, 4.0, 2.0], dtype=float),
        vx=np.array([0.0, 0.0, 0.0, 0.0], dtype=float),
        vy=np.array([-2.0, -2.0, -2.0, -2.0], dtype=float),
        position_valid=np.array([True, True, True, True]),
        velocity_valid=np.array([True, True, True, True]),
        metadata={},
    )


@pytest.fixture
def store(
    target_track: TrackSimple,
    other_track_a: TrackSimple,
    other_track_b: TrackSimple,
) -> TrackStore:
    timestamps = np.concatenate(
        [target_track.timestamps, other_track_a.timestamps, other_track_b.timestamps]
    )
    track_ids = np.array(
        [target_track.track_id] * target_track.num_samples()
        + [other_track_a.track_id] * other_track_a.num_samples()
        + [other_track_b.track_id] * other_track_b.num_samples(),
        dtype=object,
    )
    x = np.concatenate([target_track.x, other_track_a.x, other_track_b.x])
    y = np.concatenate([target_track.y, other_track_a.y, other_track_b.y])
    vx = np.concatenate([target_track.vx, other_track_a.vx, other_track_b.vx])
    vy = np.concatenate([target_track.vy, other_track_a.vy, other_track_b.vy])
    position_valid = np.concatenate(
        [target_track.position_valid, other_track_a.position_valid, other_track_b.position_valid]
    )
    velocity_valid = np.concatenate(
        [target_track.velocity_valid, other_track_a.velocity_valid, other_track_b.velocity_valid]
    )

    return TrackStore(
        timestamps=timestamps,
        track_ids=track_ids,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        position_valid=position_valid,
        velocity_valid=velocity_valid,
        timestamp_field="timestamp",
        track_id_field="person_track_id",
        coordinate_frame="layout_xy_meters",
        source_path="synthetic.csv",
        metadata={},
    )


def _make_plan(
    track: TrackSimple,
    *,
    planner_name: str = "test_planner",
    success: bool = True,
) -> PlanResult:
    return PlanResult(
        planner_name=planner_name,
        success=success,
        track=track,
        path_length_m=track.path_length_m(),
        runtime_s=0.01,
        message="ok" if success else "failed",
        metadata={},
    )


def test_evaluate_plan_perfect_match_has_high_v1_scores(
    monkeypatch: pytest.MonkeyPatch,
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    target_track: TrackSimple,
    store: TrackStore,
) -> None:
    monkeypatch.setattr(evaluation_module, "load_track_store", lambda *args, **kwargs: store)

    plan = _make_plan(target_track)
    result = evaluate_plan(layout, scene, task, robot, plan)

    assert result.success is True
    assert result.scene_type == "cross_flow"
    assert result.target_track_id == "t1"

    assert result.path_deviation_m == pytest.approx(0.0, abs=1e-9)
    assert result.react_proxy_gap == pytest.approx(0.0, abs=1e-9)

    assert result.human_likeness_score is not None
    assert result.human_likeness_score > 0.99

    assert result.social_compatibility_score is not None
    assert 0.0 <= result.social_compatibility_score <= 1.0

    assert result.task_efficiency_score is not None
    assert result.task_efficiency_score > 0.99

    assert result.overall_score is not None
    assert 0.0 <= result.overall_score <= 1.0

    # New explicit goal-completion component lives in metadata in v1.
    assert result.metadata["goal_completion_score"] is not None
    assert result.metadata["goal_completion_score"] > 0.99
    assert result.metadata["goal_distance_m"] == pytest.approx(0.0, abs=1e-9)

    # Cross-flow weighting should be present and sum to 1.
    weights = result.metadata["weights"]
    assert set(weights.keys()) == {
        "goal_completion",
        "social_compatibility",
        "task_efficiency",
        "human_likeness",
    }
    assert sum(weights.values()) == pytest.approx(1.0)


def test_evaluate_plan_populates_social_metrics_from_other_tracks(
    monkeypatch: pytest.MonkeyPatch,
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    target_track: TrackSimple,
    store: TrackStore,
) -> None:
    monkeypatch.setattr(evaluation_module, "load_track_store", lambda *args, **kwargs: store)

    robot_track = TrackSimple(
        track_id="robot",
        timestamps=target_track.timestamps,
        x=target_track.x + np.array([0.0, 0.2, 0.2, 0.0]),
        y=target_track.y + np.array([0.0, -0.1, 0.1, 0.0]),
        vx=target_track.vx,
        vy=target_track.vy,
        position_valid=np.array([True, True, True, True]),
        velocity_valid=np.array([True, True, True, True]),
        metadata={},
    )

    plan = _make_plan(robot_track)
    result = evaluate_plan(layout, scene, task, robot, plan)

    assert result.n_other_human_tracks == 2
    assert result.min_other_human_distance_m is not None
    assert result.mean_other_human_min_distance_m is not None
    assert result.flow_axis_alignment is not None
    assert result.flow_coherence is not None
    assert result.disruption_rate is not None

    assert result.social_compatibility_score is not None
    assert 0.0 <= result.social_compatibility_score <= 1.0


def test_evaluate_plan_uses_task_robot_start_goal_for_scene_scale(
    monkeypatch: pytest.MonkeyPatch,
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    target_track: TrackSimple,
    store: TrackStore,
) -> None:
    monkeypatch.setattr(evaluation_module, "load_track_store", lambda *args, **kwargs: store)

    plan = _make_plan(target_track)
    result = evaluate_plan(layout, scene, task, robot, plan)

    expected = float(
        np.hypot(
            task.robot.goal[0] - task.robot.start[0],
            task.robot.goal[1] - task.robot.start[1],
        )
    )
    assert result.scene_scale_m == pytest.approx(expected)


def test_evaluate_plan_goal_completion_drops_when_robot_misses_goal(
    monkeypatch: pytest.MonkeyPatch,
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    target_track: TrackSimple,
    store: TrackStore,
) -> None:
    monkeypatch.setattr(evaluation_module, "load_track_store", lambda *args, **kwargs: store)

    robot_track = TrackSimple(
        track_id="robot",
        timestamps=target_track.timestamps,
        x=np.array([1.0, 3.0, 5.0, 6.0], dtype=float),
        y=np.array([8.5, 6.5, 4.0, 3.5], dtype=float),
        vx=np.array([2.0, 2.0, 1.0, 1.0], dtype=float),
        vy=np.array([-2.0, -2.5, -0.5, -0.5], dtype=float),
        position_valid=np.array([True, True, True, True]),
        velocity_valid=np.array([True, True, True, True]),
        metadata={},
    )

    plan = _make_plan(robot_track)
    result = evaluate_plan(layout, scene, task, robot, plan)

    assert result.metadata["goal_distance_m"] is not None
    assert result.metadata["goal_distance_m"] > 0.5
    assert result.metadata["goal_completion_score"] is not None
    assert result.metadata["goal_completion_score"] < 1.0


def test_evaluate_plan_failed_plan_returns_structured_failure(
    monkeypatch: pytest.MonkeyPatch,
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    store: TrackStore,
) -> None:
    monkeypatch.setattr(evaluation_module, "load_track_store", lambda *args, **kwargs: store)

    plan = PlanResult(
        planner_name="test_planner",
        success=False,
        track=None,
        path_length_m=None,
        runtime_s=0.01,
        message="planner failed",
        metadata={},
    )

    result = evaluate_plan(layout, scene, task, robot, plan)

    assert result.success is False
    assert result.human_likeness_score is None
    assert result.social_compatibility_score is None
    assert result.task_efficiency_score == 0.0
    assert result.overall_score == 0.0
    assert "evaluation_error" in result.metadata


def test_evaluate_plan_ignores_invalid_robot_samples(
    monkeypatch: pytest.MonkeyPatch,
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    target_track: TrackSimple,
    store: TrackStore,
) -> None:
    monkeypatch.setattr(evaluation_module, "load_track_store", lambda *args, **kwargs: store)

    robot_track = TrackSimple(
        track_id="robot",
        timestamps=target_track.timestamps,
        x=np.array([1.0, np.nan, 5.5, 8.0], dtype=float),
        y=np.array([8.5, np.nan, 4.25, 2.0], dtype=float),
        vx=target_track.vx,
        vy=target_track.vy,
        position_valid=np.array([True, False, True, True]),
        velocity_valid=np.array([True, True, True, True]),
        metadata={},
    )

    plan = _make_plan(robot_track)
    result = evaluate_plan(layout, scene, task, robot, plan)

    assert result.success is True
    assert result.human_likeness_score is not None
    assert result.path_deviation_m is not None
    assert result.path_deviation_m >= 0.0


def test_evaluate_plan_scene_type_controls_weighting(
    monkeypatch: pytest.MonkeyPatch,
    layout: Layout,
    robot: Robot,
    target_track: TrackSimple,
    store: TrackStore,
) -> None:
    monkeypatch.setattr(evaluation_module, "load_track_store", lambda *args, **kwargs: store)

    icn_scene = Scene(
        scene_id="test.icn.scene",
        layout=SceneLayoutRef(
            layout_id="test.layout",
            path="minimal_layout.json",
            coordinate_frame="layout_xy_meters",
        ),
        tracking=SceneTracking(
            tracking_id="test.scene",
            format="csv",
            path="minimal_tracks.csv",
            timestamp_field="timestamp",
            track_id_field="person_track_id",
            coordinate_frame="layout_xy_meters",
        ),
        window=SceneWindow(
            start="1971-01-01T00:00:00Z",
            end="1971-01-01T00:00:04Z",
            duration_s=4.0,
        ),
        metadata={},
        provenance={},
    )

    icn_task = Task(
        task_id="test.icn.001",
        scene=TaskSceneRef(scene_id="test.icn.scene", path="minimal_scene.json"),
        task_type="icn",
        robot=TaskRobot(start=(1.0, 8.5), goal=(8.0, 2.0)),
        target=TaskTargetRef(track_id="t1"),
        metadata={},
        provenance={},
    )

    plan = _make_plan(target_track)
    result = evaluate_plan(layout, icn_scene, icn_task, robot, plan)

    weights = result.metadata["weights"]
    assert weights["goal_completion"] == pytest.approx(0.20)
    assert weights["social_compatibility"] == pytest.approx(0.35)
    assert weights["task_efficiency"] == pytest.approx(0.15)
    assert weights["human_likeness"] == pytest.approx(0.30)



