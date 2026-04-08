from __future__ import annotations

from .models import Layout, Polygon, Robot, Scene, Task


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

    if not scene.layout_id:
        raise ValidationError("layout_id is required")

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

    if not task.scene_id:
        raise ValidationError("scene_id is required")

    if not task.task_type:
        raise ValidationError("task_type is required")

    _validate_point("task.robot.start", task.robot.start)
    _validate_point("task.robot.goal", task.robot.goal)


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
