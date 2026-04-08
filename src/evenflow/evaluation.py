from dataclasses import dataclass, field
from typing import Any

from .models import Layout, Scene, Task, Robot, PlanResult


@dataclass(slots=True)
class EvalResult:
    success: bool
    path_length_m: float | None
    runtime_s: float | None
    num_waypoints: int
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_plan(
    layout: Layout,
    scene: Scene,
    task: Task,
    robot: Robot,
    plan: PlanResult,
) -> EvalResult:
    return EvalResult(
        success=plan.success,
        path_length_m=plan.path_length_m,
        runtime_s=plan.runtime_s,
        num_waypoints=len(plan.waypoints),
        message=plan.message,
        metadata={
            "planner": plan.planner_name,
        },
    )
