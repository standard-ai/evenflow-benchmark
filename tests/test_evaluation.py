from evenflow.evaluation import evaluate_plan
from evenflow.models import PlanResult, PlanWaypoint


def test_evaluate_plan_pass_through() -> None:
    plan = PlanResult(
        planner_name="test",
        success=True,
        waypoints=(
            PlanWaypoint(x=0, y=0),
            PlanWaypoint(x=1, y=1),
        ),
        path_length_m=1.4,
        runtime_s=0.01,
    )

    result = evaluate_plan(None, None, None, None, plan)

    assert result.success is True
    assert result.path_length_m == 1.4
    assert result.num_waypoints == 2
