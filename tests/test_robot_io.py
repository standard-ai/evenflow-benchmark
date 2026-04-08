from __future__ import annotations

from pathlib import Path

from evenflow.io import load_robot


FIXTURES = Path(__file__).parent / "fixtures"


def test_load_robot_reads_basic_fields() -> None:
    robot = load_robot(FIXTURES / "minimal_robot.json")

    assert robot.robot_id == "test.simple_disk"
    assert robot.kinematics == "holonomic"
    assert robot.radius_m == 0.25
    assert robot.max_speed_mps == 1.2
