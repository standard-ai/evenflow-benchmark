from pathlib import Path

from evenflow.io import load_task

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_task_reads_basic_fields() -> None:
    task = load_task(FIXTURES / "minimal_task.json")


    assert task.scene.scene_id == "test.scene_with_tracks"
    assert task.scene.path == "minimal_scene_with_tracks.json"
    assert task.task_type == "cross_flow"
    assert task.robot.start == (1.0, 8.5)
    assert task.robot.goal == (8.5, 2.0)
    
    assert task.target is not None
    assert task.target.track_id == "t1"
