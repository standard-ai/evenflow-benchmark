from pathlib import Path

from evenflow.io import load_scene

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_scene_reads_basic_fields() -> None:
    scene = load_scene(FIXTURES / "minimal_scene.json")

    assert scene.scene_id == "test.simple"
    assert scene.layout.layout_id == "test.minimal_layout"
    assert scene.layout.path == "../fixtures/minimal_layout.json"
    assert scene.window.duration_s == 10.0
    assert scene.flow.p_star == (5.0, 5.0)
