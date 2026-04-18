from __future__ import annotations

from pathlib import Path

from evenflow.io import load_layout, load_plan, load_scene
from evenflow.render import (
    render_layout,
    save_layout_figure,
    save_plan_figure,
    save_scene_figure,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_render_layout_returns_figure_and_axes() -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json")
    fig, ax = render_layout(layout)

    assert fig is not None
    assert ax is not None
    assert ax.get_title() == "EvenFlow Layout: test.minimal_layout"


def test_save_layout_figure_writes_png(tmp_path: Path) -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json")
    out_path = tmp_path / "layout.png"

    save_layout_figure(layout, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_plan_figure_writes_png(tmp_path: Path) -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json")
    plan = load_plan(FIXTURES / "minimal_plan.json")
    out_path = tmp_path / "plan.png"

    save_plan_figure(layout, plan, out_path)

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_save_scene_figure_with_canonical_tracks_writes_png(tmp_path: Path) -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json")
    scene = load_scene(FIXTURES / "minimal_scene_with_tracks.json")
    out_path = tmp_path / "scene_with_tracks.png"

    save_scene_figure(
        layout,
        scene,
        out_path,
        scene_json_path=FIXTURES / "minimal_scene_with_tracks.json",
        show_tracks=True,
        max_tracks=5,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
