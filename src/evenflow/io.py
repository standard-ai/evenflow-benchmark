from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import Exit, Layout, Obstacle, Scenario
from .validation import validate_layout


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


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


def load_scenario(path: str | Path) -> Scenario:
    data = _read_json(path)
    known = {"scenario_id", "layout_id", "start", "goal"}
    return Scenario(
        scenario_id=data.get("scenario_id", Path(path).stem),
        layout_id=data["layout_id"],
        start=tuple(data["start"]),
        goal=tuple(data["goal"]),
        metadata={k: v for k, v in data.items() if k not in known},
    )
