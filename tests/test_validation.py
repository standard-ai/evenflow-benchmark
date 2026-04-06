from __future__ import annotations

from pathlib import Path

import pytest

from evenflow.io import load_layout
from evenflow.models import Layout
from evenflow.validation import ValidationError, validate_layout


FIXTURES = Path(__file__).parent / "fixtures"


def test_validate_layout_accepts_valid_fixture() -> None:
    layout = load_layout(FIXTURES / "minimal_layout.json", validate=False)
    validate_layout(layout)


def test_validate_layout_requires_layout_id() -> None:
    layout = Layout(layout_id="", boundary=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])

    with pytest.raises(ValidationError, match="layout_id is required"):
        validate_layout(layout)


def test_validate_layout_rejects_too_small_boundary() -> None:
    layout = Layout(layout_id="bad", boundary=[(0.0, 0.0), (1.0, 0.0)])

    with pytest.raises(ValidationError, match="boundary must have at least 3 points"):
        validate_layout(layout)
