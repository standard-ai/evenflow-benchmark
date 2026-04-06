from __future__ import annotations

from evenflow.geometry import bounds, centroid


def test_centroid_of_rectangle() -> None:
    poly = [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]
    assert centroid(poly) == (2.0, 1.0)


def test_bounds_of_rectangle() -> None:
    poly = [(2.0, 3.0), (4.0, 3.0), (4.0, 6.0), (2.0, 6.0)]
    assert bounds(poly) == (2.0, 3.0, 4.0, 6.0)
