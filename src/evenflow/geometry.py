from __future__ import annotations

from .models import Point, Polygon


def centroid(poly: Polygon) -> Point:
    if not poly:
        raise ValueError("Cannot compute centroid of an empty polygon")
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def bounds(poly: Polygon) -> tuple[float, float, float, float]:
    if not poly:
        raise ValueError("Cannot compute bounds of an empty polygon")
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)
