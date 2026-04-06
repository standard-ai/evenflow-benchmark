from __future__ import annotations

from .models import Layout, Polygon


class ValidationError(ValueError):
    pass


def _validate_polygon(name: str, poly: Polygon) -> None:
    if len(poly) < 3:
        raise ValidationError(f"{name} must have at least 3 points")
    for i, point in enumerate(poly):
        if len(point) != 2:
            raise ValidationError(f"{name}[{i}] must be a 2D point")
        if not all(isinstance(v, (int, float)) for v in point):
            raise ValidationError(f"{name}[{i}] must contain numeric coordinates")


def validate_layout(layout: Layout) -> None:
    if not layout.layout_id:
        raise ValidationError("layout_id is required")
    _validate_polygon("boundary", layout.boundary)
    for obs in layout.obstacles:
        if not obs.id:
            raise ValidationError("Each obstacle must have an id")
        _validate_polygon(f"obstacle:{obs.id}", obs.polygon)
    for ex in layout.exits:
        if not ex.id:
            raise ValidationError("Each exit must have an id")
        _validate_polygon(f"exit:{ex.id}", ex.polygon)
