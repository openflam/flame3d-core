"""Coordinate-system transforms between COLMAP and the Model3DViewer.

The database stores bounding boxes in the COLMAP coordinate system. The
frontend 3D viewer expects axes swapped (x, y, z) -> (y, z, x). These helpers
convert in both directions and are shared by the search and component routes.
"""

from __future__ import annotations

from typing import Any


def transform_coordinates(coords: list[float]) -> list[float]:
    """Transform a single [x, y, z] from COLMAP to the Model3DViewer frame."""
    return [coords[1], coords[2], coords[0]]


def transform_bbox(bbox: dict | None, transform_type: str = "COLMAP_to_3DViewer") -> Any:
    """Transform a bounding box dict (with a ``corners`` key) between frames.

    Args:
        bbox: Bounding box dict, e.g. ``{"corners": [[x, y, z], ...]}``.
        transform_type: ``"COLMAP_to_3DViewer"`` (default) or
            ``"3DViewer_to_COLMAP"``.
    """
    if not bbox or "corners" not in bbox:
        return bbox

    if transform_type == "COLMAP_to_3DViewer":
        return {"corners": [[c[1], c[2], c[0]] for c in bbox.get("corners", [])]}
    elif transform_type == "3DViewer_to_COLMAP":
        return {"corners": [[c[2], c[0], c[1]] for c in bbox.get("corners", [])]}
    return bbox
