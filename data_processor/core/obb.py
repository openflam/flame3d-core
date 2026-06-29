"""
obb.py – Oriented bounding boxes for point sets in a gravity-aligned frame.

The repository works in a right-handed **Z-up** world frame (see
``docs/CoordinateSystem.md``).  Objects in a scanned scene sit upright on the
floor, so the most meaningful "oriented" box is one that may rotate freely about
the vertical (up) axis but stays level — i.e. its footprint is the
minimum-area rectangle enclosing the points' projection onto the ground plane,
extruded between the lowest and highest point.  A full 3-D OBB (PCA on all three
axes) would tilt boxes off-vertical for noisy or flat objects, which is rarely
what you want indoors.

The returned corner ordering matches the convention used elsewhere in the repo
and expected by the Three.js viewer (``frontend/.../BoundingBoxMesh.tsx``):

    corners 0..3  – bottom face, traversed as a cycle (0→1→2→3)
    corners 4..7  – top face, with corner ``i+4`` directly above corner ``i``

so corner ``i`` and ``i+4`` share a vertical edge.  This is a strict superset of
the axis-aligned ordering produced by
``segment3d.postsam3_pipeline.bbox_corners.get_bbox`` — an axis-aligned box is
just the special case where the optimal yaw is 0.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def _min_area_rectangle(pts2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Minimum-area enclosing rectangle of a 2-D point set.

    Uses the rotating-calipers fact that an optimal rectangle is collinear with
    an edge of the convex hull, so we test each hull edge's orientation and keep
    the smallest-area axis-aligned box in that rotated frame.

    Parameters
    ----------
    pts2d : (N, 2) ndarray

    Returns
    -------
    (corners, size) : tuple
        ``corners`` is a (4, 2) array of rectangle corners in cyclic order;
        ``size`` is the (2,) array of the rectangle's side lengths.
    """
    pts2d = np.asarray(pts2d, dtype=np.float64)

    # Degenerate footprints (coincident or collinear points) have no 2-D hull;
    # fall back to the axis-aligned rectangle, which is still a valid box.
    try:
        hull = ConvexHull(pts2d)
        hull_pts = pts2d[hull.vertices]
    except (QhullError, ValueError):
        lo = pts2d.min(axis=0)
        hi = pts2d.max(axis=0)
        corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                            [hi[0], hi[1]], [lo[0], hi[1]]])
        return corners, hi - lo

    best_area = np.inf
    best_corners = None
    best_size = None
    n = len(hull_pts)
    for i in range(n):
        edge = hull_pts[(i + 1) % n] - hull_pts[i]
        theta = np.arctan2(edge[1], edge[0])

        # Rotate the hull by -theta so this edge aligns with the x-axis.
        c, s = np.cos(-theta), np.sin(-theta)
        rot = np.array([[c, -s], [s, c]])
        aligned = hull_pts @ rot.T

        lo = aligned.min(axis=0)
        hi = aligned.max(axis=0)
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        if area < best_area:
            best_area = area
            # Rectangle corners in the aligned frame, cyclic order.
            box = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                            [hi[0], hi[1]], [lo[0], hi[1]]])
            # Rotate back into the original frame (by +theta).
            cb, sb = np.cos(theta), np.sin(theta)
            rot_back = np.array([[cb, -sb], [sb, cb]])
            best_corners = box @ rot_back.T
            best_size = hi - lo

    return best_corners, best_size


def compute_obb(points: np.ndarray, up_axis: int = 2) -> dict[str, Any]:
    """Gravity-aligned oriented bounding box for a set of 3-D points.

    The box is free to rotate about ``up_axis`` but stays upright; its footprint
    is the minimum-area rectangle enclosing the points projected onto the plane
    orthogonal to ``up_axis``, extruded between the min and max along ``up_axis``.

    Parameters
    ----------
    points : (N, 3) ndarray
        Points in a right-handed frame with ``up_axis`` pointing up.
    up_axis : int, optional
        Index (0, 1, or 2) of the vertical axis; default ``2`` (Z-up).

    Returns
    -------
    dict
        ``{"corners", "min", "max", "center", "size"}`` where:

        * ``corners`` – (8, 3) list, ordered for the viewer (see module docs).
        * ``min`` / ``max`` – the axis-aligned envelope of the oriented box
          (kept for the PostGIS spatial index, which uses an AABB envelope).
        * ``center`` – the box centroid.
        * ``size`` – oriented extents ``[footprint_w, footprint_d, height]``.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be an (N, 3) array")
    if len(points) == 0:
        raise ValueError("Cannot compute an OBB for an empty point set")

    plane_axes = [a for a in range(3) if a != up_axis]
    a0, a1 = plane_axes

    up = points[:, up_axis]
    up_min, up_max = float(up.min()), float(up.max())

    rect, rect_size = _min_area_rectangle(points[:, [a0, a1]])

    # Assemble the 8 corners: bottom face (up_min) then top face (up_max), with
    # corner i+4 directly above corner i so they share a vertical edge.
    corners = np.empty((8, 3), dtype=np.float64)
    for i, (u, v) in enumerate(rect):
        corners[i, a0] = u
        corners[i, a1] = v
        corners[i, up_axis] = up_min
        corners[i + 4, a0] = u
        corners[i + 4, a1] = v
        corners[i + 4, up_axis] = up_max

    aabb_min = corners.min(axis=0)
    aabb_max = corners.max(axis=0)
    center = corners.mean(axis=0)

    size = np.empty(3, dtype=np.float64)
    size[a0] = rect_size[0]
    size[a1] = rect_size[1]
    size[up_axis] = up_max - up_min

    return {
        "corners": corners.tolist(),
        "min": aabb_min.tolist(),
        "max": aabb_max.tolist(),
        "center": center.tolist(),
        "size": size.tolist(),
    }
