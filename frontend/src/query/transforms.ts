// Coordinate-system transforms — frontend only.
//
// The whole backend (data_processor + server) works in a single right-handed
// **Z-up** world frame, established by
// data_processor/vendor_specific/polycam.py. The server stores and returns all
// geometry in that frame, so it stays usable by any downstream application.
//
// The Three.js viewer, however, is **Y-up**. These helpers convert between the
// two and are the *only* place coordinate swaps happen on the client. They
// mirror the axis swaps that previously lived in the server's transforms.py:
//
//   world (Z-up) -> viewer (Y-up):  (x, y, z) -> (y, z, x)
//   viewer (Y-up) -> world (Z-up):  (x, y, z) -> (z, x, y)
//
// Apply worldToViewer* on data coming *from* the server before rendering, and
// viewerToWorld* on geometry going *to* the server (component edits/additions).

import type { BoundingBox } from "./types/global";

export type Vec3 = [number, number, number];

/** World (right-handed Z-up) -> viewer (Three.js Y-up). */
export function worldToViewerPoint(c: number[]): Vec3 {
  return [c[1], c[2], c[0]];
}

/** Viewer (Three.js Y-up) -> world (right-handed Z-up). */
export function viewerToWorldPoint(c: number[]): Vec3 {
  return [c[2], c[0], c[1]];
}

/** Convert a bounding box's corners from the world frame to the viewer frame. */
export function worldToViewerBbox(bbox: BoundingBox): BoundingBox {
  if (!bbox || !bbox.corners) return bbox;
  return { corners: bbox.corners.map(worldToViewerPoint) };
}

/** Convert a bounding box's corners from the viewer frame to the world frame. */
export function viewerToWorldBbox(bbox: BoundingBox): BoundingBox {
  if (!bbox || !bbox.corners) return bbox;
  return { corners: bbox.corners.map(viewerToWorldPoint) };
}
