// Coordinate-system transforms — frontend only.
//
// The whole backend (data_processor + server) works in a single right-handed
// **Z-up** world frame, established by
// data_processor/vendor_specific/polycam.py. The server stores and returns all
// geometry in that frame, so it stays usable by any downstream application.
//
// The Three.js viewer, however, is **Y-up**, and it renders mesh.glb in its raw
// Polycam/ARKit Y-up frame (GLTFLoader applies no axis change). polycam.py built
// the world frame by rotating that same Y-up mesh with _YUP_TO_ZUP:
//
//   _YUP_TO_ZUP:  (x, y, z)_viewer  ->  (x, -z, y)_world
//
// so to place world-frame geometry onto the viewer's Y-up mesh we apply the
// inverse of that rotation. (Blender's glTF importer happens to apply the exact
// same Y-up->Z-up rotation as _YUP_TO_ZUP, which is why the mesh and COLMAP line
// up there but a naive axis-swap is wrong here.)
//
//   world (Z-up) -> viewer (Y-up):  (x, y, z) -> (x,  z, -y)
//   viewer (Y-up) -> world (Z-up):  (x, y, z) -> (x, -z,  y)
//
// Apply worldToViewer* on data coming *from* the server before rendering, and
// viewerToWorld* on geometry going *to* the server (component edits/additions).

import type { BoundingBox } from "./types/global";

export type Vec3 = [number, number, number];

/** World (right-handed Z-up) -> viewer (Three.js Y-up): inverse of _YUP_TO_ZUP. */
export function worldToViewerPoint(c: number[]): Vec3 {
  return [c[0], c[2], -c[1]];
}

/** Viewer (Three.js Y-up) -> world (right-handed Z-up): _YUP_TO_ZUP. */
export function viewerToWorldPoint(c: number[]): Vec3 {
  return [c[0], -c[2], c[1]];
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
