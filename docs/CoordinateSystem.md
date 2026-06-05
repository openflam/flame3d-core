# Coordinate Systems

flame3d-core deliberately uses **two** coordinate frames. They are not an
inconsistency to be "fixed" — they serve different purposes and are bridged by a
single, fixed rotation.

| Frame | Up axis | Handedness | Used by |
| ----- | ------- | ---------- | ------- |
| **World frame** | **Z-up** | right-handed | everything on the backend: `data_processor`, COLMAP output, component bounding boxes, the PostGIS spatial tables, and the `server` query/reasoning logic |
| **glTF / mesh frame** | **Y-up** | right-handed | `mesh.glb` (and the raw Polycam `raw.glb`), per the glTF specification |

The two are related by a single fixed rotation, `_YUP_TO_ZUP`, defined in
[`data_processor/vendor_specific/polycam.py`](../data_processor/vendor_specific/polycam.py).

## Why there are two frames

### The world frame is Z-up

`data_processor/vendor_specific/polycam.py` converts the Polycam/ARKit capture
(natively **Y-up**) into a right-handed **Z-up** world frame on load, using:

```
_YUP_TO_ZUP:   (x, y, z)_mesh  ->  (x, -z, y)_world
```

Everything downstream stays in this Z-up world frame:

- **COLMAP output** written by mesh reprojection.
- **Component bounding boxes** (`bbox_corners.json` → the per-dataset PostGIS
  tables). Stored verbatim; `server/database/spatial_tables.py` applies **no**
  coordinate transform.
- **Spatial reasoning tools** in `server/search/` assume `z` is the up axis
  (e.g. `search_around_component`'s `above` / `below` filtering compares the
  `z` component of bbox centers, and the nd-GIST spatial index is built on the
  Z-up corners).

This is the canonical frame. New vendor-specific loaders should convert their
input into this same right-handed Z-up frame so the rest of the pipeline "just
works" without changes.

### The mesh stays Y-up because glTF mandates it

`mesh.glb` is **not** rotated into the world frame. The polycam step copies the
raw Polycam `raw.glb` verbatim, and the glTF specification *requires* assets to
be **Y-up**. Baking Z-up geometry into a `.glb` would produce a file that every
compliant loader (Three.js `GLTFLoader`, Blender, `<model-viewer>`, …)
misinterprets and renders lying on its side. Keeping the mesh as a standard
glTF asset is what makes it interoperable with arbitrary downstream tools.

> This is why importing `mesh.glb` and the COLMAP output into Blender lines up
> exactly: Blender's glTF importer applies the same Y-up → Z-up rotation as
> `_YUP_TO_ZUP`, landing the mesh on the world/COLMAP frame.

## The bridge

To overlay world-frame geometry (bboxes, routes, occupancy) onto the Y-up mesh,
apply the rotation that relates the two frames:

```
mesh / viewer (Y-up)  ->  world (Z-up):   (x, y, z) -> (x, -z, y)   # _YUP_TO_ZUP
world (Z-up)  ->  mesh / viewer (Y-up):   (x, y, z) -> (x,  z, -y)  # inverse
```

These are exact inverses of each other (the round-trip is the identity), and the
transform is a proper rotation (determinant +1, no mirroring): `world_z` maps to
viewer "up" and the horizontal plane is preserved.

## Who applies the bridge

The **backend never applies viewer-specific transforms** — the server stores and
returns all geometry in the Z-up world frame, so it remains usable by any
downstream consumer, not just this frontend.

The **frontend** is the Three.js viewer and works in the mesh's Y-up frame. It
applies the bridge in exactly one place,
[`frontend/src/query/transforms.ts`](../frontend/src/query/transforms.ts):

- **Inbound** (server → viewer): `worldToViewerBbox` is applied to search
  results, custom bboxes, the annotations overlay, and the occupancy grid before
  rendering.
- **Outbound** (viewer → server): `viewerToWorldBbox` is applied to gizmo-edited
  and newly-added component bboxes before they are POSTed back, so the server
  always receives world-frame geometry.

Note that `transforms.ts` is not coupled to Polycam specifically — it is a
generic **"glTF Y-up ↔ world Z-up"** bridge, valid for any standard glTF mesh
served alongside the Z-up world frame.

## A downstream consumer's checklist

If you consume the server's data outside this frontend:

- `GET /api/load_mesh` returns a **Y-up** glTF binary.
- Component bboxes (from `/api/search`, `/api/search_stream`,
  `/api/download_all_components`) and COLMAP output are **Z-up**.
- To align them, rotate the mesh frame to world with
  `(x, y, z) -> (x, -z, y)`, or rotate the world geometry to the mesh frame with
  `(x, y, z) -> (x, z, -y)`.
