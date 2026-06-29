"""
scannetpp.py – Load a ScanNet++ scene and run mesh reprojection.

Expected input
--------------
Place an unpacked ScanNet++ scene at ``data/<dataset_name>/`` with (at least)
the following subset of the official ScanNet++ layout::

    data/<dataset_name>/
    ├── scans/
    │   └── mesh_aligned_0.05.ply          # reconstructed 3D mesh (vertex colors)
    └── dslr/
        ├── colmap/
        │   ├── cameras.txt                # OPENCV_FISHEYE intrinsics (unused here)
        │   ├── images.txt                 # camera poses (world == mesh frame)
        │   └── points3D.txt
        ├── nerfstudio/
        │   └── transforms_undistorted.json  # PINHOLE intrinsics for undistorted imgs
        └── resized_undistorted_images/    # undistorted (pinhole) DSLR images

Why undistorted images
----------------------
The raw ScanNet++ DSLR captures use an ``OPENCV_FISHEYE`` camera model, but
:func:`mesh_reprojection` projects with a simple pinhole model.  ScanNet++ ships
pre-undistorted images (``resized_undistorted_images/``) together with the
matching PINHOLE intrinsics in ``nerfstudio/transforms_undistorted.json``, so we
reproject against those.  The camera *poses* are unchanged by undistortion, so
we take them straight from ``dslr/colmap/images.txt`` as instructed by the
ScanNet++ docs (the colmap poses live in the same frame as the mesh).

Coordinate convention
---------------------
ScanNet++ aligned meshes and the colmap camera poses are already in a
right-handed **Z-up** world frame, and the colmap camera convention (x-right,
y-down, z-forward) matches what :func:`mesh_reprojection` expects.  No
coordinate transform is therefore applied to any backend data.

The single exception is ``mesh.glb``: the Three.js viewer renders the glb
*raw* (Y-up) and transforms all other geometry from world Z-up into that Y-up
frame.  To stay consistent with the Polycam path (whose glb is natively Y-up),
the exported glb is rotated Z-up -> Y-up before writing — see
``frontend/src/query/transforms.ts``.

Usage::

    from data_processor.vendor_specific.scannetpp import process_scannetpp

    result = process_scannetpp("scannetpp_09c1414f1b")
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from config_io import (
    get_data_path,
    get_images_output_path,
    get_output_path,
)
from data_processor.core.mesh_reprojection import mesh_reprojection
from data_processor.core.obb import compute_obb
from segment3d.utils.read_write_model import read_images_text

# ═══════════════════════════════════════════════════════════════════════════
# Coordinate-system helpers
# ═══════════════════════════════════════════════════════════════════════════

# Rotation that maps the world Z-up frame to the Three.js viewer's Y-up frame:
#   (x, y, z)_world -> (x, z, -y)_viewer
# This is the inverse of polycam's _YUP_TO_ZUP, mirrored by
# ``worldToViewerPoint`` in frontend/src/query/transforms.ts.  It is applied
# *only* to the exported mesh.glb so the viewer renders it aligned with the
# (world-frame) component bounding boxes; all other data stays Z-up.
_ZUP_TO_YUP = np.array(
    [[1,  0,  0],
     [0,  0,  1],
     [0, -1,  0]],
    dtype=np.float64,
)


# ═══════════════════════════════════════════════════════════════════════════
# COLMAP pose parsing
# ═══════════════════════════════════════════════════════════════════════════

def _parse_colmap_poses(images_txt: Path) -> dict[str, np.ndarray]:
    """Parse ``images.txt`` and return ``{image_name: c2w (4×4)}``.

    Uses COLMAP's own reader (:func:`segment3d.utils.read_write_model.read_images_text`),
    which yields each image's world-to-camera quaternion (``qvec``, scalar-first)
    and translation (``tvec``).  COLMAP stores world-to-camera, so we invert it
    to the camera-to-world matrix that :func:`mesh_reprojection` expects.
    """
    images = read_images_text(str(images_txt))
    if not images:
        raise ValueError(f"No camera poses parsed from {images_txt}")

    poses: dict[str, np.ndarray] = {}
    for img in images.values():
        R_w2c = Rotation.from_quat(img.qvec, scalar_first=True).as_matrix()
        t_w2c = np.asarray(img.tvec, dtype=np.float64)

        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, :3] = R_w2c.T
        c2w[:3, 3] = -R_w2c.T @ t_w2c
        poses[img.name] = c2w
    return poses


# ═══════════════════════════════════════════════════════════════════════════
# Intrinsics (undistorted / PINHOLE)
# ═══════════════════════════════════════════════════════════════════════════

def _load_pinhole_intrinsics(transforms_json: Path) -> tuple[np.ndarray, dict[str, bool]]:
    """Read the shared PINHOLE intrinsics and per-frame ``is_bad`` flags.

    Returns
    -------
    (intrinsics, is_bad) : tuple
        ``intrinsics`` is the 3×3 pinhole matrix shared by every undistorted
        frame.  ``is_bad`` maps image filename -> bad-frame flag (frames not
        listed default to *not* bad).
    """
    with open(transforms_json) as f:
        meta = json.load(f)

    intrinsics = np.array(
        [[meta["fl_x"], 0,            meta["cx"]],
         [0,            meta["fl_y"], meta["cy"]],
         [0,            0,            1         ]],
        dtype=np.float64,
    )

    is_bad: dict[str, bool] = {}
    for frame in meta.get("frames", []) + meta.get("test_frames", []):
        is_bad[frame["file_path"]] = bool(frame.get("is_bad", False))

    return intrinsics, is_bad


# ═══════════════════════════════════════════════════════════════════════════
# File-loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_image(image_path: Path) -> np.ndarray:
    """Load an RGB image as a uint8 numpy array (H, W, 3)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_mesh(mesh_path: Path) -> o3d.geometry.TriangleMesh:
    """Load the aligned ScanNet++ mesh (kept in its native Z-up frame)."""
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")
    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# Frame assembly
# ═══════════════════════════════════════════════════════════════════════════

def _build_all_frames(
    poses: dict[str, np.ndarray],
    intrinsics: np.ndarray,
    images_dir: Path,
    is_bad: dict[str, bool],
    image_sample_prob: float = 1.0,
    sample_seed: int = 0,
) -> list[dict[str, Any]]:
    """Build the ``image_pose_depth`` list for every usable frame.

    A frame is used when it has a colmap pose, a matching undistorted image on
    disk, and is not flagged ``is_bad``.  Depth is omitted on purpose:
    :func:`mesh_reprojection` renders depth from the mesh itself
    (``use_rendered_depth=True``), so no captured depth map is needed.

    Parameters
    ----------
    image_sample_prob : float, optional
        Fraction of usable frames (in ``(0, 1]``) to keep.  Default ``1.0``
        keeps every frame; lower values give a quick, sparse test run.
    sample_seed : int, optional
        Seed for the frame-sampling RNG, for reproducible subsets.
    """
    if not 0.0 < image_sample_prob <= 1.0:
        raise ValueError("image_sample_prob must be in (0, 1]")

    usable: list[str] = []
    skipped_missing = 0
    skipped_bad = 0
    for name in sorted(poses):
        if is_bad.get(name, False):
            skipped_bad += 1
            continue
        if not (images_dir / name).is_file():
            skipped_missing += 1
            continue
        usable.append(name)

    if image_sample_prob < 1.0:
        rng = np.random.default_rng(sample_seed)
        keep = rng.random(len(usable)) < image_sample_prob
        sampled = [name for name, k in zip(usable, keep) if k]
        print(f"Sampling {len(sampled)}/{len(usable)} frames "
              f"(image_sample_prob={image_sample_prob})")
    else:
        sampled = usable

    frames = [
        {
            "image": _load_image(images_dir / name),
            "extrinsics": poses[name],
            "intrinsics": intrinsics,
            "name": name,
        }
        for name in sampled
    ]

    print(
        f"Built {len(frames)} frames "
        f"(skipped {skipped_bad} bad, {skipped_missing} missing images)"
    )
    if not frames:
        raise FileNotFoundError(
            f"No usable frames found — no images in {images_dir} matched the "
            "colmap poses (or sampling removed them all)."
        )
    return frames


# ═══════════════════════════════════════════════════════════════════════════
# Mesh.glb export
# ═══════════════════════════════════════════════════════════════════════════

def _write_mesh_glb(mesh: o3d.geometry.TriangleMesh, dataset_name: str) -> None:
    """Write ``mesh.glb`` rotated from world Z-up into the viewer's Y-up frame.

    The reprojection mesh stays untouched; we rotate a copy so the Three.js
    viewer (which renders the glb raw) lines up with the world-frame component
    boxes — exactly as the Polycam path's natively-Y-up glb does.
    """
    viewer_mesh = o3d.geometry.TriangleMesh(mesh)
    viewer_mesh.rotate(_ZUP_TO_YUP, center=(0, 0, 0))

    mesh_out = get_output_path(dataset_name) / "mesh.glb"
    if not o3d.io.write_triangle_mesh(str(mesh_out), viewer_mesh):
        raise RuntimeError(f"Failed to write mesh.glb to {mesh_out}")
    print(f"Wrote mesh to {mesh_out}")


def _link_images(images_dir: Path, dataset_name: str) -> None:
    """Symlink ``outputs/<dataset>/images`` -> the undistorted image directory."""
    images_link = get_images_output_path(dataset_name)
    if images_link.is_symlink():
        images_link.unlink()
    elif images_link.is_dir():
        shutil.rmtree(images_link)
    elif images_link.exists():
        images_link.unlink()

    if images_dir.exists():
        images_link.symlink_to(images_dir.resolve(), target_is_directory=True)


# ═══════════════════════════════════════════════════════════════════════════
# Ground-truth segmentation (per-instance, from the segment annotations)
# ═══════════════════════════════════════════════════════════════════════════

def _build_segment_lookup(
    seg_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an inverse map from segment ID to the vertices it contains.

    ``segments.json`` gives ``seg_indices[v] = segment_id`` per vertex; we need
    the reverse.  Sorting the vertices by segment ID groups each segment's
    vertices into one contiguous run, so a segment's vertices can later be sliced
    out with a single ``searchsorted`` — no per-segment scan of the full array.

    Returns ``(order, unique_segs, bounds)`` where ``order`` lists vertex indices
    sorted by segment, ``unique_segs`` are the sorted distinct segment IDs, and
    segment ``unique_segs[p]`` owns ``order[bounds[p]:bounds[p + 1]]``.
    """
    order = np.argsort(seg_indices, kind="stable")
    seg_sorted = seg_indices[order]
    unique_segs, first = np.unique(seg_sorted, return_index=True)
    bounds = np.append(first, len(seg_sorted))
    return order, unique_segs, bounds


def _vertices_for_segments(
    segments: np.ndarray,
    order: np.ndarray,
    unique_segs: np.ndarray,
    bounds: np.ndarray,
) -> np.ndarray:
    """Return all vertex indices belonging to the given *segments*."""
    pos = np.searchsorted(unique_segs, segments)
    pos = pos[(pos < len(unique_segs)) & (unique_segs[np.clip(pos, 0, len(unique_segs) - 1)] == segments)]
    if len(pos) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([order[bounds[p]:bounds[p + 1]] for p in pos])


def _generate_groundtruth_segmentation(dataset_name: str) -> int:
    """Write ``bbox_corners.json`` + ``component_captions.json`` from GT instances.

    Shows the ground truth *as-is*, one component **per annotated instance** (not
    per class): each entry of ``scans/segments_anno.json``'s ``segGroups`` is one
    object, so multiple plants become multiple components.  An instance's
    vertices are gathered via ``scans/segments.json`` (``segIndices`` maps vertex
    -> segment) and the object's ``segments`` list; its caption is the instance
    ``label`` and its bbox a gravity-aligned oriented box
    (:func:`data_processor.core.obb.compute_obb`).  No clustering, filtering, or
    discarding is applied.  The files feed ``create_tables`` exactly like the ML
    pipeline's outputs do.

    Returns the number of components written.
    """
    data_dir = get_data_path(dataset_name)
    mesh_path = data_dir / "scans" / "mesh_aligned_0.05.ply"
    segments_json = data_dir / "scans" / "segments.json"
    anno_json = data_dir / "scans" / "segments_anno.json"
    for p in (mesh_path, segments_json, anno_json):
        if not p.is_file():
            raise FileNotFoundError(f"Required ground-truth file not found: {p}")

    coords = np.asarray(o3d.io.read_triangle_mesh(str(mesh_path)).vertices)
    with open(segments_json) as f:
        seg_indices = np.asarray(json.load(f)["segIndices"], dtype=np.int64)
    with open(anno_json) as f:
        seg_groups = json.load(f)["segGroups"]
    print(f"Loaded {len(coords)} vertices and {len(seg_groups)} GT instances")

    order, unique_segs, bounds = _build_segment_lookup(seg_indices)

    bbox_results: list[dict[str, Any]] = []
    caption_results: list[dict[str, Any]] = []
    comp_id = 0

    for group in seg_groups:
        segments = np.asarray(group.get("segments", []), dtype=np.int64)
        verts = _vertices_for_segments(segments, order, unique_segs, bounds)
        if len(verts) == 0:
            continue
        name = group.get("label", "")

        bbox_results.append({
            "connected_comp_id": comp_id,
            "class_name": name,
            "object_id": group.get("objectId", group.get("id")),
            "num_point3d_ids": int(len(verts)),
            "num_points_used": int(len(verts)),
            "num_filtered": 0,
            "bbox": compute_obb(coords[verts]),
        })
        caption_results.append({
            "component_id": comp_id,
            "caption": name,
            "num_images_used": 0,
            "crop_filenames": [],
        })
        comp_id += 1

    out_dir = get_output_path(dataset_name)
    with open(out_dir / "bbox_corners.json", "w") as f:
        json.dump(bbox_results, f, indent=2)
    with open(out_dir / "component_captions.json", "w") as f:
        json.dump(caption_results, f, indent=2)

    print(f"Ground-truth segmentation: wrote {comp_id} components to "
          f"{out_dir/'bbox_corners.json'} and "
          f"{out_dir/'component_captions.json'}")
    return comp_id


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def process_scannetpp(
    dataset_name: str,
    depth_tolerance: float = 0.05,
    point_sample_prob: float = 1.0,
    image_sample_prob: float = 1.0,
    sample_seed: int = 0,
    use_groundtruth_segmentation: bool = False,
) -> dict[str, Any]:
    """End-to-end ScanNet++ pipeline: load → mesh reproject → export glb.

    1. Loads ``scans/mesh_aligned_0.05.ply`` (kept in its Z-up frame).
    2. Reads camera poses from ``dslr/colmap/images.txt`` and the shared
       PINHOLE intrinsics from ``dslr/nerfstudio/transforms_undistorted.json``.
    3. Builds one frame per usable undistorted image and calls
       :func:`mesh_reprojection`, writing COLMAP files to
       ``outputs/<dataset_name>/colmap/``.
    4. Writes ``mesh.glb`` (Y-up for the viewer) and symlinks the image dir.

    Parameters
    ----------
    dataset_name : str
        Identifier for the dataset (matches the directory under ``data/``).
    depth_tolerance : float, optional
        Relative depth tolerance forwarded to :func:`mesh_reprojection`.
    point_sample_prob : float, optional
        Fraction of mesh vertices (in ``(0, 1]``) to project as 3D points.
        Forwarded to :func:`mesh_reprojection`.  Use a small value (e.g.
        ``0.05``) for a quick end-to-end test run.
    image_sample_prob : float, optional
        Fraction of usable frames (in ``(0, 1]``) to reproject.  Use a small
        value for a quick end-to-end test run.
    sample_seed : int, optional
        Seed shared by the vertex- and frame-sampling RNGs, for reproducibility.
    use_groundtruth_segmentation : bool, optional
        When *True*, additionally write ``bbox_corners.json`` and
        ``component_captions.json`` directly from the ScanNet++ instance
        annotations (``scans/segments.json`` + ``scans/segments_anno.json``) —
        one component **per annotated object instance**, shown as-is (no
        clustering, filtering, or discarding) — bypassing the ML
        segmentation/captioning steps.  Run the pipeline with ``steps_to_run =
        ["scannetpp_process", "create_tables"]`` to build a queryable scene
        straight from ground truth.

    Returns
    -------
    dict
        The result dict from :func:`mesh_reprojection`.
    """
    data_dir = get_data_path(dataset_name)
    dslr_dir = data_dir / "dslr"
    mesh_path = data_dir / "scans" / "mesh_aligned_0.05.ply"
    images_dir = dslr_dir / "resized_undistorted_images"
    images_txt = dslr_dir / "colmap" / "images.txt"
    transforms_json = dslr_dir / "nerfstudio" / "transforms_undistorted.json"

    # Step 1 – load mesh (stays in its native Z-up frame)
    mesh = _load_mesh(mesh_path)
    print(f"Loaded mesh: {len(mesh.vertices)} vertices")

    # Step 2 – poses (colmap) and shared pinhole intrinsics (undistorted)
    poses = _parse_colmap_poses(images_txt)
    print(f"Parsed {len(poses)} colmap poses")
    intrinsics, is_bad = _load_pinhole_intrinsics(transforms_json)

    # Step 3 – assemble frames (optionally sampling a fraction for testing)
    frames = _build_all_frames(
        poses, intrinsics, images_dir, is_bad,
        image_sample_prob=image_sample_prob,
        sample_seed=sample_seed,
    )

    # Step 4 – run reprojection (writes COLMAP output automatically)
    result = mesh_reprojection(
        mesh=mesh,
        image_pose_depth=frames,
        depth_tolerance=depth_tolerance,
        dataset_name=dataset_name,
        point_sample_prob=point_sample_prob,
        sample_seed=sample_seed,
    )

    # Step 5 – export mesh.glb (Y-up) and symlink the image directory
    _write_mesh_glb(mesh, dataset_name)
    _link_images(images_dir, dataset_name)

    # Step 6 – optionally derive components directly from the semantic mesh
    if use_groundtruth_segmentation:
        _generate_groundtruth_segmentation(dataset_name)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Config-dict entry point
# ═══════════════════════════════════════════════════════════════════════════

def process_scannetpp_from_config(config: dict, dataset_name: str) -> dict:
    """Run the ScanNet++ pipeline using the config dictionary.

    Reads parameters from ``config["scannetpp"]``.  The dataset name is passed
    separately (it is not read from the config).
    """
    scannetpp_cfg = config.get("scannetpp", {})
    return process_scannetpp(
        dataset_name,
        depth_tolerance=scannetpp_cfg.get("depth_tolerance", 0.05),
        point_sample_prob=scannetpp_cfg.get("point_sample_prob", 1.0),
        image_sample_prob=scannetpp_cfg.get("image_sample_prob", 1.0),
        sample_seed=scannetpp_cfg.get("sample_seed", 0),
        use_groundtruth_segmentation=scannetpp_cfg.get(
            "use_groundtruth_segmentation", False),
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from config_cli import parse_config_args

    config, dataset_name = parse_config_args(
        description="Process a ScanNet++ scene and write COLMAP files.",
    )
    process_scannetpp_from_config(config, dataset_name)
