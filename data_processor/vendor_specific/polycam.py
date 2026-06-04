"""
polycam.py – Load a Polycam raw-data export and run mesh reprojection.

Expected input
--------------
Place ``input.zip`` (the Polycam raw-data export) at::

    data/<dataset_name>/input.zip

The ZIP is extracted into ``data/<dataset_name>/polycam_data/``.

Polycam raw-data directory layout (after extraction)::

    polycam_data/
    ├── keyframes/
    │   ├── corrected_cameras/   # globally-optimised camera JSON files
    │   ├── images/              # captured images
    │   └── depth/               # 16-bit PNG depth maps (millimetres)
    ├── mesh_info.json           # mesh metadata incl. alignmentTransform
    └── raw.glb                  # reconstructed 3D mesh

Coordinate convention
---------------------
Polycam / ARKit uses **Y-up**.  The core algorithms in this repository
expect **right-handed Z-up**.  This module converts all data on load.

Usage::

    from data_processor.vendor_specific.polycam import process_polycam

    result = process_polycam("my_scan")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d

from config_io import (
    get_colmap_output_path,
    get_data_path,
    get_images_output_path,
    get_output_path,
)
from data_processor.core.mesh_reprojection import mesh_reprojection
from data_processor.utils.extract_zip import extract_zip

# ═══════════════════════════════════════════════════════════════════════════
# Coordinate-system helpers
# ═══════════════════════════════════════════════════════════════════════════

# Rotation matrix that maps Y-up (ARKit) → Z-up:
#   X →  X
#   Y →  Z
#   Z → -Y
_YUP_TO_ZUP = np.array(
    [[1,  0,  0,  0],
     [0,  0, -1,  0],
     [0,  1,  0,  0],
     [0,  0,  0,  1]],
    dtype=np.float64,
)

# Rotation matrix that maps OpenGL/ARKit camera to OpenCV camera:
#   X →  X
#   Y → -Y
#   Z → -Z
_GL_TO_CV_CAM = np.array(
    [[1,  0,  0,  0],
     [0, -1,  0,  0],
     [0,  0, -1,  0],
     [0,  0,  0,  1]],
    dtype=np.float64,
)


def _convert_pose_yup_to_zup(c2w_yup: np.ndarray) -> np.ndarray:
    """Convert a 4×4 camera-to-world matrix from Y-up to Z-up."""
    return _YUP_TO_ZUP @ c2w_yup


def _convert_mesh_yup_to_zup(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Rotate an Open3D mesh from Y-up to Z-up **in-place** and return it."""
    R = _YUP_TO_ZUP[:3, :3]
    mesh.rotate(R, center=(0, 0, 0))
    return mesh


# ═══════════════════════════════════════════════════════════════════════════
# File-loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_camera_json(
    json_path: Path,
    alignment: np.ndarray | None = None,
) -> dict[str, Any]:
    """Parse a single Polycam camera JSON file.

    Parameters
    ----------
    json_path : Path
        Path to the camera JSON.
    alignment : (4, 4) ndarray, optional
        The ``alignmentTransform`` from ``mesh_info.json``.  When provided
        the camera pose is transformed into the aligned mesh frame
        *before* the Y-up → Z-up conversion.

    Returns a dict with ``"intrinsics"`` (3×3) and ``"extrinsics"`` (4×4,
    in the aligned Z-up frame).
    """
    with open(json_path) as f:
        cam = json.load(f)

    intrinsics = np.array(
        [[cam["fx"], 0,         cam["cx"]],
         [0,         cam["fy"], cam["cy"]],
         [0,         0,         1        ]],
        dtype=np.float64,
    )

    # Extrinsics stored as t_00 … t_23 (row-major, last row [0,0,0,1] omitted).
    c2w_yup = np.array(
        [[cam["t_00"], cam["t_01"], cam["t_02"], cam["t_03"]],
         [cam["t_10"], cam["t_11"], cam["t_12"], cam["t_13"]],
         [cam["t_20"], cam["t_21"], cam["t_22"], cam["t_23"]],
         [0,           0,           0,           1           ]],
        dtype=np.float64,
    )

    # Bring the camera into the aligned mesh frame (still Y-up).
    if alignment is not None:
        c2w_yup = alignment @ c2w_yup

    c2w_zup = _convert_pose_yup_to_zup(c2w_yup)

    # Convert camera coordinate system from OpenGL/ARKit (Y-up, Z-back)
    # to OpenCV/COLMAP (Y-down, Z-forward).
    c2w_zup = c2w_zup @ _GL_TO_CV_CAM

    return {"intrinsics": intrinsics, "extrinsics": c2w_zup}


def _load_image(image_path: Path) -> np.ndarray:
    """Load an RGB image as a uint8 numpy array (H, W, 3)."""
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _load_depth(depth_path: Path) -> np.ndarray:
    """Load a Polycam 16-bit PNG depth map and convert to metres (float32)."""
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(f"Could not read depth map: {depth_path}")
    return depth_mm.astype(np.float32) / 1000.0


def _load_mesh(polycam_dir: Path) -> o3d.geometry.TriangleMesh:
    """Load ``raw.glb`` and convert from Y-up to Z-up.

    The mesh is kept in its axis-aligned coordinate system.  Camera poses
    are aligned to this frame separately (see :func:`_load_camera_json`).
    """
    mesh_path = polycam_dir / "raw.glb"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    _convert_mesh_yup_to_zup(mesh)
    return mesh


def _load_alignment_transform(polycam_dir: Path) -> np.ndarray | None:
    """Read the ``alignmentTransform`` from ``mesh_info.json``.

    The matrix is stored as a 16-element list in **column-major** order.
    Returns a 4×4 numpy array, or *None* if the file is absent.
    """
    info_path = polycam_dir / "mesh_info.json"
    if not info_path.is_file():
        return None

    with open(info_path) as f:
        info = json.load(f)

    flat = info.get("alignmentTransform")
    if flat is None:
        return None

    # Column-major → reshape transposed.
    return np.array(flat, dtype=np.float64).reshape(4, 4).T


# ═══════════════════════════════════════════════════════════════════════════
# Frame assembly
# ═══════════════════════════════════════════════════════════════════════════

def _sorted_stems(directory: Path, suffix: str) -> list[str]:
    """Return sorted file stems for all files with *suffix* in *directory*."""
    return sorted(p.stem for p in directory.glob(f"*{suffix}"))


def _match_frames(polycam_dir: Path) -> list[str]:
    """Return the ordered list of frame stems that exist across cameras,
    images, and depth maps."""
    keyframes = polycam_dir / "keyframes"
    cam_dir = keyframes / "corrected_cameras"
    img_dir = keyframes / "images"
    depth_dir = keyframes / "depth"

    cam_stems = set(_sorted_stems(cam_dir, ".json"))
    img_stems = set(_sorted_stems(img_dir, ".jpg")) | set(_sorted_stems(img_dir, ".png"))
    depth_stems = set(_sorted_stems(depth_dir, ".png"))

    common = sorted(cam_stems & img_stems & depth_stems)
    if not common:
        raise FileNotFoundError(
            "No matching frames found across keyframes/corrected_cameras/, "
            "keyframes/images/ and keyframes/depth/ directories."
        )
    return common


def _find_image_file(img_dir: Path, stem: str) -> Path:
    """Return the image path for a given stem, checking common extensions."""
    for ext in (".jpg", ".jpeg", ".png"):
        p = img_dir / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"No image found for stem '{stem}' in {img_dir}")


def _build_frame(
    polycam_dir: Path,
    stem: str,
    alignment: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a single frame dict for :func:`mesh_reprojection`."""
    keyframes = polycam_dir / "keyframes"
    cam_data = _load_camera_json(
        keyframes / "corrected_cameras" / f"{stem}.json",
        alignment=alignment,
    )
    image = _load_image(_find_image_file(keyframes / "images", stem))
    depth = _load_depth(keyframes / "depth" / f"{stem}.png")

    return {
        "image": image,
        "extrinsics": cam_data["extrinsics"],
        "intrinsics": cam_data["intrinsics"],
        "depth": depth,
        "name": f"{stem}.jpg",
    }


def _build_all_frames(
    polycam_dir: Path,
    alignment: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Build the ``image_pose_depth`` list for every matched frame."""
    stems = _match_frames(polycam_dir)
    print(f"Found {len(stems)} matched frames")
    return [_build_frame(polycam_dir, stem, alignment) for stem in stems]


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def process_polycam(
    dataset_name: str,
    depth_tolerance: float = 0.05,
) -> dict[str, Any]:
    """End-to-end Polycam pipeline: extract → load → mesh reproject.

    1. Extracts ``data/<dataset_name>/input.zip`` into
       ``data/<dataset_name>/polycam_data/``.
    2. Loads the mesh, images, depth maps, and camera parameters.
    3. Converts everything from Polycam Y-up to Z-up.
    4. Calls :func:`mesh_reprojection` and writes COLMAP files to
       ``outputs/<dataset_name>/colmap/``.

    Parameters
    ----------
    dataset_name : str
        Identifier for the dataset (matches the directory name under
        ``data/``).
    depth_tolerance : float, optional
        Relative depth tolerance forwarded to
        :func:`mesh_reprojection` (default 0.05).

    Returns
    -------
    dict
        The result dict from :func:`mesh_reprojection`.
    """
    data_dir = get_data_path(dataset_name)
    zip_path = data_dir / "input.zip"
    polycam_dir = data_dir / "polycam_data"

    # Step 1 – extract (skip if already extracted)
    if polycam_dir.is_dir():
        print(f"Using existing polycam_data at {polycam_dir}")
    else:
        extract_zip(zip_path, polycam_dir)

    # Step 2 – load mesh (stays in its aligned frame)
    mesh = _load_mesh(polycam_dir)
    print(f"Loaded mesh: {len(mesh.vertices)} vertices")

    # Step 3 – load alignment and build frames in the aligned frame
    alignment = _load_alignment_transform(polycam_dir)
    frames = _build_all_frames(polycam_dir, alignment)

    # Step 4 – run reprojection (writes COLMAP output automatically)
    result = mesh_reprojection(
        mesh=mesh,
        image_pose_depth=frames,
        depth_tolerance=depth_tolerance,
        dataset_name=dataset_name,
    )

    import shutil

    # Step 5 – copy the raw mesh into the output directory as mesh.glb
    src_mesh = polycam_dir / "raw.glb"
    if src_mesh.is_file():
        mesh_out = get_output_path(dataset_name) / "mesh.glb"
        shutil.copy2(src_mesh, mesh_out)
        print(f"Copied mesh to {mesh_out}")

    # Step 6 – create symlink to images in the output directory
    src_images = polycam_dir / "keyframes" / "images"
    
    images_link = get_images_output_path(dataset_name)
    if images_link.exists() or images_link.is_symlink():
        if images_link.is_symlink():
            images_link.unlink()
        elif images_link.is_dir():
            shutil.rmtree(images_link)
        else:
            images_link.unlink()
            
    if src_images.exists():
        images_link.symlink_to(src_images.resolve(), target_is_directory=True)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Config-dict entry point
# ═══════════════════════════════════════════════════════════════════════════

def process_polycam_from_config(config: dict) -> dict:
    """Run the Polycam pipeline using the master config dictionary.

    Reads ``config["dataset_name"]`` and ``config["polycam"]``.
    """
    dataset_name = config["dataset_name"]
    polycam_cfg = config.get("polycam", {})
    depth_tolerance = polycam_cfg.get("depth_tolerance", 0.05)
    return process_polycam(dataset_name, depth_tolerance=depth_tolerance)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Process a Polycam raw-data export and write COLMAP files.",
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        default=None,
        help="Name of the dataset directory under data/ (e.g. ProjectStudio)",
    )
    parser.add_argument(
        "--depth-tolerance",
        type=float,
        default=None,
        help="Relative depth tolerance for visibility checks (default: 0.05)",
    )
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to master_config.json (overrides positional args)",
    )

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            config = json.load(f)
        process_polycam_from_config(config)
    else:
        if args.dataset_name is None:
            parser.error("dataset_name is required when --config is not provided")
        depth_tolerance = args.depth_tolerance if args.depth_tolerance is not None else 0.05
        process_polycam(args.dataset_name, depth_tolerance=depth_tolerance)

