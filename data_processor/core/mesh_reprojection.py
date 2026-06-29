"""
mesh_reprojection.py – Sample mesh vertices and project them onto images.

Coordinate system
-----------------
All data is expected in a **right-handed, Z-up** coordinate system.

Input format
------------
``mesh`` : open3d.geometry.TriangleMesh
    A 3D triangle mesh loaded via Open3D (e.g.
    ``open3d.io.read_triangle_mesh(path)``).  The mesh must contain
    vertices; vertex colors / normals are optional.

``image_pose_depth`` : list[dict]
    Each element is a dictionary with the following keys:

    * ``"image"`` (numpy.ndarray, shape ``(H, W, 3)``, dtype ``uint8``)
        – The RGB image.
    * ``"extrinsics"`` (numpy.ndarray, shape ``(4, 4)``, dtype ``float64``)
        – Camera-to-world (c2w) transformation matrix.
          Columns are [right, forward, up, translation] in a right-handed
          Z-up frame.  The function internally inverts this to obtain the
          world-to-camera matrix for projection.
    * ``"intrinsics"`` (numpy.ndarray, shape ``(3, 3)``, dtype ``float64``)
        – Camera intrinsic matrix::

              [[fx,  0, cx],
               [ 0, fy, cy],
               [ 0,  0,  1]]

    * ``"depth"`` (numpy.ndarray, shape ``(H, W)``, dtype ``float32``)
        – Depth map in the same frame as the camera (metric, along the
          camera Z-axis).  Used for visibility / occlusion checking.

``depth_tolerance`` : float, optional
    Relative tolerance for the depth check.  A projected point is
    considered visible when its projected depth is within
    ``depth_tolerance`` of the value stored in the depth map.  Default
    is 0.05 (5 %).

``dataset_name`` : str, optional
    Dataset identifier used by ``config_io`` to build the output path.
    Defaults to ``"default"``.

Output
------
The function writes COLMAP-format text files (``cameras.txt``,
``images.txt``, ``points3D.txt``) into
``<repo_root>/outputs/<dataset_name>/colmap/``.

Returns
-------
dict
    ``{
        "points3D": np.ndarray (N, 3),
        "observations": list[dict]   # per-image observations
    }``
    where each observation dict contains ``"image_id"``,
    ``"point2D"`` (M, 2), ``"point3D_ids"`` (M,).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm

from config_io import get_colmap_output_path, get_rendered_depth_output_path


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def render_depth_from_mesh(
    mesh: o3d.geometry.TriangleMesh,
    width: int,
    height: int,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    scene: o3d.t.geometry.RaycastingScene | None = None,
) -> np.ndarray:
    """Render a z-depth map of *mesh* as seen from a given camera pose.

    The returned depth map is pixel-wise aligned with an RGB image captured
    with the same ``intrinsics`` and resolution: pixel ``(v, u)`` of the
    output corresponds to pixel ``(v, u)`` of that image.  Depth is expressed
    in metric units along the camera optical (Z) axis – the same convention
    used by the ``"depth"`` entries of ``image_pose_depth`` – so it can be
    used as a drop-in replacement for a captured depth map.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        Input 3D mesh (right-handed Z-up).
    width, height : int
        Output resolution in pixels (must match the RGB image).
    extrinsics : numpy.ndarray, shape (4, 4)
        Camera-to-world (c2w) transformation matrix.
    intrinsics : numpy.ndarray, shape (3, 3)
        Pinhole intrinsic matrix ``[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]``.
    scene : open3d.t.geometry.RaycastingScene, optional
        A pre-built raycasting scene that already contains *mesh*.  Supply
        this to avoid rebuilding the BVH when rendering many frames of the
        same mesh.  If *None*, a scene is constructed from *mesh* on the fly.

    Returns
    -------
    numpy.ndarray, shape (height, width), dtype float32
        Z-depth map.  Pixels where no surface is hit are set to ``0``.
    """
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)

    # Build the raycasting scene if the caller did not provide one.
    if scene is None:
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    # Open3D expects the world-to-camera (extrinsic) matrix.
    w2c = np.linalg.inv(extrinsics)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        intrinsic_matrix=o3d.core.Tensor(intrinsics),
        extrinsic_matrix=o3d.core.Tensor(w2c),
        width_px=int(width),
        height_px=int(height),
    )

    ans = scene.cast_rays(rays)
    # ``create_rays_pinhole`` returns ray directions with a *unit z-component*
    # along the camera optical axis (they are not unit-length).  The hit
    # parameter ``t_hit`` is therefore already the depth along the optical
    # (Z) axis – exactly the convention used by the captured depth maps – so
    # no Euclidean-to-z conversion is needed.
    depth = ans["t_hit"].numpy().astype(np.float32)  # (H, W)
    # Rays that miss the mesh yield inf -> mark as invalid (0).
    depth[~np.isfinite(depth)] = 0.0
    return depth


def mesh_reprojection(
    mesh: o3d.geometry.TriangleMesh,
    image_pose_depth: list[dict[str, Any]],
    depth_tolerance: float = 0.05,
    dataset_name: str = "default",
    use_rendered_depth: bool = True,
    save_rendered_depth: bool = True,
    point_sample_prob: float = 1.0,
    sample_seed: int = 0,
) -> dict[str, Any]:
    """Sample mesh vertices and project them into every image.

    See the module-level docstring for a full description of the expected
    input format, coordinate conventions, and output structure.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh
        Input 3D mesh (right-handed Z-up).
    image_pose_depth : list[dict]
        Per-frame data – see module docstring for the dict schema.
    depth_tolerance : float, optional
        Relative depth tolerance for visibility checks (default 0.05).
    dataset_name : str, optional
        Name used for the output sub-directory (default ``"default"``).
    use_rendered_depth : bool, optional
        When *True* (default), the per-frame depth maps are rendered from
        *mesh* via :func:`render_depth_from_mesh` instead of using the
        ``"depth"`` entries of *image_pose_depth*; the passed-in depth is
        then completely ignored.
    save_rendered_depth : bool, optional
        When *True* (default) and ``use_rendered_depth`` is also *True*, each
        rendered depth map is written as a 16-bit PNG (millimetres) into
        ``outputs/<dataset_name>/rendered_images/``.
    point_sample_prob : float, optional
        Fraction of mesh vertices (in ``(0, 1]``) to keep as projected 3D
        points.  The **full** mesh is still used for depth rendering and
        occlusion checks; only the projected point cloud (and the resulting
        COLMAP tracks) becomes sparser.  Default ``1.0`` projects every vertex.
        Useful for quick end-to-end test runs.
    sample_seed : int, optional
        Seed for the vertex-sampling RNG, for reproducible subsets.

    Returns
    -------
    dict
        ``{"points3D": ndarray (N,3), "observations": [...]}``
    """

    vertices = np.asarray(mesh.vertices)  # (N, 3)
    num_points = len(vertices)
    num_images = len(image_pose_depth)

    if num_points == 0:
        raise ValueError("Mesh has no vertices – nothing to project.")

    # Optionally project only a random subset of vertices. ``candidate_idx``
    # holds the original vertex indices that are eligible to be observed; the
    # full mesh is still used for depth rendering, and ``points3D`` keeps the
    # full vertex array so these indices stay valid as point IDs.
    if not 0.0 < point_sample_prob <= 1.0:
        raise ValueError("point_sample_prob must be in (0, 1]")
    if point_sample_prob < 1.0:
        rng = np.random.default_rng(sample_seed)
        candidate_idx = np.where(rng.random(num_points) < point_sample_prob)[0]
        print(f"Sampling {len(candidate_idx)}/{num_points} mesh vertices "
              f"(point_sample_prob={point_sample_prob})")
    else:
        candidate_idx = np.arange(num_points)
    sub_vertices = vertices[candidate_idx]

    # When rendering depth from the mesh, build the raycasting scene once and
    # reuse it for every frame (avoids rebuilding the BVH per image).
    render_scene = None
    rendered_depth_dir = None
    if use_rendered_depth:
        render_scene = o3d.t.geometry.RaycastingScene()
        render_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
        if save_rendered_depth:
            rendered_depth_dir = get_rendered_depth_output_path(dataset_name)

    print(f"Projecting {num_points} vertices onto {num_images} images "
          f"(depth_tolerance={depth_tolerance})")

    # -- Collect vertex colors if available (used in points3D.txt) ----------
    if mesh.has_vertex_colors():
        vertex_colors = (np.asarray(mesh.vertex_colors) * 255).astype(np.uint8)
    else:
        vertex_colors = np.full((num_points, 3), 128, dtype=np.uint8)

    # -- Per-image projection -----------------------------------------------
    all_observations: list[dict[str, Any]] = []
    total_visible = 0

    for image_id, frame in enumerate(
        tqdm(image_pose_depth, desc="Reprojecting", unit="img"), start=1,
    ):
        image = frame["image"]
        extrinsics_c2w = np.asarray(frame["extrinsics"], dtype=np.float64)
        intrinsics = np.asarray(frame["intrinsics"], dtype=np.float64)

        if use_rendered_depth:
            # Ignore the captured depth entirely – render a pixel-aligned
            # depth map from the mesh for this exact pose / resolution.
            H, W = image.shape[:2]
            depth_map = render_depth_from_mesh(
                mesh=mesh,
                width=W,
                height=H,
                extrinsics=extrinsics_c2w,
                intrinsics=intrinsics,
                scene=render_scene,
            )
            if rendered_depth_dir is not None:
                # Save as a 16-bit PNG in millimetres (matches the captured
                # depth convention, so it round-trips via the depth loaders).
                stem = Path(frame.get("name", f"frame_{image_id:06d}")).stem
                depth_mm = np.clip(depth_map * 1000.0, 0, 65535).astype(np.uint16)
                cv2.imwrite(str(rendered_depth_dir / f"{stem}.png"), depth_mm)
        else:
            depth_map = np.asarray(frame["depth"], dtype=np.float32)
            H, W = depth_map.shape[:2]

        # World-to-camera: invert the camera-to-world matrix.
        w2c = np.linalg.inv(extrinsics_c2w)
        R = w2c[:3, :3]
        t = w2c[:3, 3]

        # --- Project vertices into the camera frame -----------------------
        pts_cam = (R @ sub_vertices.T).T + t  # (M, 3)
        depths = pts_cam[:, 2]

        # Keep only points in front of the camera.
        in_front = depths > 0
        pts_cam_valid = pts_cam[in_front]
        depths_valid = depths[in_front]
        # Map back to original vertex IDs (indices into the full point array).
        indices_valid = candidate_idx[in_front]

        if len(pts_cam_valid) == 0:
            continue

        # Project to 2D pixel coordinates.
        pts_2d_h = (intrinsics @ pts_cam_valid.T).T  # (M, 3)
        pts_2d = pts_2d_h[:, :2] / pts_2d_h[:, 2:3]  # (M, 2)

        u = pts_2d[:, 0]
        v = pts_2d[:, 1]

        # Bounds check.
        in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)

        u_valid = u[in_bounds].astype(int)
        v_valid = v[in_bounds].astype(int)
        depths_proj = depths_valid[in_bounds]
        indices_proj = indices_valid[in_bounds]
        pts_2d_proj = pts_2d[in_bounds]

        if len(u_valid) == 0:
            continue

        # Depth / occlusion check.
        depth_sampled = depth_map[v_valid, u_valid]
        # Ignore pixels with zero / invalid depth.
        valid_depth = depth_sampled > 0
        depth_diff = np.abs(depths_proj - depth_sampled)
        within_tol = depth_diff < depth_tolerance * depth_sampled
        visible = valid_depth & within_tol

        if not np.any(visible):
            continue

        n_visible = int(np.sum(visible))
        total_visible += n_visible

        obs = {
            "image_id": image_id,
            "point2D": pts_2d_proj[visible],       # (K, 2)
            "point3D_ids": indices_proj[visible],   # (K,)  – 0-based
        }
        all_observations.append(obs)

    # -- Summary stats -------------------------------------------------------
    images_with_obs = len(all_observations)
    unique_pts = len({int(pid) for obs in all_observations for pid in obs["point3D_ids"]})
    print(f"Reprojection complete:")
    print(f"  Images with observations: {images_with_obs}/{num_images}")
    print(f"  Total 2D-3D correspondences: {total_visible}")
    print(f"  Unique 3D points observed: {unique_pts}/{num_points}")

    result = {
        "points3D": vertices,
        "vertex_colors": vertex_colors,
        "observations": all_observations,
        "image_pose_depth": image_pose_depth,
    }

    # -- Write COLMAP files -------------------------------------------------
    colmap_dir = get_colmap_output_path(dataset_name)
    write_colmap(result, colmap_dir)
    print(f"COLMAP files written to {colmap_dir}")

    return result


# ═══════════════════════════════════════════════════════════════════════════
# COLMAP writer
# ═══════════════════════════════════════════════════════════════════════════

def write_colmap(
    reprojection_result: dict[str, Any],
    output_path: Path | str | None = None,
    dataset_name: str = "default",
) -> Path:
    """Write 2D–3D correspondences as COLMAP text-format files.

    Produces three files inside *output_path*:

    * ``cameras.txt``  – one PINHOLE camera per image
    * ``images.txt``   – image poses and their 2D observations
    * ``points3D.txt`` – 3D points with mean reprojection color and a
      track of observing images

    Parameters
    ----------
    reprojection_result : dict
        The dictionary returned by :func:`mesh_reprojection`.
    output_path : Path or str, optional
        Directory to write into.  If *None*, it is resolved via
        :func:`config_io.get_colmap_output_path` using *dataset_name*.
    dataset_name : str, optional
        Fallback dataset name when *output_path* is not given.

    Returns
    -------
    pathlib.Path
        The directory the files were written to.
    """

    if output_path is None:
        output_path = get_colmap_output_path(dataset_name)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    points3D = reprojection_result["points3D"]
    vertex_colors = reprojection_result["vertex_colors"]
    observations = reprojection_result["observations"]
    frames = reprojection_result["image_pose_depth"]

    # -- Build a track for every 3D point -----------------------------------
    # track[point_id] = [(image_id, idx_within_image), ...]
    num_points = len(points3D)
    tracks: dict[int, list[tuple[int, int]]] = {i: [] for i in range(num_points)}

    for obs in observations:
        image_id = obs["image_id"]
        for local_idx, pt3d_id in enumerate(obs["point3D_ids"]):
            tracks[int(pt3d_id)].append((image_id, local_idx))

    # -- cameras.txt --------------------------------------------------------
    _write_cameras_txt(output_path / "cameras.txt", frames)

    # -- images.txt ---------------------------------------------------------
    _write_images_txt(output_path / "images.txt", observations, frames)

    # -- points3D.txt -------------------------------------------------------
    _write_points3D_txt(
        output_path / "points3D.txt", points3D, vertex_colors, tracks,
    )

    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a unit quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.
    """
    m = R
    trace = np.trace(m)

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _write_cameras_txt(path: Path, frames: list[dict[str, Any]]) -> None:
    """Write ``cameras.txt`` – one PINHOLE camera entry per image.

    COLMAP format (text)::

        # Camera list with one line of data per camera:
        #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        # For PINHOLE: PARAMS = fx, fy, cx, cy
    """
    with open(path, "w") as f:
        f.write(
            "# Camera list with one line of data per camera:\n"
            "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            f"# Number of cameras: {len(frames)}\n"
        )
        for cam_id, frame in enumerate(frames, start=1):
            K = np.asarray(frame["intrinsics"], dtype=np.float64)
            image = frame["image"]
            H, W = image.shape[:2]
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            f.write(f"{cam_id} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")


def _write_images_txt(
    path: Path,
    observations: list[dict[str, Any]],
    frames: list[dict[str, Any]],
) -> None:
    """Write ``images.txt``.

    COLMAP text format::

        # Image list with two lines of data per image:
        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #   POINTS2D[] as (X, Y, POINT3D_ID) – POINT3D_ID = -1 if unmatched
    """
    # Build a quick lookup: image_id -> observation
    obs_by_image: dict[int, dict] = {o["image_id"]: o for o in observations}

    with open(path, "w") as f:
        f.write(
            "# Image list with two lines of data per image:\n"
            "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            f"# Number of images: {len(frames)}\n"
        )

        for image_id, frame in enumerate(frames, start=1):
            c2w = np.asarray(frame["extrinsics"], dtype=np.float64)
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            qw, qx, qy, qz = _rotation_matrix_to_quaternion(R)
            tx, ty, tz = t

            camera_id = image_id  # one camera per image
            name = frame.get("name", f"frame_{image_id:06d}.jpg")

            f.write(
                f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} "
                f"{camera_id} {name}\n"
            )

            # 2D observations line.
            obs = obs_by_image.get(image_id)
            if obs is not None:
                pts2d = obs["point2D"]
                ids = obs["point3D_ids"]
                parts = []
                for (x, y), pid in zip(pts2d, ids):
                    # COLMAP uses 1-based POINT3D_ID; 0-based internally
                    # so we add 1 to match.
                    parts.append(f"{x:.4f} {y:.4f} {int(pid) + 1}")
                f.write(" ".join(parts) + "\n")
            else:
                f.write("\n")


def _write_points3D_txt(
    path: Path,
    points3D: np.ndarray,
    vertex_colors: np.ndarray,
    tracks: dict[int, list[tuple[int, int]]],
) -> None:
    """Write ``points3D.txt``.

    COLMAP text format::

        # 3D point list with one line of data per point:
        #   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as
        #       (IMAGE_ID, POINT2D_IDX)
    """
    with open(path, "w") as f:
        f.write(
            "# 3D point list with one line of data per point:\n"
            "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, "
            "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            f"# Number of points: {len(points3D)}\n"
        )

        for pid in range(len(points3D)):
            track = tracks[pid]
            if not track:
                # Only write points that are observed in at least one image.
                continue

            x, y, z = points3D[pid]
            r, g, b = vertex_colors[pid]
            error = 0.0  # placeholder – no BA has been run

            track_str = " ".join(
                f"{img_id} {pt2d_idx}" for img_id, pt2d_idx in track
            )

            # COLMAP POINT3D_IDs are 1-based.
            f.write(
                f"{pid + 1} {x} {y} {z} {r} {g} {b} {error} {track_str}\n"
            )
