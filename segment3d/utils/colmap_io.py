"""
COLMAP I/O utilities for loading and indexing 3D reconstruction data.

This module provides functions to load COLMAP models and create convenient
indexed representations for images and 3D points.
"""

import numpy as np
from typing import Dict, Tuple, Any, List
from .read_write_model import read_model


def load_colmap_model(model_dir: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load a COLMAP model from the specified directory.

    Args:
        model_dir: Path to the directory containing COLMAP reconstruction files
                  (cameras.txt/bin, images.txt/bin, points3D.txt/bin)

    Returns:
        Tuple of (cameras, images, points3D) dictionaries as returned by read_model()
        - cameras: dict of camera_id -> Camera namedtuples
        - images: dict of image_id -> Image namedtuples
        - points3D: dict of point3d_id -> Point3D namedtuples
    """
    cameras, images, points3D = read_model(model_dir)  # type: ignore
    return cameras, images, points3D


def index_image_metadata(images: Dict) -> Dict[int, Dict[str, Any]]:
    """
    Create an indexed representation of image metadata for efficient access.

    Args:
        images: Dictionary of image_id -> Image namedtuples from COLMAP

    Returns:
        Dictionary mapping image_id to metadata dict with keys:
        - "name": str - image filename
        - "xys": np.ndarray(N, 2) - 2D keypoint coordinates
        - "point3D_ids": np.ndarray(N,) - corresponding 3D point IDs
    """
    image_metadata = {}

    for image_id, image in images.items():
        image_metadata[image_id] = {
            "name": image.name,
            "xys": np.array(image.xys),
            "point3D_ids": np.array(image.point3D_ids),
        }

    return image_metadata


def index_point3d(points3D: Dict) -> Dict[int, Dict[str, Any]]:
    """
    Create an indexed representation of 3D point data for efficient access.

    Args:
        points3D: Dictionary of point3d_id -> Point3D namedtuples from COLMAP

    Returns:
        Dictionary mapping point3d_id to point data dict with keys:
        - "xyz": np.ndarray(3,) - 3D coordinates
        - "rgb": np.ndarray(3,) - RGB color values
        - "error": float - reprojection error
    """
    point3d_index = {}

    for point3d_id, point3d in points3D.items():
        point3d_index[point3d_id] = {
            "xyz": np.array(point3d.xyz),
            "rgb": np.array(point3d.rgb),
        }

    return point3d_index


def reverse_index_points3D(points3D: Dict) -> Dict[int, List[int]]:
    """
    Create a reverse index mapping 3D point IDs to the images they appear in.

    Args:
        points3D: Dictionary of point3d_id -> Point3D namedtuples from COLMAP

    Returns:
        Dictionary mapping point3d_id to list of image_ids where the point is visible
    """
    reverse_index = {}

    for point3d_id, point3d in points3D.items():
        # point3d.image_ids is an array of image IDs where this 3D point is observed
        image_ids = (
            point3d.image_ids.tolist()
            if hasattr(point3d.image_ids, "tolist")
            else list(point3d.image_ids)
        )
        reverse_index[point3d_id] = image_ids

    return reverse_index
