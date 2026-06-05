"""Debug an object instance by visualizing its 3D points in a point cloud.

Reads object_3d_associations.json to get the list of 3D point IDs for the
requested instance, then builds a PLY point cloud where the instance points
are coloured red and all other scene points are white.

The instance name format is: {object_name}_seq_{seq_id}_{object_id}
e.g. battery_seq_0_5, Boxes_seq_0_3

Usage:
    python instance.py --dataset_name ProjectLabStudio_inv_method --instance_name battery_seq_0_5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config_io import get_colmap_output_path, get_output_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pattern: <object_name>_seq_<seq_id>_<object_id>
_INSTANCE_ID_RE = re.compile(r"^(.+)_seq_(\d+)_(\d+)$")


def parse_instance_name(instance_name: str) -> Tuple[str, str, str]:
    """Split 'battery_seq_0_5' → ('battery', 'seq_0', '5').

    Works for multi-word object names like 'plastic_container_seq_0_8'.

    Returns:
        (object_name, seq_name, object_id_str)
    """
    m = _INSTANCE_ID_RE.match(instance_name)
    if not m:
        raise ValueError(
            f"Cannot parse instance name: {instance_name!r}. "
            "Expected format: {{object_name}}_seq_{{seq_id}}_{{object_id}}"
        )
    object_name, seq_num, object_id = m.group(1), m.group(2), m.group(3)
    return object_name, f"seq_{seq_num}", object_id


def get_instance_point3d_ids(
    associations: dict, object_name: str, seq_name: str, object_id_str: str
) -> Optional[List[int]]:
    """Look up point3D IDs for the given instance in the associations dict.

    The associations dict has the structure:
        { object_name: { seq_name: { object_id_str: [point3D_id, ...] } } }

    Returns the list of point3D IDs, or None if the instance is not found.
    """
    obj_entry = associations.get(object_name)
    if obj_entry is None:
        return None
    seq_entry = obj_entry.get(seq_name)
    if seq_entry is None:
        return None
    return seq_entry.get(object_id_str)


# ---------------------------------------------------------------------------
# PLY writer (identical to the one in component.py)
# ---------------------------------------------------------------------------


def write_ply(path: Path, xyzs: np.ndarray, rgbs: np.ndarray) -> None:
    """Write a binary little-endian PLY file with float32 XYZ and uint8 RGB."""
    n = len(xyzs)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    data = np.empty(n, dtype=dtype)
    data["x"] = xyzs[:, 0].astype(np.float32)
    data["y"] = xyzs[:, 1].astype(np.float32)
    data["z"] = xyzs[:, 2].astype(np.float32)
    data["red"] = rgbs[:, 0]
    data["green"] = rgbs[:, 1]
    data["blue"] = rgbs[:, 2]

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def debug_instance(dataset_name: str, instance_name: str) -> None:
    # Parse instance name
    try:
        object_name, seq_name, object_id_str = parse_instance_name(instance_name)
    except ValueError as exc:
        sys.exit(str(exc))

    print(
        f"Instance: {instance_name!r}  →  "
        f"object={object_name!r}, seq={seq_name!r}, id={object_id_str!r}"
    )

    # Resolve paths
    outputs_dir = get_output_path(dataset_name)

    colmap_model_dir = get_colmap_output_path(dataset_name)
    if not colmap_model_dir.exists():
        sys.exit(f"COLMAP model dir not found: {colmap_model_dir}")

    # Load object_3d_associations.json
    associations_path = (
        outputs_dir / "object_level_masks" / "object_3d_associations.json"
    )
    if not associations_path.exists():
        sys.exit(f"object_3d_associations.json not found at {associations_path}")

    with associations_path.open("r", encoding="utf-8") as f:
        associations = json.load(f)

    point3d_ids = get_instance_point3d_ids(
        associations, object_name, seq_name, object_id_str
    )
    if point3d_ids is None:
        # Provide helpful diagnostic information
        available_objects = list(associations.keys())
        sys.exit(
            f"Instance not found in associations.\n"
            f"  object_name={object_name!r} — available objects: {available_objects}\n"
            f"  If object_name looks correct, check the seq/id values."
        )

    instance_point_set = set(point3d_ids)
    print(f"Found {len(instance_point_set)} 3D point(s) for this instance.")

    # Load COLMAP model
    print("Loading COLMAP model...")
    from segment3d.utils.colmap_io import load_colmap_model

    _, _, points3D = load_colmap_model(str(colmap_model_dir))
    print(f"Total scene points: {len(points3D)}")

    all_ids = list(points3D.keys())
    xyzs = np.array([points3D[pid].xyz for pid in all_ids], dtype=np.float32)

    # COLMAP is Z-up; convert to Y-up for standard 3D viewers:
    #   new X =  old X,  new Y =  old Z,  new Z = -old Y
    xyzs = np.stack([xyzs[:, 0], xyzs[:, 2], -xyzs[:, 1]], axis=1)

    # Colour: red for instance points, white for the rest
    rgbs = np.full((len(all_ids), 3), 255, dtype=np.uint8)
    for i, pid in enumerate(all_ids):
        if pid in instance_point_set:
            rgbs[i] = (255, 0, 0)

    # Save PLY
    out_dir = outputs_dir / "debug_instances"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / f"{instance_name}.ply"
    write_ply(ply_path, xyzs, rgbs)
    print(f"Point cloud saved to: {ply_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize an object instance's 3D points in the scene point cloud. "
            "Instance points are coloured red; all others are white."
        )
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g. ProjectLabStudio_inv_method)",
    )
    parser.add_argument(
        "--instance_name",
        type=str,
        required=True,
        help=(
            "Instance name in the format {object_name}_seq_{seq_id}_{object_id} "
            "(e.g. battery_seq_0_5)"
        ),
    )
    args = parser.parse_args()
    debug_instance(
        dataset_name=args.dataset_name,
        instance_name=args.instance_name,
    )


if __name__ == "__main__":
    main()
