"""Compute bounding box corners for sets of 3D points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from ..utils.colmap_io import load_colmap_model
from config_io import get_output_path, get_colmap_output_path


def get_bbox(
    point3D_ids: List[int], colmap_model: any, percentile: float = 95.0
) -> Dict[str, Any]:
    """
    Compute bounding box corners for a set of 3D point IDs.

    Filters out outliers by cutting off points beyond the specified percentile
    on all axes, then returns the corners of the bounding box.

    Args:
        point3D_ids: List of COLMAP 3D point IDs
        colmap_model_dir: Path to COLMAP model directory
        percentile: Percentile threshold for outlier removal (default: 95.0)

    Returns:
        Dictionary containing:
        - "corners": np.ndarray(8, 3) - Eight corners of the bounding box
        - "min": np.ndarray(3,) - Minimum coordinates [x, y, z]
        - "max": np.ndarray(3,) - Maximum coordinates [x, y, z]
        - "center": np.ndarray(3,) - Center of bounding box
        - "size": np.ndarray(3,) - Size of bounding box [width, height, depth]
        - "num_points": int - Number of points after filtering
        - "num_filtered": int - Number of points filtered out
    """
    # Load COLMAP model
    cameras, images, points3D = colmap_model

    # Collect 3D coordinates for valid point IDs
    coords_list = []
    for point_id in point3D_ids:
        if point_id in points3D:
            coords_list.append(points3D[point_id].xyz)

    if len(coords_list) == 0:
        raise ValueError("No valid 3D points found for the given IDs")

    # Convert to numpy array (N, 3)
    coords = np.array(coords_list)
    num_original = len(coords)

    # Filter outliers: keep only points within percentile range on all axes
    percentile_low = (100.0 - percentile) / 2.0
    percentile_high = 100.0 - percentile_low

    # Compute percentile bounds for each axis
    min_bounds = np.percentile(coords, percentile_low, axis=0)
    max_bounds = np.percentile(coords, percentile_high, axis=0)

    # Keep points within bounds on all axes
    mask = np.all((coords >= min_bounds) & (coords <= max_bounds), axis=1)
    filtered_coords = coords[mask]

    if len(filtered_coords) == 0:
        raise ValueError("All points were filtered out as outliers")

    # Compute bounding box
    bbox_min = filtered_coords.min(axis=0)
    bbox_max = filtered_coords.max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2.0
    bbox_size = bbox_max - bbox_min

    # Compute 8 corners of the bounding box
    # Order: (min_x, min_y, min_z), (max_x, min_y, min_z), ...
    corners = np.array(
        [
            [bbox_min[0], bbox_min[1], bbox_min[2]],  # 0: min, min, min
            [bbox_max[0], bbox_min[1], bbox_min[2]],  # 1: max, min, min
            [bbox_max[0], bbox_max[1], bbox_min[2]],  # 2: max, max, min
            [bbox_min[0], bbox_max[1], bbox_min[2]],  # 3: min, max, min
            [bbox_min[0], bbox_min[1], bbox_max[2]],  # 4: min, min, max
            [bbox_max[0], bbox_min[1], bbox_max[2]],  # 5: max, min, max
            [bbox_max[0], bbox_max[1], bbox_max[2]],  # 6: max, max, max
            [bbox_min[0], bbox_max[1], bbox_max[2]],  # 7: min, max, max
        ]
    )

    return {
        "corners": corners,
        "min": bbox_min,
        "max": bbox_max,
        "center": bbox_center,
        "size": bbox_size,
        "num_points": len(filtered_coords),
        "num_filtered": num_original - len(filtered_coords),
    }


def get_all_bbox_corners_cli(dataset_name: str, percentile: float = 95.0) -> None:
    """
    Compute bounding box corners for all connected components.

    Args:
        dataset_name: Name of the dataset
        percentile: Percentile threshold for outlier removal (default: 95.0)
    """

    # Resolve paths
    outputs_dir = get_output_path(dataset_name)
    colmap_model_dir = get_colmap_output_path(dataset_name)

    print(f"Outputs directory: {outputs_dir}")
    print(f"COLMAP model directory: {colmap_model_dir}")
    print(f"Outlier filter percentile: {percentile}%")

    # Load connected components
    components_path = outputs_dir / "connected_components.json"
    if not components_path.exists():
        raise FileNotFoundError(
            f"Connected components file not found: {components_path}\n"
            "Please run src.mask_graph first."
        )

    print(f"\nLoading connected components from: {components_path}")
    with components_path.open("r", encoding="utf-8") as f:
        connected_components = json.load(f)

    print(f"Found {len(connected_components)} connected components")
    
    # Load COLMAP model
    print("\nLoading COLMAP model...")
    colmap_model = load_colmap_model(colmap_model_dir)

    # Compute bounding boxes for each component
    bbox_results = []

    for component in connected_components:
        comp_id = component["connected_comp_id"]
        point3D_ids = component["set_of_point3DIds"]

        print(f"\nProcessing component {comp_id} ({len(point3D_ids)} points)...")

        try:
            bbox_info = get_bbox(point3D_ids, colmap_model, percentile=percentile)

            # Convert numpy arrays to lists for JSON serialization
            bbox_result = {
                "connected_comp_id": comp_id,
                "num_point3d_ids": len(point3D_ids),
                "num_points_used": int(bbox_info["num_points"]),
                "num_filtered": int(bbox_info["num_filtered"]),
                "bbox": {
                    "corners": bbox_info["corners"].tolist(),
                    "min": bbox_info["min"].tolist(),
                    "max": bbox_info["max"].tolist(),
                    "center": bbox_info["center"].tolist(),
                    "size": bbox_info["size"].tolist(),
                },
            }

            bbox_results.append(bbox_result)

            print(f"  Bounding box computed:")
            print(f"    Points used: {bbox_info['num_points']}/{len(point3D_ids)}")
            print(f"    Filtered out: {bbox_info['num_filtered']}")
            print(
                f"    Center: [{bbox_info['center'][0]:.3f}, {bbox_info['center'][1]:.3f}, {bbox_info['center'][2]:.3f}]"
            )
            print(
                f"    Size: [{bbox_info['size'][0]:.3f}, {bbox_info['size'][1]:.3f}, {bbox_info['size'][2]:.3f}]"
            )

        except Exception as e:
            print(f"  Error computing bounding box for component {comp_id}: {e}")
            continue

    # Save results
    output_path = outputs_dir / "bbox_corners.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(bbox_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Saved bounding box corners to: {output_path}")
    print(
        f"Successfully processed {len(bbox_results)}/{len(connected_components)} components"
    )

    # Save summary statistics
    if bbox_results:
        sizes = [np.array(r["bbox"]["size"]) for r in bbox_results]
        volumes = [np.prod(s) for s in sizes]

        stats = {
            "total_components": len(bbox_results),
            "percentile_filter": percentile,
            "volume_statistics": {
                "min": float(min(volumes)),
                "max": float(max(volumes)),
                "mean": float(np.mean(volumes)),
                "median": float(np.median(volumes)),
            },
            "size_statistics": {
                "min_size": np.min(sizes, axis=0).tolist(),
                "max_size": np.max(sizes, axis=0).tolist(),
                "mean_size": np.mean(sizes, axis=0).tolist(),
            },
        }

        stats_path = outputs_dir / "bbox_stats.json"
        with stats_path.open("w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved bounding box statistics to: {stats_path}")


def main() -> None:
    """Main entry point for CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute bounding box corners for all connected components"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        required=True,
        help="Name of the dataset",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=95.0,
        help="Percentile threshold for outlier removal (default: 95.0)",
    )

    args = parser.parse_args()

    if args.percentile <= 0 or args.percentile > 100:
        parser.error("Percentile must be between 0 and 100")

    get_all_bbox_corners_cli(dataset_name=args.dataset_name, percentile=args.percentile)


if __name__ == "__main__":
    main()
