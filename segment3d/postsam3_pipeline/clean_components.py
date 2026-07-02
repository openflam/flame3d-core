"""
Clean connected components using DBSCAN clustering on 3D point positions.

For each connected component in connected_components.json:
  1. Fetch the 3D coordinates of all point3D IDs from the COLMAP model.
  2. Run DBSCAN to cluster those points in 3D space.
  3. Remove noise points (label == -1) from each component.
  4. If DBSCAN finds more than one cluster, split the component into separate
     components — one per cluster.

The result is written back to connected_components.json (overwriting it) and
a backup of the original is saved as connected_components_pre_clean.json.
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

from ..utils.colmap_io import load_colmap_model
from config_io import get_output_path, get_colmap_output_path


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _get_coords(point3d_ids: List[int], points3D: Dict) -> Tuple[np.ndarray, List[int]]:
    """Return (N, 3) coordinates array and the corresponding valid point IDs."""
    coords, valid_ids = [], []
    for pid in point3d_ids:
        if pid in points3D:
            coords.append(points3D[pid].xyz)
            valid_ids.append(pid)
    if not coords:
        return np.empty((0, 3), dtype=float), []
    return np.array(coords, dtype=float), valid_ids


# ---------------------------------------------------------------------------
# Point-count statistics
# ---------------------------------------------------------------------------

_PERCENTILE_LEVELS = [1, 5, 10, 25, 50, 75, 90, 95, 99]


def point_count_stats(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarise the distribution of 3D-point counts across components.

    Counts the length of each component's ``set_of_point3DIds`` and returns
    total, mean/std, min/max, and a set of percentiles over those per-component
    counts. Used for both the pre-clean and post-clean stages so the two are
    directly comparable.
    """
    counts = np.array(
        [len(c["set_of_point3DIds"]) for c in components], dtype=np.int64
    )
    n = int(counts.size)
    if n == 0:
        return {
            "num_components": 0,
            "total_points": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "percentiles": {},
        }
    percentiles = np.percentile(counts, _PERCENTILE_LEVELS)
    return {
        "num_components": n,
        "total_points": int(counts.sum()),
        "mean": float(counts.mean()),
        "std": float(counts.std()),
        "min": int(counts.min()),
        "max": int(counts.max()),
        "percentiles": {
            str(level): float(value)
            for level, value in zip(_PERCENTILE_LEVELS, percentiles)
        },
    }


def clean_component(
    component: Dict[str, Any],
    points3D: Dict,
    eps: float,
    min_samples: int,
    split_components: bool = False,
) -> List[Dict[str, Any]]:
    """
    Clean a single connected component with DBSCAN.

    Returns a list of component dicts:
    - If all points are noise → returns an empty list (component removed).
    - If one cluster       → returns a single component (noise stripped).
    - If multiple clusters → returns one component per cluster (split),
                             only if *split_components* is True; otherwise
                             all inlier points are kept in a single component.
    """
    comp_id = component["connected_comp_id"]
    instance_ids = component["instance_ids"]
    point3d_ids: List[int] = component["set_of_point3DIds"]
    edges = component.get("edges", [])

    coords, valid_ids = _get_coords(point3d_ids, points3D)

    if len(coords) == 0:
        print(f"  Component {comp_id}: no valid 3D points — removed.")
        return []

    if len(coords) == 1:
        # Can't run DBSCAN on a single point; keep as-is.
        return [{**component, "edges": edges}]

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    unique_labels = set(labels)
    cluster_labels = sorted(lbl for lbl in unique_labels if lbl != -1)

    noise_count = int((labels == -1).sum())

    if not cluster_labels:
        # Everything is noise
        print(f"  Component {comp_id}: all {len(coords)} points are noise — removed.")
        return []

    if len(cluster_labels) == 1:
        # One cluster: strip noise, keep instance_ids unchanged
        kept_ids = [vid for vid, lbl in zip(valid_ids, labels) if lbl != -1]
        if noise_count:
            print(
                f"  Component {comp_id}: removed {noise_count} noise points "
                f"({len(kept_ids)} kept)."
            )
        else:
            print(f"  Component {comp_id}: 1 cluster, no noise.")
        return [
            {
                "connected_comp_id": comp_id,
                "instance_ids": instance_ids,
                "set_of_point3DIds": kept_ids,
                "edges": edges,
            }
        ]

    # Multiple clusters
    if not split_components:
        # Keep all inlier points in the original component without splitting
        kept_ids = [vid for vid, lbl in zip(valid_ids, labels) if lbl != -1]
        if noise_count:
            print(
                f"  Component {comp_id}: {len(cluster_labels)} clusters found — "
                f"noise only mode, removed {noise_count} noise points "
                f"({len(kept_ids)} kept, not split)."
            )
        else:
            print(
                f"  Component {comp_id}: {len(cluster_labels)} clusters found — "
                f"noise only mode, no noise removed."
            )
        return [
            {
                "connected_comp_id": comp_id,
                "instance_ids": instance_ids,
                "set_of_point3DIds": kept_ids,
                "edges": edges,
            }
        ]

    # Multiple clusters → split
    print(
        f"  Component {comp_id}: split into {len(cluster_labels)} clusters "
        f"(noise removed: {noise_count} pts)."
    )
    sub_components: List[Dict[str, Any]] = []
    for sub_idx, lbl in enumerate(cluster_labels):
        kept_ids = [vid for vid, l in zip(valid_ids, labels) if l == lbl]
        sub_components.append(
            {
                "connected_comp_id": comp_id,  # re-numbered after all components processed
                "instance_ids": instance_ids,  # all original instance IDs kept on each split
                "_split_from": comp_id,
                "_split_sub": sub_idx,
                "set_of_point3DIds": kept_ids,
                "edges": edges,
            }
        )
    return sub_components


def clean_connected_components(
    dataset_name: str,
    eps: float = 0.1,
    min_samples: int = 5,
    min_points: int = 20,
    split_components: bool = False,
) -> None:
    """
    Load connected_components.json, apply DBSCAN cleaning to every component,
    and save the cleaned result back to the same file.

    Args:
        dataset_name:     Dataset name (folder under data/ and outputs/).
        eps:              DBSCAN neighbourhood radius (in COLMAP world units, metres
                          if the model is metric).
        min_samples:      Minimum neighbours for a core point in DBSCAN.
        min_points:       Drop components with fewer than this many points after
                          cleaning (default: 20).
        split_components: When True, components with multiple DBSCAN clusters are
                          split into separate components.  When False (default),
                          only noise points (label == -1) are removed and
                          multi-cluster components are kept as a single component.
    """
    outputs_dir = get_output_path(dataset_name)
    colmap_model_dir = get_colmap_output_path(dataset_name)

    components_path = outputs_dir / "connected_components.json"
    if not components_path.exists():
        raise FileNotFoundError(
            f"connected_components.json not found: {components_path}\n"
            "Run the mask_graph pipeline first."
        )

    # ---- backup ----
    backup_path = outputs_dir / "connected_components_pre_clean.json"
    if not backup_path.exists():
        import shutil

        shutil.copy2(components_path, backup_path)
        print(f"Backup saved → {backup_path}")
    else:
        print(f"Backup already exists at {backup_path} (not overwriting).")

    print(f"\nLoading connected components from {components_path} …")
    with components_path.open("r", encoding="utf-8") as fh:
        connected_components: List[Dict[str, Any]] = json.load(fh)
    print(f"  {len(connected_components)} components loaded.")

    # ---- pre-clean point-count stats ----
    stats_path = outputs_dir / "component_point_stats.json"
    stats: Dict[str, Any] = {"pre_clean": point_count_stats(connected_components)}
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Pre-clean point-count stats saved → {stats_path}")

    print(f"\nLoading COLMAP model from {colmap_model_dir} …")
    cameras, images, points3D = load_colmap_model(colmap_model_dir)
    print(f"  {len(points3D)} 3D points loaded.")

    print(
        f"\nRunning DBSCAN (eps={eps}, min_samples={min_samples}, split_components={split_components}) …\n"
    )

    all_output: List[Dict[str, Any]] = []
    removed = 0
    split = 0

    for component in connected_components:
        result = clean_component(
            component,
            points3D,
            eps=eps,
            min_samples=min_samples,
            split_components=split_components,
        )
        if len(result) == 0:
            removed += 1
        elif len(result) > 1:
            split += 1
        all_output.extend(result)

    # Remove internal bookkeeping keys before filtering
    for comp in all_output:
        comp.pop("_split_from", None)
        comp.pop("_split_sub", None)

    # Drop components with fewer than min_points points
    before_filter = len(all_output)
    all_output = [c for c in all_output if len(c["set_of_point3DIds"]) >= min_points]
    too_small = before_filter - len(all_output)

    # Re-number component IDs sequentially
    for new_id, comp in enumerate(all_output):
        comp["connected_comp_id"] = new_id

    print(
        f"\nSummary:"
        f"\n  Original components : {len(connected_components)}"
        f"\n  Removed (all noise) : {removed}"
        f"\n  Split               : {split}"
        f"\n  Removed (< {min_points} pts)     : {too_small}"
        f"\n  Output components   : {len(all_output)}"
    )

    with components_path.open("w", encoding="utf-8") as fh:
        json.dump(all_output, fh, indent=2)
    print(f"\nSaved cleaned components → {components_path}")

    # ---- post-clean point-count stats ----
    stats["cleaned"] = point_count_stats(all_output)
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Point-count stats (pre-clean + cleaned) saved → {stats_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean connected components using DBSCAN: remove noise points and "
            "split multi-cluster components."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Dataset name (must be recognised by config.py).",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.1,
        help="DBSCAN neighbourhood radius in world units.",
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=5,
        help="DBSCAN minimum samples per core point.",
    )
    parser.add_argument(
        "--min_points",
        type=int,
        default=20,
        help="Drop components with fewer than this many points after cleaning.",
    )
    parser.add_argument(
        "--split_components",
        action="store_true",
        default=False,
        help=(
            "Split components with multiple DBSCAN clusters into separate "
            "components.  By default only noise points are removed and "
            "multi-cluster components are kept intact."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    clean_connected_components(
        dataset_name=args.dataset_name,
        eps=args.eps,
        min_samples=args.min_samples,
        min_points=args.min_points,
        split_components=args.split_components,
    )
