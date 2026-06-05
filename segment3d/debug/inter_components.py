"""
inter_components.py - Inspect cross-component instance overlap for two connected components.

Prints two tables matching the output of temp.ipynb cell 2:
  1. Calculated values  – live-computed voxel Jaccard, containment, and CLIP distance.
  2. Unmerged edges     – records from unmerged_edges.json for matching pairs, with
                         the stored Jaccard, CLIP distance, and rejection reason.

Usage:
    python debug/inter_components.py --dataset_name <name> --comp_id_a <int> --comp_id_b <int>
    python debug/inter_components.py --dataset_name <name> --comp_id_a <int> --comp_id_b <int> --fresh-calculate

By default (no --fresh-calculate), only unmerged_edges.json is read; no COLMAP data or CLIP
model is loaded.  Pass --fresh-calculate to also recompute voxel Jaccard, containment, and
CLIP distance from scratch.

Voxel size is taken from default_params.DEFAULT_PARAMETERS["voxel_size_cm"].
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config_io import get_colmap_output_path, get_output_path
from segment3d.utils.colmap_io import load_colmap_model
from segment3d.postsam3_pipeline.default_params import DEFAULT_PARAMETERS


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare instances across two connected components."
    )
    p.add_argument(
        "--dataset_name",
        required=True,
        help="Dataset name (folder under data/ and outputs/)",
    )
    p.add_argument(
        "--comp_id_a", required=True, type=int, help="First connected-component ID"
    )
    p.add_argument(
        "--comp_id_b", required=True, type=int, help="Second connected-component ID"
    )
    p.add_argument(
        "--voxel_size_cm",
        type=float,
        default=DEFAULT_PARAMETERS["voxel_size_cm"],
        help=f"Voxel side length in cm (default: {DEFAULT_PARAMETERS['voxel_size_cm']})",
    )
    p.add_argument(
        "--fresh-calculate",
        action="store_true",
        default=False,
        help="Recompute voxel Jaccard and CLIP embeddings from scratch. "
        "When omitted (default), values are read directly from unmerged_edges.json.",
    )
    return p.parse_args()


# ── CLIP model ────────────────────────────────────────────────────────────────
def load_clip(model_name: str, pretrained: str, device: str):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval().to(device)
    return model, preprocess


# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_instance_id(iid: str):
    """'container_seq_1_2'  →  ('container', 'seq_1', '2')"""
    m = re.match(r"^(.+)_(seq_\d+)_(\d+)$", iid)
    if m:
        return m.group(1), m.group(2), m.group(3)
    raise ValueError(f"Cannot parse instance id: {iid!r}")


def instance_to_image_path(iid: str, dataset_name: str) -> Path:
    label, seq_key, idx = parse_instance_id(iid)
    base_path = (
        get_output_path(dataset_name)
        / "graph_node_mask_images"
        / f"{label}__{seq_key}__{idx}"
    )
    return base_path.with_suffix(
        ".png"
    )  # default to PNG (matches mask_graph.py output)


def instance_to_point_ids(iid: str, obj_assoc: dict) -> set:
    label, seq_key, idx = parse_instance_id(iid)
    try:
        return set(obj_assoc[label][seq_key][idx])
    except KeyError:
        print(
            f"  WARNING: {iid} not found in object_3d_associations ({label}/{seq_key}/{idx})"
        )
        return set()


def to_voxel(
    xyz: np.ndarray, min_xyz: np.ndarray, extent: np.ndarray, grid_xyz: np.ndarray
):
    frac = (xyz - min_xyz) / extent
    ix, iy, iz = (frac * grid_xyz).astype(np.int64).clip(0, grid_xyz - 1)
    return (int(ix), int(iy), int(iz))


def point_ids_to_voxels(
    pid_set: set,
    points3D: dict,
    min_xyz: np.ndarray,
    extent: np.ndarray,
    grid_xyz: np.ndarray,
) -> set:
    return {
        to_voxel(
            np.array(points3D[pid].xyz, dtype=np.float64), min_xyz, extent, grid_xyz
        )
        for pid in pid_set
        if pid in points3D
    }


def voxel_jaccard(va: set, vb: set) -> float:
    union = va | vb
    return len(va & vb) / len(union) if union else 0.0


def containment(va: set, vb: set) -> float:
    denom = min(len(va), len(vb))
    return len(va & vb) / denom if denom else 0.0


def clip_embedding(img_path: Path, model, preprocess, device: str) -> np.ndarray | None:
    if not img_path.exists():
        return None
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat[0].cpu().float().numpy()


# ── Unmerged edges ────────────────────────────────────────────────────────────
def load_unmerged_edges(dataset_name: str) -> dict:
    """
    Load unmerged_edges.json and return a bidirectional lookup:
        (node1, node2) → entry dict  (both orderings stored)

    Returns an empty dict if the file does not exist.
    """
    path = get_output_path(dataset_name) / "unmerged_edges.json"
    if not path.exists():
        print(f"WARNING: {path} not found – unmerged edge info unavailable")
        return {}
    with open(path) as f:
        entries = json.load(f)
    lookup: dict = {}
    for entry in entries:
        key_ab = (entry["node1"], entry["node2"])
        key_ba = (entry["node2"], entry["node1"])
        lookup[key_ab] = entry
        lookup[key_ba] = entry
    print(f"\n\nLoaded {len(entries)} unmerged edge(s)")
    return lookup


def print_unmerged_edges(
    instances_a: list,
    instances_b: list,
    unmerged_lookup: dict,
    comp_id_a: int,
    comp_id_b: int,
) -> None:
    """
    Print the unmerged-edges table for cross-component pairs, mirroring
    Section 2 of temp.ipynb cell 2.
    """
    print(
        f"\n── Unmerged edges from file (comp {comp_id_a} \u2194 comp {comp_id_b}) "
        "───────────────────────"
    )
    print(
        f"{'Instance A':<30}  {'Instance B':<30}  {'Jaccard':>8}  "
        f"{'CLIPdist':>10}  {'Rejection reason'}"
    )
    print("-" * 100)
    found_any = False
    for iid_a, _ in instances_a:
        for iid_b, _ in instances_b:
            entry = unmerged_lookup.get((iid_a, iid_b))
            if entry:
                found_any = True
                u_jac = entry.get("geometric_jaccard", entry.get("jaccard"))
                u_clip = entry.get("clip_distance")
                reason = entry.get("rejection_reason", "unknown")
                jac_str = f"{u_jac:.4f}" if u_jac is not None else "    n/a"
                clip_str = f"{u_clip:.4f}" if u_clip is not None else "       n/a"
                print(
                    f"{iid_a:<30}  {iid_b:<30}  {jac_str:>8}  {clip_str:>10}  {reason}"
                )
    if not found_any:
        print(
            "  (no matching entries found in unmerged_edges.json for these component pairs)"
        )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    dataset_name = args.dataset_name
    comp_id_a = args.comp_id_a
    comp_id_b = args.comp_id_b
    voxel_size_cm: float = args.voxel_size_cm

    fresh_calculate: bool = args.fresh_calculate

    # ── Load connected components (always needed) ─────────────────────────────
    cc_path = get_output_path(dataset_name) / "connected_components.json"
    with open(cc_path) as f:
        components = json.load(f)
    comp_map = {c["connected_comp_id"]: c for c in components}

    comp_a_info = comp_map[comp_id_a]
    comp_b_info = comp_map[comp_id_b]
    all_instances = [(iid, comp_id_a) for iid in comp_a_info["instance_ids"]] + [
        (iid, comp_id_b) for iid in comp_b_info["instance_ids"]
    ]
    print(f"\nComponent {comp_id_a} instances: {comp_a_info['instance_ids']}")
    print(f"Component {comp_id_b} instances: {comp_b_info['instance_ids']}")

    instances_a = [(iid, cid) for iid, cid in all_instances if cid == comp_id_a]
    instances_b = [(iid, cid) for iid, cid in all_instances if cid == comp_id_b]

    if fresh_calculate:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model_name = DEFAULT_PARAMETERS["clip_model"]
        clip_pretrained = DEFAULT_PARAMETERS["clip_pretrained"]

        # ── Load COLMAP model ─────────────────────────────────────────────────
        colmap_model_dir = get_colmap_output_path(dataset_name)
        _, _, points3D = load_colmap_model(str(colmap_model_dir))
        print(f"Loaded {len(points3D)} COLMAP 3D points")

        # ── Load object_3d_associations ───────────────────────────────────────
        assoc_path = (
            get_output_path(dataset_name)
            / "object_level_masks"
            / "object_3d_associations.json"
        )
        with open(assoc_path) as f:
            obj_assoc = json.load(f)

        # ── Voxel grid ────────────────────────────────────────────────────────
        all_scene_xyz = np.stack(
            [np.array(pt.xyz, dtype=np.float64) for pt in points3D.values()]
        )
        min_xyz = all_scene_xyz.min(axis=0)
        extent = all_scene_xyz.max(axis=0) - min_xyz
        extent = np.where(extent == 0, 1.0, extent)
        voxel_size_m = voxel_size_cm / 100.0
        grid_xyz = np.maximum(np.ceil(extent / voxel_size_m).astype(np.int64), 1)
        print(f"\nScene bbox: {min_xyz} → {min_xyz + extent}")
        print(f"Voxel grid: {grid_xyz}  (voxel_size={voxel_size_cm} cm)")

        # ── Load CLIP model ───────────────────────────────────────────────────
        print(
            f"\nLoading CLIP model {clip_model_name!r} ({clip_pretrained}) on {device} …"
        )
        clip_m, preprocess = load_clip(clip_model_name, clip_pretrained, device)

        # ── Per-instance data ─────────────────────────────────────────────────
        print(
            "\n── Instance data ────────────────────────────────────────────────────────"
        )
        instance_data: dict = {}
        for iid, comp_id in all_instances:
            img_path = instance_to_image_path(iid, dataset_name)
            point_ids = instance_to_point_ids(iid, obj_assoc)
            voxels = point_ids_to_voxels(point_ids, points3D, min_xyz, extent, grid_xyz)
            clip_emb = clip_embedding(img_path, clip_m, preprocess, device)
            instance_data[iid] = {
                "comp": comp_id,
                "pts": point_ids,
                "voxels": voxels,
                "clip": clip_emb,
            }
            img_ok = "✓" if img_path.exists() else "✗ MISSING"
            print(
                f"  [{comp_id:>3}] {iid:<30}  pts={len(point_ids):>5}  voxels={len(voxels):>4}  img={img_ok}"
            )

        # Section 1: Calculated values
        print(
            f"\n── Calculated values (comp {comp_id_a} \u2194 comp {comp_id_b}) "
            "──────────────────────────────"
        )
        print(
            f"{'Instance A':<30}  {'Instance B':<30}  {'Jaccard':>8}  {'Contain':>8}  {'CLIPdist':>10}"
        )
        print("-" * 92)
        for iid_a, _ in instances_a:
            for iid_b, _ in instances_b:
                jac = voxel_jaccard(
                    instance_data[iid_a]["voxels"], instance_data[iid_b]["voxels"]
                )
                cnt = containment(
                    instance_data[iid_a]["voxels"], instance_data[iid_b]["voxels"]
                )
                ea, eb = instance_data[iid_a]["clip"], instance_data[iid_b]["clip"]
                cdist = (
                    (1.0 - float(np.dot(ea, eb)))
                    if (ea is not None and eb is not None)
                    else float("nan")
                )
                print(
                    f"{iid_a:<30}  {iid_b:<30}  {jac:>8.4f}  {cnt:>8.4f}  {cdist:>10.4f}"
                )
    else:
        print("\n(Skipping live calculation — use --fresh-calculate to recompute.)")

    # Section 2: Unmerged edges from file
    unmerged_lookup = load_unmerged_edges(dataset_name)
    print_unmerged_edges(
        instances_a, instances_b, unmerged_lookup, comp_id_a, comp_id_b
    )


if __name__ == "__main__":
    main()
