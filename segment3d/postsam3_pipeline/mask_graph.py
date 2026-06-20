"""
mask_graph.py - Build a graph connecting per-object SAM3 instances across sequences
based on shared COLMAP 3D points.

Reads:
    <outputs_dir>/object_level_masks/object_3d_associations.json

produced by associate2d3d.py.  Each node in the graph is one
(obj_slug, seq_key, obj_id) triple.  Edges connect nodes from *different*
sequences that share enough 3D points (Jaccard ≥ tau or overlap ≥ K).
The merge is object-name-agnostic: nodes from different objects may be linked.

Outputs (written to <outputs_dir>/):
    mask_graph.gpickle             – NetworkX graph (pickle)
    connected_components.json      – list of merged instances with point3D IDs
    mask_graph_stats.json          – summary statistics
    unmerged_edges.json            – candidate pairs that had spatial overlap but were
                                     rejected (Jaccard below threshold or CLIP distance
                                     exceeded), with their jaccard / clip_distance scores

Usage (from the segment3d/ directory):
    python -m src.per_object_sam3.mask_graph --dataset ProjectLabStudio_inv_method
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import open_clip
from PIL import Image as PILImage
from pycocotools import mask as mask_utils
from scipy import sparse
import torch
from tqdm import tqdm

from config_io import (
    get_output_path,
    get_images_output_path,
    get_colmap_output_path,
    reset_dir,
)
from ..utils.colmap_io import load_colmap_model

# Node type: (seq_key, obj_slug, obj_id_str)
# Using seq_key as the first element so the "same-sequence" skip is node[0] == node[0],
# mirroring the "same image" guard in src/mask_graph.py.
Node = Tuple[str, str, str]


# ---------------------------------------------------------------------------
# Mask image extractor
# ---------------------------------------------------------------------------


def get_mask_image(
    node: Node,
    per_instance_sets: Dict[Node, Set[int]],
    images: Dict[int, Any],
    masks_base_dir: Path,
    images_dir: Path,
    save_dir: Optional[Path] = None,
) -> "Optional[Any]":  # Optional[PIL.Image.Image]
    """
    Return a PIL Image of the masked region for *node*.

    Frame selection: for each frame JSON in
    ``masks_base_dir / obj_slug / seq_key / *.json``, count how many of the
    node’s COLMAP 3D point IDs are also observed in the corresponding COLMAP
    image.  The frame with the highest overlap is preferred.  Frames are
    visited in descending overlap order until one is found that contains an
    annotation for the node’s ``obj_id``.

    The RLE mask for that annotation is decoded, applied to the RGB image, and
    the bounding-box crop (background pixels zeroed) is returned.

    Returns ``None`` if no suitable frame is found.
    """

    seq_key, obj_slug, obj_id_str = node
    obj_id_int = int(obj_id_str)
    node_point_ids = per_instance_sets.get(node, set())

    mask_seq_dir = masks_base_dir / obj_slug / seq_key
    if not mask_seq_dir.is_dir():
        return None

    # Map frame stem -> COLMAP Image namedtuple
    stem_to_image: Dict[str, Any] = {
        Path(img.name).stem: img for img in images.values()
    }

    # ------------------------------------------------------------------
    # Pass 1: count node-point overlap per frame (no JSON reading).
    # ------------------------------------------------------------------
    candidates: List[Tuple[int, Path]] = []
    for frame_path in sorted(mask_seq_dir.glob("*.json")):
        colmap_img = stem_to_image.get(frame_path.stem)
        if colmap_img is None:
            continue
        img_point_ids = {int(pid) for pid in colmap_img.point3D_ids if int(pid) != -1}
        count = len(node_point_ids & img_point_ids)
        candidates.append((count, frame_path))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # Pass 2: walk best-first, read JSONs until we find our obj_id.
    # ------------------------------------------------------------------
    best_rle: Optional[Dict[str, Any]] = None
    best_frame_stem: Optional[str] = None

    for _count, frame_path in candidates:
        with frame_path.open("r", encoding="utf-8") as fh:
            annotations = json.load(fh)
        for ann in annotations:
            if int(ann["obj_id"]) == obj_id_int:
                best_rle = ann["segmentation"]
                best_frame_stem = frame_path.stem
                break
        if best_rle is not None:
            break

    if best_rle is None or best_frame_stem is None:
        return None

    # ------------------------------------------------------------------
    # Load the RGB image, apply mask, crop to bounding box.
    # ------------------------------------------------------------------
    img_path: Optional[Path] = None
    for ext in (".jpg", ".jpeg", ".png"):
        p = images_dir / (best_frame_stem + ext)
        if p.exists():
            img_path = p
            break

    if img_path is None:
        return None

    rgb = PILImage.open(img_path).convert("RGB")
    mask_arr = mask_utils.decode(best_rle).astype(bool)  # (H, W)

    rows = np.any(mask_arr, axis=1)
    cols = np.any(mask_arr, axis=0)
    if not rows.any() or not cols.any():
        return None

    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1])
    cmin = int(np.where(cols)[0][0])
    cmax = int(np.where(cols)[0][-1])

    rgb_arr = np.array(rgb)
    cropped = rgb_arr[rmin : rmax + 1, cmin : cmax + 1].copy()
    mask_crop = mask_arr[rmin : rmax + 1, cmin : cmax + 1]
    cropped[~mask_crop] = 0  # zero out background pixels

    result = PILImage.fromarray(cropped)

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{obj_slug}__{seq_key}__{obj_id_str}.png"
        result.save(save_path, "PNG")

    return result


# ---------------------------------------------------------------------------
# CLIP image encoder
# ---------------------------------------------------------------------------


def compute_clip_image_embeddings(
    nodes: List[Node],
    per_instance_sets: Dict[Node, Set[int]],
    images: Dict[int, Any],
    masks_base_dir: Path,
    images_dir: Path,
    save_dir: Optional[Path] = None,
) -> Dict[Node, np.ndarray]:
    """
    Encode the representative mask image for each node with OpenCLIP ViT-H-14
    and return ``node → unit-normalised float32 vector`` (shape ``(1024,)``).

    For each node :func:`get_mask_image` is called to obtain the best-frame
    masked crop; crops that can't be resolved are silently skipped.

    Args:
        save_dir: If provided, each resolved mask-crop image is saved as a
            lossless PNG to this directory under the name
            ``{obj_slug}__{seq_key}__{obj_id_str}.png``.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k"
    )
    model.eval().to(device)

    embeddings: Dict[Node, np.ndarray] = {}
    missing = 0
    iterator = tqdm(nodes, desc="CLIP image embeddings", unit="node", dynamic_ncols=True)
    for node in iterator:
        img = get_mask_image(
            node,
            per_instance_sets,
            images,
            masks_base_dir,
            images_dir,
            save_dir=save_dir,
        )
        if img is None:
            missing += 1
            continue
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.encode_image(tensor)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        embeddings[node] = feat.squeeze(0).cpu().float().numpy()

    print(
        f"  Computed image embeddings for {len(embeddings)}/{len(nodes)} node(s) "
        f"({missing} skipped – no suitable frame found)."
    )
    return embeddings


# ---------------------------------------------------------------------------
# Load associations
# ---------------------------------------------------------------------------


def load_associations(
    associations_path: Path, min_points: int = 1
) -> Tuple[Dict[Node, Set[int]], Dict[Node, str]]:
    """
    Load object_3d_associations.json and return:
      - per_instance_sets  : Node → set of 3D point IDs
      - node_to_instance_id: Node → human-readable instance key string

    Nodes with fewer than *min_points* 3D associations are dropped.
    """
    with associations_path.open("r", encoding="utf-8") as fh:
        data: Dict[str, Dict[str, Dict[str, List[int]]]] = json.load(fh)

    per_instance_sets: Dict[Node, Set[int]] = {}
    node_to_instance_id: Dict[Node, str] = {}

    for obj_slug, seq_map in data.items():
        for seq_key, obj_map in seq_map.items():
            for obj_id_str, point_list in obj_map.items():
                if len(point_list) < min_points:
                    continue
                node: Node = (seq_key, obj_slug, obj_id_str)
                per_instance_sets[node] = set(point_list)
                node_to_instance_id[node] = f"{obj_slug}_{seq_key}_{obj_id_str}"

    return per_instance_sets, node_to_instance_id


# ---------------------------------------------------------------------------
# Edge building  (same sparse-matrix logic as src/mask_graph.py)
# ---------------------------------------------------------------------------


def build_edges_scipy(
    per_instance_sets: Dict[Node, Set[int]],
    K: int = 5,
    tau: float = 0.2,
    clip_node_embeddings: Dict[Node, np.ndarray] = {},
    clip_distance_threshold: float = 0.8,
) -> Tuple[
    List[Tuple[Node, Node, Dict[str, Any]]],
    List[Tuple[Node, Node, Dict[str, Any]]],
]:
    """
    Build edges using a sparse incidence matrix + M·Mᵀ intersection trick.

    Two nodes are connected when:
        overlap ≥ K  OR  Jaccard(A, B) ≥ tau

    A CLIP-based semantic guard is always applied: a candidate edge is
    discarded if the cosine distance between the two nodes' mask-image
    embeddings exceeds *clip_distance_threshold*:
        cosine_distance = 1 − cosine_similarity

    Returns:
        A tuple ``(edges, unmerged_edges)`` where *unmerged_edges* contains
        candidate pairs that had at least one shared 3D point but were
        rejected, annotated with a ``rejection_reason`` field.
    """
    nodes_list = list(per_instance_sets.keys())
    if not nodes_list:
        return [], []

    # --- collect all unique 3D point IDs ---
    all_points: Set[int] = set()
    total_assoc = 0
    for s in per_instance_sets.values():
        all_points.update(s)
        total_assoc += len(s)

    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    num_nodes = len(nodes_list)
    num_points = len(all_points)

    print(f"Constructing sparse matrix ({num_nodes} instances × {num_points} points)…")

    # --- build COO arrays ---
    row_inds = np.zeros(total_assoc, dtype=np.int32)
    col_inds = np.zeros(total_assoc, dtype=np.int32)
    cur = 0
    for ni, node in enumerate(nodes_list):
        for pt in per_instance_sets[node]:
            row_inds[cur] = ni
            col_inds[cur] = point_to_idx[pt]
            cur += 1

    data = np.ones(total_assoc, dtype=np.int32)
    M = sparse.csr_matrix((data, (row_inds, col_inds)), shape=(num_nodes, num_points))

    print("Computing intersection matrix (M · Mᵀ)…")
    intersection_matrix = M.dot(M.T)

    set_sizes = np.array(M.sum(axis=1)).flatten()

    print("Filtering edges and calculating Jaccard…")
    upper_tri = sparse.triu(intersection_matrix, k=1).tocoo()

    edges: List[Tuple[Node, Node, Dict[str, Any]]] = []
    unmerged_edges: List[Tuple[Node, Node, Dict[str, Any]]] = []
    for i, j, overlap in zip(upper_tri.row, upper_tri.col, upper_tri.data):
        node1 = nodes_list[i]
        node2 = nodes_list[j]

        union_size = set_sizes[i] + set_sizes[j] - overlap
        jaccard_sim = float(overlap / union_size) if union_size > 0 else 0.0

        emb1 = clip_node_embeddings.get(node1)
        emb2 = clip_node_embeddings.get(node2)
        clip_dist: Optional[float] = None
        if emb1 is not None and emb2 is not None:
            clip_dist = 1.0 - float(np.dot(emb1, emb2))

        if overlap >= K or jaccard_sim >= tau:
            # CLIP semantic guard: skip if mask images are too dissimilar.
            if clip_dist is not None and clip_dist > clip_distance_threshold:
                unmerged_edges.append(
                    (
                        node1,
                        node2,
                        {
                            "overlap": int(overlap),
                            "jaccard": jaccard_sim,
                            "clip_distance": clip_dist,
                            "rejection_reason": "clip_distance_exceeded",
                        },
                    )
                )
                continue

            edges.append(
                (
                    node1,
                    node2,
                    {
                        "overlap": int(overlap),
                        "jaccard": jaccard_sim,
                        "clip_distance": clip_dist,
                    },
                )
            )
        else:
            unmerged_edges.append(
                (
                    node1,
                    node2,
                    {
                        "overlap": int(overlap),
                        "jaccard": jaccard_sim,
                        "clip_distance": clip_dist,
                        "rejection_reason": "jaccard_below_threshold",
                    },
                )
            )

    return edges, unmerged_edges


# ---------------------------------------------------------------------------
# Geometric (voxel-based) edge building
# ---------------------------------------------------------------------------


def build_edges_geometric_intersection(
    per_instance_sets: Dict[Node, Set[int]],
    points3D: Dict[int, Any],
    voxel_size_cm: float = 50.0,
    tau: float = 0.8,
    clip_node_embeddings: Dict[Node, np.ndarray] = {},
    clip_distance_threshold: float = 0.8,
) -> Tuple[
    List[Tuple[Node, Node, Dict[str, Any]]],
    List[Tuple[Node, Node, Dict[str, Any]]],
]:
    """
    Build edges using voxel-based geometric intersection.

    Each instance's 3D point set is projected onto a voxel grid whose cell
    side length is *voxel_size_cm* centimetres (point coordinates are assumed
    to be in metres).  The grid spans the global bounding box of **all points
    in the COLMAP model** (not just the points belonging to the instances under
    comparison).  This ensures voxel indices are consistent across all instance
    pairs and that the grid is not artificially shrunk to fit only the
    instances being compared (which would inflate Jaccard similarity).

    Two instances are connected when their voxel-occupancy Jaccard similarity
    meets the threshold:

        Jaccard(A, B) ≥ tau

    The same sparse incidence matrix + M·Mᵀ trick employed in
    :func:`build_edges_scipy` is used, but the columns represent voxels
    instead of raw point IDs.

    Args:
        per_instance_sets:  Node → set of COLMAP 3D point IDs.
        points3D:           COLMAP ``points3D`` dict
                            (``point_id → Point3D`` namedtuple with ``.xyz``).
        voxel_size_cm:      Side length of each voxel in centimetres
                            (point coordinates are in metres).  The per-axis
                            grid resolution is derived as
                            ``ceil(extent_m / (voxel_size_cm / 100))``.
        tau:                Min Jaccard similarity of voxel-occupancy sets.
        clip_node_embeddings: Mapping of Node → unit-vector
                            (from :func:`compute_clip_image_embeddings`).
                            Candidate edges whose two node embeddings have
                            cosine distance ``> clip_distance_threshold``
                            are always discarded.
        clip_distance_threshold: Maximum allowed cosine distance between
                            the OpenCLIP ViT-H-14 image embeddings of the two
                            nodes' representative mask crops.

    Returns:
        A tuple ``(edges, unmerged_edges)`` where *unmerged_edges* contains
        candidate pairs that had at least one shared voxel but were rejected,
        annotated with a ``rejection_reason`` field
        (``"jaccard_below_threshold"`` or ``"clip_distance_exceeded"``).
        Each entry carries ``voxel_overlap`` (int), ``geometric_jaccard``
        (float), and ``clip_distance`` (float or None).
    """
    nodes_list = list(per_instance_sets.keys())
    if not nodes_list:
        return [], []

    # ------------------------------------------------------------------
    # Global bounding box: use ALL points in the COLMAP model so that
    # voxel indices are stable and not biased toward the instances being
    # compared.
    # ------------------------------------------------------------------
    if not points3D:
        print("Warning: points3D is empty; returning no edges.")
        return [], []

    all_scene_xyz = np.stack(
        [np.array(pt.xyz, dtype=np.float64) for pt in points3D.values()], axis=0
    )  # (N_scene, 3)
    min_xyz = all_scene_xyz.min(axis=0)
    max_xyz = all_scene_xyz.max(axis=0)
    extent = max_xyz - min_xyz
    # Avoid division-by-zero on degenerate axes.
    extent = np.where(extent == 0, 1.0, extent)

    voxel_size_m = voxel_size_cm / 100.0
    grid_xyz = np.maximum(np.ceil(extent / voxel_size_m).astype(np.int64), 1)
    grid_x, grid_y, grid_z = int(grid_xyz[0]), int(grid_xyz[1]), int(grid_xyz[2])

    print(
        f"Voxel grid: {grid_x}×{grid_y}×{grid_z} (voxel_size={voxel_size_cm}cm)  "
        f"bbox [{min_xyz[0]:.3f}, {min_xyz[1]:.3f}, {min_xyz[2]:.3f}] → "
        f"[{max_xyz[0]:.3f}, {max_xyz[1]:.3f}, {max_xyz[2]:.3f}]"
    )

    def _point_to_voxel(xyz: np.ndarray) -> int:
        """Map an XYZ coordinate to a flat voxel index."""
        frac = (xyz - min_xyz) / extent  # values in [0, 1]
        ix, iy, iz = (frac * grid_xyz).astype(np.int64).clip(0, grid_xyz - 1)
        return int(ix * grid_y * grid_z + iy * grid_z + iz)

    # ------------------------------------------------------------------
    # Resolve instance point IDs → xyz (skip IDs absent from the model).
    # ------------------------------------------------------------------
    all_instance_point_ids: Set[int] = set()
    for s in per_instance_sets.values():
        all_instance_point_ids.update(s)

    point_xyz: Dict[int, np.ndarray] = {
        pid: np.array(points3D[pid].xyz, dtype=np.float64)
        for pid in all_instance_point_ids
        if pid in points3D
    }

    if not point_xyz:
        print(
            "Warning: no valid 3D coordinates found for any instance; returning no edges."
        )
        return []

    # ------------------------------------------------------------------
    # Compute per-instance voxel occupancy sets.
    # ------------------------------------------------------------------
    per_instance_voxels: Dict[Node, Set[int]] = {}
    for node, point_ids in per_instance_sets.items():
        voxel_set: Set[int] = set()
        for pid in point_ids:
            if pid in point_xyz:
                voxel_set.add(_point_to_voxel(point_xyz[pid]))
        if voxel_set:
            per_instance_voxels[node] = voxel_set

    # Work only with nodes that have at least one valid voxel.
    valid_nodes = [n for n in nodes_list if n in per_instance_voxels]
    if not valid_nodes:
        return [], []

    # ------------------------------------------------------------------
    # Build sparse incidence matrix M  (instances × voxels).
    # ------------------------------------------------------------------
    all_voxels: Set[int] = set()
    total_assoc = 0
    for s in per_instance_voxels.values():
        all_voxels.update(s)
        total_assoc += len(s)

    voxel_to_idx = {v: i for i, v in enumerate(all_voxels)}
    num_nodes = len(valid_nodes)
    num_voxels = len(all_voxels)

    print(
        f"Constructing sparse voxel matrix "
        f"({num_nodes} instances × {num_voxels} occupied voxels)…"
    )

    row_inds = np.zeros(total_assoc, dtype=np.int32)
    col_inds = np.zeros(total_assoc, dtype=np.int32)
    cur = 0
    for ni, node in enumerate(valid_nodes):
        for v in per_instance_voxels[node]:
            row_inds[cur] = ni
            col_inds[cur] = voxel_to_idx[v]
            cur += 1

    data = np.ones(total_assoc, dtype=np.int32)
    M = sparse.csr_matrix((data, (row_inds, col_inds)), shape=(num_nodes, num_voxels))

    print("Computing voxel intersection matrix (M · Mᵀ)…")
    intersection_matrix = M.dot(M.T)

    set_sizes = np.array(M.sum(axis=1)).flatten()

    print("Filtering edges and calculating geometric Jaccard…")
    upper_tri = sparse.triu(intersection_matrix, k=1).tocoo()

    edges: List[Tuple[Node, Node, Dict[str, Any]]] = []
    unmerged_edges: List[Tuple[Node, Node, Dict[str, Any]]] = []
    for i, j, overlap in zip(upper_tri.row, upper_tri.col, upper_tri.data):
        node1 = valid_nodes[i]
        node2 = valid_nodes[j]

        # Do not merge objects identified as different instances by SAM3.
        if (node1[0] == node2[0]) and (node1[1] == node2[1]) and (node1[2] != node2[2]):
            continue

        union_size = set_sizes[i] + set_sizes[j] - overlap
        geo_jac = float(overlap / union_size) if union_size > 0 else 0.0

        emb1 = clip_node_embeddings.get(node1)
        emb2 = clip_node_embeddings.get(node2)
        clip_dist: Optional[float] = None
        if emb1 is not None and emb2 is not None:
            clip_dist = 1.0 - float(np.dot(emb1, emb2))

        if geo_jac >= tau:
            # CLIP semantic guard: skip if mask images are too dissimilar.
            if clip_dist is not None and clip_dist > clip_distance_threshold:
                unmerged_edges.append(
                    (
                        node1,
                        node2,
                        {
                            "voxel_overlap": int(overlap),
                            "geometric_jaccard": geo_jac,
                            "clip_distance": clip_dist,
                            "rejection_reason": "clip_distance_exceeded",
                        },
                    )
                )
                continue

            edges.append(
                (
                    node1,
                    node2,
                    {
                        "voxel_overlap": int(overlap),
                        "geometric_jaccard": geo_jac,
                        "clip_distance": clip_dist,
                    },
                )
            )
        else:
            unmerged_edges.append(
                (
                    node1,
                    node2,
                    {
                        "voxel_overlap": int(overlap),
                        "geometric_jaccard": geo_jac,
                        "clip_distance": clip_dist,
                        "rejection_reason": "jaccard_below_threshold",
                    },
                )
            )

    return edges, unmerged_edges


# ---------------------------------------------------------------------------
# NetworkX conversion
# ---------------------------------------------------------------------------


def to_networkx(
    nodes: List[Node],
    edges: List[Tuple[Node, Node, Dict[str, Any]]],
    node_to_instance_id: Dict[Node, str],
) -> nx.Graph:
    G = nx.Graph()
    for node in nodes:
        seq_key, obj_slug, obj_id_str = node
        G.add_node(
            node,
            seq_key=seq_key,
            obj_slug=obj_slug,
            obj_id=obj_id_str,
            instance_id=node_to_instance_id.get(node, ""),
        )
    for node1, node2, attrs in edges:
        G.add_edge(node1, node2, **attrs)
    return G


# ---------------------------------------------------------------------------
# Constrained connected components
# ---------------------------------------------------------------------------


def constrained_connected_components(G: nx.Graph) -> List[Set[Node]]:
    """
    Compute connected components of *G* while enforcing the SAM3 distinctness
    constraint: two nodes that share the same ``(seq_key, obj_slug)`` but have
    different ``obj_id`` values are **never** placed in the same component,
    even transitively through intermediate nodes.

    Algorithm: union-find where merging two components is rejected when their
    ``(seq_key, obj_slug)`` key-sets overlap.  This guarantees that for every
    ``(seq_key, obj_slug)`` key, at most one ``obj_id`` appears in each
    resulting component.

    Returns a list of sets, one set per component (including singletons),
    mirroring the output of :func:`networkx.connected_components`.
    """
    # Initialise each node as its own component.
    parent: Dict[Node, Node] = {n: n for n in G.nodes()}
    # For each root: the set of (seq_key, obj_slug) keys present in its component.
    seq_slug_keys: Dict[Node, Set[Tuple[str, str]]] = {
        n: {(n[0], n[1])} for n in G.nodes()
    }
    # For each root: the member-node set (used for final output).
    members: Dict[Node, Set[Node]] = {n: {n} for n in G.nodes()}

    def _find(x: Node) -> Node:
        """Path-compressed root lookup."""
        while parent[x] is not x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    def _union(a: Node, b: Node) -> bool:
        """
        Attempt to merge the components of *a* and *b*.

        Returns ``True`` if the merge was performed (or they were already in
        the same component), ``False`` if rejected by the constraint.
        """
        ra, rb = _find(a), _find(b)
        if ra is rb:
            return True  # already the same component
        # Reject if the two components share any (seq_key, obj_slug) key —
        # that means they both contain a node from the same sequence/slug
        # pair, which by definition have different obj_ids (nodes are unique).
        if seq_slug_keys[ra] & seq_slug_keys[rb]:
            return False
        # Union by size: merge the smaller component into the larger.
        if len(members[ra]) < len(members[rb]):
            ra, rb = rb, ra
        parent[rb] = ra
        seq_slug_keys[ra] |= seq_slug_keys[rb]
        members[ra] |= members[rb]
        del seq_slug_keys[rb]
        del members[rb]
        return True

    # Sort edges by decreasing merge confidence so that the highest-quality
    # association always wins when a conflict triangle forces a choice.
    # Primary key: geometric_jaccard or jaccard (higher = more confident).
    # Secondary key: clip_distance (lower = more similar images).
    def _edge_sort_key(uvd: tuple) -> tuple:
        attrs = uvd[2]
        jac = attrs.get("geometric_jaccard", attrs.get("jaccard", 0.0)) or 0.0
        clip = attrs.get("clip_distance") or 1.0
        return (-jac, clip)

    sorted_edges = sorted(G.edges(data=True), key=_edge_sort_key)

    rejected = 0
    for u, v, _ in sorted_edges:
        if not _union(u, v):
            rejected += 1

    if rejected:
        print(
            f"  constrained_connected_components: blocked {rejected} transitive "
            "merge(s) that would have joined SAM3-distinct instances."
        )

    # Collect one set per unique root.
    roots = {_find(n) for n in G.nodes()}
    return [members[r] for r in roots]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def build_object_mask_graph(
    dataset_name: str,
    K: int = 30,
    tau: float = 0.8,
    min_points: int = 1,
    min_points_in_3d_segment: int = 10,
    intersection_type: str = "geometric",
    voxel_size_cm: float = 50.0,
    clip_distance_threshold: float = 0.8,
    save_segment_images: bool = False,
) -> None:
    """
    Build the object mask graph.

    Args:
        intersection_type: How to measure instance overlap.
            ``"geometric"``  – voxelise each instance's 3D point cloud and
                              compute Jaccard over the voxel-occupancy sets
                              (calls :func:`build_edges_geometric_intersection`).
            ``"id_based"``   – compute Jaccard directly over the raw COLMAP
                              point-ID sets (calls :func:`build_edges_scipy`).
        voxel_size_cm: Side length of each voxel in centimetres used when
            *intersection_type* is ``"geometric"``.
        clip_distance_threshold: Maximum cosine distance between OpenCLIP
            ViT-H-14 image embeddings of the two nodes' representative mask
            crops for an edge to be kept.  Two nodes whose mask-image embeddings
            have cosine distance greater than this value are **never** merged,
            even if their spatial overlap passes the Jaccard / K threshold.
        save_segment_images: When ``True``, save each node's representative
            masked-crop image as a JPEG to
            ``outputs/{dataset}/graph_node_mask_images/``.
    """
    if intersection_type not in ("geometric", "id_based"):
        raise ValueError(
            f"intersection_type must be 'geometric' or 'id_based', got {intersection_type!r}"
        )

    outputs_dir = get_output_path(dataset_name)
    obj_level_dir = outputs_dir / "object_level_masks"

    associations_path = obj_level_dir / "object_3d_associations.json"
    if not associations_path.is_file():
        raise FileNotFoundError(
            f"Associations file not found: {associations_path}\n"
            "Run associate2d3d.py first."
        )

    print(f"Loading associations from {associations_path}…")
    per_instance_sets, node_to_instance_id = load_associations(
        associations_path, min_points=min_points
    )

    total_nodes = len(per_instance_sets)
    total_pts = sum(len(s) for s in per_instance_sets.values())
    print(f"Loaded {total_nodes} instances, {total_pts} total point associations.")
    if total_nodes:
        print(f"Average points per instance: {total_pts / total_nodes:.1f}")

    # ----- Pre-compute CLIP image embeddings --------------------------------

    colmap_model_dir = get_colmap_output_path(dataset_name)
    images_dir = get_images_output_path(dataset_name)
    masks_base_dir = obj_level_dir / "masks"

    print(f"Loading COLMAP model for CLIP image embeddings from {colmap_model_dir}…")
    cameras, images, points3D = load_colmap_model(str(colmap_model_dir))

    all_nodes = list(per_instance_sets.keys())
    print(
        f"Computing CLIP image embeddings for {len(all_nodes)} node(s) "
        f"(clip_distance_threshold={clip_distance_threshold})…"
    )
    save_dir: Optional[Path] = None
    if save_segment_images:
        # Clear stale node images from a previous run so leftover files for
        # nodes that no longer exist don't accumulate.
        save_dir = reset_dir(outputs_dir / "graph_node_mask_images")
        print(f"  Saving node mask images → {save_dir}")
    clip_node_embeddings: Dict[Node, np.ndarray] = compute_clip_image_embeddings(
        all_nodes,
        per_instance_sets,
        images,
        masks_base_dir,
        images_dir,
        save_dir=save_dir,
    )

    # ----- Build edges -------------------------------------------------------
    print(
        f"\nBuilding edges (K={K}, tau={tau}, intersection_type={intersection_type!r})…"
    )
    if intersection_type == "geometric":
        edges, unmerged_edges = build_edges_geometric_intersection(
            per_instance_sets,
            points3D=points3D,
            voxel_size_cm=voxel_size_cm,
            tau=tau,
            clip_node_embeddings=clip_node_embeddings,
            clip_distance_threshold=clip_distance_threshold,
        )
    else:
        edges, unmerged_edges = build_edges_scipy(
            per_instance_sets,
            K=K,
            tau=tau,
            clip_node_embeddings=clip_node_embeddings,
            clip_distance_threshold=clip_distance_threshold,
        )
    print(f"Created {len(edges)} edges, {len(unmerged_edges)} unmerged candidate(s).")

    # ----- NetworkX graph ---------------------------------------------------
    print("\nCreating NetworkX graph…")
    nodes = list(per_instance_sets.keys())
    G = to_networkx(nodes, edges, node_to_instance_id)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    ccs = constrained_connected_components(G)
    if G.number_of_edges() > 0:
        print(f"Connected components: {len(ccs)}")
        print(f"Largest component: {max(len(c) for c in ccs)} nodes")
        degrees = [d for _, d in G.degree()]
        print(f"Average degree: {sum(degrees) / len(degrees):.2f}  Max: {max(degrees)}")

    # ----- Save graph -------------------------------------------------------
    graph_path = outputs_dir / "mask_graph.gpickle"
    with graph_path.open("wb") as fh:
        pickle.dump(G, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"\nSaved graph → {graph_path}")

    # ----- Save unmerged edges ----------------------------------------------
    unmerged_path = outputs_dir / "unmerged_edges.json"
    unmerged_list = [
        {
            "node1": node_to_instance_id.get(n1, "|".join(n1)),
            "node2": node_to_instance_id.get(n2, "|".join(n2)),
            **attrs,
        }
        for n1, n2, attrs in unmerged_edges
    ]
    with unmerged_path.open("w", encoding="utf-8") as fh:
        json.dump(unmerged_list, fh, indent=2)
    print(f"Saved unmerged edges  → {unmerged_path}  ({len(unmerged_list)} entries)")

    # ----- Load existing component IDs for stable incremental assignment -----
    components_path = outputs_dir / "connected_components.json"
    # Map: frozenset(instance_ids) → existing connected_comp_id
    existing_id_map: Dict[frozenset, int] = {}
    max_existing_id: int = -1
    if components_path.exists():
        try:
            with components_path.open("r", encoding="utf-8") as fh:
                existing_list: List[Dict[str, Any]] = json.load(fh)
            for comp in existing_list:
                key = frozenset(comp["instance_ids"])
                existing_id_map[key] = comp["connected_comp_id"]
                max_existing_id = max(max_existing_id, comp["connected_comp_id"])
            if existing_id_map:
                print(
                    f"Loaded {len(existing_id_map)} existing component ID(s) "
                    f"(max id={max_existing_id}) for stable reassignment."
                )
        except Exception as exc:
            print(
                f"Warning: could not load existing components for ID reuse ({exc}); "
                "all IDs will be freshly assigned."
            )
            existing_id_map = {}
            max_existing_id = -1

    next_new_id: int = max_existing_id + 1

    # ----- Connected components with merged 3D points -----------------------
    print("\nExtracting connected components…")
    components_list = []

    if G.number_of_nodes() > 0:
        for component in ccs:
            point3d_union: Set[int] = set()
            instance_ids: List[str] = []
            for node in component:
                point3d_union.update(per_instance_sets[node])
                if node in node_to_instance_id:
                    instance_ids.append(node_to_instance_id[node])

            if len(point3d_union) < min_points_in_3d_segment:
                continue

            sorted_instances = sorted(instance_ids)
            new_key = frozenset(sorted_instances)

            # Exact match with existing component
            if new_key in existing_id_map:
                comp_id = existing_id_map[new_key]
            else:
                # Subset match: an existing component grew by absorbing new instances
                # (new_key is a superset of some existing component's instance set)
                parent_id = None
                for existing_key, existing_id in existing_id_map.items():
                    if existing_key.issubset(new_key):
                        parent_id = existing_id
                        break
                if parent_id is not None:
                    comp_id = parent_id
                else:
                    # Genuinely new component – assign a fresh ID
                    comp_id = next_new_id
                    next_new_id += 1

            # Collect intra-component edges with their properties.
            component_edges: List[Dict[str, Any]] = []
            for node_a, node_b, edge_attrs in G.edges(component, data=True):
                # G.edges(component) may yield edges twice for undirected graphs;
                # only keep the canonical (node_a < node_b) direction.
                if node_a > node_b:
                    node_a, node_b = node_b, node_a
                id_a = node_to_instance_id.get(node_a, "__".join(node_a))
                id_b = node_to_instance_id.get(node_b, "__".join(node_b))
                jaccard_val = edge_attrs.get(
                    "jaccard", edge_attrs.get("geometric_jaccard")
                )
                component_edges.append(
                    {
                        "node1": id_a,
                        "node2": id_b,
                        "jaccard": jaccard_val,
                        "clip_distance": edge_attrs.get("clip_distance"),
                    }
                )
            # Deduplicate (node_a > node_b swap above may still produce duplicates
            # if NetworkX yields the same edge from both endpoint views).
            seen_edges: set = set()
            deduped_edges: List[Dict[str, Any]] = []
            for e in component_edges:
                key = (e["node1"], e["node2"])
                if key not in seen_edges:
                    seen_edges.add(key)
                    deduped_edges.append(e)

            components_list.append(
                {
                    "connected_comp_id": comp_id,
                    "instance_ids": sorted_instances,
                    "set_of_point3DIds": sorted(point3d_union),
                    "edges": deduped_edges,
                }
            )

        # Sort output by component ID for readability / stable ordering
        components_list.sort(key=lambda c: c["connected_comp_id"])

        new_count = sum(
            1
            for c in components_list
            if frozenset(c["instance_ids"]) not in existing_id_map
            and not any(
                ek.issubset(frozenset(c["instance_ids"])) for ek in existing_id_map
            )
        )
        print(
            f"Found {len(components_list)} component(s) with ≥ {min_points_in_3d_segment} points "
            f"({new_count} new, {len(components_list) - new_count} existing)."
        )
        if components_list:
            sizes = [len(c["set_of_point3DIds"]) for c in components_list]
            print(
                f"  Largest: {max(sizes)}  Smallest: {min(sizes)}  Avg: {sum(sizes)/len(sizes):.1f}"
            )

    with components_path.open("w", encoding="utf-8") as fh:
        json.dump(components_list, fh, indent=2)
    print(f"Saved connected components → {components_path}")

    # ----- Statistics -------------------------------------------------------
    stats: Dict[str, Any] = {
        "parameters": {
            "K": K,
            "tau": tau,
            "min_points": min_points,
            "min_points_in_3d_segment": min_points_in_3d_segment,
            "intersection_type": intersection_type,
            "voxel_size_cm": voxel_size_cm,
            "clip_distance_threshold": clip_distance_threshold,
        },
        "nodes": {"total": total_nodes},
        "edges": {"total": G.number_of_edges()},
        "point_associations": {
            "total": total_pts,
            "average_per_instance": total_pts / total_nodes if total_nodes else 0,
        },
    }
    if G.number_of_edges() > 0:
        degrees = [d for _, d in G.degree()]
        stats["graph_statistics"] = {
            "connected_components": len(ccs),
            "largest_component_nodes": max(len(c) for c in ccs),
            "average_degree": sum(degrees) / len(degrees) if degrees else 0,
            "max_degree": max(degrees) if degrees else 0,
        }

    stats_path = outputs_dir / "mask_graph_stats.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    print(f"Saved statistics    → {stats_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build mask graph from per-object SAM3 3D associations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Dataset name.")
    parser.add_argument(
        "--K", type=int, default=30, help="Min overlap count threshold."
    )
    parser.add_argument("--tau", type=float, default=0.8, help="Min Jaccard threshold.")
    parser.add_argument(
        "--min-points",
        type=int,
        default=1,
        help="Min 3D points for a node to be included.",
    )
    parser.add_argument(
        "--min-points-segment",
        type=int,
        default=10,
        help="Min 3D points in a connected component to be reported.",
    )
    parser.add_argument(
        "--intersection-type",
        choices=["geometric", "id_based"],
        default="geometric",
        help="How to measure instance overlap: 'geometric' (voxel Jaccard) or 'id_based' (point-ID Jaccard).",
    )
    parser.add_argument(
        "--voxel-size-cm",
        type=float,
        default=10.0,
        help="Voxel side length in centimetres (used when --intersection-type=geometric). "
        "Point coordinates are assumed to be in metres.",
    )
    parser.add_argument(
        "--clip-distance-threshold",
        type=float,
        default=0.8,
        metavar="DIST",
        help="Maximum cosine distance between OpenCLIP ViT-H-14 image embeddings of two "
        "nodes' mask crops for them to be allowed to merge.  Range (0, 1].",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    build_object_mask_graph(
        dataset_name=args.dataset,
        K=args.K,
        tau=args.tau,
        min_points=args.min_points,
        min_points_in_3d_segment=args.min_points_segment,
        intersection_type=args.intersection_type,
        voxel_size_cm=args.voxel_size_cm,
        clip_distance_threshold=args.clip_distance_threshold,
    )
