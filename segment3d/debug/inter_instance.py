"""
inter_instance.py – Find and display a path between two instances within a
connected component.

Reads the ``edges`` field from connected_components.json for the given
component, builds an undirected graph, and finds the shortest hop-count path
from *source_inst_id* to *dest_inst_id* using BFS.

Each edge of the path is annotated with its stored Jaccard similarity and
CLIP distance.

Usage:
    python debug/inter_instance.py --dataset_name ProjectLabStudio_inv_method \\
        --component_id 0 --source_inst_id Boxes_seq_0_1 --dest_inst_id box_seq_1_7
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from config_io import get_output_path


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Show the shortest path between two instances in a connected component."
    )
    p.add_argument(
        "--dataset_name",
        required=True,
        help="Dataset name (folder under data/ and outputs/).",
    )
    p.add_argument(
        "--component_id",
        required=True,
        type=int,
        help="Connected-component ID to inspect.",
    )
    p.add_argument("--source_inst_id", required=True, help="Starting instance ID.")
    p.add_argument("--dest_inst_id", required=True, help="Destination instance ID.")
    return p.parse_args()


# ── Graph helpers ─────────────────────────────────────────────────────────────

# Edge attributes keyed by (node1, node2) with node1 < node2 (lexicographic).
EdgeKey = Tuple[str, str]


def _edge_key(a: str, b: str) -> EdgeKey:
    return (a, b) if a < b else (b, a)


def build_graph(
    edges: List[Dict],
) -> Tuple[Dict[str, List[str]], Dict[EdgeKey, Dict]]:
    """Return an adjacency list and an edge-attribute lookup."""
    adj: Dict[str, List[str]] = defaultdict(list)
    attrs: Dict[EdgeKey, Dict] = {}

    for e in edges:
        n1, n2 = e["node1"], e["node2"]
        adj[n1].append(n2)
        adj[n2].append(n1)
        key = _edge_key(n1, n2)
        attrs[key] = {k: v for k, v in e.items() if k not in ("node1", "node2")}

    return adj, attrs


def bfs_path(
    adj: Dict[str, List[str]],
    source: str,
    dest: str,
) -> Optional[List[str]]:
    """BFS shortest-hop path; returns list of node IDs or None if unreachable."""
    if source == dest:
        return [source]

    visited = {source}
    queue: deque[List[str]] = deque([[source]])

    while queue:
        path = queue.popleft()
        current = path[-1]
        for neighbour in adj.get(current, []):
            if neighbour == dest:
                return path + [neighbour]
            if neighbour not in visited:
                visited.add(neighbour)
                queue.append(path + [neighbour])

    return None


# ── Display ───────────────────────────────────────────────────────────────────

# ANSI colours
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def _fmt_attrs(attrs: Dict) -> str:
    parts = []
    if "jaccard" in attrs:
        parts.append(f"jaccard={attrs['jaccard']:.4f}")
    if "clip_distance" in attrs:
        parts.append(f"clip_dist={attrs['clip_distance']:.6f}")
    for k, v in attrs.items():
        if k not in ("jaccard", "clip_distance"):
            parts.append(f"{k}={v}")
    return "  ".join(parts)


def print_path(
    path: List[str],
    attrs: Dict[EdgeKey, Dict],
    component_id: int,
) -> None:
    hops = len(path) - 1
    print()
    print(
        f"{_BOLD}Component {component_id}{_RESET}  –  "
        f"path length {_BOLD}{hops}{_RESET} hop{'s' if hops != 1 else ''}"
    )
    print()

    col_w = max(len(n) for n in path)

    for i, node in enumerate(path):
        # Node line
        marker = (
            f"{_GREEN}▶{_RESET}"
            if i == 0
            else (f"{_GREEN}★{_RESET}" if i == len(path) - 1 else " ")
        )
        print(f"  {marker} {_BOLD}{node:{col_w}}{_RESET}")

        # Edge line (between this node and the next)
        if i < len(path) - 1:
            key = _edge_key(node, path[i + 1])
            edge_info = _fmt_attrs(attrs.get(key, {}))
            print(f"  {_DIM}  │  {edge_info}{_RESET}")

    print()


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()

    outputs_dir = get_output_path(args.dataset_name)
    components_path = outputs_dir / "connected_components.json"

    if not components_path.exists():
        sys.exit(f"ERROR: {components_path} not found.")

    with components_path.open("r", encoding="utf-8") as fh:
        components = json.load(fh)

    # Find the target component
    component = next(
        (c for c in components if c["connected_comp_id"] == args.component_id),
        None,
    )
    if component is None:
        sys.exit(
            f"ERROR: component_id {args.component_id} not found in "
            f"{components_path}"
        )

    instance_ids: List[str] = component.get("instance_ids", [])
    edges: List[Dict] = component.get("edges", [])

    # Validate requested nodes
    for iid, label in [
        (args.source_inst_id, "source_inst_id"),
        (args.dest_inst_id, "dest_inst_id"),
    ]:
        if iid not in instance_ids:
            print(
                f"WARNING: {label} '{iid}' is not listed in component "
                f"{args.component_id}'s instance_ids."
            )

    if not edges:
        sys.exit(f"ERROR: component {args.component_id} has no edges stored.")

    adj, attr_map = build_graph(edges)

    path = bfs_path(adj, args.source_inst_id, args.dest_inst_id)

    if path is None:
        print(
            f"\nNo path found between '{args.source_inst_id}' and "
            f"'{args.dest_inst_id}' in component {args.component_id}.\n"
            f"They may belong to disconnected sub-graphs within the component."
        )
        return

    print_path(path, attr_map, args.component_id)


if __name__ == "__main__":
    main()
