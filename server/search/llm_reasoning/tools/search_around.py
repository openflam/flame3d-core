"""Search around component tool."""

from __future__ import annotations

from typing import Any
import json

from server.database import spatial as database
from .utils import _get_dataset_name


def _get_bbox_json(bbox_str: str | dict | None) -> dict:
    if isinstance(bbox_str, dict):
        return bbox_str
    if isinstance(bbox_str, str) and bbox_str:
        try:
            return json.loads(bbox_str)
        except Exception:
            return {}
    return {}


def _is_within_xy_limits(target_bbox: dict, n_bbox: dict) -> bool:
    target_min = target_bbox.get("min")
    target_max = target_bbox.get("max")
    n_min = n_bbox.get("min")
    n_max = n_bbox.get("max")
    
    if target_min and target_max and n_min and n_max:
        return (target_min[0] <= n_max[0] and target_max[0] >= n_min[0] and
                target_min[1] <= n_max[1] and target_max[1] >= n_min[1])
    
    return True


def search_around_component(
    component_id: int,
    radius: float,
    search_term: str | None = None,
    direction: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Find components within a given radius using the spatial index,
    optionally filtered by a BM25 search term.

    Args:
        component_id: The ID of the central component.
        radius: The search radius in meters.
        search_term: Optional text to filter the neighbors by.
        dataset_name: Dataset to search; falls back to DATASET_NAME env var.

    Returns:
        Dict returning matched component IDs and a reason string.
    """
    resolved_dataset = _get_dataset_name(dataset_name)


    # Get all components in a radius around the selected component
    neighbors = database.fetch_components_in_radius(resolved_dataset, component_id, radius)
    
    if direction in ("above", "below"):
        target_info = database.fetch_component_info(resolved_dataset, component_id)
        target_z = None
        if target_info:
            target_bbox = _get_bbox_json(target_info.get("bbox_json"))
            if "center" in target_bbox:
                target_z = target_bbox["center"][2]
                
        if target_z is not None:
            filtered_neighbors = []
            for n in neighbors:
                n_bbox = _get_bbox_json(n.get("bbox_json"))
                if "center" in n_bbox:
                    n_z = n_bbox["center"][2]
                    
                    if direction == "above" and n_z <= target_z:
                        continue
                    if direction == "below" and n_z >= target_z:
                        continue
                        
                    if not _is_within_xy_limits(target_bbox, n_bbox):
                        continue
                        
                    filtered_neighbors.append(n)
            neighbors = filtered_neighbors
    if not neighbors:
        return {
            "components": [],
            "reason": f"No neighboring components found within {radius} meters."
        }
        
    neighbor_ids = [n["component_id"] for n in neighbors]

    if search_term and search_term.strip():
        result = database.bm25_search(
            dataset_name=resolved_dataset,
            search_terms=[search_term],
            top_k=len(neighbor_ids),
            apply_elbow=False,
            candidate_ids=neighbor_ids
        )
        
        components = []
        for r in result.get("results", []):
            components.append({
                "id": r["component_id"],
                "caption": r.get("caption", ""),
                "bbox": r.get("bbox", {})
            })
            
        return {
            "components": components,
            "reason": f"Filtered {len(neighbor_ids)} neighbors using BM25. " + result.get("reason", "")
        }
    else:
        components = []
        for n in neighbors:
            components.append({
                "id": n["component_id"],
                "caption": n.get("caption", ""),
                "bbox": _get_bbox_json(n.get("bbox_json"))
            })
            
        return {
            "components": components,
            "reason": f"Found {len(neighbor_ids)} neighboring components within {radius}m."
        }


SEARCH_AROUND_COMPONENT_TOOL = {
    "type": "function",
    "name": "search_around_component",
    "description": "Find components within a specific radius of a target component, optionally filtered by a search term.",
    "parameters": {
        "type": "object",
        "properties": {
            "component_id": {
                "type": "integer",
                "description": "The ID of the target component to search around.",
            },
            "radius": {
                "type": "number",
                "description": "The search radius in meters.",
            },
            "search_term": {
                "type": "string",
                "description": "Optional text to filter the neighboring components by.",
            },
            "direction": {
                "type": "string",
                "description": "Optional direction to filter components relative to the target ('above' or 'below').",
                "enum": ["above", "below"]
            },
        },
        "required": ["component_id", "radius"],
        "additionalProperties": False,
    },
}

SEARCH_AROUND_COMPONENT_THINKING_TEXT = "Searching for objects around a target object..."

SEARCH_AROUND_COMPONENT_DESCRIPTION = """- **Search Around Tool:** Use `search_around_component` to find objects located within a specified radius of a known component.
  - You can optionally filter these neighboring components using a `search_term`. When providing a search term, apply the same **BM25 Creative Expansion** rule as above (e.g., use a descriptive, creatively expanded string to catch synonyms or related items).
  - You can optionally filter the neighbors vertically using the `direction` argument (either "above" or "below") if you specifically want components located higher or lower than the known component."""
