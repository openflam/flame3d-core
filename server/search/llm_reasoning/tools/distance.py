"""Distance tool."""

from __future__ import annotations

import json
import math

from server.database import spatial as database
from .utils import _get_dataset_name


def get_distance(
    component_id_1: int,
    component_id_2: int,
    dataset_name: str | None = None,
) -> dict[str, float]:
    """
    Calculate the Euclidean distance between the centers of two components.

    Args:
            component_id_1: The ID of the first component.
            component_id_2: The ID of the second component.
            dataset_name: Dataset to search; falls back to DATASET_NAME env var.

    Returns:
            The distance in meters as a float.
    """
    resolved_dataset = _get_dataset_name(dataset_name)
    components = database.fetch_components_by_ids(resolved_dataset, [component_id_1, component_id_2])

    if len(components) != 2:
        raise ValueError("Could not find both components in the database.")

    comp1 = next((c for c in components if c["component_id"] == component_id_1), None)
    comp2 = next((c for c in components if c["component_id"] == component_id_2), None)

    if not comp1 or not comp2:
        raise ValueError("Could not find both components in the database.")

    try:
        bbox1 = json.loads(comp1["bbox_json"]) if comp1["bbox_json"] else {}
        bbox2 = json.loads(comp2["bbox_json"]) if comp2["bbox_json"] else {}
    except json.JSONDecodeError:
        raise ValueError("Invalid bbox JSON in the database.")

    corners1 = bbox1.get("corners")
    corners2 = bbox2.get("corners")

    if not corners1 or not corners2:
        raise ValueError("Missing corners in component bounding box.")

    def get_center(corners):
        return [
            sum(c[0] for c in corners) / len(corners),
            sum(c[1] for c in corners) / len(corners),
            sum(c[2] for c in corners) / len(corners),
        ]

    center1 = get_center(corners1)
    center2 = get_center(corners2)

    distance = math.sqrt(
        (center1[0] - center2[0]) ** 2 +
        (center1[1] - center2[1]) ** 2 +
        (center1[2] - center2[2]) ** 2
    )

    return {"distance": distance}


GET_DISTANCE_TOOL = {
    "type": "function",
    "name": "get_distance",
    "description": "Calculate the Euclidean distance between the centers of two components' bounding boxes in meters.",
    "parameters": {
        "type": "object",
        "properties": {
            "component_id_1": {
                "type": "integer",
                "description": "The ID of the first component.",
            },
            "component_id_2": {
                "type": "integer",
                "description": "The ID of the second component.",
            },
        },
        "required": ["component_id_1", "component_id_2"],
        "additionalProperties": False,
    },
}

GET_DISTANCE_THINKING_TEXT = "Calculating distance between objects..."

GET_DISTANCE_DESCRIPTION = """- **Distance Evaluation Tool:** Use `get_distance` to compute the spatial Euclidean distance (in meters) between any two known components in the scene.
  - Use this tool when the query requires comparing spatial proximity, answering questions about how far apart two items are, or verifying if two objects are close to each other."""
