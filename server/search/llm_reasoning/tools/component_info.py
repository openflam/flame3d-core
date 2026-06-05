"""Component info tool."""

from __future__ import annotations

import json
from typing import Any

from server.database import spatial as database
from .utils import _get_dataset_name


def get_component_info(
    component_id: int,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Get detailed information about a specific component.

    Args:
            component_id: The ID of the component to fetch.
            dataset_name: Dataset to search; falls back to DATASET_NAME env var.

    Returns:
            Dict containing component info or an error message.
    """
    resolved_dataset = _get_dataset_name(dataset_name)
    info = database.fetch_component_info(resolved_dataset, component_id)

    if not info:
        return {"error": f"Component with ID {component_id} not found."}

    # Parse bbox_json if possible
    try:
        info["bbox"] = json.loads(info["bbox_json"]) if info.get("bbox_json") else {}
    except Exception:
        info["bbox"] = {}
        
    info.pop("bbox_json", None)

    return info


GET_COMPONENT_INFO_TOOL = {
    "type": "function",
    "name": "get_component_info",
    "description": "Get detailed information (caption, bounding box, etc.) for a specific component by ID.",
    "parameters": {
        "type": "object",
        "properties": {
            "component_id": {
                "type": "integer",
                "description": "The ID of the component.",
            },
        },
        "required": ["component_id"],
        "additionalProperties": False,
    },
}

GET_COMPONENT_INFO_THINKING_TEXT = "Fetching component details..."

GET_COMPONENT_INFO_DESCRIPTION = """- **Component Info Tool:** Use `get_component_info` to retrieve details (caption, bounding box, etc.) for a specific known component by its ID.
  - This is useful when you have a component ID from the user prompt, previous search, or calculation, but need to check its description or properties."""
