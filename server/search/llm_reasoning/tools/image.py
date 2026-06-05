"""Image tool."""

from __future__ import annotations

import base64
import json
from typing import Any

from config_io import PATHS
from .utils import _get_dataset_name


def get_images(
    component_id: int,
    num_images: int,
    dataset_name: str | None = None,
) -> dict:
    """
    Get the top N crop images for a component.

    Args:
        component_id: The ID of the component.
        num_images: The number of top images to return.
        dataset_name: Dataset to search; falls back to DATASET_NAME env var.

    Returns:
        A dictionary containing an 'images' list with Base64 encoded images.
    """
    resolved_dataset = _get_dataset_name(dataset_name)
    outputs_dir = PATHS["outputs"] / resolved_dataset
    manifest_path = outputs_dir / "crops" / "manifest.json"

    if not manifest_path.exists():
        return {"error": f"manifest.json not found for dataset {resolved_dataset}"}

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception as e:
        return {"error": f"Failed to load manifest: {e}"}

    comp_str = str(component_id)
    if comp_str not in manifest:
        return {"error": f"Component ID {component_id} not found in manifest."}

    component_data: dict[str, Any] = manifest[comp_str]
    crops: list[dict[str, Any]] = component_data.get("crops", [])

    # Sort crops by fraction_visible descending
    top_crops = sorted(crops, key=lambda x: x.get("fraction_visible", 0.0), reverse=True)[:num_images]

    images = []
    for crop in top_crops:
        crop_filename = crop.get("crop_filename")
        if not crop_filename:
            continue
            
        image_path = outputs_dir / "crops" / f"component_{component_id}" / crop_filename
        if image_path.exists():
            try:
                with open(image_path, "rb") as img_file:
                    image_data = img_file.read()
                    image_base64 = base64.b64encode(image_data).decode("utf-8")
                    images.append(f"data:image/jpeg;base64,{image_base64}")
            except Exception as e:
                pass

    return {"images": images}


GET_IMAGES_TOOL = {
    "type": "function",
    "name": "get_images",
    "description": "Get the top N views of a specific component. The images show the component from different angles. Returns a dictionary containing an 'images' list with Base64 encoded data URLs.",
    "parameters": {
        "type": "object",
        "properties": {
            "component_id": {
                "type": "integer",
                "description": "The ID of the component to get images for.",
            },
            "num_images": {
                "type": "integer",
                "description": "The number of images to return.",
            },
        },
        "required": ["component_id", "num_images"],
        "additionalProperties": False,
    },
}

GET_IMAGES_THINKING_TEXT = "Retrieving images of component..."

GET_IMAGES_DESCRIPTION = """- **Image Tool:** Use `get_images` to get the top N views of a specific component. The images show the component from different angles. Returns a dictionary containing an 'images' list with Base64 encoded data URLs. This can be used to look at the appearance of the component like its color, shape, brand, etc. It can also be used to see the state of the component (e.g. if it is open or closed, on or off, messy or tidy, etc.). Generally, 2-3 images are enough to get a good sense of the component."""
