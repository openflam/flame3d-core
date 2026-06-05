"""component_edit.py – Component mutation endpoints.

  * ``/api/update_component`` – edit a component's caption and/or bbox
  * ``/api/add_component``    – create a new component (optionally with an image)
  * ``/api/delete_component`` – remove a component

Incoming bboxes are in the Model3DViewer frame and are converted back to the
COLMAP frame before being persisted.
"""

from __future__ import annotations

import base64
import json

from flask import Blueprint, current_app, jsonify, request

from config_io import PATHS
from server.database import spatial as database
from server.search.transforms import transform_bbox

component_edit_bp = Blueprint("component_edit", __name__, url_prefix="/api")


@component_edit_bp.route("/update_component", methods=["POST"])
def update_component():
    """Update the caption and/or bbox of an existing component."""
    data = request.json
    dataset_name = data.get("dataset_name")
    component_id = data.get("component_id")
    new_caption = data.get("caption")
    new_bbox = data.get("bbox")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400
    if not database.check_dataset_exists(dataset_name):
        return jsonify({"error": f"Dataset '{dataset_name}' not found in database."}), 404
    if not component_id:
        return jsonify({"error": "No component_id provided"}), 400
    if new_caption is None and new_bbox is None:
        return jsonify({"error": "At least one of 'caption' or 'bbox' must be provided"}), 400
    if new_bbox is not None and "corners" not in new_bbox:
        return jsonify({"error": "bbox must have 'corners' key"}), 400

    if new_bbox is not None:
        new_bbox = transform_bbox(new_bbox, transform_type="3DViewer_to_COLMAP")

    try:
        comp_id_int = int(component_id)
    except (ValueError, TypeError):
        return jsonify({"error": f"Invalid component_id: {component_id}"}), 400

    updated = database.update_component(dataset_name, comp_id_int, new_caption, new_bbox)
    if updated == 0:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    result = {"component_id": component_id}
    if new_caption is not None:
        result["caption"] = new_caption
    if new_bbox is not None:
        result["bbox"] = new_bbox
    return jsonify(result)


@component_edit_bp.route("/add_component", methods=["POST"])
def add_component():
    """Create a new component with an auto-generated id."""
    data = request.json
    dataset_name = data.get("dataset_name")
    caption = data.get("caption", "")
    bbox = data.get("bbox", {})
    image_base64 = data.get("image_base64")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400
    if not database.check_dataset_exists(dataset_name):
        return jsonify({"error": f"Dataset '{dataset_name}' not found in database."}), 404
    if bbox and "corners" not in bbox:
        return jsonify({"error": "bbox must have 'corners' key if provided"}), 400

    if bbox:
        bbox = transform_bbox(bbox, transform_type="3DViewer_to_COLMAP")

    component_id = database.get_next_component_id(dataset_name)

    best_crop = ""
    if image_base64:
        try:
            crops_base_dir = PATHS["outputs"] / dataset_name / "crops"
            crops_dir = crops_base_dir / f"component_{component_id}"
            crops_dir.mkdir(parents=True, exist_ok=True)

            best_crop = "user_upload.jpg"
            image_path = crops_dir / best_crop
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_base64))

            # Record the crop in the manifest so get_component_info can find it.
            manifest_path = crops_base_dir / "manifest.json"
            manifest_data = {}
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r") as f:
                        manifest_data = json.load(f)
                except Exception as exc:  # noqa: BLE001
                    current_app.logger.warning("Could not read manifest.json: %s", exc)

            manifest_data[str(component_id)] = {
                "crops": [{"crop_filename": best_crop, "fraction_visible": 1.0}]
            }
            with open(manifest_path, "w") as f:
                json.dump(manifest_data, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            current_app.logger.warning(
                "Failed to save image for component %s: %s", component_id, exc
            )

    added = database.add_row(dataset_name, component_id, caption, bbox, best_crop)
    if added == 0:
        return jsonify({"error": "Failed to add component"}), 500

    return jsonify(
        {
            "component_id": str(component_id),
            "caption": caption,
            "bbox": bbox,
            "best_crop": best_crop,
            "added": True,
        }
    )


@component_edit_bp.route("/delete_component", methods=["DELETE"])
def delete_component():
    """Delete a component by id."""
    data = request.json
    dataset_name = data.get("dataset_name")
    component_id = data.get("component_id")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400
    if not component_id:
        return jsonify({"error": "No component_id provided"}), 400
    if not database.check_dataset_exists(dataset_name):
        return jsonify({"error": f"Dataset '{dataset_name}' not found in database."}), 404

    try:
        comp_id_int = int(component_id)
    except (ValueError, TypeError):
        return jsonify({"error": f"Invalid component_id: {component_id}"}), 400

    deleted = database.delete_component(dataset_name, comp_id_int)
    if deleted == 0:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    return jsonify({"component_id": component_id, "deleted": True})
