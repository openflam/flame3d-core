"""component_view.py – Read-only component + mesh endpoints.

  * ``/api/get_component_info``     – caption + best crop image for one component
  * ``/api/download_all_components`` – every component's id + bbox (annotations)
  * ``/api/load_mesh``              – the dataset's reconstructed mesh (.glb)
"""

from __future__ import annotations

import base64
import json
import re

from flask import Blueprint, current_app, jsonify, request, send_file

from config_io import PATHS, get_data_path
from server.database import spatial as database

component_view_bp = Blueprint("component_view", __name__, url_prefix="/api")

# Dataset names become directory names; keep them path-safe.
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@component_view_bp.route("/get_component_info", methods=["GET"])
def get_component_info():
    """Return the caption and highest-visibility crop image for a component."""
    dataset_name = request.args.get("dataset_name")
    component_id = request.args.get("component_id")

    if not dataset_name:
        return jsonify({"error": "dataset_name query parameter is required"}), 400
    if not component_id:
        return jsonify({"error": "No component_id provided"}), 400
    if not database.check_dataset_exists(dataset_name):
        return jsonify({"error": f"Dataset '{dataset_name}' not found in database."}), 404

    try:
        comp_id_int = int(component_id)
    except ValueError:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    row = database.fetch_component_info(dataset_name, comp_id_int)
    if row is None:
        return jsonify({"error": f"Component ID {component_id} not found"}), 404

    caption = row["caption"] or "No caption available"
    crop_filename = row["best_crop"]

    if not crop_filename:
        return jsonify(
            {
                "component_id": component_id,
                "caption": caption,
                "image_base64": None,
                "message": "No crops available for this component",
            }
        )

    image_path = (
        PATHS["outputs"]
        / dataset_name
        / "crops"
        / f"component_{component_id}"
        / crop_filename
    )

    image_base64 = None
    if image_path.exists():
        try:
            with open(image_path, "rb") as img_file:
                image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            current_app.logger.warning("Error reading crop image %s: %s", image_path, exc)
    else:
        current_app.logger.warning("Crop image not found at %s", image_path)

    return jsonify(
        {
            "component_id": component_id,
            "caption": caption,
            "crop_filename": crop_filename,
            "image_base64": image_base64,
        }
    )


@component_view_bp.route("/download_all_components", methods=["GET"])
def download_all_components():
    """Return every component's id + bbox (used for the annotations overlay)."""
    dataset_name = request.args.get("dataset_name")
    if not dataset_name:
        return jsonify({"error": "dataset_name query parameter is required"}), 400
    if not database.check_dataset_exists(dataset_name):
        return jsonify({"error": f"Dataset '{dataset_name}' not found in database."}), 404

    rows = database.fetch_all_components(dataset_name)

    result = []
    for row in rows:
        if row.get("bbox_json"):
            try:
                bbox = json.loads(row["bbox_json"])
            except json.JSONDecodeError:
                bbox = {}
        else:
            bbox = {}
        result.append({"connected_comp_id": row["component_id"], "bbox": bbox})

    response = jsonify(result)
    response.headers["Content-Disposition"] = "attachment; filename=bbox_corners.json"
    return response


@component_view_bp.route("/load_mesh", methods=["GET"])
def load_mesh():
    """Serve the dataset's reconstructed mesh.

    Prefers ``outputs/<name>/mesh.glb`` (produced by the processing pipeline),
    falling back to ``outputs/<name>/raw.glb`` and the raw polycam export.
    """
    dataset_name = request.args.get("dataset_name")
    if not dataset_name:
        return jsonify({"error": "dataset_name query parameter is required"}), 400
    if not _VALID_NAME.match(dataset_name):
        return jsonify({"error": "invalid dataset_name"}), 400

    candidates = [
        PATHS["outputs"] / dataset_name / "mesh.glb",
        PATHS["outputs"] / dataset_name / "raw.glb",
        get_data_path(dataset_name) / "polycam_data" / "raw.glb",
    ]
    for mesh_path in candidates:
        if mesh_path.is_file():
            return send_file(mesh_path, mimetype="model/gltf-binary")

    return jsonify({"error": f"Mesh file not found for dataset {dataset_name}"}), 404
