"""query.py – Scan querying + 3D viewer endpoints.

These back the "Query" UI (the scan-to-map viewer integrated into the
frontend). Only ``/api/load_mesh`` is wired to real data for now — it serves
the ``mesh.glb`` produced by the data-processing pipeline. The remaining
routes are placeholders returning empty/echo responses so the UI is fully
navigable; the real tool-based search backend will be plugged in later.

Routes
------
GET  /api/get_providers_list          → available query methods (always Tools)
POST /api/search_stream               → SSE stream of thinking/result events
GET  /api/load_mesh?dataset_name=...  → the dataset's mesh.glb (real)
GET  /api/get_component_info          → metadata for a single component
GET  /api/download_all_components     → all annotated components
POST /api/update_component            → edit a component's caption/bbox
DELETE /api/delete_component          → remove a component
POST /api/add_component               → add a custom component
"""

from __future__ import annotations

import json
import re

from flask import Blueprint, Response, jsonify, request, send_file

from config_io import PATHS

query_bp = Blueprint("query", __name__, url_prefix="/api")

# Dataset names map to directory names; keep them to a safe character set so a
# crafted name can't escape the outputs/ directory.
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@query_bp.route("/get_providers_list", methods=["GET"])
def get_providers_list():
    """Only tool-based querying is supported."""
    return jsonify({"providers": ["Tools"]}), 200


@query_bp.route("/search_stream", methods=["POST"])
def search_stream():
    """Placeholder Server-Sent-Events stream.

    Emits a short 'thinking' message followed by an empty 'result'. The real
    tool-calling search backend will replace this.
    """
    body = request.get_json(silent=True) or {}
    dataset_name = body.get("dataset_name", "")

    def generate():
        thinking = {
            "type": "thinking",
            "content": "Tool-based search is not wired up yet.\n",
        }
        yield f"data: {json.dumps(thinking)}\n\n"

        result = {
            "type": "result",
            "data": {
                "reason": (
                    f"Search for dataset '{dataset_name}' is a placeholder. "
                    "The tool-based search backend will be integrated later."
                ),
                "search_time_ms": 0,
                "components": [],
                "custom_bboxes": [],
            },
        }
        yield f"data: {json.dumps(result)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@query_bp.route("/load_mesh", methods=["GET"])
def load_mesh():
    """Serve the dataset's reconstructed mesh (``outputs/<name>/mesh.glb``)."""
    name = request.args.get("dataset_name", "")
    if not _VALID_NAME.match(name):
        return jsonify({"error": "invalid dataset_name"}), 400

    mesh_path = PATHS["outputs"] / name / "mesh.glb"
    if not mesh_path.is_file():
        return jsonify({"error": f"mesh not found for dataset '{name}'"}), 404

    return send_file(mesh_path, mimetype="model/gltf-binary")


@query_bp.route("/get_component_info", methods=["GET"])
def get_component_info():
    """Placeholder component metadata."""
    component_id = request.args.get("component_id", "")
    return (
        jsonify(
            {
                "component_id": component_id,
                "caption": "",
                "image_name": "",
                "image_base64": None,
                "fraction_visible": 0,
                "image_width": 0,
                "image_height": 0,
            }
        ),
        200,
    )


@query_bp.route("/download_all_components", methods=["GET"])
def download_all_components():
    """Placeholder: no annotated components yet."""
    return jsonify([]), 200


@query_bp.route("/update_component", methods=["POST"])
def update_component():
    """Placeholder: echo the update back."""
    body = request.get_json(silent=True) or {}
    return (
        jsonify(
            {
                "component_id": body.get("component_id"),
                "caption": body.get("caption"),
                "bbox": body.get("bbox"),
            }
        ),
        200,
    )


@query_bp.route("/delete_component", methods=["DELETE"])
def delete_component():
    """Placeholder: pretend the delete succeeded."""
    body = request.get_json(silent=True) or {}
    return jsonify({"component_id": body.get("component_id"), "deleted": True}), 200


@query_bp.route("/add_component", methods=["POST"])
def add_component():
    """Placeholder: echo a created component."""
    body = request.get_json(silent=True) or {}
    return (
        jsonify(
            {
                "component_id": "placeholder",
                "caption": body.get("caption", ""),
                "bbox": body.get("bbox"),
                "best_crop": "",
                "added": True,
            }
        ),
        200,
    )
