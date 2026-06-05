"""search.py – Scene query endpoints.

Two query paths are exposed (mirroring the integrated query UI):

  * ``/api/search_stream`` – tool-calling LLM agent, streamed over SSE. This is
    the primary path used by the frontend ("Tools").
  * ``/api/search``        – one-shot BM25 keyword search.

The original scan-to-map server also offered full-context and RAG OpenAI
searches, CLIP image search, robot step-planning, and directions; those are
intentionally not integrated here.
"""

from __future__ import annotations

import json
import queue
import threading
import time

from flask import Blueprint, current_app, jsonify, request

from server.database import spatial as database
from server.search.process_query import process_query
from server.search.semantic_search import BM25Provider
from server.search.transforms import transform_bbox

search_bp = Blueprint("search", __name__, url_prefix="/api")

# Default model used by the tool-calling agent when the client doesn't specify
# one. Matches the original search server.
DEFAULT_TOOLS_MODEL = "gpt-5.4"


@search_bp.route("/get_providers_list", methods=["GET"])
def get_providers_list():
    """Return the available query methods. Only Tools + BM25 are integrated."""
    return jsonify({"providers": ["Tools", "BM25"]})


@search_bp.route("/search", methods=["POST"])
def search():
    """One-shot BM25 search returning the matched components and their bboxes."""
    dataset_name = request.json.get("dataset_name")
    search_query = request.json.get("query")
    method = request.json.get("method", "BM25")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400
    if not search_query or len(search_query) == 0:
        return jsonify({"error": "No query provided"}), 400
    if method != "BM25":
        return (
            jsonify({"error": f"Unsupported method '{method}'. Only 'BM25' is supported by /search."}),
            400,
        )
    if not database.check_dataset_exists(dataset_name):
        return jsonify({"error": f"Dataset '{dataset_name}' not found in database."}), 404

    # Only text queries are supported.
    query_item = search_query[0]
    if query_item.get("type") != "text":
        return jsonify({"error": "BM25 search only supports text queries"}), 400
    query_value = query_item.get("value")
    if not query_value:
        return jsonify({"error": "Invalid query format"}), 400

    provider = BM25Provider(dataset_name)
    result_data = process_query(query_value, dataset_name, provider)

    bboxes = result_data["bbox"]
    component_ids = result_data["component_ids"]
    custom_bboxes = result_data.get("custom_bboxes", [])

    # Fetch fresh captions for the matched components.
    if component_ids:
        caption_rows = database.fetch_components_by_ids(dataset_name, component_ids)
        id_to_caption = {row["component_id"]: row["caption"] for row in caption_rows}
    else:
        id_to_caption = {}

    components = []
    for bbox, comp_id in zip(bboxes, component_ids):
        components.append(
            {
                "bbox": transform_bbox(bbox),
                "caption": id_to_caption.get(comp_id, "No caption available"),
                "component_id": str(comp_id),
            }
        )

    transformed_custom_bboxes = []
    for custom_bbox in custom_bboxes:
        if isinstance(custom_bbox, list):
            bbox_dict = {"corners": custom_bbox}
        elif isinstance(custom_bbox, dict) and "corners" in custom_bbox:
            bbox_dict = custom_bbox
        else:
            continue
        transformed_custom_bboxes.append(transform_bbox(bbox_dict))

    return jsonify(
        {
            "reason": result_data["reason"],
            "search_time_ms": result_data["search_time_ms"],
            "components": components,
            "custom_bboxes": transformed_custom_bboxes,
        }
    )


@search_bp.route("/search_stream", methods=["POST"])
def search_stream():
    """Stream the tool-calling agent's reasoning + final result over SSE."""
    # Import lazily so the (heavy) litellm/agent stack is only loaded when a
    # streaming search actually runs, not at every app boot.
    from server.search.llm_reasoning.llm_agent import LLMAgent

    dataset_name = request.json.get("dataset_name")
    search_query = request.json.get("query")
    model_name = request.json.get("model_name") or request.json.get("model")
    active_model = model_name or DEFAULT_TOOLS_MODEL
    tools = request.json.get("tools")

    if not dataset_name:
        return jsonify({"error": "dataset_name is required"}), 400
    if not search_query or len(search_query) == 0:
        return jsonify({"error": "No query provided"}), 400
    if not database.check_dataset_exists(dataset_name):
        return jsonify({"error": f"Dataset '{dataset_name}' not found in database."}), 404

    query_item = search_query[0]
    if query_item.get("type") != "text":
        return jsonify({"error": "Streaming only supports text queries"}), 400
    query_input = query_item.get("value")
    if not query_input:
        return jsonify({"error": f"No query string provided for {active_model}"}), 400

    q: queue.Queue = queue.Queue()

    def on_stream_event(event):
        q.put({"type": "event", "data": event})

    def run_agent():
        try:
            agent = LLMAgent(model=active_model, allowed_tools=tools)
            result = agent.answer_query_stream(
                query=query_input,
                dataset_name=dataset_name,
                on_stream_event=on_stream_event,
            )
            q.put({"type": "result", "data": result})
        except Exception as exc:  # noqa: BLE001
            q.put({"type": "error", "error": str(exc)})

    threading.Thread(target=run_agent, daemon=True).start()

    def generate():
        start_time = time.perf_counter()
        while True:
            item = q.get()
            if item["type"] == "event":
                yield f"data: {json.dumps(item['data'])}\n\n"
            elif item["type"] == "error":
                yield f"data: {json.dumps({'type': 'error', 'error': item['error']})}\n\n"
                break
            elif item["type"] == "result":
                result = item["data"]
                search_time_ms = (time.perf_counter() - start_time) * 1000

                component_ids = result.get("component_ids", [])
                custom_bboxes = result.get("custom_bboxes", [])
                reason = result.get("reason", "")

                valid_bboxes = []
                valid_component_ids = []
                bbox_map: dict = {}

                if component_ids:
                    rows = database.fetch_components_by_ids(dataset_name, component_ids)
                    for row in rows:
                        try:
                            bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else {}
                        except json.JSONDecodeError:
                            bbox = {}
                        bbox_map[row["component_id"]] = {"bbox": bbox, "caption": row["caption"]}

                    for comp_id in component_ids:
                        if comp_id in bbox_map:
                            valid_bboxes.append(bbox_map[comp_id]["bbox"])
                            valid_component_ids.append(comp_id)

                if not valid_bboxes:
                    current_app.logger.warning(
                        "No valid component IDs found for %s; using first component.",
                        active_model,
                    )
                    row = database.fetch_first_component(dataset_name)
                    if row:
                        try:
                            bbox = json.loads(row["bbox_json"]) if row["bbox_json"] else {}
                        except json.JSONDecodeError:
                            bbox = {}
                        valid_bboxes = [bbox]
                        valid_component_ids = [row["component_id"]]
                        bbox_map = {row["component_id"]: {"caption": row["caption"]}}

                components = []
                for bbox, comp_id in zip(valid_bboxes, valid_component_ids):
                    components.append(
                        {
                            "bbox": transform_bbox(bbox),
                            "caption": bbox_map.get(comp_id, {}).get("caption") or "No caption available",
                            "component_id": str(comp_id),
                        }
                    )

                transformed_custom_bboxes = []
                for custom_bbox in custom_bboxes:
                    if isinstance(custom_bbox, list):
                        bbox_dict = {"corners": custom_bbox}
                    elif isinstance(custom_bbox, dict) and "corners" in custom_bbox:
                        bbox_dict = custom_bbox
                    else:
                        continue
                    transformed_custom_bboxes.append(transform_bbox(bbox_dict))

                final_result = {
                    "reason": reason,
                    "search_time_ms": search_time_ms,
                    "components": components,
                    "custom_bboxes": transformed_custom_bboxes,
                }
                yield f"data: {json.dumps({'type': 'result', 'data': final_result})}\n\n"
                break

    return current_app.response_class(generate(), mimetype="text/event-stream")
