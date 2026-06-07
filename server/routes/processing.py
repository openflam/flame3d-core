"""processing.py – Upload + data-processing endpoints.

Routes
------
GET  /api/config              → default_config.json contents (for pre-filling)
GET  /api/config/schema       → config_schema.json (field types + allowed values)
GET  /api/steps?source=...    → ordered pipeline steps for a data source
POST /api/upload              → save uploaded zip + config, start pipeline
GET  /api/jobs/<job_id>       → live status of a running/finished job
GET    /api/datasets               → all datasets + their index status
GET    /api/datasets/<name>        → a single dataset's index row
DELETE /api/datasets/<name>        → soft-delete (mark_delete); files purged later
GET    /api/datasets/<name>/config → config used to process the dataset
POST   /api/datasets/<name>/reprocess → re-run (in place or as a named copy)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from flask import Blueprint, jsonify, request

from config_io import PATHS, get_data_path
from server.database import (
    get_dataset,
    list_datasets,
    mark_deleted,
    reconcile_existing,
)
from server.utils.job_manager import job_manager

processing_bp = Blueprint("processing", __name__, url_prefix="/api")

# Default config + its schema, shipped with the server.
_SERVER_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _SERVER_DIR / "default_config.json"
_CONFIG_SCHEMA_PATH = _SERVER_DIR / "config_schema.json"

# Dataset names become directory names under data/ and outputs/, so keep them
# to a safe character set (no path separators, no "..").
_VALID_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _valid_dataset_name(name: str) -> bool:
    return bool(name) and ".." not in name and _VALID_NAME.match(name) is not None


@processing_bp.route("/config", methods=["GET"])
def get_default_config():
    """Return the default_config.json so the UI can pre-fill the form."""
    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return jsonify(config), 200


@processing_bp.route("/config/schema", methods=["GET"])
def get_config_schema():
    """Return config_schema.json so the UI can render the config form."""
    with open(_CONFIG_SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema = json.load(f)
    return jsonify(schema), 200


@processing_bp.route("/steps", methods=["GET"])
def list_steps():
    """Return the ordered pipeline steps for the given data source."""
    from server.utils.pipeline_steps import get_pipeline_steps

    source = request.args.get("source", "polycam")
    try:
        steps = get_pipeline_steps(source)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify([{"name": s.name, "description": s.description} for s in steps]), 200


@processing_bp.route("/upload", methods=["POST"])
def upload_and_process():
    """Save the uploaded zip + config to ``data/<dataset>/`` and start the run.

    Expects a ``multipart/form-data`` body with:
      • ``file``   — the Polycam raw-data export zip
      • ``config`` — the master config as a JSON string
    """
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    upload = request.files["file"]
    if not upload.filename:
        return jsonify({"error": "No file selected"}), 400

    raw_config = request.form.get("config")
    if not raw_config:
        return jsonify({"error": "Missing config field"}), 400

    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        return jsonify({"error": f"Invalid config JSON: {exc}"}), 400

    dataset_name = config.get("dataset_name")
    if not dataset_name:
        return jsonify({"error": "config.dataset_name is required"}), 400

    # Persist the upload as data/<dataset>/input.zip — the path the polycam
    # processor expects.
    data_dir = get_data_path(dataset_name)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "input.zip"
    upload.save(str(zip_path))

    # A fresh upload should reprocess from scratch — drop any stale extraction.
    polycam_dir = data_dir / "polycam_data"
    if polycam_dir.exists():
        import shutil

        shutil.rmtree(polycam_dir)

    # Write the config next to the data so subprocess steps can read it.
    config_path = data_dir / "master_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    job_id = job_manager.create_job(config, str(config_path))

    return jsonify({"job_id": job_id}), 202


@processing_bp.route("/jobs/<job_id>", methods=["GET"])
def get_job_status(job_id: str):
    """Return a snapshot of the job's progress."""
    job = job_manager.get_job(job_id)
    if job is None:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job), 200


@processing_bp.route("/datasets", methods=["GET"])
def get_datasets():
    """Return all datasets and their status.

    On-disk dataset directories that predate the index are reconciled in as
    ``complete`` so they appear in the list too.
    """
    data_root = PATHS["data"]
    if data_root.is_dir():
        existing = [p.name for p in data_root.iterdir() if p.is_dir()]
        reconcile_existing(existing)
    return jsonify(list_datasets()), 200


@processing_bp.route("/datasets/<name>", methods=["GET"])
def get_dataset_status(name: str):
    """Return a single dataset's index row."""
    dataset = get_dataset(name)
    if dataset is None:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify(dataset), 200


@processing_bp.route("/datasets/<name>", methods=["DELETE"])
def delete_dataset(name: str):
    """Soft-delete a dataset.

    Marks the row ``marked_delete`` so it disappears from the UI immediately.
    The on-disk files under data/ and outputs/ are left in place and purged
    later by the ``clean_storage`` script.
    """
    if not mark_deleted(name):
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify({"dataset_name": name, "status": "marked_delete"}), 200


@processing_bp.route("/datasets/<name>/config", methods=["GET"])
def get_dataset_config(name: str):
    """Return the config that was used to process *name*.

    Reads ``data/<name>/master_config.json``; falls back to the shipped default
    (with ``dataset_name`` filled in) if that dataset has no saved config.
    """
    saved = get_data_path(name) / "master_config.json"
    if saved.is_file():
        with open(saved, "r", encoding="utf-8") as f:
            return jsonify(json.load(f)), 200

    with open(_DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    config["dataset_name"] = name
    return jsonify(config), 200


@processing_bp.route("/datasets/<name>/reprocess", methods=["POST"])
def reprocess_dataset(name: str):
    """Re-run the pipeline for *name* with (possibly) edited parameters.

    Body (JSON):
      • ``config``     — the full master config to run with (its ``steps_to_run``
                         controls which steps execute)
      • ``as_copy``    — if true, run on a fresh copy instead of in place
      • ``new_name``   — name for the copy (required when ``as_copy``)

    In-place re-runs reuse the dataset's existing data/ and outputs/.  Copies
    are cloned (by the worker) from the source dataset first, leaving the
    original untouched.
    """
    body = request.get_json(silent=True) or {}
    config = body.get("config")
    if not isinstance(config, dict):
        return jsonify({"error": "Missing or invalid 'config'"}), 400

    config = dict(config)  # don't mutate the request payload

    if body.get("as_copy"):
        new_name = (body.get("new_name") or "").strip()
        if not _valid_dataset_name(new_name):
            return jsonify({
                "error": "Invalid copy name (use letters, digits, '.', '_', '-')"
            }), 400
        if get_dataset(new_name) is not None or get_data_path(new_name).exists():
            return jsonify({"error": f"Dataset {new_name!r} already exists"}), 409

        config["dataset_name"] = new_name
        config_path = get_data_path(new_name) / "master_config.json"
        job_id = job_manager.create_job(config, str(config_path), copy_from=name)
        return jsonify({"dataset_name": new_name, "job_id": job_id}), 202

    # In-place re-run.
    config["dataset_name"] = name
    config_path = get_data_path(name) / "master_config.json"
    job_id = job_manager.create_job(config, str(config_path))
    return jsonify({"dataset_name": name, "job_id": job_id}), 202
