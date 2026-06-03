"""processing.py – Upload + data-processing endpoints.

Routes
------
GET  /api/config              → default master_config.json contents
GET  /api/steps?source=...    → ordered pipeline steps for a data source
POST /api/upload              → save uploaded zip + config, start pipeline
GET  /api/jobs/<job_id>       → live status of a running/finished job
"""

from __future__ import annotations

import json
from pathlib import Path

from flask import Blueprint, jsonify, request

from config_io import get_data_path
from server.utils.job_manager import job_manager

processing_bp = Blueprint("processing", __name__, url_prefix="/api")

# Default config shipped with the server.
_MASTER_CONFIG_PATH = Path(__file__).resolve().parent.parent / "master_config.json"


@processing_bp.route("/config", methods=["GET"])
def get_default_config():
    """Return the default master_config.json so the UI can pre-fill the form."""
    with open(_MASTER_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return jsonify(config), 200


@processing_bp.route("/steps", methods=["GET"])
def list_steps():
    """Return the ordered pipeline steps for the given data source."""
    from server.utils.data_process import get_pipeline_steps

    source = request.args.get("source", "polycam")
    try:
        steps = get_pipeline_steps(source)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify([
        {"name": s.name, "description": s.description} for s in steps
    ]), 200


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
