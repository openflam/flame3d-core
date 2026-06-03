"""tasks.py – Celery task that runs the data-processing pipeline.

This module is imported by the Celery *worker* only (via the ``include`` list
in :mod:`server.celery_app`).  It pulls in the full ML stack through
:mod:`server.utils.data_process`, so it must never be imported by the Flask web
process — that process enqueues work by task *name* and reads results from the
backend.

Live progress
=============
The task publishes a custom ``PROGRESS`` state whose ``meta`` carries the same
shape the frontend expects: ``dataset_name``, ``data_source``, ``status``, and
a ``steps`` list.  Every time a pipeline step starts or finishes, the meta is
re-published so a polling client sees the stepper advance in real time.

Failures are captured *inside* the returned meta (logical ``status: "failed"``)
rather than raised, so the rich per-step state survives to the UI instead of
being replaced by a Celery traceback.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List

from server.celery_app import celery_app

# ── Logical job/step statuses (mirrors the frontend's StepStatus) ───────────

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

TASK_NAME = "run_pipeline"


@celery_app.task(bind=True, name=TASK_NAME)
def run_pipeline_task(
    self,
    config: Dict[str, Any],
    config_path: str,
) -> Dict[str, Any]:
    """Run the full pipeline for *config*, publishing per-step progress."""
    # Heavy imports happen here, inside the worker, not at module import time.
    from server.utils.data_process import process_data
    from server.utils.pipeline_steps import get_pipeline_steps

    data_source = config["data_source"]
    steps = get_pipeline_steps(data_source)

    # Steps before ``start_from_step`` are reported as skipped.
    start_from = config.get("start_from_step")
    step_names = [s.name for s in steps]
    skip_before = step_names.index(start_from) if start_from in step_names else 0

    step_records: List[Dict[str, Any]] = []
    for idx, step in enumerate(steps):
        step_records.append({
            "name": step.name,
            "description": step.description,
            "status": "skipped" if idx < skip_before else STATUS_PENDING,
            "duration_seconds": None,
            "error": None,
        })

    meta: Dict[str, Any] = {
        "dataset_name": config.get("dataset_name"),
        "data_source": data_source,
        "status": STATUS_RUNNING,
        "steps": step_records,
        "error": None,
    }

    def publish() -> None:
        self.update_state(state="PROGRESS", meta=meta)

    publish()

    def on_progress(event: Dict[str, Any]) -> None:
        name = event.get("name")
        kind = event.get("event")
        for record in step_records:
            if record["name"] != name:
                continue
            if kind == "step_start":
                record["status"] = STATUS_RUNNING
            elif kind == "step_complete":
                record["status"] = (
                    STATUS_COMPLETED if event.get("success") else STATUS_FAILED
                )
                record["duration_seconds"] = event.get("duration_seconds")
                record["error"] = event.get("error")
            break
        publish()

    try:
        result = process_data(
            config,
            config_path=config_path,
            progress_callback=on_progress,
        )
        meta["status"] = STATUS_COMPLETED if result.success else STATUS_FAILED
        if not result.success and meta["error"] is None:
            failed = next(
                (s for s in step_records if s["status"] == STATUS_FAILED),
                None,
            )
            meta["error"] = failed["error"] if failed else "Pipeline failed"
    except Exception as exc:  # noqa: BLE001 — report any failure to the UI
        traceback.print_exc()
        meta["status"] = STATUS_FAILED
        meta["error"] = str(exc)
        for record in step_records:
            if record["status"] == STATUS_RUNNING:
                record["status"] = STATUS_FAILED
                record["error"] = str(exc)

    # Returning the meta marks the Celery task SUCCESS with result == meta;
    # the logical pass/fail lives in meta["status"].
    return meta
