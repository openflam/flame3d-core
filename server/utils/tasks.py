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

import json
import shutil
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from server.celery_app import celery_app

# ── Logical job/step statuses (mirrors the frontend's StepStatus) ───────────

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

TASK_NAME = "run_pipeline"


def _prepare_inputs(
    config: Dict[str, Any],
    config_path: str,
    copy_from: Optional[str],
) -> None:
    """Stage inputs before the pipeline runs.

    When *copy_from* is set ("process as copy"), clone that dataset's ``data/``
    and ``outputs/`` directories into this dataset so the run operates on an
    independent copy and the original is left untouched.  Symlinks are
    preserved (the original — and thus their targets — still exists).

    Always (re)writes the on-disk ``master_config.json`` so the subprocess
    pipeline steps read the exact parameters and ``steps_to_run`` chosen for
    this run.
    """
    from config_io import PATHS

    dataset_name = config["dataset_name"]

    if copy_from:
        for root in (PATHS["data"], PATHS["outputs"]):
            src = root / copy_from
            dst = root / dataset_name
            if src.exists():
                shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)

    cfg_path = Path(config_path)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


@celery_app.task(bind=True, name=TASK_NAME)
def run_pipeline_task(
    self,
    config: Dict[str, Any],
    config_path: str,
    copy_from: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the full pipeline for *config*, publishing per-step progress.

    If *copy_from* is given, this dataset is first cloned from that one (see
    :func:`_prepare_inputs`) — used by "process as copy".
    """
    # Heavy imports happen here, inside the worker, not at module import time.
    from server.utils.data_process import process_data
    from server.utils.pipeline_steps import get_pipeline_steps

    dataset_name = config["dataset_name"]
    data_source = config["data_source"]
    steps = get_pipeline_steps(data_source)

    # Steps not in ``steps_to_run`` are reported as skipped. An empty/null
    # ``steps_to_run`` means "run every step".
    steps_to_run = config.get("steps_to_run")
    selected = set(steps_to_run) if steps_to_run else None

    step_records: List[Dict[str, Any]] = []
    for step in steps:
        will_run = selected is None or step.name in selected
        step_records.append({
            "name": step.name,
            "description": step.description,
            "status": STATUS_PENDING if will_run else "skipped",
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
        _prepare_inputs(config, config_path, copy_from)
        result = process_data(
            config,
            dataset_name,
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

    # Flip the durable dataindex row to its terminal status. Wrapped so a DB
    # hiccup never masks the pipeline result (which still rides back in meta).
    try:
        from server.database import STATUS_COMPLETE, STATUS_FAILED as DB_FAILED
        from server.database import set_status

        db_status = STATUS_COMPLETE if meta["status"] == STATUS_COMPLETED else DB_FAILED
        set_status(config["dataset_name"], db_status, error=meta.get("error"))
    except Exception:  # noqa: BLE001 — index update is best-effort
        traceback.print_exc()

    # Returning the meta marks the Celery task SUCCESS with result == meta;
    # the logical pass/fail lives in meta["status"].
    return meta
