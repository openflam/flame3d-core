"""job_manager.py – Run data-processing pipelines in the background and track
their live progress so the frontend can poll for status.

A single :class:`JobManager` instance holds an in-memory registry of jobs.
Each job runs :func:`server.utils.data_process.process_data` on a daemon
thread, and a progress callback updates the job's per-step status as the
pipeline advances.  The frontend polls ``GET /api/jobs/<id>`` to render a
stepper.

State is in-memory only — it is reset whenever the Flask process restarts.
That is acceptable for a single-user dev tool; persisting jobs would require a
database and is out of scope here.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from typing import Any, Dict, List, Optional

# NOTE: `data_process` pulls in the full ML stack (torch, vllm, open3d, …) at
# import time. We import it lazily inside the methods below so the Flask server
# boots quickly and only loads those heavy deps when a job actually runs.


# ── Status constants ───────────────────────────────────────────────────────

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


class JobManager:
    """Thread-safe registry of background pipeline jobs."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # ── Creation ───────────────────────────────────────────────────────────

    def create_job(
        self,
        config: Dict[str, Any],
        config_path: str,
    ) -> str:
        """Register a new job and kick off its background thread.

        Returns the generated job id.
        """
        from server.utils.data_process import get_pipeline_steps

        job_id = uuid.uuid4().hex
        data_source = config["data_source"]

        # Pre-build the step list so the UI can show every step up front.
        steps = get_pipeline_steps(data_source)
        start_from = config.get("start_from_step")
        step_names = [s.name for s in steps]
        skip_before = (
            step_names.index(start_from)
            if start_from in step_names
            else 0
        )

        step_records: List[Dict[str, Any]] = []
        for idx, step in enumerate(steps):
            # Steps before start_from_step are reported as skipped.
            status = "skipped" if idx < skip_before else STATUS_PENDING
            step_records.append({
                "name": step.name,
                "description": step.description,
                "status": status,
                "duration_seconds": None,
                "error": None,
            })

        job = {
            "id": job_id,
            "dataset_name": config["dataset_name"],
            "data_source": data_source,
            "status": STATUS_RUNNING,
            "steps": step_records,
            "error": None,
            "created_at": time.time(),
            "finished_at": None,
        }

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run,
            args=(job_id, config, config_path),
            daemon=True,
        )
        thread.start()

        return job_id

    # ── Background execution ───────────────────────────────────────────────

    def _run(
        self,
        job_id: str,
        config: Dict[str, Any],
        config_path: str,
    ) -> None:
        """Execute the pipeline, updating job state as steps progress."""
        from server.utils.data_process import process_data

        def on_progress(event: Dict[str, Any]) -> None:
            kind = event.get("event")
            name = event.get("name")
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None:
                    return
                for record in job["steps"]:
                    if record["name"] != name:
                        continue
                    if kind == "step_start":
                        record["status"] = STATUS_RUNNING
                    elif kind == "step_complete":
                        record["status"] = (
                            STATUS_COMPLETED
                            if event.get("success")
                            else STATUS_FAILED
                        )
                        record["duration_seconds"] = event.get("duration_seconds")
                        record["error"] = event.get("error")
                    break

        try:
            result = process_data(
                config,
                config_path=config_path,
                progress_callback=on_progress,
            )
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["status"] = (
                        STATUS_COMPLETED if result.success else STATUS_FAILED
                    )
                    job["finished_at"] = time.time()
                    if not result.success and job["error"] is None:
                        # Surface the first failing step's error.
                        failed = next(
                            (s for s in job["steps"]
                             if s["status"] == STATUS_FAILED),
                            None,
                        )
                        job["error"] = (
                            failed["error"] if failed else "Pipeline failed"
                        )
        except Exception as exc:  # noqa: BLE001 — report any failure to the UI
            traceback.print_exc()
            with self._lock:
                job = self._jobs.get(job_id)
                if job is not None:
                    job["status"] = STATUS_FAILED
                    job["error"] = str(exc)
                    job["finished_at"] = time.time()
                    # Mark any still-running step as failed.
                    for record in job["steps"]:
                        if record["status"] == STATUS_RUNNING:
                            record["status"] = STATUS_FAILED
                            record["error"] = str(exc)

    # ── Queries ────────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return a snapshot of the job, or ``None`` if unknown."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            # Return a shallow copy so callers can't mutate internal state.
            return {
                **job,
                "steps": [dict(s) for s in job["steps"]],
            }


# Module-level singleton shared across requests.
job_manager = JobManager()
