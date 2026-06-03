"""job_manager.py – Celery-backed job management.

Pipeline runs are dispatched onto a Celery task queue (RabbitMQ broker) and
their live progress is read back from the Redis result backend.  The Flask web
process only *sends* tasks (by name) and *reads* results — the heavy pipeline,
along with its ML import stack, executes in a separate Celery worker process.
See :mod:`server.utils.tasks` for the task itself and :mod:`server.celery_app`
for the broker/backend wiring.

State therefore survives Flask restarts (it lives in Redis), and multiple web
workers can serve status for the same job.
"""

from __future__ import annotations

from typing import Any, Dict

from celery.result import AsyncResult

from server.celery_app import celery_app

# ── Logical statuses returned to the frontend ──────────────────────────────

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

# Celery task name (must match the @task(name=...) in server.utils.tasks).
_TASK_NAME = "run_pipeline"


def _empty_job(job_id: str) -> Dict[str, Any]:
    """A normalized snapshot for a job with no progress meta yet."""
    return {
        "id": job_id,
        "dataset_name": "",
        "data_source": "",
        "status": STATUS_PENDING,
        "steps": [],
        "error": None,
    }


class JobManager:
    """Dispatches pipeline jobs to Celery and normalizes their status."""

    def create_job(self, config: Dict[str, Any], config_path: str) -> str:
        """Enqueue a pipeline run and return its Celery task id (= job id)."""
        result = celery_app.send_task(_TASK_NAME, args=[config, config_path])
        return result.id

    def get_job(self, job_id: str) -> Dict[str, Any]:
        """Return a normalized snapshot of a job's progress.

        Maps Celery's task states onto the ``{id, dataset_name, data_source,
        status, steps, error}`` shape the frontend consumes.  Note that Celery
        cannot distinguish an unknown id from a still-queued one — both report
        as ``pending``.
        """
        res = AsyncResult(job_id, app=celery_app)
        state = res.state
        job = _empty_job(job_id)

        # Queued (or unknown) — no worker has touched it yet.
        if state == "PENDING":
            return job

        # Picked up but no custom progress published yet.
        if state in ("RECEIVED", "STARTED", "RETRY"):
            job["status"] = STATUS_RUNNING
            return job

        info = res.info  # PROGRESS → meta dict; SUCCESS → return value; FAILURE → exc

        # Hard failure (unhandled exception) — no rich meta survives.
        if state == "FAILURE":
            job["status"] = STATUS_FAILED
            job["error"] = str(info) if info else "Task failed"
            return job

        # PROGRESS or SUCCESS both carry our meta dict.
        if isinstance(info, dict):
            job.update(
                {
                    "dataset_name": info.get("dataset_name") or "",
                    "data_source": info.get("data_source") or "",
                    "status": info.get(
                        "status",
                        STATUS_COMPLETED if state == "SUCCESS" else STATUS_RUNNING,
                    ),
                    "steps": info.get("steps", []),
                    "error": info.get("error"),
                }
            )
        elif state == "SUCCESS":
            job["status"] = STATUS_COMPLETED

        return job


# Module-level singleton shared across requests.
job_manager = JobManager()
