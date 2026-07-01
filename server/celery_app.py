"""celery_app.py – Celery application for the flame3d-core pipeline.

The broker is RabbitMQ and the result backend is Redis; both run as services in
``docker-compose.yml``.  Connection URLs come from the environment so the same
code works locally and in Docker.

The actual pipeline task lives in :mod:`server.utils.tasks` and is registered
via the ``include`` list — it is only imported by the *worker*, never by the
Flask web process (which talks to Celery purely by task name through
``send_task``).  That keeps the web server free of the heavy ML import stack.
"""

from __future__ import annotations

import os

from celery import Celery

# RabbitMQ broker (amqp) and Redis result backend.
_BROKER_URL = os.environ.get(
    "CELERY_BROKER_URL", "amqp://guest:guest@rabbitmq:5672//"
)
_RESULT_BACKEND = os.environ.get(
    "CELERY_RESULT_BACKEND", "redis://redis:6379/0"
)

celery_app = Celery(
    "flame3d",
    broker=_BROKER_URL,
    backend=_RESULT_BACKEND,
    include=["server.utils.tasks"],
)

celery_app.conf.update(
    # Surface a STARTED state between "received" and the first progress update.
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    # Store args/kwargs/worker info alongside results.
    result_extended=True,
    # GPU pipeline: one job at a time, don't prefetch extra work.
    worker_prefetch_multiplier=1,
    # Ack the message as soon as it's picked up (before running), NOT after.
    # With acks_late the broker keeps the message live for the whole run, so a
    # worker crash or `docker compose restart segmentation-worker` leaves it
    # unacknowledged and RabbitMQ redelivers it — restarting the pipeline. We
    # want a failed/interrupted run to stay dead, so ack early instead.
    task_acks_late=False,
    # Don't requeue a task whose worker died mid-run (redundant with the early
    # ack above, but explicit: a lost worker must never resurrect the job).
    task_reject_on_worker_lost=False,
    # Retry connecting to the broker on startup (broker may boot after worker).
    broker_connection_retry_on_startup=True,
    # Keep finished results in Redis for a day so the UI can read them.
    result_expires=86400,
)
