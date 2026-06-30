# Debug commands

## Inspect & unstick the pipeline queue / dataset state

Pipeline runs are Celery tasks (RabbitMQ broker, Redis result backend); the
dataset's `processing` / `complete` / `failed` status is a row in the PostGIS
`dataindex` table. These can drift apart — e.g. a hard
`docker compose restart segmentation-worker` mid-run leaves the task un-acked, so
RabbitMQ **redelivers** it on restart and the worker re-runs a "zombie" job,
while the dataset row sits stuck on `processing`. The commands below let you see
what is actually queued/running and repair the state by hand. See
[`server/utils/job_manager.py`](../server/utils/job_manager.py) and
[`server/database/dataindex.py`](../server/database/dataindex.py).

### See what the worker is doing

```bash
# Tasks currently executing (look for "redelivered": True / "acknowledged": False
# — that flags a zombie re-run from a mid-task restart):
docker compose exec segmentation-worker \
  conda run --no-capture-output -n flame3d-core \
  celery -A server.celery_app inspect active

# Tasks prefetched by the worker but not started yet:
docker compose exec segmentation-worker \
  conda run --no-capture-output -n flame3d-core \
  celery -A server.celery_app inspect reserved

# Messages still waiting in the broker queue (not yet sent to any worker):
docker compose exec rabbitmq rabbitmqctl list_queues name messages
```

### Check a dataset's stored status / a job's live status

```bash
# Durable dataindex row (status, job_id, error) — run from the server container:
docker compose exec -T server python -c \
  "from server.database.dataindex import get_dataset; print(get_dataset('ProjectStudio'))"

# Normalized live job snapshot the UI consumes (status + per-step progress):
docker compose exec -T server python -c \
  "from server.utils.job_manager import job_manager; print(job_manager.get_job('<job_id>'))"
```

### Kill a stuck / zombie task

Revoking with `terminate=True` frees the (single) worker slot so the next queued
job can start. Do this from the `server` container — it shares the broker but
not the ML stack, so it imports quickly:

```bash
docker compose exec -T server python -c \
  "from server.celery_app import celery_app; \
   celery_app.control.revoke('<job_id>', terminate=True, signal='SIGKILL'); \
   print('revoked')"
```

To drop _every_ unstarted message (use with care — this discards all queued
runs):

```bash
docker compose exec segmentation-worker \
  conda run --no-capture-output -n flame3d-core \
  celery -A server.celery_app purge -f
```

### Manually fix a stuck dataset row

When the worker died without writing a terminal status, set it yourself so the UI
stops showing a spinner. `set_status` takes `complete` / `failed` (plus an
optional error message); `mark_deleted` hides it from the list.

```bash
docker compose exec -T server python -c \
  "from server.database.dataindex import set_status; \
   set_status('ProjectStudio', 'failed', 'Worker restarted mid-run; re-run to retry.')"
```

> Note: clicking **Reprocess** in the UI calls `upsert_processing`, which resets
> the row back to `processing` and enqueues a fresh job. If that job then sits in
> `Queued`, a zombie task is almost certainly still holding the worker slot —
> revoke it (above) and the reprocess will start.

### Avoiding the problem

Prefer `docker compose stop segmentation-worker` (waits for a warm shutdown that
re-queues cleanly) over `restart`, which yanks the worker mid-task and triggers
the redelivery described above.
