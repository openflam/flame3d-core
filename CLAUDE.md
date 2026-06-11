# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Flame3D turns a vendor 3D capture (currently Polycam) into a queryable scene of
**components** — connected groups of masks the pipeline believes are single
physical objects — each with a caption, a 3D bounding box, best-view crops, and
a CLIP embedding, stored in PostGIS. On top of that sits a Flask API and a
Three.js frontend for spatial, semantic, and LLM-driven search of the scene.

## Commands

Everything runs in Docker (GPU pipeline needs the NVIDIA Container Toolkit).

```bash
docker compose build                 # rebuild images — only needed when DEPENDENCIES change
docker compose up [-d]               # start the stack (server, worker, frontend, rabbitmq, redis, postgis)
docker compose restart               # pick up CODE changes (project is volume-mounted at /app)
curl http://localhost:5005/health    # API health check
docker compose exec segmentation-worker conda env list   # verify the two conda envs
```

Rebuild scope: the **server** image uses `server/requirements.txt`; the
**segmentation-worker** image uses the repo-root `requirements.txt`. Editing one
only requires rebuilding that service (e.g. `docker compose up -d --build segmentation-worker`).

Run a single pipeline step manually (inside the worker, picking the right conda env):

```bash
docker compose exec segmentation-worker conda run --no-capture-output -n flame3d-core \
  python -m segment3d.identify_objects.orchestrator \
  --config server/default_config.json --dataset-name MyDataset
```

Run the whole pipeline directly (bypassing Celery): `python -m server.utils.data_process --config server/default_config.json --dataset-name MyDataset`.

There is no automated test suite in this repo.

## Architecture

### Two-process split (the central design constraint)

The Flask **server** and the Celery **segmentation-worker** are deliberately
separate processes with separate images. The server only enqueues jobs and
reads status; it talks to Celery purely **by task name via `send_task`** (broker
RabbitMQ, result backend Redis) and must **never import the heavy ML stack**
(torch / vLLM / open3d / SAM3). This is why metadata and orchestration are split:

- `server/utils/pipeline_steps.py` — step *metadata* only (names, descriptions,
  conda env, module path). No heavy deps; safe for the web process to import.
- `server/utils/data_process.py` — the actual orchestration. Imports the ML
  stack; **only the worker imports it**.

When adding a pipeline step or changing how steps are dispatched, respect this
boundary — putting a heavy import where the Flask process can reach it will break
the lightweight server image.

### Pipeline

`get_pipeline_steps(data_source)` returns the ordered `PipelineStep` list for a
data source (only `polycam` today). `process_data()` runs them in order, aborting
on the first failure (downstream steps depend on upstream outputs).

Each step runs either **inline** (a `*_from_config(config, dataset_name)`
function call) or as a **`conda run` subprocess**. Subprocess steps exist for
**GPU-memory isolation**: launching a fresh process guarantees the CUDA context
and PyTorch allocations are released before the next step starts (running them
inline leaked GPU memory into the long-lived worker). Two conda envs live in the
worker image: `flame3d-core` (most steps) and `sam3` (the SAM3 step only).

`steps_to_run` in the config is an optional whitelist of step names to run in
pipeline order; `null`/empty runs all. Selecting a subset means *you* are
responsible for ensuring the inputs each step needs already exist on disk.

The Polycam step order: `polycam_process → identify_objects → normalize_labels →
sam3_segmentation → postsam3_pipeline → captioning → create_tables`.

### Config system

- `server/config_schema.json` drives both the UI's config form and validation
  (per-field `type`, optional `possible_values`, `nullable`).
- `server/default_config.json` holds the shipped default values.
- **`dataset_name` is passed separately**, never read from the config — every
  step takes it as an explicit argument (`--dataset-name` on the CLI, its own
  field in the UI). `default_config.json` carries a `dataset_name` field but
  steps ignore it.
- Each step exposes a `*_from_config(config, dataset_name)` entry point and a
  CLI (`python -m <module> --config <path> --dataset-name <name>`), wired via the
  helpers in `config_cli.py`.
- `docs/Configs.md` documents every field **and** how to tune component
  count/quality (the `tau` / `voxel_size_cm` / `clip_distance_threshold` graph
  knobs vs. the `component_dbscan_*` / `split_components` cleaning knobs). Read it
  before changing pipeline parameters.

### Paths

`config_io.py` is the single source of truth for the on-disk layout: raw inputs
live in `data/<dataset>/`, all generated artifacts in `outputs/<dataset>/...`
(images, colmap, objects_inventory, etc.). Use its helpers rather than
hard-coding paths.

### Coordinate systems (read `docs/CoordinateSystem.md`)

The backend works entirely in a right-handed **Z-up world frame** — COLMAP
output, component bboxes, PostGIS tables, and all spatial reasoning in
`server/search/` assume `z` is up. The exception is `mesh.glb`, which stays
**Y-up** because the glTF spec requires it. The Y-up↔Z-up rotation
(`_YUP_TO_ZUP` in `data_processor/vendor_specific/polycam.py`) is applied **only**
in the frontend (`frontend/src/query/transforms.ts`); the backend never applies
viewer-specific transforms, so its data stays usable by any consumer. A new
vendor loader must convert its capture into this same Z-up frame.

### Search & LLM access

`server/search/` holds the query stack: a tool-calling LLM agent
(`llm_reasoning/`, which drives the model through `utils/llm_call.py`'s
`LLMCaller`), semantic search (`semantic_search/`, BM25 + CLIP), and PostGIS
spatial queries. `utils/llm_call.py` is a thin LiteLLM wrapper shared across the
repo; `model` is a **required** argument (no default — callers specify it). LLM
API keys come from the repo-root `.env`, injected into the `server` and
`segmentation-worker` containers via `env_file` in `docker-compose.yml` — code
does not load `.env` itself.
