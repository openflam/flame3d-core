# Flame3D

## Getting Started

### Prerequisites

- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed

### Build & Run

```bash
# Build the Docker image (only needed once, or when dependencies change)
docker compose build

# Start the server
docker compose up

# Start in detached mode
docker compose up -d
```

### Test

```bash
# Health check
curl http://localhost:5005/health

# Verify conda environments
docker compose exec flame3d-core conda env list
```

### Development

Code changes are volume-mounted into the container — just restart to pick them up:

```bash
docker compose restart
```

## Process Data

All algorithms in core assume a right-handed Z-up coordinate system. Vendor specific code does whatever is needed to convert vendor data to this coordinate frame.

Algorithms implemented:

- RGB + Depth + Pose + Mesh (e.g., Polycam): Use mesh reprojection to project mesh vertices onto images to find 2D-3D correspondence.

### Starting the pipeline from the command line

The web UI starts a run by uploading a capture zip + config to the server's
`POST /api/upload` route. When the data already lives on the machine you can do
the same thing without the UI using `scripts/start_pipeline.sh`, which uploads
to that same route on the running stack (so `docker compose up` must already be
running).

It works for any supported data source (`polycam`, `scannetpp`) — both expect
the capture packed as a single `input.zip`, which is what the script uploads.

```bash
# Data already packed as a zip
scripts/start_pipeline.sh \
  --data ~/captures/kitchen.zip \
  --config server/default_config.json \
  --wait

# Data as a directory (its contents are zipped, then uploaded)
scripts/start_pipeline.sh \
  --data ~/captures/kitchen_export/ \
  --config my_config.json
```

The config must contain `dataset_name` (the directory name used under `data/`
and `outputs/`) and `data_source`. The script prints the `job_id`; pass
`--wait` to poll until the run completes, or track it yourself:

```bash
curl http://localhost:5005/api/jobs/<job_id>
```

Pass `--host` if the server is not at `http://localhost:5005`. Run
`scripts/start_pipeline.sh --help` for all options.

### Reprocessing an existing dataset

Once a dataset has been processed, `scripts/reprocess.sh` re-runs the pipeline
against its existing `data/` and `outputs/` — no re-upload needed. It posts to
the server's `POST /api/datasets/<name>/reprocess` route (the UI's "reprocess"
button), so the stack must be running.

The config's `steps_to_run` selects which steps execute (`null`/empty = all).
When running a subset, the inputs those steps need must already exist on disk
from a previous run — see `docs/Configs.md`.

```bash
# Re-run in place (e.g. only the steps listed in the config's steps_to_run)
scripts/reprocess.sh \
  --dataset kitchen \
  --config data/kitchen/master_config.json \
  --wait

# Try new parameters on a copy, leaving the original untouched
scripts/reprocess.sh \
  --dataset kitchen \
  --config tweaked.json \
  --as-copy --new-name kitchen_v2
```

The script prints the `job_id` (and the target dataset name, which differs when
`--as-copy` is used). Run `scripts/reprocess.sh --help` for all options.
