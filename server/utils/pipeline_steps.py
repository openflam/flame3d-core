"""pipeline_steps.py – Lightweight pipeline step *metadata*.

This module describes *what* the pipeline steps are (their names, descriptions,
conda env, and module path) without importing any of the heavy ML machinery
that actually *runs* them.  It has no third-party dependencies, so the Flask
web process can import it (e.g. to serve ``GET /api/steps``) without pulling in
torch / vLLM / open3d / SAM3.

The orchestration that executes these steps lives in
:mod:`server.utils.data_process`, which is imported only by the Celery worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List

# Conda environments the steps run in (inside the worker image).
_FLAME3D_ENV = "flame3d-core"
_SAM3_ENV = "sam3"


# ── Data source enum ──────────────────────────────────────────────────────

class DataSource(str, Enum):
    """Supported data-source types."""

    POLYCAM = "polycam"


# ── Pipeline step descriptor ─────────────────────────────────────────────

@dataclass
class PipelineStep:
    """Describes a single step in the processing pipeline."""

    name: str
    description: str
    conda_env: str
    module: str
    run_as_subprocess: bool = False


# ── Polycam pipeline builder ─────────────────────────────────────────────

def _build_polycam_steps() -> List[PipelineStep]:
    """Build the ordered list of steps for a Polycam dataset."""

    return [
        # Step 1 — Polycam data processing (flame3d-core, inline)
        PipelineStep(
            name="polycam_process",
            description="Extract and process Polycam raw-data export",
            conda_env=_FLAME3D_ENV,
            module="data_processor.vendor_specific.polycam",
            run_as_subprocess=False,
        ),
        # Step 2 — Identify objects in frames (flame3d-core, subprocess)
        PipelineStep(
            name="identify_objects",
            description="Run VLM object identification on every frame",
            conda_env=_FLAME3D_ENV,
            module="segment3d.identify_objects.orchestrator",
            run_as_subprocess=True,
        ),
        # Step 3 — Normalize labels (flame3d-core, subprocess for GPU isolation)
        # Loads an OpenCLIP model onto the GPU; run it as a subprocess so the
        # CUDA context (and PyTorch's cached allocations) are fully released on
        # exit. Running it inline leaked GPU memory into the long-lived Celery
        # worker, starving the later captioning step.
        PipelineStep(
            name="normalize_labels",
            description="Normalize object labels via CLIP clustering",
            conda_env=_FLAME3D_ENV,
            module="segment3d.identify_objects.normalize_labels",
            run_as_subprocess=True,
        ),
        # Step 4 — SAM3 segmentation (sam3 env, subprocess for GPU isolation)
        PipelineStep(
            name="sam3_segmentation",
            description="Run SAM3 video predictor on per-object frame sequences",
            conda_env=_SAM3_ENV,
            module="segment3d.sam3_runner",
            run_as_subprocess=True,
        ),
        # Step 5 — Post-SAM3 pipeline (flame3d-core, subprocess for GPU isolation)
        PipelineStep(
            name="postsam3_pipeline",
            description="Associate masks with 3D points, build graph, clean, bbox, crop",
            conda_env=_FLAME3D_ENV,
            module="segment3d.postsam3_pipeline.postsam3_pipeline",
            run_as_subprocess=True,
        ),
        # Step 6 — Captioning (flame3d-core, subprocess for GPU isolation)
        PipelineStep(
            name="captioning",
            description="Generate captions for each component using VLM",
            conda_env=_FLAME3D_ENV,
            module="segment3d.captioning.orchestrator",
            run_as_subprocess=True,
        ),
    ]


# ── Public API ────────────────────────────────────────────────────────────

_PIPELINE_BUILDERS: Dict[DataSource, Callable[[], List[PipelineStep]]] = {
    DataSource.POLYCAM: _build_polycam_steps,
}


def get_pipeline_steps(data_source: str) -> List[PipelineStep]:
    """Return the ordered list of pipeline steps for *data_source*.

    Useful for UIs that need to display the full step list before the run
    starts.  Raises ``ValueError`` for unsupported sources.
    """
    try:
        ds = DataSource(data_source.lower())
    except ValueError:
        supported = ", ".join(d.value for d in DataSource)
        raise ValueError(
            f"Unsupported data source: {data_source!r}. Supported: {supported}"
        )

    builder = _PIPELINE_BUILDERS.get(ds)
    if builder is None:
        raise ValueError(f"No pipeline builder registered for {ds!r}")
    return builder()
