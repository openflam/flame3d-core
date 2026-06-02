"""
data_process.py – Orchestrate the full data-processing pipeline.

Takes a config dictionary (or path to master_config.json) containing the
dataset name, data source type, and all per-step parameters, then runs every
pipeline step in the correct conda environment inside the running Docker
container.

Currently supported data sources:
  • polycam

Pipeline steps (Polycam)
========================
1. ``data_processor.vendor_specific.polycam``   (flame3d-core env)
2. ``segment3d.identify_objects.orchestrator``   (flame3d-core env)
3. ``segment3d.identify_objects.normalize_labels`` (flame3d-core env)
4. ``segment3d.sam3_runner``                     (sam3 env) – run as subprocess
5. ``segment3d.postsam3_pipeline.postsam3_pipeline`` (flame3d-core env) – run as subprocess
6. ``segment3d.captioning.orchestrator``         (flame3d-core env) – run as subprocess

Steps 4–6 are launched as shell subprocesses (via ``conda run``) rather than
Python function calls so that each step can fully release GPU memory before
the next step starts.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_processor.vendor_specific.polycam import process_polycam_from_config
from segment3d.identify_objects.orchestrator import identify_all_frames_from_config
from segment3d.identify_objects.normalize_labels import normalize_labels_from_config

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
# Working directory inside the Docker container (volume-mounted project root).
_WORKDIR = Path("/app")

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


# ── Result container ──────────────────────────────────────────────────────

@dataclass
class StepResult:
    """Outcome of a single pipeline step."""

    name: str
    success: bool
    duration_seconds: float
    return_code: Optional[int] = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Outcome of the entire pipeline run."""

    dataset_name: str
    data_source: str
    steps: List[StepResult] = field(default_factory=list)
    success: bool = False
    total_duration_seconds: float = 0.0


# ── Subprocess runner ────────────────────────────────────────────────────

def _run_step_subprocess(
    step: PipelineStep,
    config_path: str,
    *,
    cwd: Path = _WORKDIR,
) -> StepResult:
    """Run a pipeline step as a ``conda run`` subprocess.

    The subprocess receives ``--config <path>`` pointing to the same
    master_config.json on disk.  stdout/stderr are streamed so the caller
    can follow along in real time.
    """
    cmd = [
        "conda", "run",
        "--no-capture-output",
        "-n", step.conda_env,
        "python", "-m", step.module,
        "--config", config_path,
    ]

    logger.info("Running: %s", " ".join(cmd))
    start = time.monotonic()

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            # Stream output rather than capturing – keeps GPU-heavy steps visible.
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        elapsed = time.monotonic() - start

        if result.returncode != 0:
            return StepResult(
                name=step.name,
                success=False,
                duration_seconds=elapsed,
                return_code=result.returncode,
                error=f"Process exited with code {result.returncode}",
            )

        return StepResult(
            name=step.name,
            success=True,
            duration_seconds=elapsed,
            return_code=0,
        )

    except Exception as exc:
        elapsed = time.monotonic() - start
        return StepResult(
            name=step.name,
            success=False,
            duration_seconds=elapsed,
            error=str(exc),
        )


# ── Inline step dispatch ─────────────────────────────────────────────────

# Maps step name → the from_config callable for inline steps.
_INLINE_HANDLERS: Dict[str, Any] = {
    "polycam_process": process_polycam_from_config,
    "identify_objects": identify_all_frames_from_config,
    "normalize_labels": normalize_labels_from_config,
}


def _run_step_inline(step: PipelineStep, config: Dict[str, Any]) -> StepResult:
    """Run a pipeline step by calling its ``*_from_config`` function directly.

    This is used for lighter steps that don't require GPU-memory isolation.
    """
    start = time.monotonic()

    try:
        handler = _INLINE_HANDLERS.get(step.name)
        if handler is None:
            raise ValueError(f"No inline handler for step: {step.name}")

        handler(config)

        elapsed = time.monotonic() - start
        return StepResult(name=step.name, success=True, duration_seconds=elapsed)

    except Exception as exc:
        elapsed = time.monotonic() - start
        logger.exception("Step %s failed", step.name)
        return StepResult(
            name=step.name,
            success=False,
            duration_seconds=elapsed,
            error=str(exc),
        )


def _run_step(
    step: PipelineStep,
    config: Dict[str, Any],
    config_path: str,
) -> StepResult:
    """Dispatch a step to the appropriate runner."""
    if step.run_as_subprocess:
        return _run_step_subprocess(step, config_path)
    return _run_step_inline(step, config)


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
        # Step 2 — Identify objects in frames (flame3d-core, inline)
        PipelineStep(
            name="identify_objects",
            description="Run VLM object identification on every frame",
            conda_env=_FLAME3D_ENV,
            module="segment3d.identify_objects.orchestrator",
            run_as_subprocess=True,
        ),
        # Step 3 — Normalize labels (flame3d-core, inline)
        PipelineStep(
            name="normalize_labels",
            description="Normalize object labels via CLIP clustering",
            conda_env=_FLAME3D_ENV,
            module="segment3d.identify_objects.normalize_labels",
            run_as_subprocess=False,
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

_PIPELINE_BUILDERS = {
    DataSource.POLYCAM: _build_polycam_steps,
}


def process_data(
    config: Dict[str, Any],
    config_path: Optional[str] = None,
) -> PipelineResult:
    """Run the full processing pipeline using *config*.

    Parameters
    ----------
    config:
        The master configuration dictionary.  Must contain at least
        ``"dataset_name"`` and ``"data_source"``.  Per-step parameters are
        read from nested dicts (e.g. ``config["polycam"]``).
    config_path:
        Filesystem path to the master_config.json file.  Passed to
        subprocess steps via ``--config``.  Required when the pipeline
        includes subprocess steps.

    Returns
    -------
    PipelineResult
        Aggregated result with per-step timing and success information.
    """
    dataset_name = config["dataset_name"]
    data_source_str = config["data_source"]
    start_from_step = config.get("start_from_step")

    try:
        data_source = DataSource(data_source_str.lower())
    except ValueError:
        supported = ", ".join(ds.value for ds in DataSource)
        raise ValueError(
            f"Unsupported data source: {data_source_str!r}. "
            f"Supported sources: {supported}"
        )

    builder = _PIPELINE_BUILDERS.get(data_source)
    if builder is None:
        raise ValueError(f"No pipeline builder registered for {data_source!r}")

    steps = builder()
    pipeline_result = PipelineResult(
        dataset_name=dataset_name,
        data_source=data_source.value,
    )

    # Optionally skip steps before start_from_step.
    if start_from_step is not None:
        step_names = [s.name for s in steps]
        if start_from_step not in step_names:
            valid = ", ".join(step_names)
            raise ValueError(
                f"Unknown step name: {start_from_step!r}. Valid steps: {valid}"
            )
        skip_idx = step_names.index(start_from_step)
        skipped = steps[:skip_idx]
        steps = steps[skip_idx:]
        for skipped_step in skipped:
            logger.info(
                "Skipping step: %s (resuming from %s)",
                skipped_step.name,
                start_from_step,
            )

    # Verify config_path is available if any subprocess steps remain.
    has_subprocess_steps = any(s.run_as_subprocess for s in steps)
    if has_subprocess_steps and not config_path:
        raise ValueError(
            "config_path is required when the pipeline includes subprocess "
            "steps (SAM3, postSAM3, captioning).  Pass the path to your "
            "master_config.json."
        )

    pipeline_start = time.monotonic()

    for i, step in enumerate(steps, 1):
        total = len(steps)
        logger.info(
            "\n%s\n  STEP %d/%d: %s — %s\n%s",
            "=" * 72, i, total, step.name, step.description, "=" * 72,
        )
        print(
            f"\n{'=' * 72}\n"
            f"  STEP {i}/{total}: {step.name} — {step.description}\n"
            f"{'=' * 72}\n"
        )

        step_result = _run_step(step, config, config_path or "")
        pipeline_result.steps.append(step_result)

        if step_result.success:
            logger.info(
                "✓ %s completed in %.1fs", step.name, step_result.duration_seconds,
            )
            print(f"✓ {step.name} completed in {step_result.duration_seconds:.1f}s")
        else:
            logger.error(
                "✗ %s failed after %.1fs: %s",
                step.name,
                step_result.duration_seconds,
                step_result.error,
            )
            print(
                f"✗ {step.name} failed after {step_result.duration_seconds:.1f}s: "
                f"{step_result.error}"
            )
            # Abort on first failure — downstream steps depend on earlier outputs.
            break

    pipeline_result.total_duration_seconds = time.monotonic() - pipeline_start
    pipeline_result.success = all(s.success for s in pipeline_result.steps)

    _print_summary(pipeline_result)
    return pipeline_result


# ── Pretty-print summary ─────────────────────────────────────────────────

def _print_summary(result: PipelineResult) -> None:
    """Print a human-readable summary of the pipeline run."""
    mins = result.total_duration_seconds / 60
    status = "SUCCESS" if result.success else "FAILED"

    print(f"\n{'=' * 72}")
    print(f"  PIPELINE {status}")
    print(f"  Dataset: {result.dataset_name}  |  Source: {result.data_source}")
    print(f"  Total time: {result.total_duration_seconds:.1f}s ({mins:.1f} min)")
    print(f"{'=' * 72}")

    for step in result.steps:
        icon = "✓" if step.success else "✗"
        line = f"  {icon} {step.name:<25s} {step.duration_seconds:>8.1f}s"
        if step.error:
            line += f"  — {step.error}"
        print(line)

    print()


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point for the data-processing pipeline."""
    # Build the list of valid step names for help text.
    _example_steps = [s.name for s in _build_polycam_steps()]

    parser = argparse.ArgumentParser(
        description="Run the flame3d-core data-processing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""\
examples:
  python -m server.utils.data_process --config server/master_config.json
  python -m server.utils.data_process --config server/master_config.json \\
      --start-from-step sam3_segmentation

pipeline steps (polycam):
  {', '.join(_example_steps)}
""",
    )

    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="Path to master_config.json",
    )
    parser.add_argument(
        "--start-from-step",
        default=None,
        metavar="STEP",
        help="Resume the pipeline from this step (overrides config). "
        f"Valid steps: {', '.join(_example_steps)}",
    )

    args = parser.parse_args()

    config_path = str(Path(args.config).resolve())
    with open(config_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = json.load(f)

    # CLI overrides
    if args.start_from_step is not None:
        config["start_from_step"] = args.start_from_step

    result = process_data(config, config_path=config_path)
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
