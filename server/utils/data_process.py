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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from data_processor.vendor_specific.polycam import process_polycam_from_config
from segment3d.identify_objects.orchestrator import identify_all_frames_from_config
from segment3d.identify_objects.normalize_labels import normalize_labels_from_config

# Lightweight step metadata lives in its own module so the Flask server can
# import it without pulling in the heavy ML stack above.
from server.utils.pipeline_steps import (
    DataSource,
    PipelineStep,
    get_pipeline_steps,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────
# Working directory inside the Docker container (volume-mounted project root).
_WORKDIR = Path("/app")


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
        # Stream output through a pipe rather than handing sys.stdout/stderr
        # directly to the child. Under a Celery worker, sys.stdout/stderr are
        # ``LoggingProxy`` objects with no ``fileno()``, which subprocess needs
        # for an inherited fd. Reading the pipe and re-emitting via write()
        # keeps GPU-heavy steps visible and works under both Celery and a TTY.
        with subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
            returncode = proc.wait()

        elapsed = time.monotonic() - start

        if returncode != 0:
            return StepResult(
                name=step.name,
                success=False,
                duration_seconds=elapsed,
                return_code=returncode,
                error=f"Process exited with code {returncode}",
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


# ── Public API ────────────────────────────────────────────────────────────

# Callback invoked as steps start/finish so callers can track live progress.
# Receives a dict with at least ``event`` ("step_start" | "step_complete"),
# ``name``, ``index`` (1-based), and ``total``.  On "step_complete" it also
# carries ``success``, ``duration_seconds``, and ``error``.
ProgressCallback = Callable[[Dict[str, Any]], None]


def process_data(
    config: Dict[str, Any],
    config_path: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
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

    # Resolve + validate the data source and its ordered step list.
    steps = get_pipeline_steps(data_source_str)
    data_source = DataSource(data_source_str.lower())

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

        if progress_callback is not None:
            progress_callback({
                "event": "step_start",
                "name": step.name,
                "description": step.description,
                "index": i,
                "total": total,
            })

        step_result = _run_step(step, config, config_path or "")
        pipeline_result.steps.append(step_result)

        if progress_callback is not None:
            progress_callback({
                "event": "step_complete",
                "name": step.name,
                "index": i,
                "total": total,
                "success": step_result.success,
                "duration_seconds": step_result.duration_seconds,
                "error": step_result.error,
            })

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
    _example_steps = [s.name for s in get_pipeline_steps("polycam")]

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
