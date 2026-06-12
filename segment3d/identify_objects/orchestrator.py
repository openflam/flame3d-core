"""
Model-agnostic orchestrator for the objects inventory pipeline.

For each frame in the dataset, runs a VLM with IDENTIFY_COMPONENT_PROMPT and
stores the resulting list of tangible objects per frame as a JSON file at:

    outputs/<dataset_name>/objects_inventory/objects_inventory.json
"""

from __future__ import annotations

import argparse
import json
import time as time_module
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from config_io import get_images_output_path, get_output_path

from .identifier_base import Identifier, create_identifier
from ..utils.save_runtime_stats import save_runtime_stats

# Supported image extensions (case-insensitive)
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _discover_frames(images_dir: Path) -> List[tuple[str, Path]]:
    """
    Discover all image frames in a directory.

    Args:
        images_dir: Directory containing image files

    Returns:
        Sorted list of (frame_name, image_path) tuples where frame_name is the
        filename stem (e.g., "frame_00001").
    """
    frames = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() in _IMAGE_EXTENSIONS:
            frames.append((image_path.stem, image_path))
    return frames


def _frame_result_path(frames_dir: Path, frame_name: str) -> Path:
    """Path to the per-frame checkpoint file for a given frame."""
    return frames_dir / f"{frame_name}.json"


def _save_frame_result(frames_dir: Path, result) -> None:
    """
    Atomically write a single frame's result to its own JSON file.

    Writing to a temporary file and renaming it into place ensures a crash
    mid-write cannot leave a partial/corrupt checkpoint behind.
    """
    payload = {
        "frame_name": result.frame_name,
        "image_path": result.image_path,
        "objects": result.objects,
        "error": result.error,
    }
    final_path = _frame_result_path(frames_dir, result.frame_name)
    tmp_path = final_path.with_name(final_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(final_path)


def _load_frame_objects(frames_dir: Path, frame_name: str) -> Optional[List[str]]:
    """
    Load a frame's saved object list from its checkpoint file.

    Returns None if the file is missing or unreadable (treated as not yet
    computed), so it will be recomputed on the next run.
    """
    path = _frame_result_path(frames_dir, frame_name)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("objects", [])
    except (json.JSONDecodeError, OSError):
        return None


def identify_all_frames_cli(
    dataset_name: str,
    identifier_type: str = "vllm",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: int = 0,
    max_frames: Optional[int] = None,
    batch_size: int = 32,
    skip_processed_frames: bool = True,
    rate_limits: Optional[Dict] = None,
    **identifier_kwargs,
) -> None:
    """
    Identify objects in every frame of a dataset and save the results.

    This is the main orchestrator function that:
    1. Loads the dataset configuration and discovers image frames
    2. Creates an identifier instance
    3. Batches frames and calls the identifier, checkpointing each frame's
       result to outputs/<dataset_name>/objects_inventory/frames/<frame>.json
       as soon as it is ready (so progress survives a mid-run crash; frames
       already checkpointed are skipped to avoid duplicate computation)
    4. Assembles the per-frame checkpoints into
       outputs/<dataset_name>/objects_inventory/objects_inventory.json

    Args:
        dataset_name: Name of the dataset to process
        identifier_type: Type of identifier to use (e.g., "vllm")
        model: Name of the model to use
        device: GPU device ID to use for inference
        max_frames: Maximum number of frames to process (None for all)
        batch_size: Number of frames to process in each batch
        skip_processed_frames: If True (default), frames that already have a
            saved checkpoint are skipped (resume support). If False, every
            frame is recomputed and its checkpoint overwritten.
        rate_limits: Per-model RPM/TPM limits passed to the LLM API identifier's
            scheduler (None disables limiting). Ignored by the vLLM identifier.
        **identifier_kwargs: Additional arguments to pass to the identifier
    """

    # Load configuration

    images_dir = get_images_output_path(dataset_name)
    outputs_dir = get_output_path(dataset_name)
    inventory_dir = outputs_dir / "objects_inventory"
    frames_dir = inventory_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images directory:   {images_dir}")
    print(f"Output directory:   {inventory_dir}")
    print(f"Identifier type:    {identifier_type}  (supported: vllm, llmapi)")
    print(f"Model:              {model}")
    print(f"Device:             {device}")
    print(f"Batch size:         {batch_size}")

    # Discover frames
    all_frames = _discover_frames(images_dir)
    if not all_frames:
        raise FileNotFoundError(
            f"No image files found in {images_dir}. "
            "Supported extensions: " + ", ".join(sorted(_IMAGE_EXTENSIONS))
        )

    total_discovered = len(all_frames)
    if max_frames is not None:
        all_frames = all_frames[:max_frames]
        print(
            f"\nLimiting to {len(all_frames)} frames (out of {total_discovered} total)"
        )
    else:
        print(f"\nFound {len(all_frames)} frames to process")

    total_frames = len(all_frames)

    # Skip frames already checkpointed on a previous run (resume support),
    # unless skip_processed_frames is disabled (then recompute everything).
    if skip_processed_frames:
        frames_to_process = [
            (frame_name, image_path)
            for frame_name, image_path in all_frames
            if not _frame_result_path(frames_dir, frame_name).exists()
        ]
        already_done = total_frames - len(frames_to_process)
        if already_done:
            print(
                f"Resuming: {already_done} frame(s) already computed, "
                f"{len(frames_to_process)} remaining"
            )
    else:
        frames_to_process = list(all_frames)
        print("skip_processed_frames is False: recomputing all frames")

    # Process the remaining frames in batches, checkpointing each frame's
    # result to its own file as soon as it is ready.
    newly_processed = 0
    start_time = time_module.time()

    if frames_to_process:
        identifier = create_identifier(
            identifier_type=identifier_type,
            model=model,
            device=device,
            rate_limits=rate_limits,
            **identifier_kwargs,
        )

        remaining = len(frames_to_process)
        for batch_start in range(0, remaining, batch_size):
            batch_end = min(batch_start + batch_size, remaining)
            batch = frames_to_process[batch_start:batch_end]

            print(
                f"\n{'='*60}\n"
                f"Processing batch [{batch_start + 1}-{batch_end}] / {remaining} frames\n"
                f"{'='*60}"
            )

            try:
                batch_results = identifier.identify_batch(batch)

                for result in batch_results:
                    objects_preview = ", ".join(result.objects[:5])
                    if len(result.objects) > 5:
                        objects_preview += f", ... ({len(result.objects)} total)"

                    if result.error:
                        # Leave uncheckpointed so it is retried on the next run.
                        print(f"  [{result.frame_name}] ERROR: {result.error}")
                    else:
                        _save_frame_result(frames_dir, result)
                        newly_processed += 1
                        print(f"  [{result.frame_name}] {objects_preview}")

            except Exception as e:
                print(f"Error processing batch [{batch_start + 1}-{batch_end}]: {e}")
                traceback.print_exc()
                # Frames in this batch stay uncheckpointed and are retried later.

        # Clean up identifier resources
        try:
            identifier.cleanup()
        except Exception as e:
            print(f"Warning: Error during identifier cleanup: {e}")
    else:
        print("\nAll frames already computed; assembling inventory from checkpoints.")

    # Timing statistics (covers only frames computed in this run)
    end_time = time_module.time()
    total_runtime = end_time - start_time
    fps = newly_processed / total_runtime if total_runtime > 0 else 0

    # Assemble the final inventory by reading every per-frame checkpoint.
    # Frames without a checkpoint (never computed or persistently failing)
    # default to an empty object list.
    per_frame_objects: Dict[str, List[str]] = {}
    for frame_name, _ in all_frames:
        objects = _load_frame_objects(frames_dir, frame_name)
        per_frame_objects[frame_name] = objects if objects is not None else []

    total_processed = sum(
        1
        for frame_name, _ in all_frames
        if _frame_result_path(frames_dir, frame_name).exists()
    )

    # Save the assembled per-frame object lists
    output_path = inventory_dir / "objects_inventory.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(per_frame_objects, f, indent=2)

    # Object count statistics
    object_counts = [len(v) for v in per_frame_objects.values()]
    avg_objects = sum(object_counts) / len(object_counts) if object_counts else 0.0
    min_objects = min(object_counts) if object_counts else 0
    max_objects = max(object_counts) if object_counts else 0

    step_stats = {
        "duration_seconds": total_runtime,
        "status": "completed",
        "parameters": {
            "total_frames_processed": total_processed,
            "frames_per_second": fps,
            "batch_size": batch_size,
            "identifier_type": identifier_type,
            "model": model,
            "device": device,
            "avg_objects_per_frame": avg_objects,
            "min_objects_per_frame": min_objects,
            "max_objects_per_frame": max_objects,
        },
    }
    save_runtime_stats(dataset_name, "identify_frames", step_stats)

    stats_path = outputs_dir / "runtime_stats.json"

    print(f"\n{'='*60}")
    print(f"Objects inventory complete!")
    print(f"Processed {total_processed}/{total_frames} frames")
    print(f"Results saved to: {output_path}")
    print(f"Statistics saved to: {stats_path}")
    print(f"\nTiming statistics:")
    print(
        f"  Total runtime: {total_runtime:.2f} seconds ({total_runtime / 60:.2f} minutes)"
    )
    print(f"  Frames per second: {fps:.2f}")

    if object_counts:
        print(f"\nObjects per frame statistics:")
        print(f"  Average: {avg_objects:.1f}")
        print(f"  Min: {min_objects}")
        print(f"  Max: {max_objects}")


def identify_all_frames_from_config(config: dict, dataset_name: str) -> None:
    """Run the object identification pipeline using the config dictionary.

    Reads parameters from ``config["identify_objects"]``.  The dataset name is
    passed separately (it is not read from the config).
    """
    io_cfg = config.get("identify_objects", {})
    identify_all_frames_cli(
        dataset_name=dataset_name,
        identifier_type=io_cfg.get("identifier_type", "vllm"),
        model=io_cfg.get("model", "Qwen/Qwen3-VL-8B-Instruct"),
        device=io_cfg.get("device", 0),
        max_frames=io_cfg.get("max_frames"),
        batch_size=io_cfg.get("batch_size", 32),
        skip_processed_frames=io_cfg.get("skip_processed_frames", True),
        rate_limits=config.get("rate_limits"),
    )


def main() -> None:
    """Main entry point for CLI."""
    from config_cli import parse_config_args

    config, dataset_name = parse_config_args(
        description="Build an objects inventory by running a VLM on every frame",
    )
    identify_all_frames_from_config(config, dataset_name)


if __name__ == "__main__":
    main()
