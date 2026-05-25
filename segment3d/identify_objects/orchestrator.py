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


def identify_all_frames_cli(
    dataset_name: str,
    identifier_type: str = "vllm",
    model: str = "Qwen/Qwen3-VL-8B-Instruct",
    device: int = 0,
    max_frames: Optional[int] = None,
    batch_size: int = 32,
    **identifier_kwargs,
) -> None:
    """
    Identify objects in every frame of a dataset and save the results.

    This is the main orchestrator function that:
    1. Loads the dataset configuration and discovers image frames
    2. Creates an identifier instance
    3. Batches frames and calls the identifier
    4. Saves the per-frame object lists to
       outputs/<dataset_name>/objects_inventory/objects_inventory.json

    Args:
        dataset_name: Name of the dataset to process
        identifier_type: Type of identifier to use (e.g., "vllm")
        model: Name of the model to use
        device: GPU device ID to use for inference
        max_frames: Maximum number of frames to process (None for all)
        batch_size: Number of frames to process in each batch
        **identifier_kwargs: Additional arguments to pass to the identifier
    """

    # Load configuration

    images_dir = get_images_output_path(dataset_name)
    outputs_dir = get_output_path(dataset_name)
    inventory_dir = outputs_dir / "objects_inventory"
    inventory_dir.mkdir(parents=True, exist_ok=True)

    print(f"Images directory:   {images_dir}")
    print(f"Output directory:   {inventory_dir}")
    print(f"Identifier type:    {identifier_type}  (supported: vllm, openai)")
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

    if max_frames is not None:
        all_frames = all_frames[:max_frames]
        print(f"\nLimiting to {max_frames} frames (out of {len(all_frames)} total)")
    else:
        print(f"\nFound {len(all_frames)} frames to process")

    # Create identifier instance
    identifier = create_identifier(
        identifier_type=identifier_type,
        model=model,
        device=device,
        **identifier_kwargs,
    )

    # Process frames in batches
    per_frame_objects: Dict[str, List[str]] = {}
    total_frames = len(all_frames)
    start_time = time_module.time()

    for batch_start in range(0, total_frames, batch_size):
        batch_end = min(batch_start + batch_size, total_frames)
        batch = all_frames[batch_start:batch_end]

        print(
            f"\n{'='*60}\n"
            f"Processing batch [{batch_start + 1}-{batch_end}] / {total_frames} frames\n"
            f"{'='*60}"
        )

        try:
            batch_results = identifier.identify_batch(batch)

            for result in batch_results:
                per_frame_objects[result.frame_name] = result.objects

                objects_preview = ", ".join(result.objects[:5])
                if len(result.objects) > 5:
                    objects_preview += f", ... ({len(result.objects)} total)"

                if result.error:
                    print(f"  [{result.frame_name}] ERROR: {result.error}")
                else:
                    print(f"  [{result.frame_name}] {objects_preview}")

        except Exception as e:
            print(f"Error processing batch [{batch_start + 1}-{batch_end}]: {e}")
            traceback.print_exc()
            # Record empty lists for frames in this batch so we don't lose them
            for frame_name, _ in batch:
                if frame_name not in per_frame_objects:
                    per_frame_objects[frame_name] = []

    # Clean up identifier resources
    try:
        identifier.cleanup()
    except Exception as e:
        print(f"Warning: Error during identifier cleanup: {e}")

    # Save statistics

    # Timing statistics
    end_time = time_module.time()
    total_runtime = end_time - start_time
    total_processed = len(per_frame_objects)
    fps = total_processed / total_runtime if total_runtime > 0 else 0

    # Save per-frame object lists
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
        }
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


def main() -> None:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Build an objects inventory by running a VLM on every frame"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to process",
    )
    parser.add_argument(
        "--identifier-type",
        type=str,
        default="vllm",
        choices=["vllm", "openai"],
        help="Type of identifier to use: 'vllm' (local GPU inference) or 'openai' (OpenAI API). Default: vllm",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model to use (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process (default: process all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of frames to process in each batch (default: 32)",
    )

    args = parser.parse_args()

    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    identify_all_frames_cli(
        dataset_name=args.dataset,
        identifier_type=args.identifier_type,
        model=args.model,
        device=args.device,
        max_frames=args.max_frames,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
