"""
Model-agnostic orchestrator for component captioning pipeline.
"""

from __future__ import annotations

import argparse
import json
import time as time_module
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_io import get_output_path
from ..utils.save_runtime_stats import save_runtime_stats
from .captioner_base import Captioner, create_captioner


def get_top_images(
    component_id: str | int, manifest_data: Dict[str, Any], n: int = 5
) -> List[Dict[str, Any]]:
    """
    Get the top n images with the most visible points for a component.

    Args:
        component_id: ID of the component
        manifest_data: Dictionary loaded from manifest.json
        n: Number of top images to return (default: 5)

    Returns:
        List of dictionaries containing crop information, sorted by visible points
    """
    comp_id_str = str(component_id)

    if comp_id_str not in manifest_data:
        raise ValueError(f"Component {comp_id_str} not found in manifest")

    component_data = manifest_data[comp_id_str]
    crops = component_data.get("crops", [])

    # Sort by visible_points descending
    sorted_crops = sorted(crops, key=lambda x: x.get("visible_points", 0), reverse=True)

    # Return top n
    return sorted_crops[:n]


def caption_all_components_cli(
    dataset_name: str,
    n_images: int = 1,
    captioner_type: str = "vllm",
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    device: int = 0,
    max_components: Optional[int] = None,
    batch_size: int = 16,
    **captioner_kwargs,
) -> None:
    """
    Caption all components by reading manifest.json and bbox_corners.json.

    This is the main orchestrator function that:
    1. Loads the manifest and configuration
    2. Creates a captioner instance
    3. Batches components and calls the captioner
    4. Saves results and statistics

    Args:
        dataset_name: Name of the dataset to process
        n_images: Number of top images to use for each component
        captioner_type: Type of captioner to use (e.g., "vllm")
        model: Name of the model to use
        device: GPU device ID to use for inference
        max_components: Maximum number of components to process (None for all)
        batch_size: Number of components to process in each batch
        **captioner_kwargs: Additional arguments to pass to the captioner
    """
    outputs_dir = get_output_path(dataset_name)
    crops_dir = outputs_dir / "crops"

    print(f"Outputs directory: {outputs_dir}")
    print(f"Crops directory: {crops_dir}")
    print(f"Captioner type: {captioner_type}")
    print(f"Model: {model}")
    print(f"Device: {device}")
    print(f"Images per component: {n_images}")
    print(f"Batch size: {batch_size}")

    # Create captioner instance
    captioner = create_captioner(
        captioner_type=captioner_type,
        model=model,
        device=device,
        **captioner_kwargs,
    )

    # Load manifest
    manifest_path = crops_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Please run the crop generation step first."
        )

    print(f"\nLoading manifest from: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Get list of component IDs to process
    component_ids = sorted(manifest_data.keys(), key=int)

    # Limit the number of components if specified
    if max_components is not None:
        component_ids = component_ids[:max_components]
        print(
            f"\nLimiting to {max_components} components (out of {len(manifest_data)} total)"
        )
    else:
        print(f"\nFound {len(component_ids)} components to caption")

    # Caption components in batches
    all_captions = []
    total_components = len(component_ids)

    # Start timing
    start_time = time_module.time()

    for batch_start in range(0, total_components, batch_size):
        batch_end = min(batch_start + batch_size, total_components)
        batch_component_ids = component_ids[batch_start:batch_end]

        print(
            f"\n{'='*60}\nProcessing batch [{batch_start + 1}-{batch_end}] of {total_components} components\n{'='*60}"
        )

        # Prepare batch data and store top_images_info for later use
        batch_data = []
        component_metadata = {}  # Store top_images_info for each component
        for component_id in batch_component_ids:
            try:
                top_images_info = get_top_images(
                    component_id, manifest_data, n=n_images
                )
                batch_data.append((component_id, top_images_info))
                component_metadata[component_id] = top_images_info
            except Exception as e:
                print(f"Error preparing component {component_id}: {e}")
                continue

        if not batch_data:
            print("No valid components in this batch, skipping")
            continue

        # Process batch using the captioner
        try:
            batch_results = captioner.caption_batch(batch_data, crops_dir)

            # Process results
            for result in batch_results:
                component_id = result.component_id
                caption = result.caption
                comp_id_int = int(component_id)

                # Get the top images info from metadata (already retrieved)
                top_images_info = component_metadata.get(component_id, [])
                crop_filenames = [img["crop_filename"] for img in top_images_info]

                print(
                    f"\nComponent {component_id}:\n  Images: {', '.join(crop_filenames[:3])}{'...' if len(crop_filenames) > 3 else ''}"
                )
                print(
                    f"  Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}"
                )

                component_result = {
                    "component_id": comp_id_int,
                    "caption": caption,
                    "num_images_used": len(top_images_info),
                    "crop_filenames": crop_filenames,
                }

                all_captions.append(component_result)

        except Exception as e:
            print(f"Error processing batch: {e}")
            traceback.print_exc()
            continue

    # Clean up captioner resources
    try:
        captioner.cleanup()
    except Exception as e:
        print(f"Warning: Error during captioner cleanup: {e}")

    # Calculate timing statistics
    end_time = time_module.time()
    total_runtime = end_time - start_time
    total_captions_generated = len(all_captions)

    # Calculate rates
    captions_per_second = (
        total_captions_generated / total_runtime if total_runtime > 0 else 0
    )
    time_per_caption = (
        total_runtime / total_captions_generated if total_captions_generated > 0 else 0
    )

    # Save captions
    output_path = outputs_dir / "component_captions.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_captions, f, indent=2)

    # Save caption statistics
    
    step_stats = {
        "duration_seconds": total_runtime,
        "status": "completed",
        "parameters": {
            "total_captions_generated": total_captions_generated,
            "captions_per_second": captions_per_second,
            "time_per_caption_seconds": time_per_caption,
            "batch_size": batch_size,
            "n_images_per_component": n_images,
            "captioner_type": captioner_type,
            "model": model,
            "device": device,
        }
    }
    
    save_runtime_stats(dataset_name, "captioning", step_stats)
    stats_output_path = outputs_dir / "runtime_stats.json"

    print(f"\n{'='*60}")
    print(f"Captioning complete!")
    print(f"Successfully captioned {len(all_captions)}/{len(manifest_data)} components")
    print(f"Captions saved to: {output_path}")
    print(f"Statistics saved to: {stats_output_path}")

    # Print timing statistics
    print(f"\nTiming statistics:")
    print(
        f"  Total runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)"
    )
    print(f"  Total captions generated: {total_captions_generated}")
    print(f"  Captions per second: {captions_per_second:.2f}")
    print(f"  Time per caption: {time_per_caption:.2f} seconds")

    # Print caption length statistics
    caption_lengths = [len(c["caption"]) for c in all_captions]
    if caption_lengths:
        print(f"\nCaption length statistics:")
        print(
            f"  Average length: {sum(caption_lengths) / len(caption_lengths):.1f} characters"
        )
        print(f"  Min length: {min(caption_lengths)} characters")
        print(f"  Max length: {max(caption_lengths)} characters")


def main() -> None:
    """Main entry point for CLI."""

    parser = argparse.ArgumentParser(
        description="Caption components using a Vision Language Model"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the dataset to process",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=1,
        help="Number of top images to use per component (default: 1)",
    )
    parser.add_argument(
        "--captioner-type",
        type=str,
        default="vllm",
        help="Type of captioner to use (default: vllm)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model to use (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID to use (default: 0)",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Maximum number of components to process (default: process all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Number of components to process in each batch (default: 512)",
    )

    args = parser.parse_args()

    if args.n_images < 1:
        parser.error("--n-images must be at least 1")

    if args.max_components is not None and args.max_components < 1:
        parser.error("--max-components must be at least 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    caption_all_components_cli(
        dataset_name=args.dataset,
        n_images=args.n_images,
        captioner_type=args.captioner_type,
        model=args.model,
        device=args.device,
        max_components=args.max_components,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
