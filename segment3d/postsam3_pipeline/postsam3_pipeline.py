#!/usr/bin/env python3
"""
Post-SAM3 pipeline script that runs all steps after SAM3 segmentation.

This script orchestrates the per-object pipeline starting from 2D-3D association:
1. Associate per-object SAM3 masks with COLMAP 3D points
2. Build object mask connectivity graph
3. Clean connected components (DBSCAN noise removal + splitting)
4. Compute 3D bounding boxes for each connected component
5. Segment (crop) images using connected component masks
6. Generate captions with VLM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .associate2d3d import associate_per_object
from .clean_components import clean_connected_components
from .default_params import DEFAULT_PARAMETERS
from .dummy_caption import generate_dummy_captions
from .mask_graph import build_object_mask_graph
from .segment_crops import segment_crops_cli
from .bbox_corners import get_all_bbox_corners_cli
# from ..captioning import caption_all_components_cli
# from ..clip_embed import generate_clip_embeddings_cli
from ..utils.save_runtime_stats import save_runtime_stats

from config_io import get_images_output_path, get_output_path, get_colmap_output_path


def print_step_header(step_num: int, total_steps: int, title: str) -> None:
    """Print a formatted step header."""
    print("\n" + "=" * 80)
    print(f"STEP {step_num}/{total_steps}: {title}")
    print("=" * 80 + "\n")


def print_step_complete(elapsed_time: float) -> None:
    """Print step completion message."""
    print(f"\n✓ Step completed in {elapsed_time:.2f} seconds")


def run_pipeline(
    dataset_name: str,
    skip_association: bool = False,
    skip_graph: bool = False,
    skip_clean: bool = False,
    skip_bbox: bool = False,
    skip_segment_crops: bool = False,
    skip_caption: bool = False,
    skip_clip: bool = False,
    # Mask graph parameters
    K: int = DEFAULT_PARAMETERS["K"],
    tau: float = DEFAULT_PARAMETERS["tau"],
    min_points: int = DEFAULT_PARAMETERS["min_points"],
    min_points_in_3d_segment: int = DEFAULT_PARAMETERS["min_points_in_3d_segment"],
    intersection_type: str = DEFAULT_PARAMETERS["intersection_type"],
    voxel_size_cm: float = DEFAULT_PARAMETERS["voxel_size_cm"],
    clip_distance_threshold: float = DEFAULT_PARAMETERS["clip_distance_threshold"],
    save_segment_images: bool = DEFAULT_PARAMETERS["save_segment_images"],
    # Segment-level DBSCAN parameters (2D-3D association noise filtering)
    segment_dbscan_eps: float = DEFAULT_PARAMETERS["segment_dbscan_eps"],
    segment_dbscan_min_samples: int = DEFAULT_PARAMETERS["segment_dbscan_min_samples"],
    # Objects to discard during 2D-3D association
    discard_objects_list: list = DEFAULT_PARAMETERS["discard_objects_list"],
    # Clean components parameters (component-level DBSCAN)
    component_dbscan_eps: float = DEFAULT_PARAMETERS["component_dbscan_eps"],
    component_dbscan_min_samples: int = DEFAULT_PARAMETERS[
        "component_dbscan_min_samples"
    ],
    component_dbscan_min_points: int = DEFAULT_PARAMETERS[
        "component_dbscan_min_points"
    ],
    split_components: bool = DEFAULT_PARAMETERS["split_components"],
    # Bounding box parameters
    percentile: float = DEFAULT_PARAMETERS["percentile"],
    # Segment crops parameters
    crop_type: str = DEFAULT_PARAMETERS["crop_type"],
    top_n: int = DEFAULT_PARAMETERS["top_n"],
    min_fraction: float = DEFAULT_PARAMETERS["min_fraction"],
    # Captioning parameters
    caption_n_images: int = DEFAULT_PARAMETERS["caption_n_images"],
    captioner_type: str = DEFAULT_PARAMETERS["captioner_type"],
    caption_model: str = DEFAULT_PARAMETERS["caption_model"],
    caption_device: int = DEFAULT_PARAMETERS["caption_device"],
    caption_batch_size: int = DEFAULT_PARAMETERS["caption_batch_size"],
    # CLIP embedding parameters
    clip_model: str = DEFAULT_PARAMETERS["clip_model"],
    clip_pretrained: str = DEFAULT_PARAMETERS["clip_pretrained"],
    clip_batch_size: int = DEFAULT_PARAMETERS["clip_batch_size"],
    clip_device: int = DEFAULT_PARAMETERS["clip_device"],
) -> None:
    """
    Run the post-SAM3 pipeline (all steps after SAM3 segmentation).

    Args:
        dataset_name: Name of the dataset to process
        skip_association: Skip 2D-3D association step
        skip_graph: Skip object mask graph building step
        skip_clean: Skip DBSCAN component cleaning step
        skip_bbox: Skip 3D bounding box computation step
        skip_segment_crops: Skip image cropping step
        skip_caption: Skip VLM captioning step
        skip_clip: Skip CLIP embedding generation step
        K: Min overlap count threshold for mask graph edges
        tau: Min Jaccard similarity threshold for mask graph edges
        min_points: Min 3D points for a node to be included in the graph
        min_points_in_3d_segment: Min 3D points in a component to be reported
        intersection_type: How to measure instance overlap: "geometric" (voxel
            Jaccard over 3D space) or "id_based" (Jaccard over raw point IDs)
        voxel_size_cm: Voxel side length in centimetres when intersection_type="geometric"
            (point coordinates are assumed to be in metres)
        clip_distance_threshold: Max cosine distance between OpenCLIP ViT-H-14 image
            embeddings of two mask images for them to be merged.  None disables the check.
        save_segment_images: When True, save each node's representative masked-crop
            image to outputs/{dataset}/graph_node_mask_images/ for visual inspection.
        segment_dbscan_eps: DBSCAN neighbourhood radius for segment-level noise filtering during 2D-3D association
        segment_dbscan_min_samples: DBSCAN minimum samples per core point for segment-level filtering
        discard_objects_list: Object labels (case-insensitive) to skip during 2D-3D association
        component_dbscan_eps: Component-level DBSCAN neighbourhood radius in world units
        component_dbscan_min_samples: Component-level DBSCAN minimum samples per core point
        component_dbscan_min_points: Drop components with fewer than this many points after cleaning
        split_components: Split multi-cluster components into separate components (default: False)
        percentile: Percentile threshold for bbox outlier removal
        crop_type: Cropping method – ``"segment"`` uses per-object mask crops
            (segment_crops.py); ``"bbox"`` projects the 3D bounding box to 2D
            and crops to that region (project_bbox + crop_images, as in main.py)
        top_n: Number of top frames to crop per component
        min_fraction: Minimum visibility fraction to consider a frame for cropping
        caption_n_images: Number of top images to use for captioning
        captioner_type: Type of captioner to use (e.g., "vllm")
        caption_model: VLM model to use for captioning
        caption_device: GPU device ID for captioning
        caption_batch_size: Batch size for captioning inference
        clip_model: OpenCLIP model name for embeddings
        clip_pretrained: Pretrained weights for CLIP model
        clip_batch_size: Batch size for CLIP embedding generation
        clip_device: GPU device ID for CLIP embeddings
    """
    print("\n" + "=" * 80)
    print("POST-SAM3 PIPELINE (per-object)")
    print("=" * 80)

    os.environ["SCAN_TO_MAP_DATASET"] = dataset_name

    try:
        images_dir = get_images_output_path(dataset_name)
        outputs_dir = get_output_path(dataset_name)
        colmap_dir = get_colmap_output_path(dataset_name)
        print(f"\nConfiguration loaded for dataset: {dataset_name}")
        print(f"  Images directory: {images_dir}")
        print(f"  COLMAP model: {colmap_dir}")
        print(f"  Outputs directory: {outputs_dir}")
    except Exception as e:
        print(f"\nError loading configuration: {e}")
        sys.exit(1)

    print("\nPipeline Parameters:")
    if skip_association:
        print("  2D-3D association: SKIPPED")
    if skip_graph:
        print("  Mask graph building: SKIPPED")
    if skip_clean:
        print("  Component cleaning: SKIPPED")
    if skip_bbox:
        print("  3D bounding boxes: SKIPPED")
    if skip_segment_crops:
        print("  Image cropping: SKIPPED")
    if skip_caption:
        print("  VLM captioning: SKIPPED")
    if skip_clip:
        print("  CLIP embedding generation: SKIPPED")
    print(f"  Mask graph K: {K}")
    print(f"  Mask graph tau: {tau}")
    print(f"  Mask graph min_points: {min_points}")
    print(f"  Mask graph min_points_in_3d_segment: {min_points_in_3d_segment}")
    print(f"  Mask graph intersection_type: {intersection_type}")
    if intersection_type == "geometric":
        print(f"  Mask graph voxel_size_cm: {voxel_size_cm}")
    print(f"  Mask graph clip_distance_threshold: {clip_distance_threshold}")
    print(f"  Mask graph save_segment_images: {save_segment_images}")
    if not skip_association:
        print(f"  Segment DBSCAN eps: {segment_dbscan_eps}")
        print(f"  Segment DBSCAN min_samples: {segment_dbscan_min_samples}")
        print(f"  Discard objects: {discard_objects_list}")
    if not skip_clean:
        print(f"  Component DBSCAN eps: {component_dbscan_eps}")
        print(f"  Component DBSCAN min_samples: {component_dbscan_min_samples}")
        print(f"  Component DBSCAN min_points: {component_dbscan_min_points}")
        print(f"  Split components: {split_components}")
    if not skip_bbox:
        print(f"  Bbox percentile: {percentile}")
    print(f"  Crop type: {crop_type}")
    print(f"  Segment crops top_n: {top_n}")
    print(f"  Segment crops min_fraction: {min_fraction}")
    if not skip_caption:
        print(f"  Captioner type: {captioner_type}")
        print(f"  Caption model: {caption_model}")
        print(f"  Caption n_images: {caption_n_images}")
        print(f"  Caption device: {caption_device}")
        print(f"  Caption batch size: {caption_batch_size}")
    if not skip_clip:
        print(f"  CLIP model: {clip_model}")
        print(f"  CLIP pretrained: {clip_pretrained}")
        print(f"  CLIP batch size: {clip_batch_size}")
        print(f"  CLIP device: {clip_device}")

    pipeline_start = time.time()
    total_steps = 7
    current_step = 0


    try:
        # Step 1: Associate 2D-3D
        if not skip_association:
            current_step += 1
            print_step_header(
                current_step, total_steps, "Associate 2D Masks with 3D Points"
            )
            step_start = time.time()

            associate_per_object(
                dataset_name=dataset_name,
                segment_dbscan_eps=segment_dbscan_eps,
                segment_dbscan_min_samples=segment_dbscan_min_samples,
                discard_objects_list=discard_objects_list,
            )

            step_time = time.time() - step_start
            save_runtime_stats(dataset_name, "associate_2d_3d", {
                "duration_seconds": step_time,
                "status": "completed",
                "parameters": {
                    "segment_dbscan_eps": segment_dbscan_eps,
                    "segment_dbscan_min_samples": segment_dbscan_min_samples,
                    "discard_objects_list": discard_objects_list,
                }
            })
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 1: 2D-3D Association (using existing associations)")

        # Step 2: Build object mask graph
        if not skip_graph:
            current_step += 1
            print_step_header(
                current_step, total_steps, "Build Object Mask Connectivity Graph"
            )
            step_start = time.time()

            build_object_mask_graph(
                dataset_name=dataset_name,
                K=K,
                tau=tau,
                min_points=min_points,
                min_points_in_3d_segment=min_points_in_3d_segment,
                intersection_type=intersection_type,
                voxel_size_cm=voxel_size_cm,
                clip_distance_threshold=clip_distance_threshold,
                save_segment_images=save_segment_images,
            )

            step_time = time.time() - step_start
            save_runtime_stats(dataset_name, "build_mask_graph", {
                "duration_seconds": step_time,
                "status": "completed",
                "parameters": {
                    "K": K,
                    "tau": tau,
                    "min_points": min_points,
                    "min_points_in_3d_segment": min_points_in_3d_segment,
                    "intersection_type": intersection_type,
                    "voxel_size_cm": voxel_size_cm,
                    "clip_distance_threshold": clip_distance_threshold,
                    "save_segment_images": save_segment_images,
                }
            })
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 2: Mask Graph Building (using existing graph)")

        # Step 3: Clean connected components
        if not skip_clean:
            current_step += 1
            print_step_header(
                current_step, total_steps, "Clean Connected Components (DBSCAN)"
            )
            step_start = time.time()

            clean_connected_components(
                dataset_name=dataset_name,
                eps=component_dbscan_eps,
                min_samples=component_dbscan_min_samples,
                min_points=component_dbscan_min_points,
                split_components=split_components,
            )

            step_time = time.time() - step_start
            save_runtime_stats(dataset_name, "clean_components", {
                "duration_seconds": step_time,
                "status": "completed",
                "parameters": {
                    "component_dbscan_eps": component_dbscan_eps,
                    "component_dbscan_min_samples": component_dbscan_min_samples,
                    "component_dbscan_min_points": component_dbscan_min_points,
                    "split_components": split_components,
                }
            })
            print_step_complete(step_time)
        else:
            print(
                "\nSkipping Step 3: Component Cleaning (using existing connected_components.json)"
            )

        # Step 4: Compute 3D bounding boxes
        if not skip_bbox:
            current_step += 1
            print_step_header(current_step, total_steps, "Compute 3D Bounding Boxes")
            step_start = time.time()

            get_all_bbox_corners_cli(dataset_name=dataset_name, percentile=percentile)

            step_time = time.time() - step_start
            save_runtime_stats(dataset_name, "compute_bboxes", {
                "duration_seconds": step_time,
                "status": "completed",
                "parameters": {
                    "percentile": percentile,
                }
            })
            print_step_complete(step_time)
        else:
            print(
                "\nSkipping Step 4: 3D Bounding Box Computation (using existing bbox_corners.json)"
            )

        # Step 5: Segment/crop images
        if not skip_segment_crops:
            current_step += 1
            print_step_header(current_step, total_steps, "Segment and Crop Images")
            step_start = time.time()
            
            segment_crops_cli(
                dataset_name=dataset_name,
                top_n=top_n,
                min_fraction=min_fraction,
            )

            step_time = time.time() - step_start
            save_runtime_stats(dataset_name, "segment_crops", {
                "duration_seconds": step_time,
                "status": "completed",
                "parameters": {
                    "crop_type": crop_type,
                    "top_n": top_n,
                    "min_fraction": min_fraction,
                }
            })
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 5: Image Cropping (using existing crops)")

        # Step 6: Caption components
        if not skip_caption:
            current_step += 1
            print_step_header(current_step, total_steps, "Generate Captions with VLM")
            step_start = time.time()

            caption_all_components_cli(
                dataset_name=dataset_name,
                n_images=caption_n_images,
                captioner_type=captioner_type,
                model=caption_model,
                device=caption_device,
                batch_size=caption_batch_size,
            )

            step_time = time.time() - step_start
            save_runtime_stats(dataset_name, "caption_components", {
                "duration_seconds": step_time,
                "status": "completed",
                "parameters": {
                    "caption_n_images": caption_n_images,
                    "captioner_type": captioner_type,
                    "caption_model": caption_model,
                    "caption_device": caption_device,
                    "caption_batch_size": caption_batch_size,
                }
            })
            print_step_complete(step_time)
        else:
            print("\nSkipping Step 6: VLM Captioning (generating dummy captions)")
            step_start = time.time()
            generate_dummy_captions(dataset_name=dataset_name)
            step_time = time.time() - step_start
            save_runtime_stats(dataset_name, "caption_components", {
                "duration_seconds": step_time,
                "status": "skipped_dummy",
            })

        # # Step 7: Generate CLIP embeddings
        # if not skip_clip:
        #     current_step += 1
        #     print_step_header(current_step, total_steps, "Generate CLIP Embeddings")
        #     step_start = time.time()

        #     generate_clip_embeddings_cli(
        #         dataset_name=dataset_name,
        #         model_name=clip_model,
        #         pretrained=clip_pretrained,
        #         device=clip_device,
        #         batch_size=clip_batch_size,
        #     )

        #     step_time = time.time() - step_start
        #     save_runtime_stats(dataset_name, "clip_embeddings", {
        #         "duration_seconds": step_time,
        #         "status": "completed",
        #         "parameters": {
        #             "clip_model": clip_model,
        #             "clip_pretrained": clip_pretrained,
        #             "clip_batch_size": clip_batch_size,
        #             "clip_device": clip_device,
        #         }
        #     })
        #     print_step_complete(step_time)
        # else:
        #     print("\nSkipping Step 7: CLIP Embedding Generation")
        #     save_runtime_stats(dataset_name, "clip_embeddings", {
        #         "duration_seconds": 0,
        #         "status": "skipped",
        #     })

        # Pipeline complete
        total_time = time.time() - pipeline_start

        outputs_dir = get_output_path(dataset_name)
        runtime_stats_path = outputs_dir / "runtime_stats.json"

        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(
            f"\nTotal execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
        )
        print(f"\nResults saved to: {outputs_dir}")
        print(f"Runtime statistics saved to: {runtime_stats_path}")
        print("\nOutput structure:")
        print("  ├── object_level_masks/")
        print("  │   ├── masks/              - SAM3 per-object mask JSON files (input)")
        print("  │   └── object_3d_associations.json")
        print("  ├── connected_components.json")
        print("  ├── mask_graph_stats.json")
        print("  ├── bbox_corners.json       - 3D bounding box corners")
        print("  ├── bbox_stats.json")
        print("  ├── crops/                  - Masked & cropped image regions")
        print("  │   ├── component_0/")
        print("  │   ├── component_1/")
        print("  │   └── manifest.json")
        print("  ├── component_captions.json - VLM-generated captions")
        print("  ├── clip_embeddings.json    - CLIP embeddings (JSON)")
        print("  ├── clip_embeddings.npz     - CLIP embeddings (numpy)")
        print("  ├── clip_embeddings.faiss   - FAISS HNSW index")
        print("  └── runtime_stats.json")

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed at step {current_step}: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the post-SAM3 per-object pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Pipeline Steps:
  1. Associate per-object SAM3 masks with COLMAP 3D points
  2. Build object mask connectivity graph and extract connected components
  3. Clean connected components (DBSCAN noise removal + multi-cluster splitting)
  4. Compute 3D bounding boxes for each connected component
  5. Segment and crop images using connected component masks
  6. Generate captions for each component using VLM

Configuration:
  Dataset configurations are defined in segment3d/config.py
        """,
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to process (required)",
    )

    # Skip flags
    parser.add_argument(
        "--skip-association",
        action="store_true",
        help="Skip 2D-3D association step (use existing associations)",
    )
    parser.add_argument(
        "--skip-graph",
        action="store_true",
        help="Skip mask graph building step (use existing graph)",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip DBSCAN component cleaning step (use existing connected_components.json)",
    )
    parser.add_argument(
        "--skip-bbox",
        action="store_true",
        help="Skip 3D bounding box computation step (use existing bbox_corners.json)",
    )
    parser.add_argument(
        "--skip-segment-crops",
        action="store_true",
        help="Skip image cropping step (use existing crops)",
    )
    parser.add_argument(
        "--skip-caption",
        action="store_true",
        help="Skip VLM captioning step",
    )
    parser.add_argument(
        "--skip-clip",
        action="store_true",
        help="Skip CLIP embedding generation step",
    )

    # Mask graph parameters
    parser.add_argument(
        "--K",
        type=int,
        default=DEFAULT_PARAMETERS["K"],
        help=f"Min overlap count threshold for mask graph edges (default: {DEFAULT_PARAMETERS['K']})",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=DEFAULT_PARAMETERS["tau"],
        help=f"Min Jaccard similarity threshold for mask graph edges (default: {DEFAULT_PARAMETERS['tau']})",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=DEFAULT_PARAMETERS["min_points"],
        help=f"Min 3D points for a node to be included in the graph (default: {DEFAULT_PARAMETERS['min_points']})",
    )
    parser.add_argument(
        "--min-points-in-3d-segment",
        type=int,
        default=DEFAULT_PARAMETERS["min_points_in_3d_segment"],
        help=f"Min 3D points in a connected component to be reported (default: {DEFAULT_PARAMETERS['min_points_in_3d_segment']})",
    )

    # Segment-level DBSCAN parameters (2D-3D association)
    parser.add_argument(
        "--segment-dbscan-eps",
        type=float,
        default=DEFAULT_PARAMETERS["segment_dbscan_eps"],
        help=f"Segment-level DBSCAN neighbourhood radius for 2D-3D association noise filtering (default: {DEFAULT_PARAMETERS['segment_dbscan_eps']})",
    )
    parser.add_argument(
        "--segment-dbscan-min-samples",
        type=int,
        default=DEFAULT_PARAMETERS["segment_dbscan_min_samples"],
        help=f"Segment-level DBSCAN minimum samples per core point (default: {DEFAULT_PARAMETERS['segment_dbscan_min_samples']})",
    )
    parser.add_argument(
        "--discard-objects",
        nargs="+",
        default=DEFAULT_PARAMETERS["discard_objects_list"],
        metavar="LABEL",
        help=f"Object labels (case-insensitive) to exclude from 2D-3D association "
        f"(default: {DEFAULT_PARAMETERS['discard_objects_list']})",
    )

    # Mask graph intersection type
    parser.add_argument(
        "--intersection-type",
        choices=["geometric", "id_based"],
        default=DEFAULT_PARAMETERS["intersection_type"],
        help=f"How to measure instance overlap: 'geometric' (voxel Jaccard) or 'id_based' (point-ID Jaccard) "
        f"(default: {DEFAULT_PARAMETERS['intersection_type']})",
    )
    parser.add_argument(
        "--voxel-size-cm",
        type=float,
        default=DEFAULT_PARAMETERS["voxel_size_cm"],
        help=f"Voxel side length in centimetres used when --intersection-type=geometric "
        f"(point coordinates assumed in metres; default: {DEFAULT_PARAMETERS['voxel_size_cm']})",
    )
    parser.add_argument(
        "--clip-distance-threshold",
        type=float,
        default=DEFAULT_PARAMETERS["clip_distance_threshold"],
        metavar="DIST",
        help=f"Maximum cosine distance between OpenCLIP ViT-H-14 image embeddings for two "
        f"nodes to be merged.  Range (0, 1] (default: {DEFAULT_PARAMETERS['clip_distance_threshold']}).",
    )
    parser.add_argument(
        "--save-segment-images",
        action="store_true",
        default=DEFAULT_PARAMETERS["save_segment_images"],
        help="Save each node's representative masked-crop image to "
        "outputs/{dataset}/graph_node_mask_images/ for visual inspection.",
    )

    # Clean components parameters (component-level DBSCAN)
    parser.add_argument(
        "--component-dbscan-eps",
        type=float,
        default=DEFAULT_PARAMETERS["component_dbscan_eps"],
        help=f"Component-level DBSCAN neighbourhood radius in world units (default: {DEFAULT_PARAMETERS['component_dbscan_eps']})",
    )
    parser.add_argument(
        "--component-dbscan-min-samples",
        type=int,
        default=DEFAULT_PARAMETERS["component_dbscan_min_samples"],
        help=f"Component-level DBSCAN minimum samples per core point (default: {DEFAULT_PARAMETERS['component_dbscan_min_samples']})",
    )
    parser.add_argument(
        "--component-dbscan-min-points",
        type=int,
        default=DEFAULT_PARAMETERS["component_dbscan_min_points"],
        help=f"Drop components with fewer than this many points after cleaning (default: {DEFAULT_PARAMETERS['component_dbscan_min_points']})",
    )
    parser.add_argument(
        "--split-components",
        action="store_true",
        default=DEFAULT_PARAMETERS["split_components"],
        help="Split components with multiple DBSCAN clusters into separate components (default: False)",
    )

    # Bounding box parameters
    parser.add_argument(
        "--percentile",
        type=float,
        default=DEFAULT_PARAMETERS["percentile"],
        help=f"Percentile threshold for bbox outlier removal (default: {DEFAULT_PARAMETERS['percentile']})",
    )

    # Segment crops parameters
    parser.add_argument(
        "--crop-type",
        choices=["segment", "bbox"],
        default=DEFAULT_PARAMETERS["crop_type"],
        help=f"Cropping method: 'segment' uses per-object mask crops (segment_crops.py); "
        f"'bbox' projects 3D bounding box to 2D and crops to that region "
        f"(project_bbox + crop_images, as in main.py). "
        f"(default: {DEFAULT_PARAMETERS['crop_type']})",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_PARAMETERS["top_n"],
        help=f"Number of top frames to crop per component (default: {DEFAULT_PARAMETERS['top_n']})",
    )
    parser.add_argument(
        "--min-fraction",
        type=float,
        default=DEFAULT_PARAMETERS["min_fraction"],
        help=f"Minimum visibility fraction to consider a frame for cropping (default: {DEFAULT_PARAMETERS['min_fraction']})",
    )

    # Captioning parameters
    parser.add_argument(
        "--caption-n-images",
        type=int,
        default=DEFAULT_PARAMETERS["caption_n_images"],
        help=f"Number of top images to use for captioning (default: {DEFAULT_PARAMETERS['caption_n_images']})",
    )
    parser.add_argument(
        "--captioner-type",
        type=str,
        default=DEFAULT_PARAMETERS["captioner_type"],
        help=f"Type of captioner to use (default: {DEFAULT_PARAMETERS['captioner_type']})",
    )
    parser.add_argument(
        "--caption-model",
        type=str,
        default=DEFAULT_PARAMETERS["caption_model"],
        help=f"VLM model to use for captioning (default: {DEFAULT_PARAMETERS['caption_model']})",
    )
    parser.add_argument(
        "--caption-device",
        type=int,
        default=DEFAULT_PARAMETERS["caption_device"],
        help=f"GPU device ID for captioning (default: {DEFAULT_PARAMETERS['caption_device']})",
    )
    parser.add_argument(
        "--caption-batch-size",
        type=int,
        default=DEFAULT_PARAMETERS["caption_batch_size"],
        help=f"Batch size for captioning inference (default: {DEFAULT_PARAMETERS['caption_batch_size']})",
    )

    # CLIP embedding parameters
    parser.add_argument(
        "--clip-model",
        type=str,
        default=DEFAULT_PARAMETERS["clip_model"],
        help=f"OpenCLIP model name for embeddings (default: {DEFAULT_PARAMETERS['clip_model']})",
    )
    parser.add_argument(
        "--clip-pretrained",
        type=str,
        default=DEFAULT_PARAMETERS["clip_pretrained"],
        help=f"Pretrained weights for CLIP model (default: {DEFAULT_PARAMETERS['clip_pretrained']})",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=DEFAULT_PARAMETERS["clip_batch_size"],
        help=f"Batch size for CLIP embedding generation (default: {DEFAULT_PARAMETERS['clip_batch_size']})",
    )
    parser.add_argument(
        "--clip-device",
        type=int,
        default=DEFAULT_PARAMETERS["clip_device"],
        help=f"GPU device ID for CLIP embeddings (default: {DEFAULT_PARAMETERS['clip_device']})",
    )

    args = parser.parse_args()

    # Validate parameters
    if args.tau <= 0 or args.tau > 1:
        parser.error("--tau must be between 0 and 1")
    if args.min_fraction <= 0 or args.min_fraction > 1:
        parser.error("--min-fraction must be between 0 and 1")
    if args.K < 1:
        parser.error("--K must be at least 1")
    if args.min_points < 1:
        parser.error("--min-points must be at least 1")
    if args.min_points_in_3d_segment < 1:
        parser.error("--min-points-in-3d-segment must be at least 1")
    if args.top_n < 1:
        parser.error("--top-n must be at least 1")
    if args.caption_n_images < 1:
        parser.error("--caption-n-images must be at least 1")
    if args.caption_batch_size < 1:
        parser.error("--caption-batch-size must be at least 1")
    if args.clip_batch_size < 1:
        parser.error("--clip-batch-size must be at least 1")
    if args.percentile <= 0 or args.percentile > 100:
        parser.error("--percentile must be between 0 and 100")
    if args.segment_dbscan_eps <= 0:
        parser.error("--segment-dbscan-eps must be positive")
    if args.segment_dbscan_min_samples < 1:
        parser.error("--segment-dbscan-min-samples must be at least 1")
    if args.component_dbscan_eps <= 0:
        parser.error("--component-dbscan-eps must be positive")
    if args.component_dbscan_min_samples < 1:
        parser.error("--component-dbscan-min-samples must be at least 1")
    if args.component_dbscan_min_points < 1:
        parser.error("--component-dbscan-min-points must be at least 1")

    run_pipeline(
        dataset_name=args.dataset,
        skip_association=args.skip_association,
        skip_graph=args.skip_graph,
        skip_clean=args.skip_clean,
        skip_bbox=args.skip_bbox,
        skip_segment_crops=args.skip_segment_crops,
        skip_caption=args.skip_caption,
        skip_clip=args.skip_clip,
        K=args.K,
        tau=args.tau,
        min_points=args.min_points,
        min_points_in_3d_segment=args.min_points_in_3d_segment,
        segment_dbscan_eps=args.segment_dbscan_eps,
        segment_dbscan_min_samples=args.segment_dbscan_min_samples,
        discard_objects_list=args.discard_objects,
        component_dbscan_eps=args.component_dbscan_eps,
        component_dbscan_min_samples=args.component_dbscan_min_samples,
        component_dbscan_min_points=args.component_dbscan_min_points,
        split_components=args.split_components,
        percentile=args.percentile,
        crop_type=args.crop_type,
        top_n=args.top_n,
        min_fraction=args.min_fraction,
        caption_n_images=args.caption_n_images,
        captioner_type=args.captioner_type,
        caption_model=args.caption_model,
        caption_device=args.caption_device,
        caption_batch_size=args.caption_batch_size,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_batch_size=args.clip_batch_size,
        clip_device=args.clip_device,
        intersection_type=args.intersection_type,
        voxel_size_cm=args.voxel_size_cm,
        clip_distance_threshold=args.clip_distance_threshold,
        save_segment_images=args.save_segment_images,
    )
