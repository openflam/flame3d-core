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


# ---------------------------------------------------------------------------
# Config-dict entry point
# ---------------------------------------------------------------------------


def run_pipeline_from_config(config: dict) -> None:
    """Run the post-SAM3 pipeline using the master config dictionary.

    Reads ``config["dataset_name"]`` and ``config["postsam3_pipeline"]``.
    Any keys not present in the config fall back to ``DEFAULT_PARAMETERS``.
    """
    dataset_name = config["dataset_name"]
    ps_cfg = config.get("postsam3_pipeline", {})

    def _get(key: str):
        return ps_cfg.get(key, DEFAULT_PARAMETERS.get(key))

    run_pipeline(
        dataset_name=dataset_name,
        skip_association=ps_cfg.get("skip_association", False),
        skip_graph=ps_cfg.get("skip_graph", False),
        skip_clean=ps_cfg.get("skip_clean", False),
        skip_bbox=ps_cfg.get("skip_bbox", False),
        skip_segment_crops=ps_cfg.get("skip_segment_crops", False),
        skip_caption=ps_cfg.get("skip_caption", True),
        skip_clip=ps_cfg.get("skip_clip", False),
        K=_get("K"),
        tau=_get("tau"),
        min_points=_get("min_points"),
        min_points_in_3d_segment=_get("min_points_in_3d_segment"),
        intersection_type=_get("intersection_type"),
        voxel_size_cm=_get("voxel_size_cm"),
        clip_distance_threshold=_get("clip_distance_threshold"),
        save_segment_images=_get("save_segment_images"),
        segment_dbscan_eps=_get("segment_dbscan_eps"),
        segment_dbscan_min_samples=_get("segment_dbscan_min_samples"),
        discard_objects_list=_get("discard_objects_list"),
        component_dbscan_eps=_get("component_dbscan_eps"),
        component_dbscan_min_samples=_get("component_dbscan_min_samples"),
        component_dbscan_min_points=_get("component_dbscan_min_points"),
        split_components=_get("split_components"),
        percentile=_get("percentile"),
        crop_type=_get("crop_type"),
        top_n=_get("top_n"),
        min_fraction=_get("min_fraction"),
        caption_n_images=_get("caption_n_images"),
        captioner_type=_get("captioner_type"),
        caption_model=_get("caption_model"),
        caption_device=_get("caption_device"),
        caption_batch_size=_get("caption_batch_size"),
        clip_model=_get("clip_model"),
        clip_pretrained=_get("clip_pretrained"),
        clip_batch_size=_get("clip_batch_size"),
        clip_device=_get("clip_device"),
    )


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
        default=None,
        help="Dataset to process",
    )

    # Config file
    parser.add_argument(
        "--config",
        default=None,
        metavar="PATH",
        help="Path to master_config.json (overrides other args)",
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

    args = parser.parse_args()

    if args.config:
        import json as _json
        with open(args.config, "r", encoding="utf-8") as f:
            config = _json.load(f)
        run_pipeline_from_config(config)
    else:
        if args.dataset is None:
            parser.error("--dataset is required when --config is not provided")

        run_pipeline(
            dataset_name=args.dataset,
            skip_association=args.skip_association,
            skip_graph=args.skip_graph,
            skip_clean=args.skip_clean,
            skip_bbox=args.skip_bbox,
            skip_segment_crops=args.skip_segment_crops,
            skip_caption=args.skip_caption,
            skip_clip=args.skip_clip,
        )

