"""
default_params.py - Default parameter values for the per-object SAM3 pipeline.

All tuneable knobs for postsam3_pipeline are centralised here so they can be
imported and referenced consistently by the pipeline function signature, the
CLI argument parser, and any external callers.
"""

from __future__ import annotations

# fmt: off
DEFAULT_PARAMETERS: dict = {
    # ------------------------------------------------------------------
    # Segment-level DBSCAN (2D→3D association noise filtering)
    # Applied per per-frame mask *before* votes are accumulated across frames.
    # ------------------------------------------------------------------

    # Object labels (case-insensitive) whose masks should be excluded from
    # 2D-3D association entirely (e.g. structural elements that are not objects
    # of interest).
    "discard_objects_list": ["wall", "walls", "floor", "ceiling", "window"],

    # Neighbourhood radius in the same units as the COLMAP reconstruction
    # (typically metres).
    "segment_dbscan_eps": 0.5,

    # Minimum number of 3D points required to form a core sample.
    "segment_dbscan_min_samples": 5,
    
    # ------------------------------------------------------------------
    # Mask graph parameters
    # ------------------------------------------------------------------

    # How to measure instance overlap when building edges:
    #   "geometric" – voxelise each instance's 3D point cloud and compute
    #                 Jaccard over the voxel-occupancy sets.
    #   "id_based"  – compute Jaccard directly over the raw COLMAP point-ID sets.
    "intersection_type": "geometric",

    # Voxel side length in centimetres used when intersection_type="geometric".
    # Point coordinates are assumed to be in metres.
    "voxel_size_cm": 50.0,

    # Minimum shared-point overlap count for an edge to be created between two
    # mask nodes.  Used only when intersection_type="id_based": an edge is kept
    # when the number of shared COLMAP point IDs ≥ K *or* Jaccard ≥ tau.
    # Has no effect when intersection_type="geometric".
    "K": 5,

    # Minimum Jaccard similarity threshold for an edge to be kept (range 0–1].
    # When intersection_type="id_based": Jaccard is computed over raw COLMAP
    #   point-ID sets; an edge is kept when Jaccard ≥ tau *or* overlap ≥ K.
    # When intersection_type="geometric": Jaccard is computed over voxel-
    #   occupancy sets; an edge is kept only when Jaccard ≥ tau (K is ignored).
    "tau": 0.4,

    # Maximum cosine distance allowed between OpenCLIP ViT-H-14 image embeddings
    # of the two nodes' representative mask crops for an edge to be kept.
    # Range: (0, 1].  Smaller values enforce stricter visual similarity.
    "clip_distance_threshold": 0.3,

    # When True, save each node's representative masked-crop image to
    # outputs/{dataset}/graph_node_mask_images/ for visual inspection.
    "save_segment_images": True,

    # Minimum number of 3D points a mask node must have to be included in the
    # graph at all.
    "min_points": 1,

    # Minimum number of 3D points a connected component must contain to be
    # reported in the output.
    "min_points_in_3d_segment": 10,

    # ------------------------------------------------------------------
    # Component-level DBSCAN (connected-component cleaning)
    # Applied to the accumulated 3D point cloud for each connected component.
    # ------------------------------------------------------------------

    # Neighbourhood radius in world units.
    "component_dbscan_eps": 0.5,

    # Minimum samples per core point.
    "component_dbscan_min_samples": 5,

    # Discard components that have fewer than this many inlier points after
    # DBSCAN cleaning.
    "component_dbscan_min_points": 20,

    # When True, components with multiple DBSCAN clusters are split into
    # separate components.  When False (default) only noise points are removed.
    "split_components": True,

    # ------------------------------------------------------------------
    # Bounding box parameters
    # ------------------------------------------------------------------

    # Percentile used to clip outlier points before fitting the 3-D bounding
    # box (e.g. 95.0 means the most extreme 5 % of points are ignored).
    "percentile": 95.0,

    # ------------------------------------------------------------------
    # Segment crops parameters
    # ------------------------------------------------------------------

    # Cropping method.  Two options:
    #   "segment" – use the per-object mask crops from segment_crops.py.
    #   "bbox"    – project the 3D bounding box to 2D and crop to the projected
    #               region, replicating the behaviour of the main.py pipeline.
    "crop_type": "segment",

    # Number of top-ranked frames to crop per connected component.
    "top_n": 5,

    # Minimum fraction of the component's 3D points that must project into a
    # frame for that frame to be considered for cropping.
    "min_fraction": 0.05,

    # ------------------------------------------------------------------
    # Captioning parameters
    # ------------------------------------------------------------------

    # Number of top crop images passed to the VLM for generating a caption.
    "caption_n_images": 1,

    # Captioner backend to use (e.g. "vllm").
    "captioner_type": "vllm",

    # HuggingFace model ID for the VLM captioner.
    "caption_model": "Qwen/Qwen2.5-VL-7B-Instruct",

    # GPU device index for captioning inference.
    "caption_device": 0,

    # Batch size for captioning inference.
    "caption_batch_size": 32,

    # ------------------------------------------------------------------
    # CLIP embedding parameters
    # ------------------------------------------------------------------

    # OpenCLIP model name.
    "clip_model": "ViT-H-14",

    # Pretrained weights identifier for the CLIP model.
    "clip_pretrained": "laion2B-s32B-b79K",

    # Batch size for CLIP embedding generation.
    "clip_batch_size": 32,

    # GPU device index for CLIP embedding inference.
    "clip_device": 0,
}
# fmt: on
