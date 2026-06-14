"""
sam3_runner.py - Run SAM3 video predictor on per-object frame sequences.

Reads objects_to_frames.json (produced by the objects inventory step) and for
every (object, sequence) pair runs the SAM3 video predictor with a text prompt
equal to the object name.  Masks are saved to

    <outputs_dir>/object_level_masks/masks/<obj_slug>/seq_<i>/<frame_name>.json

Each JSON file is a list of per-instance annotations:
    [{"obj_id": <int>,              # SAM3 tracking ID (consistent within a sequence)
      "segmentation": <COCO RLE>,   # {"size": [H, W], "counts": "<utf-8 string>"}
      "area": <float>}              # mask area in pixels
     ...]

With --save-images, overlay images are written to

    <outputs_dir>/object_level_masks/images/<obj_slug>/seq_<i>/<frame_name>.jpg

using SAM3's render_masklet_frame visualization utility.

Usage:
    python sam3_runner.py --dataset ProjectLabStudio_inv_method
    python sam3_runner.py --dataset ProjectLabStudio_inv_method --objects "bucket" "table"
    python sam3_runner.py --dataset ProjectLabStudio_inv_method --resume
    python sam3_runner.py --dataset ProjectLabStudio_inv_method --save-images
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import orjson
import torch
from PIL import Image as _PILImage
from pycocotools import mask as mask_utils
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import load_frame, render_masklet_frame

from config_io import (
    get_colmap_output_path,
    get_images_output_path,
    get_output_path,
)
from segment3d.utils.colmap_io import load_colmap_model
from segment3d.utils.save_runtime_stats import save_runtime_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize(name: str) -> str:
    """Convert object name to a filesystem-safe directory name."""
    return name.replace(" ", "_").replace("/", "-").replace("\\", "-")


def _build_temp_jpeg_dir(
    frame_names: List[str],
    images_dir: Path,
    tmp_root: Optional[Path] = None,
) -> Tuple[Path, List[str]]:
    """
    Create a temporary directory containing symlinks named 0.jpg, 1.jpg, …
    pointing to the actual frame files.

    SAM3 expects a JPEG folder with integer-named files so it can sort them
    numerically.

    Returns:
        (tmp_dir, ordered_frame_names) where ordered_frame_names[i] is the
        original frame name corresponding to symlink i.jpg.
    """
    tmp_dir = Path(tempfile.mkdtemp(dir=tmp_root))
    for idx, frame_name in enumerate(frame_names):
        src = images_dir / f"{frame_name}.jpg"
        if not src.is_file():
            # fall back to .png
            src = images_dir / f"{frame_name}.png"
        if not src.is_file():
            raise FileNotFoundError(
                f"Image not found for frame '{frame_name}' in {images_dir}"
            )
        dst = tmp_dir / f"{idx}.jpg"
        dst.symlink_to(src)
    return tmp_dir, list(frame_names)


def _all_colmap_frame_stems(dataset_name: str) -> List[str]:
    """Return the filename stems of every image in the COLMAP model, sorted.

    COLMAP image names carry their extension (e.g. ``frame_00001.jpg``) but the
    rest of the pipeline addresses frames by stem (e.g. ``frame_00001``), so we
    strip the extension here.
    """
    colmap_dir = get_colmap_output_path(dataset_name)
    _, images, _ = load_colmap_model(str(colmap_dir))
    stems = {Path(image.name).stem for image in images.values()}
    return sorted(stems)


def _synthetic_objects_to_frames(
    objects_to_segment: List[str], dataset_name: str
) -> Dict[str, List[List[str]]]:
    """Build an in-memory objects_to_frames mapping.

    Every requested object is mapped to a single sequence containing all frames
    in the COLMAP model. Used when ``objects_to_segment`` is supplied instead of
    reading a stored ``objects_to_frames.json``.
    """
    all_frames = _all_colmap_frame_stems(dataset_name)
    if not all_frames:
        raise ValueError(
            f"No frames found in the COLMAP model for dataset {dataset_name!r}"
        )
    return {obj: [all_frames] for obj in objects_to_segment}


def _propagate_in_video(predictor: Any, session_id: str) -> Dict[int, Any]:
    """Stream propagation results from SAM3 and collect them by frame index."""
    outputs_per_frame: Dict[int, Any] = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def _mask_to_numpy(mask: Any) -> np.ndarray:
    """Convert a SAM3 mask (tensor or ndarray) to a uint8 numpy array (Fortran order)."""
    if isinstance(mask, torch.Tensor):
        arr = mask.detach().cpu().numpy()
    else:
        arr = np.asarray(mask)
    return np.asfortranarray(arr.astype(np.uint8))


def _save_sequence_masks(
    outputs_per_frame: Dict[int, Any],
    frame_names: List[str],
    masks_seq_dir: Path,
    images_seq_dir: Optional[Path] = None,
    images_dir: Optional[Path] = None,
    save_images: bool = False,
) -> None:
    """
    Persist per-frame mask predictions for one (object, sequence) pair.

    Each frame produces a JSON file – a list of per-instance annotations:
        [{"obj_id": <int>, "segmentation": <COCO RLE>, "area": <float>}, ...]

    The obj_id is the SAM3 tracking ID, which is consistent across every frame
    within the same sequence so callers can associate detections over time.

    Args:
        outputs_per_frame: {frame_local_idx: raw SAM3 output dict}
        frame_names:       original frame names (frame_names[i] ↔ local index i)
        masks_seq_dir:     directory where .json mask files are written
        images_seq_dir:    directory where overlay .jpg files are written
        images_dir:        source image directory – required when save_images=True
        save_images:       if True, also save overlay .jpg visualizations
    """
    masks_seq_dir.mkdir(parents=True, exist_ok=True)
    if save_images and images_seq_dir is not None:
        images_seq_dir.mkdir(parents=True, exist_ok=True)



    for local_idx, frame_out in outputs_per_frame.items():
        if local_idx >= len(frame_names):
            continue  # guard against unexpected indices
        frame_name = frame_names[local_idx]

        out_obj_ids = frame_out.get("out_obj_ids", [])
        out_binary_masks = frame_out.get("out_binary_masks", [])

        if out_obj_ids is None or len(out_obj_ids) == 0:
            continue  # no detections on this frame

        anns: List[Dict[str, Any]] = []
        n = len(out_obj_ids)
        for i in range(n):
            obj_id = int(out_obj_ids[i])
            mask_u8 = _mask_to_numpy(out_binary_masks[i])  # uint8, Fortran order
            if not mask_u8.any():
                continue
            rle = mask_utils.encode(mask_u8)
            rle["counts"] = rle["counts"].decode("utf-8")
            anns.append(
                {
                    "obj_id": obj_id,
                    "segmentation": rle,
                    "area": float(mask_utils.area(rle)),
                }
            )

        if not anns:
            continue

        out_path = masks_seq_dir / f"{frame_name}.json"
        with open(out_path, "wb") as fh:
            fh.write(orjson.dumps(anns))

        if save_images and images_dir is not None and images_seq_dir is not None:
            img_path = images_dir / f"{frame_name}.jpg"
            if not img_path.is_file():
                img_path = images_dir / f"{frame_name}.png"
            if img_path.is_file():
                img_np = load_frame(str(img_path))
                overlay = render_masklet_frame(img_np, frame_out, frame_idx=local_idx)
                overlay_path = images_seq_dir / f"{frame_name}.jpg"
                _PILImage.fromarray(overlay).save(overlay_path, quality=90)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_sam3(
    dataset_name: str,
    objects_filter: Optional[List[str]] = None,
    resume: bool = False,
    objects_to_frames_path: Optional[Path] = None,
    objects_to_segment: Optional[List[str]] = None,
    tmp_root: Optional[Path] = None,
    save_images: bool = False,
) -> None:
    """
    Main entry point: iterate over all objects/sequences and run SAM3.

    Args:
        dataset_name:           e.g. "ProjectLabStudio_inv_method"
        objects_filter:         if given, only process these object names
        resume:                 skip (object, seq) pairs whose output already exists
        objects_to_frames_path: override path to objects_to_frames.json
        objects_to_segment:     if given, ignore the stored objects_to_frames.json
                                (which need not exist) and instead build a synthetic
                                mapping where each of these objects is segmented
                                across every frame in the COLMAP model
        tmp_root:               directory in which to create temp JPEG folders;
                                defaults to the system temp dir
        save_images:            if True, also save overlay JPEG visualizations
    """
    start_time = time.time()
    # ----- Config & paths ---------------------------------------------------
    images_dir = get_images_output_path(dataset_name)
    outputs_dir = get_output_path(dataset_name)

    obj_level_masks_dir = outputs_dir / "object_level_masks"
    masks_base_dir = obj_level_masks_dir / "masks"
    images_base_dir = obj_level_masks_dir / "images"

    # Clear any masks from a previous run so we start from a clean slate and
    # don't append to stale output. Skipped when resuming, since resume relies
    # on previously written masks still being present.
    if not resume and obj_level_masks_dir.exists():
        print(f"Removing existing masks at {obj_level_masks_dir}")
        shutil.rmtree(obj_level_masks_dir, ignore_errors=True)

    masks_base_dir.mkdir(parents=True, exist_ok=True)
    if save_images:
        images_base_dir.mkdir(parents=True, exist_ok=True)

    # Build the objects_to_frames mapping. When objects_to_segment is provided we
    # ignore any stored objects_to_frames.json (it need not exist) and instead
    # segment each requested object across every frame in the COLMAP model.
    if objects_to_segment:
        print(
            f"Building synthetic objects_to_frames for {len(objects_to_segment)} "
            f"object(s) across all COLMAP frames."
        )
        objects_to_frames: Dict[str, List[List[str]]] = _synthetic_objects_to_frames(
            objects_to_segment, dataset_name
        )
        objects_to_frames_path = None
    else:
        # Locate objects_to_frames.json
        if objects_to_frames_path is None:
            objects_to_frames_path = (
                outputs_dir / "objects_inventory" / "objects_to_frames.json"
            )
        if not objects_to_frames_path.is_file():
            raise FileNotFoundError(
                f"objects_to_frames.json not found at {objects_to_frames_path}"
            )

        with objects_to_frames_path.open("r", encoding="utf-8") as fh:
            objects_to_frames = json.load(fh)

    # Optional filtering
    if objects_filter:
        lower_filter = {o.lower() for o in objects_filter}
        objects_to_frames = {
            k: v for k, v in objects_to_frames.items() if k.lower() in lower_filter
        }
        if not objects_to_frames:
            raise ValueError(
                f"None of the requested objects found in objects_to_frames.json: "
                f"{objects_filter}"
            )

    # Collect all (obj_name, seq_idx, frame_list) work items
    work_items: List[Tuple[str, int, List[str]]] = []
    for obj_name, sequences in objects_to_frames.items():
        for seq_idx, frame_list in enumerate(sequences):
            if not frame_list:
                continue
            if resume:
                seq_out_dir = masks_base_dir / _sanitize(obj_name) / f"seq_{seq_idx}"
                if seq_out_dir.is_dir() and any(seq_out_dir.glob("*.json")):
                    print(
                        f"[resume] Skipping {obj_name!r} seq {seq_idx} "
                        f"(output already exists)"
                    )
                    continue
            work_items.append((obj_name, seq_idx, frame_list))

    total = len(work_items)
    if total == 0:
        print("No work items to process.")
        return

    print(
        f"Processing {total} (object, sequence) pairs "
        f"across {len(objects_to_frames)} objects."
    )

    # ----- Build predictor --------------------------------------------------
    print("Building SAM3 video predictor…")
    gpus_to_use = list(range(torch.cuda.device_count()))
    if not gpus_to_use:
        print("[warn] No GPUs detected; running on CPU (will be slow).")
    predictor = build_sam3_video_predictor(
        gpus_to_use=gpus_to_use if gpus_to_use else None
    )

    # ----- Process work items -----------------------------------------------
    session_id: Optional[str] = None
    current_tmp_dir: Optional[Path] = None

    def _close_session() -> None:
        nonlocal session_id
        if session_id is not None:
            try:
                predictor.handle_request(
                    request=dict(type="close_session", session_id=session_id)
                )
            except Exception as exc:
                print(f"[warn] close_session failed: {exc}")
            session_id = None

    def _cleanup_tmp() -> None:
        nonlocal current_tmp_dir
        if current_tmp_dir is not None and current_tmp_dir.exists():
            shutil.rmtree(current_tmp_dir, ignore_errors=True)
            current_tmp_dir = None

    try:
        for item_idx, (obj_name, seq_idx, frame_list) in enumerate(work_items):
            _cleanup_tmp()
            _close_session()

            print(
                f"[{item_idx + 1}/{total}] object={obj_name!r}  "
                f"seq={seq_idx}  frames={len(frame_list)}"
            )

            # Build temp JPEG folder
            try:
                tmp_dir, ordered_names = _build_temp_jpeg_dir(
                    frame_list, images_dir, tmp_root
                )
            except FileNotFoundError as exc:
                print(f"  [skip] {exc}")
                continue
            current_tmp_dir = tmp_dir

            # Start SAM3 session on this temp folder
            try:
                response = predictor.handle_request(
                    request=dict(
                        type="start_session",
                        resource_path=str(tmp_dir),
                    )
                )
                session_id = response["session_id"]
            except Exception as exc:
                print(f"  [error] start_session failed: {exc}")
                continue

            # Add text prompt on frame 0
            try:
                predictor.handle_request(
                    request=dict(
                        type="add_prompt",
                        session_id=session_id,
                        frame_index=0,
                        text=obj_name,
                    )
                )
            except Exception as exc:
                print(f"  [error] add_prompt failed: {exc}")
                continue

            # Propagate through the entire sequence
            try:
                outputs_per_frame = _propagate_in_video(predictor, session_id)
            except Exception as exc:
                print(f"  [error] propagate_in_video failed: {exc}")
                continue

            if not outputs_per_frame:
                print(f"  [warn] No outputs returned for {obj_name!r} seq {seq_idx}")
                continue

            # Persist masks
            obj_slug = _sanitize(obj_name)
            masks_seq_dir = masks_base_dir / obj_slug / f"seq_{seq_idx}"
            images_seq_dir = (
                images_base_dir / obj_slug / f"seq_{seq_idx}" if save_images else None
            )
            _save_sequence_masks(
                outputs_per_frame,
                ordered_names,
                masks_seq_dir,
                images_seq_dir=images_seq_dir,
                images_dir=images_dir,
                save_images=save_images,
            )

            n_saved = sum(1 for _ in masks_seq_dir.glob("*.json"))
            n_imgs = (
                sum(1 for _ in images_seq_dir.glob("*.jpg"))
                if save_images and images_seq_dir
                else 0
            )
            suffix = f", {n_imgs} overlay images" if save_images else ""
            print(f"  Saved {n_saved} mask files{suffix} → {masks_seq_dir}")

    finally:
        _cleanup_tmp()
        _close_session()
        print("Shutting down SAM3 predictor…")
        try:
            predictor.shutdown()
        except Exception as exc:
            print(f"[warn] predictor.shutdown() failed: {exc}")

    print(f"\nDone. Masks saved under:  {masks_base_dir}")
    if save_images:
        print(f"      Images saved under: {images_base_dir}")

    end_time = time.time()
    duration = end_time - start_time
    
    step_stats = {
        "duration_seconds": duration,
        "status": "completed",
        "parameters": {
            "objects_filter": objects_filter,
            "resume": resume,
            "objects_to_frames_path": str(objects_to_frames_path) if objects_to_frames_path else None,
            "objects_to_segment": objects_to_segment,
            "tmp_root": str(tmp_root) if tmp_root else None,
            "save_images": save_images,
        }
    }
    
    save_runtime_stats(dataset_name, "sam3_runner", step_stats)


# ---------------------------------------------------------------------------
# Config-dict entry point
# ---------------------------------------------------------------------------


def run_sam3_from_config(config: dict, dataset_name: str) -> None:
    """Run the SAM3 pipeline using the config dictionary.

    Reads parameters from ``config["sam3"]``.  The dataset name is passed
    separately (it is not read from the config).
    """
    sam_cfg = config.get("sam3", {})

    objects_json = sam_cfg.get("objects_to_frames_path")
    tmp_root_str = sam_cfg.get("tmp_root")

    run_sam3(
        dataset_name=dataset_name,
        objects_filter=sam_cfg.get("objects_filter"),
        resume=sam_cfg.get("resume", False),
        objects_to_frames_path=Path(objects_json) if objects_json else None,
        objects_to_segment=sam_cfg.get("objects_to_segment"),
        tmp_root=Path(tmp_root_str) if tmp_root_str else None,
        save_images=sam_cfg.get("save_images", False),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from config_cli import parse_config_args

    config, dataset_name = parse_config_args(
        description="Run SAM3 video predictor on per-object frame sequences.",
    )
    run_sam3_from_config(config, dataset_name)
