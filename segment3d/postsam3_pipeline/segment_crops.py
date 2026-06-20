"""
Segment crops from images using connected component masks.

For each connected component in connected_components.json:
  1. Parse instance IDs to locate the corresponding mask directory
     (object_level_masks/masks/<object>/<seq>/<frame>.json)
  2. Use COLMAP data to rank frames by fraction of the component's 3D points visible
  3. For the top-N frames, decode the COCO-RLE mask, apply it to the image
     (white background outside mask), crop to the mask bounding box, and save.

Output layout mirrors crop_images.py:
    <outputs_dir>/crops/component_<id>/<frame_stem>_<instance_id>_masked_crop.jpg
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pycocotools import mask as mask_utils

from ..utils.colmap_io import load_colmap_model, index_image_metadata
from config_io import (
    get_images_output_path,
    get_colmap_output_path,
    get_output_path,
    reset_dir,
)


# ---------------------------------------------------------------------------
# Instance-ID helpers
# ---------------------------------------------------------------------------


def parse_instance_id(instance_id: str) -> Tuple[str, str, int]:
    """
    Parse an instance ID into its components.

    Instance IDs follow the pattern ``<object>_<seq_N>_<instance_idx>``,
    e.g. ``drill_seq_0_0`` → (``"drill"``, ``"seq_0"``, 0).

    Args:
        instance_id: Instance identifier string.

    Returns:
        Tuple of (object_class, seq_name, instance_index).

    Raises:
        ValueError: If the ID does not match the expected pattern.
    """
    # Split from the right to isolate the trailing integer index
    # e.g. "drill_seq_0_0" → last token "0", remainder "drill_seq_0"
    parts = instance_id.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse instance_id: {instance_id!r}")
    prefix, idx_str = parts
    if not idx_str.isdigit():
        raise ValueError(f"Instance index is not an integer in: {instance_id!r}")
    instance_index = int(idx_str)

    # prefix is e.g. "drill_seq_0" – split out the seq part (last two tokens)
    prefix_parts = prefix.rsplit("_", 2)
    if len(prefix_parts) < 3:
        raise ValueError(f"Cannot parse seq from instance_id: {instance_id!r}")
    # prefix_parts[-2:] = ["seq", "0"]  → seq_name = "seq_0"
    object_class = "_".join(prefix_parts[:-2])
    seq_name = "_".join(prefix_parts[-2:])

    return object_class, seq_name, instance_index


# ---------------------------------------------------------------------------
# COLMAP visibility helpers
# ---------------------------------------------------------------------------


def build_point3d_to_images(
    image_metadata: Dict[int, Dict[str, Any]],
) -> Dict[int, List[str]]:
    """
    Build a reverse index from 3-D point ID to the list of image names that observe it.

    Args:
        image_metadata: Output of :func:`index_image_metadata`.

    Returns:
        Mapping from ``point3D_id`` to list of image filename strings.
    """
    index: Dict[int, List[str]] = defaultdict(list)
    for meta in image_metadata.values():
        name: str = meta["name"]
        for pid in meta["point3D_ids"]:
            pid_int = int(pid)
            if pid_int == -1:
                continue
            index[pid_int].append(name)
    return dict(index)


def rank_frames_by_visibility(
    point3d_ids: List[int],
    point3d_to_images: Dict[int, List[str]],
    available_frames: Optional[set] = None,
) -> List[Tuple[str, float]]:
    """
    Rank image frames by how many of the given 3-D point IDs they observe.

    Args:
        point3d_ids: The 3-D point IDs belonging to this component.
        point3d_to_images: Reverse index from :func:`build_point3d_to_images`.
        available_frames: If given, restrict to frames in this set.

    Returns:
        List of ``(image_name, fraction)`` pairs sorted descending by fraction.
        ``fraction`` = visible_count / len(point3d_ids).
    """
    if not point3d_ids:
        return []

    counter: Dict[str, int] = defaultdict(int)
    total = len(point3d_ids)

    for pid in point3d_ids:
        for img_name in point3d_to_images.get(pid, []):
            if available_frames is None or img_name in available_frames:
                counter[img_name] += 1

    ranked = sorted(counter.items(), key=lambda kv: kv[1], reverse=True)
    return [(name, count / total) for name, count in ranked]


# ---------------------------------------------------------------------------
# Mask decoding
# ---------------------------------------------------------------------------


def decode_rle_mask(segmentation: Dict[str, Any]) -> np.ndarray:
    """
    Decode a COCO-RLE segmentation dict into a binary mask array.

    Args:
        segmentation: Dict with ``"size"`` and ``"counts"`` keys.

    Returns:
        ``np.ndarray`` of shape ``(H, W)`` with dtype ``uint8`` (0/1).

    Raises:
        RuntimeError: If ``pycocotools`` is not installed.
    """
    rle = {
        "size": segmentation["size"],
        "counts": segmentation["counts"],
    }
    decoded: np.ndarray = mask_utils.decode(rle)  # type: ignore[attr-defined]
    return decoded.astype(np.uint8)


def load_instance_mask(
    masks_dir: Path,
    object_class: str,
    seq_name: str,
    frame_stem: str,
    instance_index: int,
) -> Optional[np.ndarray]:
    """
    Load and decode the mask for a specific instance in a specific frame.

    Args:
        masks_dir: Root ``masks`` directory (contains per-object subdirs).
        object_class: Object class name (e.g. ``"drill"``).
        seq_name: Sequence name (e.g. ``"seq_0"``).
        frame_stem: Frame file stem without extension (e.g. ``"frame_00002"``).
        instance_index: Integer index matching ``obj_id`` in the mask JSON.

    Returns:
        Binary mask array ``(H, W)`` or ``None`` if the file/entry is missing.
    """
    mask_path = masks_dir / object_class / seq_name / f"{frame_stem}.json"
    if not mask_path.exists():
        return None

    with mask_path.open("r", encoding="utf-8") as fh:
        entries: List[Dict[str, Any]] = json.load(fh)

    for entry in entries:
        if entry.get("obj_id") == instance_index:
            return decode_rle_mask(entry["segmentation"])

    return None


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def apply_mask_white_background(
    image: np.ndarray, mask: np.ndarray, only_masked: bool = False
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Apply a binary mask to an image, setting masked-out pixels to white.

    Args:
        image: BGR image array ``(H, W, 3)``.
        mask:  Binary mask array ``(H, W)`` with 1=keep, 0=white-out.

    Returns:
        Tuple of:
        - masked image array ``(H, W, 3)``
        - bounding box ``(x_min, y_min, x_max, y_max)`` of the mask region.

    Raises:
        ValueError: If the mask contains no foreground pixels.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        raise ValueError("Mask has no foreground pixels.")

    y_min, y_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    x_min, x_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))
    # make x_max / y_max exclusive
    y_max += 1
    x_max += 1

    out = image.copy()
    if only_masked:
        out[mask == 0] = 255  # white background

    return out, (x_min, y_min, x_max, y_max)


def crop_to_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop an image to an axis-aligned bounding box.

    Args:
        image: Image array.
        bbox:  ``(x_min, y_min, x_max, y_max)`` with exclusive max values.

    Returns:
        Cropped sub-image.
    """
    x_min, y_min, x_max, y_max = bbox
    return image[y_min:y_max, x_min:x_max]


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def segment_crops_cli(
    dataset_name: str,
    top_n: int = 5,
    min_fraction: float = 0.05,
) -> None:
    """
    Main entry point: generate masked & cropped images for every connected component.

    For each component in ``connected_components.json``:

    1. Parse each ``instance_id`` to find its masks directory.
    2. Use COLMAP to rank frames by the fraction of the component's 3-D points
       visible in them.
    3. For the top-*n* frames that have a mask file for the instance, apply the
       mask (white background), crop to the mask bbox, and save to
       ``<outputs_dir>/crops/component_<id>/<instance_id>/<frame>_masked_crop.jpg``.

    Args:
        dataset_name: Name of the dataset to process.
        top_n: Number of top frames to use per component.
        min_fraction: Minimum visibility fraction to consider a frame.
    """
    images_dir = get_images_output_path(dataset_name)
    colmap_model_dir = get_colmap_output_path(dataset_name)
    outputs_dir = get_output_path(dataset_name)

    # Object-level masks live one level above the "masks" sub-directory
    object_masks_root = outputs_dir / "object_level_masks" / "masks"
    if not object_masks_root.exists():
        raise FileNotFoundError(
            f"Object-level masks directory not found: {object_masks_root}"
        )

    connected_components_path = outputs_dir / "connected_components.json"
    if not connected_components_path.exists():
        raise FileNotFoundError(
            f"connected_components.json not found: {connected_components_path}"
        )

    # -----------------------------------------------------------------------
    # Load COLMAP model and build the point → images reverse index
    # -----------------------------------------------------------------------
    print("Loading COLMAP model …")
    cameras, images, points3D = load_colmap_model(str(colmap_model_dir))
    image_metadata = index_image_metadata(images)

    print("Building 3D-point → image reverse index …")
    point3d_to_images = build_point3d_to_images(image_metadata)

    # Build name → COLMAP image_id lookup for manifest metadata
    name_to_image_id: Dict[str, int] = {
        meta["name"]: img_id for img_id, meta in image_metadata.items()
    }

    # -----------------------------------------------------------------------
    # Load connected components
    # -----------------------------------------------------------------------
    print(f"Loading connected components from {connected_components_path} …")
    with connected_components_path.open("r", encoding="utf-8") as fh:
        components: List[Dict[str, Any]] = json.load(fh)

    # Clear any crops from a previous run so we start from a clean slate.
    # Otherwise component_<id> directories for components that no longer exist
    # (e.g. after a parameter change reduced the component count) would linger
    # and be picked up by downstream captioning / DB steps.
    crops_dir = reset_dir(outputs_dir / "crops")

    # manifest: comp_id (str) → {component_id, total_crops, crops: [...]}
    final_manifest: Dict[str, Any] = {}

    total_saved = 0
    total_components = len(components)

    for comp in components:
        comp_id: int = comp["connected_comp_id"]
        instance_ids: List[str] = comp["instance_ids"]
        point3d_ids: List[int] = comp["set_of_point3DIds"]

        print(
            f"\n[Component {comp_id}/{total_components - 1}] "
            f"instances={instance_ids}, points={len(point3d_ids)}"
        )

        # ------------------------------------------------------------------
        # Parse all instances upfront; skip any that are malformed or missing
        # ------------------------------------------------------------------
        parsed_instances: List[Tuple[str, str, str, int]] = []
        for instance_id in instance_ids:
            try:
                object_class, seq_name, instance_index = parse_instance_id(instance_id)
            except ValueError as exc:
                print(f"  Skipping {instance_id}: {exc}")
                continue

            seq_mask_dir = object_masks_root / object_class / seq_name
            if not seq_mask_dir.exists():
                print(f"  Mask directory not found: {seq_mask_dir}")
                continue

            parsed_instances.append(
                (instance_id, object_class, seq_name, instance_index)
            )

        if not parsed_instances:
            print("  No valid instances for this component.")
            continue

        # ------------------------------------------------------------------
        # Build the union of available frames across all instances, then rank
        # by the component's combined 3-D point visibility.
        # ------------------------------------------------------------------
        available_image_names: set = set()
        for _, object_class, seq_name, _ in parsed_instances:
            seq_mask_dir = object_masks_root / object_class / seq_name
            for p in seq_mask_dir.glob("*.json"):
                available_image_names.add(f"{p.stem}.jpg")
                available_image_names.add(f"{p.stem}.png")

        ranked_frames = rank_frames_by_visibility(
            point3d_ids, point3d_to_images, available_frames=available_image_names
        )

        candidate_frames = [
            (img_name, frac) for img_name, frac in ranked_frames if frac >= min_fraction
        ]

        if not candidate_frames:
            print(f"  No frames with sufficient visibility for component {comp_id}")
            continue

        top_frames = candidate_frames[:top_n]
        print(
            f"  Selected {len(top_frames)} frames: "
            + ", ".join(f"{n}({f:.1%})" for n, f in top_frames)
        )

        # Output directory is flat per component – no instance subdirectory
        out_comp_dir = crops_dir / f"component_{comp_id}"
        out_comp_dir.mkdir(parents=True, exist_ok=True)

        comp_key = str(comp_id)
        if comp_key not in final_manifest:
            final_manifest[comp_key] = {
                "component_id": comp_id,
                "total_crops": 0,
                "crops": [],
            }
        crop_index = 0

        for img_name, fraction in top_frames:
            frame_stem = Path(img_name).stem  # e.g. "frame_00005"
            image_path = images_dir / img_name

            if not image_path.exists():
                print(f"    Image not found: {image_path}")
                continue

            img = cv2.imread(str(image_path))
            if img is None:
                print(f"    Could not read image: {image_path}")
                continue

            # For each instance, produce a separate crop from this frame
            for instance_id, object_class, seq_name, instance_index in parsed_instances:
                mask = load_instance_mask(
                    object_masks_root,
                    object_class,
                    seq_name,
                    frame_stem,
                    instance_index,
                )
                if mask is None:
                    # This frame may not have a mask for every instance – skip silently
                    continue

                # Resize mask if image dimensions differ (shouldn't normally happen)
                cur_mask = mask
                if cur_mask.shape[:2] != img.shape[:2]:
                    cur_mask = cv2.resize(
                        cur_mask,
                        (img.shape[1], img.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                try:
                    masked_img, bbox = apply_mask_white_background(img, cur_mask)
                except ValueError:
                    print(f"    Empty mask for {instance_id} in {frame_stem}")
                    continue

                cropped = crop_to_bbox(masked_img, bbox)

                # Filename encodes both frame and instance for uniqueness
                out_filename = f"{frame_stem}_{instance_id}_masked_crop.jpg"
                out_path = out_comp_dir / out_filename

                success = cv2.imwrite(str(out_path), cropped)
                if success:
                    total_saved += 1
                    visible_points = round(fraction * len(point3d_ids))
                    final_manifest[comp_key]["crops"].append(
                        {
                            "crop_filename": out_filename,
                            "source_image": img_name,
                            "instance_id": instance_id,
                            "crop_index": crop_index,
                            "crop_coordinates": list(bbox),
                            "image_id": name_to_image_id.get(img_name),
                            "fraction_visible": fraction,
                            "visible_points": visible_points,
                            "total_points": len(point3d_ids),
                        }
                    )
                    final_manifest[comp_key]["total_crops"] += 1
                    crop_index += 1
                    print(
                        f"    Saved {out_filename}  "
                        f"(visibility={fraction:.2%}, bbox={bbox})"
                    )
                else:
                    print(f"    Failed to write {out_path}")

    # Save manifest
    manifest_path = crops_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(final_manifest, fh, indent=2)

    print(f"\nDone. Total crops saved: {total_saved}")
    print(f"Manifest saved to {manifest_path}")
    print(f"Output directory: {crops_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate masked & cropped images for connected components."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of the dataset (e.g. ProjectLabStudio_inv_method).",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=5,
        help="Number of top frames to crop per component (default: 5).",
    )
    parser.add_argument(
        "--min_fraction",
        type=float,
        default=0.05,
        help="Minimum fraction of component points visible in a frame (default: 0.05).",
    )
    args = parser.parse_args()

    segment_crops_cli(
        dataset_name=args.dataset_name,
        top_n=args.top_n,
        min_fraction=args.min_fraction,
    )


if __name__ == "__main__":
    main()
