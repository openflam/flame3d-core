"""Debug a connected component by visualizing all contributing masks.

Reads connected_components.json and the masks in the masks directory to
render every mask that contributed to the specified component.

Supports two connected_components.json formats:
- ``mask_id_set`` (e.g. "frame_00001_mask_5"): reads from the SAM masks dir.
- ``instance_ids`` (e.g. "Boxes_seq_0_0"): reads from
  ``<outputs_dir>/object_level_masks/masks/<class>/<seq>/frame_XXXXX.json``.

Usage:
    python debug_component.py --dataset_name ProjectLabStudio_NoNeg --component_id 5
    python debug_component.py --dataset_name ProjectLabStudio_NoNeg --component_id 5 --display
    python debug_component.py --dataset_name ProjectLabStudio_inv_method --component_id 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional imports – fail early with a helpful message if missing
# ---------------------------------------------------------------------------
try:
    from pycocotools import mask as mask_utils
except ImportError:
    sys.exit("pycocotools is required.  Install with: pip install pycocotools")

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    sys.exit("Pillow is required.  Install with: pip install Pillow")

from config_io import (
    get_colmap_output_path,
    get_images_output_path,
    get_output_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OVERLAY_COLOR = (255, 80, 0)  # orange-red for the mask fill
OVERLAY_ALPHA = 0.55  # blend amount


def parse_mask_id(mask_id: str):
    """Split 'frame_00001_mask_5' → ('frame_00001', 5)."""
    frame_stem, _, idx_str = mask_id.rpartition("_mask_")
    if not frame_stem:
        raise ValueError(f"Cannot parse mask_id: {mask_id!r}")
    return frame_stem, int(idx_str)


# Pattern: <class_name>_seq_<seq_num>_<obj_id>
_INSTANCE_ID_RE = re.compile(r"^(.+)_seq_(\d+)_(\d+)$")


def parse_instance_id(instance_id: str) -> Tuple[str, str, int]:
    """Split 'Boxes_seq_0_0' → ('Boxes', 'seq_0', 0).

    Works for multi-word class names like 'plastic_container_seq_0_8'.
    """
    m = _INSTANCE_ID_RE.match(instance_id)
    if not m:
        raise ValueError(f"Cannot parse instance_id: {instance_id!r}")
    class_name, seq_num, obj_id = m.group(1), m.group(2), m.group(3)
    return class_name, f"seq_{seq_num}", int(obj_id)


def load_object_level_frame_masks(
    obj_masks_dir: Path, class_name: str, seq_name: str
) -> Dict[str, List[dict]]:
    """Load all per-frame JSON files for a given class/sequence.

    Returns a dict mapping frame_stem (e.g. 'frame_00061') to a list of mask
    entries (each entry has 'obj_id', 'segmentation', 'area').
    """
    seq_dir = obj_masks_dir / class_name / seq_name
    if not seq_dir.exists():
        print(f"  [warn] Sequence directory not found: {seq_dir}")
        return {}
    result: Dict[str, List[dict]] = {}
    # Per-frame JSONs are named by image stem (e.g. '50791824728.json'), matching
    # how associate2d3d enumerates them — not a fixed 'frame_XXXXX' pattern.
    for json_file in sorted(seq_dir.glob("*.json")):
        frame_stem = json_file.stem
        with json_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        result[frame_stem] = data if isinstance(data, list) else []
    return result


def load_masks_json(masks_dir: Path, frame_stem: str) -> Optional[List[dict]]:
    """Load the masks list for a given frame stem, or return None if missing."""
    path = masks_dir / f"{frame_stem}_masks.json"
    if not path.exists():
        print(f"  [warn] Mask file not found: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    raise ValueError(f"Unexpected format in {path}")


def decode_mask(rle_entry: dict) -> np.ndarray:
    """Decode a pycocotools RLE/polygon entry to a boolean numpy array."""
    seg = rle_entry["segmentation"]
    # If it's already an RLE dict with size/counts use decode directly;
    # otherwise encode it first (polygon).
    if isinstance(seg, dict):
        return mask_utils.decode(seg).astype(bool)
    # Polygon format – need height/width from the outer dict if available
    raise ValueError("Polygon segmentation not supported; expected RLE dict.")


def render_mask_image(
    mask_bool: np.ndarray,
    mask_id: str,
    orig_image: Optional[Image.Image] = None,
    only_mask: bool = False,
) -> Image.Image:
    """
    Create an RGB image showing the mask as a semi-transparent coloured overlay.
    If orig_image is provided it is used as the background; otherwise falls back
    to a plain white background.
    If only_mask is True, show only the original image pixels inside the mask;
    everything outside is white.
    """
    H, W = mask_bool.shape

    if orig_image is not None:
        # Resize original to match mask dimensions if needed
        if (orig_image.height, orig_image.width) != (H, W):
            orig_image = orig_image.resize((W, H), Image.LANCZOS)
        bg = np.array(orig_image.convert("RGB"), dtype=np.uint8)
    else:
        bg = np.full((H, W, 3), 255, dtype=np.uint8)

    if only_mask:
        # Show original pixels inside mask; white outside
        blended = np.full((H, W, 3), 255, dtype=np.uint8)
        blended[mask_bool] = bg[mask_bool]
    else:
        # Apply overlay colour where mask is True
        overlay = bg.copy()
        overlay[mask_bool] = OVERLAY_COLOR

        # Blend only the masked region; leave the rest as-is
        alpha = np.where(mask_bool[:, :, None], OVERLAY_ALPHA, 0.0)
        blended = (overlay * alpha + bg * (1 - alpha)).astype(np.uint8)

        # Draw a red border along the mask boundary (dilate XOR erode)
        from scipy.ndimage import binary_dilation, binary_erosion

        border = binary_dilation(mask_bool, iterations=2) & ~binary_erosion(
            mask_bool, iterations=2
        )
        blended[border] = (220, 0, 0)  # red border

    img = Image.fromarray(blended, mode="RGB")

    # Draw label
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except OSError:
        font = ImageFont.load_default()

    label = mask_id
    # Semi-transparent label background
    bbox = draw.textbbox((4, 4), label, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2], fill=(0, 0, 0))
    draw.text((4, 4), label, fill=(255, 255, 255), font=font)

    return img


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------


def write_ply(path: Path, xyzs: np.ndarray, rgbs: np.ndarray) -> None:
    """Write a binary little-endian PLY file with float32 XYZ and uint8 RGB."""
    n = len(xyzs)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # Build a structured array: 3x float32 + 3x uint8 = 15 bytes per point
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    data = np.empty(n, dtype=dtype)
    data["x"] = xyzs[:, 0].astype(np.float32)
    data["y"] = xyzs[:, 1].astype(np.float32)
    data["z"] = xyzs[:, 2].astype(np.float32)
    data["red"] = rgbs[:, 0]
    data["green"] = rgbs[:, 1]
    data["blue"] = rgbs[:, 2]

    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def load_original_image(images_dir: Path, frame_stem: str) -> Optional[Image.Image]:
    """Try common extensions and return a PIL image, or None if not found."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = images_dir / f"{frame_stem}{ext}"
        if p.exists():
            return Image.open(p).convert("RGB")
    print(f"  [warn] Original image not found for {frame_stem} in {images_dir}")
    return None


def debug_component(
    dataset_name: str, component_id: int, display: bool = False, only_mask: bool = False
) -> None:
    outputs_dir = get_output_path(dataset_name)

    # This pipeline stores SAM masks per object instance (the instance_ids
    # format handled below), not as flat ``<frame>_masks.json`` files, so there
    # is no standalone masks dir. It stays None unless a component happens to be
    # stored in the legacy mask_id_set format.
    masks_dir = None

    images_dir = get_images_output_path(dataset_name)
    if not images_dir.exists():
        images_dir = None
        print(
            "[warn] Could not resolve images_dir – overlays will use white background."
        )

    colmap_model_dir = get_colmap_output_path(dataset_name)
    if not colmap_model_dir.exists():
        colmap_model_dir = None
        print(
            "[warn] Could not resolve colmap_model_dir – skipping point cloud export."
        )

    components_path = outputs_dir / "connected_components.json"
    if not components_path.exists():
        sys.exit(f"connected_components.json not found at {components_path}")

    with components_path.open("r", encoding="utf-8") as f:
        components = json.load(f)

    # Find the requested component
    component = None
    for c in components:
        if c.get("connected_comp_id") == component_id:
            component = c
            break

    if component is None:
        ids = [c.get("connected_comp_id") for c in components]
        sys.exit(
            f"Component {component_id} not found.  "
            f"Available IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}"
        )

    mask_id_set: List[str] = component.get("mask_id_set", [])
    instance_ids: List[str] = component.get("instance_ids", [])

    if not mask_id_set and not instance_ids:
        sys.exit(
            f"Component {component_id} has neither 'mask_id_set' nor 'instance_ids'.  "
            "Re-run the mask_graph pipeline to regenerate connected_components.json "
            "with provenance tracking."
        )

    # Output directory
    out_dir = outputs_dir / "debug_components" / f"component_{component_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered: List[Image.Image] = []
    all_label_ids: List[str] = []

    # -----------------------------------------------------------------------
    # Path A: mask_id_set  →  SAM masks directory
    # -----------------------------------------------------------------------
    if mask_id_set:
        print(
            f"Component {component_id}: {len(mask_id_set)} contributing mask(s) [mask_id_set]"
        )

        if masks_dir is None:
            sys.exit(
                "masks_dir is not configured for this dataset but is required for "
                "the mask_id_set format."
            )

        # Group mask_ids by frame so we load each file only once
        by_frame: Dict[str, List[tuple]] = defaultdict(list)
        for mask_id in sorted(mask_id_set):
            frame_stem, mask_idx = parse_mask_id(mask_id)
            by_frame[frame_stem].append((mask_idx, mask_id))

        for frame_stem in sorted(by_frame.keys()):
            masks_list = load_masks_json(masks_dir, frame_stem)
            if masks_list is None:
                continue

            orig_image = (
                load_original_image(images_dir, frame_stem) if images_dir else None
            )

            for mask_idx, mask_id in sorted(by_frame[frame_stem]):
                if mask_idx >= len(masks_list):
                    print(
                        f"  [warn] mask index {mask_idx} out of range for {frame_stem} "
                        f"(file has {len(masks_list)} masks)"
                    )
                    continue

                try:
                    mask_bool = decode_mask(masks_list[mask_idx])
                except Exception as exc:
                    print(f"  [warn] Failed to decode {mask_id}: {exc}")
                    continue

                img = render_mask_image(
                    mask_bool, mask_id, orig_image=orig_image, only_mask=only_mask
                )
                rendered.append(img)
                all_label_ids.append(mask_id)

                out_path = out_dir / f"{mask_id}.png"
                img.save(out_path)
                print(f"  Saved {out_path}")

    # -----------------------------------------------------------------------
    # Path B: instance_ids  →  object_level_masks directory
    # -----------------------------------------------------------------------
    elif instance_ids:
        print(
            f"Component {component_id}: {len(instance_ids)} instance(s) [instance_ids]"
        )

        obj_masks_dir = outputs_dir / "object_level_masks" / "masks"
        if not obj_masks_dir.exists():
            sys.exit(
                f"object_level_masks/masks directory not found at {obj_masks_dir}. "
                "Make sure the object_level_masks pipeline has been run."
            )

        for instance_id in sorted(instance_ids):
            try:
                class_name, seq_name, obj_id = parse_instance_id(instance_id)
            except ValueError as exc:
                print(f"  [warn] {exc}")
                continue

            frame_masks = load_object_level_frame_masks(
                obj_masks_dir, class_name, seq_name
            )
            if not frame_masks:
                continue

            for frame_stem in sorted(frame_masks.keys()):
                # Find the entry matching obj_id
                entry = None
                for item in frame_masks[frame_stem]:
                    if item.get("obj_id") == obj_id:
                        entry = item
                        break
                if entry is None:
                    continue  # this instance not visible in this frame

                try:
                    mask_bool = decode_mask(entry)
                except Exception as exc:
                    label = f"{instance_id}_{frame_stem}"
                    print(f"  [warn] Failed to decode {label}: {exc}")
                    continue

                orig_image = (
                    load_original_image(images_dir, frame_stem) if images_dir else None
                )
                label = f"{instance_id}_{frame_stem}"
                img = render_mask_image(
                    mask_bool, label, orig_image=orig_image, only_mask=only_mask
                )
                rendered.append(img)
                all_label_ids.append(label)

                out_path = out_dir / f"{label}.png"
                img.save(out_path)
                print(f"  Saved {out_path}")

    if not rendered:
        print("No masks could be rendered.")
        return

    print(f"\nAll images saved to: {out_dir}")

    # -----------------------------------------------------------------------
    # Point cloud export
    # -----------------------------------------------------------------------
    if colmap_model_dir is not None:
        print("\nBuilding point cloud...")
        from segment3d.utils.colmap_io import load_colmap_model

        _, _, points3D = load_colmap_model(str(colmap_model_dir))

        component_ids = set(component["set_of_point3DIds"])

        all_ids = list(points3D.keys())
        xyzs = np.array([points3D[pid].xyz for pid in all_ids], dtype=np.float32)

        # COLMAP is Z-up; convert to Y-up for standard 3D viewers:
        #   new X =  old X,  new Y =  old Z,  new Z = -old Y
        xyzs = np.stack([xyzs[:, 0], xyzs[:, 2], -xyzs[:, 1]], axis=1)
        rgbs = np.zeros((len(all_ids), 3), dtype=np.uint8)

        for i, pid in enumerate(all_ids):
            if pid in component_ids:
                rgbs[i] = (255, 0, 0)  # red – component points
            else:
                rgbs[i] = (255, 255, 255)  # white – everything else

        ply_path = out_dir / f"component_{component_id}.ply"
        write_ply(ply_path, xyzs, rgbs)
        print(f"Point cloud saved to: {ply_path}")

    if display:
        try:
            import matplotlib.pyplot as plt

            fig, axes = (
                plt.subplots(1, len(rendered), figsize=(4 * len(rendered), 4))
                if len(rendered) > 1
                else plt.subplots(1, 1, figsize=(6, 6))
            )

            axes_flat = [axes] if len(rendered) == 1 else axes.flat
            for ax, img, label in zip(axes_flat, rendered, all_label_ids):
                ax.imshow(img)
                ax.set_title(label, fontsize=7)
                ax.axis("off")

            fig.suptitle(
                f"Component {component_id} – {len(rendered)} masks", fontsize=12
            )
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib not available – skipping display.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize all masks that contributed to a connected component"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g. ProjectLabStudio_NoNeg)",
    )
    parser.add_argument(
        "--component_id",
        type=int,
        required=True,
        help="The connected_comp_id to inspect",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show images interactively using matplotlib",
    )
    parser.add_argument(
        "--only-mask",
        action="store_true",
        help="Show only the original image pixels inside the mask; the rest is white",
    )
    args = parser.parse_args()
    debug_component(
        dataset_name=args.dataset_name,
        component_id=args.component_id,
        display=args.display,
        only_mask=args.only_mask,
    )


if __name__ == "__main__":
    main()
