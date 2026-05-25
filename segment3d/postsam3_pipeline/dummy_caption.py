"""
Generate a dummy component_captions.json that uses "Dummy Caption" for every
component.  The output has the same schema as the real captioning step so that
downstream consumers (CLIP embedding, database creation, etc.) can run without
a GPU or a running VLM.

Schema written:
    [
        {
            "component_id": <int>,
            "caption": "Dummy Caption",
            "num_images_used": 1,
            "crop_filenames": ["<randomly_chosen_crop>.jpg"]
        },
        ...
    ]

Component IDs and crop filenames are sourced from (in priority order):
    1. outputs/<dataset>/crops/manifest.json   – one crop chosen at random per component
    2. outputs/<dataset>/connected_components.json – list with "connected_comp_id"
       (no crop filenames available in this fallback)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config_io import get_output_path

DUMMY_CAPTION_TEXT = "Dummy Caption"


def generate_dummy_captions(dataset_name: str, seed: Optional[int] = None) -> None:
    """
    Write a dummy ``component_captions.json`` for *dataset_name*.

    The file is placed at ``outputs/<dataset>/component_captions.json``,
    matching the exact location that the real captioning step produces.

    When the crops manifest is available, one crop filename is chosen at
    random for each component and stored in ``crop_filenames``.  Otherwise
    ``crop_filenames`` is left empty.

    Args:
        dataset_name: Name of the dataset to process (must exist in config.py).
        seed: Optional random seed for reproducible crop selection.

    Raises:
        FileNotFoundError: If neither the crops manifest nor
            connected_components.json can be found.
    """
    # Resolve outputs directory via config_io.
    outputs_dir = get_output_path(dataset_name)

    rng = random.Random(seed)

    manifest_path = outputs_dir / "crops" / "manifest.json"
    if manifest_path.exists():
        component_entries = _entries_from_manifest(manifest_path, rng)
    else:
        cc_path = outputs_dir / "connected_components.json"
        if cc_path.exists():
            component_entries = _entries_from_connected_components(cc_path)
        else:
            raise FileNotFoundError(
                f"Cannot determine component IDs: neither\n"
                f"  {manifest_path}\nnor\n  {cc_path}\nexists.\n"
                "Run at least the 'clean components' step before generating dummy captions."
            )

    captions: List[Dict[str, Any]] = [
        {
            "component_id": cid,
            "caption": DUMMY_CAPTION_TEXT,
            "num_images_used": len(crop_filenames),
            "crop_filenames": crop_filenames,
        }
        for cid, crop_filenames in component_entries
    ]

    output_path = outputs_dir / "component_captions.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2)

    print(f"Wrote {len(captions)} dummy captions to: {output_path}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _entries_from_manifest(
    manifest_path: Path, rng: random.Random
) -> List[Tuple[int, List[str]]]:
    """Return sorted (component_id, [random_crop_filename]) pairs from the manifest.

    Each component gets exactly one randomly selected crop filename.
    Components with no crops get an empty list.
    """
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest: Dict[str, Any] = json.load(f)

    entries: List[Tuple[int, List[str]]] = []
    for key in sorted(manifest.keys(), key=int):
        cid = int(key)
        crops = manifest[key].get("crops", [])
        if crops:
            chosen = rng.choice(crops)["crop_filename"]
            entries.append((cid, [chosen]))
        else:
            entries.append((cid, []))
    return entries


def _entries_from_connected_components(
    cc_path: Path,
) -> List[Tuple[int, List[str]]]:
    """Return sorted (component_id, []) pairs from connected_components.json.

    No crop filenames are available without a manifest.
    """
    with cc_path.open("r", encoding="utf-8") as f:
        components: List[Dict[str, Any]] = json.load(f)
    return sorted((int(c["connected_comp_id"]), []) for c in components)
