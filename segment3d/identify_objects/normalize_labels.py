"""
Normalize object inventory labels via lemmatization + CLIP-based semantic clustering.

For each cluster of semantically similar lemmatized labels, the label whose
embedding is closest to the cluster centroid is chosen as the representative.
Every label in objects_inventory.json is then replaced by its cluster
representative, producing objects_inventory_compact.json with the same
frame-keyed list structure.

Usage (CLI):
    python -m segment3d.src.objects_inventory.normalize_labels --dataset <name>

    Optional flags:
      --distance-threshold FLOAT   Cosine-distance threshold for agglomerative
                                   clustering (default: 0.12).
      --clip-model STR             OpenCLIP model name (default: ViT-H-14).
      --clip-pretrained STR        OpenCLIP pretrained tag (default: laion2B-s32B-b79K).
      --device STR                 'cuda', 'cpu', or 'cuda:<id>' (default: auto).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import time
from datetime import datetime

import numpy as np
import open_clip
import spacy
from spacy.tokens import Doc
import torch
from sklearn.cluster import AgglomerativeClustering

from config_io import get_output_path
from ..utils.save_runtime_stats import save_runtime_stats


# ---------------------------------------------------------------------------
# Lemmatisation
# ---------------------------------------------------------------------------


def _load_nlp() -> spacy.language.Language:
    """Load the spaCy English model, with a helpful error if missing."""
    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Install it with:  python -m spacy download en_core_web_sm"
        ) from exc


def lemmatize_label(label: str, nlp: spacy.language.Language) -> str:
    """Lowercase, strip trailing punctuation, and lemmatize every token as a noun.

    Forcing POS=NOUN prevents verbs like 'saw' from being lemmatized as 'see'.
    """

    cleaned = label.strip().rstrip(".").lower()
    words = cleaned.split()
    if not words:
        return cleaned

    # Build a Doc with explicit NOUN tags so the lemmatizer treats every
    # token as a noun rather than inferring (possibly wrong) POS from context.
    spaces = [True] * (len(words) - 1) + [False]
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    for token in doc:
        token.pos_ = "NOUN"

    lemmatizer = nlp.get_pipe("lemmatizer")
    doc = lemmatizer(doc)
    return " ".join(token.lemma_ for token in doc)


# ---------------------------------------------------------------------------
# CLIP embeddings
# ---------------------------------------------------------------------------


def embed_labels(
    labels: List[str],
    model_name: str = "ViT-H-14",
    pretrained: str = "laion2B-s32B-b79K",
    device: Optional[str] = None,
) -> Tuple[np.ndarray, str]:
    """
    Compute L2-normalised CLIP text embeddings for *labels*.

    Returns
    -------
    embeddings : np.ndarray, shape (N, D), float32
    device_str : str  – the device actually used
    """
    if device is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_str = device

    model, _, _ = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device_str).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    try:
        with torch.no_grad():
            tokens = tokenizer(labels).to(device_str)
            features = model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)

        embeddings = features.cpu().numpy().astype(np.float32)
    finally:
        # Free the GPU model and cached allocations so this call doesn't leak
        # memory into a long-lived process (e.g. when run inline rather than as
        # an isolated subprocess).
        del model, tokenizer
        if "features" in locals():
            del features
        if "tokens" in locals():
            del tokens
        if device_str.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return embeddings, device_str


# ---------------------------------------------------------------------------
# Clustering & representative selection
# ---------------------------------------------------------------------------


def cluster_labels(
    embeddings: np.ndarray,
    distance_threshold: float = 0.12,
) -> np.ndarray:
    """
    Agglomerative clustering (average linkage, cosine distance).

    Returns array of integer cluster IDs, shape (N,).
    """
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    return clusterer.fit_predict(embeddings)


def pick_representatives(
    labels: List[str],
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
) -> Dict[str, str]:
    """
    For every cluster, pick the label whose embedding is closest to the
    cluster centroid (cosine similarity).

    Returns
    -------
    label_to_representative : dict[str, str]
        Maps each lemmatized label → its cluster representative.
    """
    # Group indices by cluster
    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, cid in enumerate(cluster_ids):
        groups[int(cid)].append(idx)

    label_to_rep: Dict[str, str] = {}
    for cid, indices in groups.items():
        cluster_embeddings = embeddings[indices]  # (k, D)
        centroid = cluster_embeddings.mean(axis=0)  # (D,)
        # Cosine similarity = dot product (embeddings are already L2-normed)
        sims = cluster_embeddings @ centroid  # (k,)
        best_local = int(np.argmax(sims))
        representative = labels[indices[best_local]]
        for idx in indices:
            label_to_rep[labels[idx]] = representative

    return label_to_rep


# ---------------------------------------------------------------------------
# Main normalisation pipeline
# ---------------------------------------------------------------------------


def normalize_inventory(
    inventory_path: Path,
    output_path: Path,
    distance_threshold: float = 0.12,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    device: Optional[str] = None,
    skip_lemmatize: bool = False,
) -> Dict[str, str]:
    """
    Full pipeline: load inventory → (lemmatize) → embed → cluster →
    replace labels → write compact inventory.

    When *skip_lemmatize* is ``True`` the lemmatization step is bypassed and
    raw labels are used directly for embedding and clustering.

    Returns the label_to_representative mapping.
    """
    # ---- 1. Load inventory -------------------------------------------------
    print(f"\nLoading inventory from: {inventory_path}")
    with inventory_path.open("r", encoding="utf-8") as f:
        inventory: Dict[str, List[str]] = json.load(f)

    # ---- 2. Collect all raw labels -----------------------------------------
    all_raw_labels: List[str] = [
        label for labels in inventory.values() for label in labels if label.strip()
    ]
    print(f"Total raw label occurrences: {len(all_raw_labels)}")

    # ---- 3. Lemmatize -------------------------------------------------------
    if skip_lemmatize:
        print("Skipping lemmatization (--skip-lemmatize set).")
        nlp = None
        raw_to_lemma: Dict[str, str] = {label: label for label in set(all_raw_labels)}
    else:
        print("Loading spaCy model...")
        nlp = _load_nlp()
        raw_to_lemma = {}
        for label in all_raw_labels:
            if label not in raw_to_lemma:
                raw_to_lemma[label] = lemmatize_label(label, nlp)

    unique_lemmas = sorted(set(raw_to_lemma.values()))
    print(
        f"Unique {'raw' if skip_lemmatize else 'lemmatized'} labels: {len(unique_lemmas)}"
    )

    # ---- 4. CLIP embeddings ------------------------------------------------
    print(
        f"Embedding {len(unique_lemmas)} labels with {clip_model} ({clip_pretrained})..."
    )
    embeddings, device_str = embed_labels(
        unique_lemmas, model_name=clip_model, pretrained=clip_pretrained, device=device
    )
    print(f"  device: {device_str}, embedding shape: {embeddings.shape}")

    # ---- 5. Cluster --------------------------------------------------------
    print(f"Clustering (cosine distance threshold={distance_threshold})...")
    cluster_ids = cluster_labels(embeddings, distance_threshold=distance_threshold)
    n_clusters = int(cluster_ids.max()) + 1
    print(f"  → {n_clusters} clusters")

    # ---- 6. Pick representatives -------------------------------------------
    lemma_to_rep = pick_representatives(unique_lemmas, embeddings, cluster_ids)

    # Compose full mapping: raw label → representative
    raw_to_rep: Dict[str, str] = {
        raw: lemma_to_rep[lemma] for raw, lemma in raw_to_lemma.items()
    }

    # ---- 7. Rewrite inventory ----------------------------------------------
    compact: Dict[str, List[str]] = {}
    for frame, labels in inventory.items():
        normalized: List[str] = []
        seen: set = set()
        for label in labels:
            if not label.strip():
                continue
            if label in raw_to_rep:
                rep = raw_to_rep[label]
            elif skip_lemmatize:
                rep = label
            else:
                rep = lemmatize_label(label, nlp)  # type: ignore[arg-type]
            if rep not in seen:
                seen.add(rep)
                normalized.append(rep)
        compact[frame] = normalized

    # ---- 8. Write output ---------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(compact, f, indent=2)
    print(f"\nCompact inventory written to: {output_path}")
    print(f"  Frames: {len(compact)}")

    return raw_to_rep


# ---------------------------------------------------------------------------
# Hole filling
# ---------------------------------------------------------------------------


def fill_frame_holes(
    compact_path: Path,
    output_path: Path,
    max_gap: int = 3,
) -> Dict[str, int]:
    """
    Fill short gaps in each object's frame sequence.

    For every unique label in the compact inventory, the function collects the
    sorted list of frame numbers in which that label appears.  When two
    consecutive appearances are separated by a gap of *at most* ``max_gap``
    frames, every intermediate frame is backfilled with the label.

    Parameters
    ----------
    compact_path : Path
        Path to ``objects_inventory_compact.json``.
    output_path : Path
        Destination path for the filled JSON. Overwrites
        ``objects_inventory_compact.json`` in place by default.
    max_gap : int
        Maximum number of consecutive missing frames to fill (default: 3).

    Returns
    -------
    dict[str, int]
        Mapping ``label → number of frames added`` for every label that had at
        least one hole filled.
    """

    print(f"\nLoading compact inventory from: {compact_path}")
    with compact_path.open("r", encoding="utf-8") as f:
        compact: Dict[str, List[str]] = json.load(f)

    # Parse frame number from key, e.g. "frame_00042" → 42
    def _frame_num(key: str) -> int:
        m = re.search(r"(\d+)$", key)
        if not m:
            raise ValueError(f"Cannot parse frame number from key: {key!r}")
        return int(m.group(1))

    # Sort frames numerically and build a reverse map: number → key string
    sorted_frame_keys = sorted(compact.keys(), key=_frame_num)
    num_to_key: Dict[int, str] = {_frame_num(k): k for k in sorted_frame_keys}
    all_frame_nums = sorted(num_to_key.keys())

    # Position-based index so gap checks work with non-sequential keys
    # (e.g. timestamp-based frame IDs)
    num_to_pos: Dict[int, int] = {n: i for i, n in enumerate(all_frame_nums)}

    # ---- Build label → sorted list of frame numbers ----------------------
    label_to_frames: Dict[str, List[int]] = defaultdict(list)
    for key, labels in compact.items():
        fn = _frame_num(key)
        for label in labels:
            label_to_frames[label].append(fn)

    for label in label_to_frames:
        label_to_frames[label].sort()

    # ---- Find and fill holes ----------------------------------------------
    # Work on a mutable copy: frame_num → set of labels
    frame_label_sets: Dict[int, set] = {
        _frame_num(k): set(v) for k, v in compact.items()
    }

    fill_counts: Dict[str, int] = defaultdict(int)

    for label, frame_nums in label_to_frames.items():
        for i in range(len(frame_nums) - 1):
            a, b = frame_nums[i], frame_nums[i + 1]
            # Use sorted position to measure gap, not raw frame numbers
            gap = num_to_pos[b] - num_to_pos[a] - 1
            if 0 < gap <= max_gap:
                # Fill every frame between positions of a and b
                for pos in range(num_to_pos[a] + 1, num_to_pos[b]):
                    fn = all_frame_nums[pos]
                    if fn in frame_label_sets:  # only fill existing frames
                        frame_label_sets[fn].add(label)
                        fill_counts[label] += 1

    # ---- Reconstruct inventory preserving original label order ------------
    filled: Dict[str, List[str]] = {}
    for key in sorted_frame_keys:
        fn = _frame_num(key)
        original_labels = compact[key]
        new_labels_set = frame_label_sets[fn] - set(original_labels)
        filled[key] = original_labels + sorted(new_labels_set)

    # ---- Write output ------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filled, f, indent=2)

    total_filled = sum(fill_counts.values())
    print(f"Hole-filled inventory written to: {output_path}")
    print(f"  Max gap filled: {max_gap} frame(s)")
    print(f"  Labels with holes filled: {len(fill_counts)}")
    print(f"  Total (label, frame) entries added: {total_filled}")

    return dict(fill_counts)


def fill_holes_cli(
    dataset_name: str,
    max_gap: int = 3,
) -> None:

    outputs_dir = get_output_path(dataset_name=dataset_name)
    inventory_dir = outputs_dir / "objects_inventory"

    compact_path = inventory_dir / "objects_inventory_compact.json"

    if not compact_path.exists():
        raise FileNotFoundError(
            f"Compact inventory not found: {compact_path}\n"
            "Please run the normalize step first."
        )

    fill_frame_holes(
        compact_path=compact_path, output_path=compact_path, max_gap=max_gap
    )


# ---------------------------------------------------------------------------
# Objects-to-frames index
# ---------------------------------------------------------------------------


def build_objects_to_frames(
    compact_path: Path,
    output_path: Path,
    min_sequence_length: int = 5,
) -> Dict[str, List[List[str]]]:
    """
    Build an inverted index: object → list of consecutive frame-key sequences.

    For each object label, the full list of frames in which it appears is split
    into runs of *consecutive* frame numbers.  Only runs whose length exceeds
    ``min_sequence_length`` are kept.

    Parameters
    ----------
    compact_path : Path
        Path to ``objects_inventory_compact.json`` (after hole-filling).
    output_path : Path
        Destination path for ``objects_to_frames.json``.
    min_sequence_length : int
        Minimum number of frames a consecutive run must have to be included
        (default: 5, i.e. sequences of length > 5 → strictly longer than 5).

    Returns
    -------
    dict[str, list[list[str]]]
        Mapping ``label → [[frame_key, ...], ...]`` for qualifying sequences.
    """

    print(f"\nLoading compact inventory from: {compact_path}")
    with compact_path.open("r", encoding="utf-8") as f:
        compact: Dict[str, List[str]] = json.load(f)

    def _frame_num(key: str) -> int:
        m = re.search(r"(\d+)$", key)
        if not m:
            raise ValueError(f"Cannot parse frame number from key: {key!r}")
        return int(m.group(1))

    # Sort frames numerically; build num → key map
    sorted_frame_keys = sorted(compact.keys(), key=_frame_num)
    num_to_key: Dict[int, str] = {_frame_num(k): k for k in sorted_frame_keys}

    # Position-based index so consecutiveness checks work with non-sequential
    # keys (e.g. timestamp-based frame IDs)
    all_frame_nums_sorted = sorted(_frame_num(k) for k in compact.keys())
    num_to_pos: Dict[int, int] = {n: i for i, n in enumerate(all_frame_nums_sorted)}

    # ---- Collect sorted frame numbers per label --------------------------
    label_to_framenums: Dict[str, List[int]] = defaultdict(list)
    for key, labels in compact.items():
        fn = _frame_num(key)
        for label in labels:
            label_to_framenums[label].append(fn)

    for label in label_to_framenums:
        label_to_framenums[label].sort()

    # ---- Split each label's frame list into consecutive runs -------------
    result: Dict[str, List[List[str]]] = {}

    for label, frame_nums in sorted(label_to_framenums.items()):
        runs: List[List[str]] = []
        current_run: List[str] = [num_to_key[frame_nums[0]]]

        for i in range(1, len(frame_nums)):
            # Two frames are "consecutive" if they are adjacent in the
            # sorted frame list, regardless of their raw numeric values.
            if num_to_pos[frame_nums[i]] == num_to_pos[frame_nums[i - 1]] + 1:
                current_run.append(num_to_key[frame_nums[i]])
            else:
                if len(current_run) > min_sequence_length:
                    runs.append(current_run)
                current_run = [num_to_key[frame_nums[i]]]

        # Flush the last run
        if len(current_run) > min_sequence_length:
            runs.append(current_run)

        if runs:
            result[label] = runs

    # ---- Write output ----------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    total_seqs = sum(len(seqs) for seqs in result.values())
    print(f"Objects-to-frames index written to: {output_path}")
    print(f"  Objects with qualifying sequences: {len(result)}")
    print(f"  Total sequences (length > {min_sequence_length}): {total_seqs}")

    return result


def objects_to_frames_cli(
    dataset_name: str,
    min_sequence_length: int = 5,
) -> None:
    outputs_dir = get_output_path(dataset_name=dataset_name)
    inventory_dir = outputs_dir / "objects_inventory"

    compact_path = inventory_dir / "objects_inventory_compact.json"
    output_path = inventory_dir / "objects_to_frames.json"

    if not compact_path.exists():
        raise FileNotFoundError(
            f"Compact inventory not found: {compact_path}\n"
            "Please run the normalize step first."
        )

    build_objects_to_frames(
        compact_path=compact_path,
        output_path=output_path,
        min_sequence_length=min_sequence_length,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def normalize_inventory_cli(
    dataset_name: str,
    distance_threshold: float = 0.12,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    device: Optional[str] = None,
    skip_lemmatize: bool = False,
) -> None:
    outputs_dir = get_output_path(dataset_name=dataset_name)
    inventory_dir = outputs_dir / "objects_inventory"

    inventory_path = inventory_dir / "objects_inventory.json"
    output_path = inventory_dir / "objects_inventory_compact.json"

    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Inventory not found: {inventory_path}\n"
            "Please run the objects-inventory identification step first."
        )

    normalize_inventory(
        inventory_path=inventory_path,
        output_path=output_path,
        distance_threshold=distance_threshold,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        device=device,
        skip_lemmatize=skip_lemmatize,
    )


def normalize_labels_from_config(config: dict, dataset_name: str) -> None:
    """Run the full normalize-labels pipeline using the config dictionary.

    Reads parameters from ``config["normalize_labels"]``.  The dataset name is
    passed separately (it is not read from the config).
    Runs: normalize → fill_holes → objects_to_frames (unless disabled).
    """
    nl_cfg = config.get("normalize_labels", {})

    start_time = time.time()

    normalize_inventory_cli(
        dataset_name=dataset_name,
        distance_threshold=nl_cfg.get("distance_threshold", 0.12),
        clip_model=nl_cfg.get("clip_model", "ViT-B-32"),
        clip_pretrained=nl_cfg.get("clip_pretrained", "openai"),
        device=nl_cfg.get("device"),
        skip_lemmatize=nl_cfg.get("skip_lemmatize", False),
    )

    if nl_cfg.get("fill_holes", True):
        fill_holes_cli(
            dataset_name=dataset_name,
            max_gap=nl_cfg.get("max_gap", 3),
        )

    if nl_cfg.get("objects_to_frames", True):
        objects_to_frames_cli(
            dataset_name=dataset_name,
            min_sequence_length=nl_cfg.get("min_sequence_length", 5),
        )

    end_time = time.time()
    duration = end_time - start_time

    step_stats = {
        "duration_seconds": duration,
        "status": "completed",
        "parameters": {
            "normalize_distance_threshold": nl_cfg.get("distance_threshold", 0.12),
            "normalize_clip_model": nl_cfg.get("clip_model", "ViT-B-32"),
            "normalize_clip_pretrained": nl_cfg.get("clip_pretrained", "openai"),
            "normalize_skip_lemmatize": nl_cfg.get("skip_lemmatize", False),
            "normalize_fill_holes": nl_cfg.get("fill_holes", True),
            "normalize_max_gap": nl_cfg.get("max_gap", 3),
            "normalize_objects_to_frames": nl_cfg.get("objects_to_frames", True),
            "normalize_min_sequence_length": nl_cfg.get("min_sequence_length", 5),
        }
    }

    save_runtime_stats(dataset_name, "normalize_labels", step_stats)


def main() -> None:
    """CLI entry point: run the normalize-labels step from a config file."""
    from config_cli import parse_config_args

    config, dataset_name = parse_config_args(
        description="Normalize object labels via CLIP clustering.",
    )
    normalize_labels_from_config(config, dataset_name)


if __name__ == "__main__":
    main()
