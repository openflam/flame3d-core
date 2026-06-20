"""config_io.py – Central I/O path configuration for flame3d-core."""

import shutil
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# ── Edit these to change default directory layout ──────────────────────────
PATHS = {
    "data": _REPO_ROOT / "data",
    "outputs": _REPO_ROOT / "outputs",
}


def reset_dir(path: Path) -> Path:
    """Delete *path* if it exists, then recreate it as an empty directory.

    Pipeline steps call this before writing a directory of per-item artifacts
    (e.g. ``crops/``) so a re-run starts from a clean slate instead of mixing
    fresh outputs with leftovers from a previous run (e.g. a ``component_<id>``
    crop directory for a component that no longer exists).
    """
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_data_path(dataset_name: str) -> Path:
    """Return ``data/<dataset_name>/``."""
    return PATHS["data"] / dataset_name


def get_output_path(dataset_name: str) -> Path:
    """Return ``outputs/<dataset_name>/``, creating it if needed."""
    out_dir = PATHS["outputs"] / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def get_colmap_output_path(dataset_name: str) -> Path:
    """Return ``outputs/<dataset_name>/colmap/``, creating it if needed."""
    colmap_dir = get_output_path(dataset_name) / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    return colmap_dir


def get_images_output_path(dataset_name: str) -> Path:
    """Return ``outputs/<dataset_name>/images/``."""
    return get_output_path(dataset_name) / "images"


def get_rendered_depth_output_path(dataset_name: str) -> Path:
    """Return ``outputs/<dataset_name>/rendered_images/``, creating it if needed."""
    rendered_dir = get_output_path(dataset_name) / "rendered_images"
    rendered_dir.mkdir(parents=True, exist_ok=True)
    return rendered_dir
