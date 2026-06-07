"""config_io.py – Central I/O path configuration for flame3d-core."""

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# ── Edit these to change default directory layout ──────────────────────────
PATHS = {
    "data": _REPO_ROOT / "data",
    "outputs": _REPO_ROOT / "outputs",
}


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
