"""
extract_zip.py – Extract a ZIP archive to a target directory.

Usage::

    from data_processor.utils.extract_zip import extract_zip

    extracted_dir = extract_zip(
        zip_path=Path("data/my_dataset/input.zip"),
        dest_dir=Path("data/my_dataset/polycam_data"),
    )
"""

from __future__ import annotations

import zipfile
from pathlib import Path


def extract_zip(zip_path: Path | str, dest_dir: Path | str) -> Path:
    """Extract *zip_path* into *dest_dir*.

    Parameters
    ----------
    zip_path : Path or str
        Path to the ``.zip`` file.
    dest_dir : Path or str
        Directory to extract into.  Created if it does not exist.

    Returns
    -------
    pathlib.Path
        The *dest_dir* path (absolute).

    Raises
    ------
    FileNotFoundError
        If *zip_path* does not exist.
    zipfile.BadZipFile
        If the file is not a valid ZIP archive.
    """
    zip_path = Path(zip_path).resolve()
    dest_dir = Path(dest_dir).resolve()

    if not zip_path.is_file():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    print(f"Extracted {zip_path.name} → {dest_dir}")
    return dest_dir
