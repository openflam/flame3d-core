"""clean_storage.py – Purge files for datasets soft-deleted in the UI.

Deleting a dataset in the frontend only flips its ``dataindex`` row to
``marked_delete`` (so it vanishes from the list); the files are intentionally
left on disk.  This script is the reaper: it finds every ``marked_delete``
dataset and removes its directories under both ``data/`` and ``outputs/``, then
drops the index row.

Usage
-----
    # inside the running stack (server image has psycopg2 + the mounts)
    docker compose exec server python -m server.clean_storage

    # preview without deleting anything
    docker compose exec server python -m server.clean_storage --dry-run

Run it on a schedule (cron) or by hand — it is safe to re-run.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from config_io import PATHS
from server.database import delete_row, list_marked_delete


def _safe_target(root: Path, dataset_name: str) -> Path | None:
    """Resolve ``root/dataset_name`` and refuse anything outside *root*.

    Guards against a malformed/hostile dataset name (e.g. containing ``..``)
    escaping the data/outputs roots.
    """
    root = root.resolve()
    target = (root / dataset_name).resolve()
    if target == root or root not in target.parents:
        return None
    return target


def _remove(target: Path, *, dry_run: bool) -> bool:
    """Delete *target* directory if it exists. Returns True if it existed."""
    if not target.exists():
        return False
    if dry_run:
        print(f"  [dry-run] would remove {target}")
        return True
    shutil.rmtree(target)
    print(f"  removed {target}")
    return True


def clean_storage(dry_run: bool = False) -> int:
    """Purge files for all ``marked_delete`` datasets. Returns count purged."""
    datasets = list_marked_delete()
    if not datasets:
        print("No datasets marked for deletion.")
        return 0

    data_root = PATHS["data"]
    outputs_root = PATHS["outputs"]

    purged = 0
    for row in datasets:
        name = row["dataset_name"]
        print(f"Cleaning '{name}'...")

        for root in (data_root, outputs_root):
            target = _safe_target(root, name)
            if target is None:
                print(
                    f"  ! skipping unsafe path for {name!r} under {root}",
                    file=sys.stderr,
                )
                continue
            _remove(target, dry_run=dry_run)

        if not dry_run:
            delete_row(name)
            print(f"  dropped index row for '{name}'")
        purged += 1

    verb = "would be purged" if dry_run else "purged"
    print(f"\nDone. {purged} dataset(s) {verb}.")
    return purged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Purge files for datasets marked for deletion."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be deleted without touching anything.",
    )
    args = parser.parse_args()
    clean_storage(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
