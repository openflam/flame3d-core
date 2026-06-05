"""
create_tables.py

Connects to the PostGIS service and creates one table per dataset found in
../outputs/.  Each table stores the bounding box as a native PostGIS 3-D 
geometry so it can be spatially indexed and queried.

Spatial design
--------------
The 3-D AABB for each component is stored as a LINESTRING Z that connects the
min corner to the max corner (the space diagonal of the box).  The 3-D
bounding box of this line segment is *identical* to the original AABB, so a
GIST index built with the ``gist_geometry_ops_nd`` operator class indexes the
full 3-D extent.  Two indexes are created:

  * nd-GIST  (gist_geometry_ops_nd)  — accelerates the &&& operator (3-D
    bounding-box overlap) and other N-D predicates.
  * 2-D GIST (default)               — accelerates the && operator and all
    standard PostGIS 2-D spatial functions on the XY footprint.

Schema (one table per dataset, named after the sanitized dataset name):
    component_id   INTEGER  PRIMARY KEY
    caption        TEXT
    bbox_json      TEXT     -- raw bbox JSON kept for backwards-compatibility
    best_crop      TEXT     -- crop filename with highest fraction_visible
    bbox_geom      geometry(LineStringZ, 0)  -- 3-D AABB diagonal, SRID=0

Usage
-----
    # populate every dataset found in ../outputs/
    python create_tables.py

    # populate one or more specific datasets
    python create_tables.py CFA_Aatmi ArenaLabSemanticNeg

Environment variables (defaults match docker-compose.yml):
    POSTGRES_HOST      default: localhost
    POSTGRES_PORT      default: 5432
    POSTGRES_DB        default: scantomap
    POSTGRES_USER      default: scantomap
    POSTGRES_PASSWORD  default: scantomap
"""

import json
import re
import sys
import numpy as np

import psycopg2
from psycopg2.extras import execute_values

from config_io import PATHS
from server.database.connection import DATABASE_URL

# ── paths ────────────────────────────────────────────────────────────────────

OUTPUTS_DIR = PATHS["outputs"]


# ── database connection ───────────────────────────────────────────────────────


def _pg_conn() -> psycopg2.extensions.connection:
    return psycopg2.connect(DATABASE_URL)


# ── helpers ───────────────────────────────────────────────────────────────────


def _table_name(dataset_name: str) -> str:
    """Return a safe PostgreSQL identifier for a dataset."""
    return re.sub(r"[^a-z0-9_]", "_", dataset_name.lower())


def _linestring_z(bbox: dict) -> str | None:
    """
    Build a WKT LINESTRING Z from the corners of an OBB.

    The space diagonal min→max has the exact same 3-D bounding box as the
    full axis-aligned box, so a GIST nd-index on this geometry is equivalent
    to indexing the box itself without storing all eight corners.
    """
    corners = bbox.get("corners")
    if not corners or not isinstance(corners, list) or len(corners) == 0:
        return None
        
    corners_np = np.array(corners)
    min_coords = corners_np.min(axis=0)
    max_coords = corners_np.max(axis=0)
    
    return f"LINESTRING Z ({min_coords[0]} {min_coords[1]} {min_coords[2]}, {max_coords[0]} {max_coords[1]} {max_coords[2]})"


def _best_crop(manifest_data: dict, comp_id: int) -> str | None:
    entry = manifest_data.get(str(comp_id)) or manifest_data.get(comp_id)
    if not entry:
        return None
    crops = entry.get("crops", [])
    if not crops:
        return None
    best = max(crops, key=lambda c: c.get("fraction_visible", 0))
    return best.get("crop_filename")


# ── data loading ──────────────────────────────────────────────────────────────


def _load_dataset(dataset_name: str) -> list[tuple]:
    """
    Read component_captions.json, bbox_corners.json and crops/manifest.json
    from outputs/<dataset_name>/ and return a list of row tuples:

        (component_id, caption, bbox_json, best_crop, bbox_wkt | None)
    """
    outputs_dir = OUTPUTS_DIR / dataset_name

    captions_path = outputs_dir / "component_captions.json"
    bbox_path = outputs_dir / "bbox_corners.json"
    manifest_path = outputs_dir / "crops" / "manifest.json"

    if not captions_path.exists():
        raise FileNotFoundError(f"Missing: {captions_path}")
    if not bbox_path.exists():
        raise FileNotFoundError(f"Missing: {bbox_path}")

    with open(captions_path, "r") as f:
        captions_list = json.load(f)

    with open(bbox_path, "r") as f:
        bbox_data = json.load(f)

    manifest_data: dict = {}
    if manifest_path.exists():
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
    else:
        print(f"  Warning: crops/manifest.json not found — best_crop will be NULL")

    # connected_comp_id → bbox dict
    bbox_lookup = {item["connected_comp_id"]: item["bbox"] for item in bbox_data}

    rows: list[tuple] = []
    for item in captions_list:
        comp_id = item["component_id"]
        caption = item.get("caption", "")
        bbox = bbox_lookup.get(comp_id, {})
        rows.append(
            (
                comp_id,
                caption,
                json.dumps(bbox),
                _best_crop(manifest_data, comp_id),
                _linestring_z(bbox),
            )
        )

    return rows


# ── DDL & DML ─────────────────────────────────────────────────────────────────


def _create_table(cur: psycopg2.extensions.cursor, table: str) -> None:
    # Ensure the PostGIS extension is available in this database.
    cur.execute("CREATE EXTENSION IF NOT EXISTS postgis;")

    cur.execute(f'DROP TABLE IF EXISTS "{table}";')
    cur.execute(
        f"""
        CREATE TABLE "{table}" (
            component_id  INTEGER PRIMARY KEY,
            caption       TEXT,
            bbox_json     TEXT,
            best_crop     TEXT,
            bbox_geom     geometry(LineStringZ, 0)
        );
        """
    )

    # ── 3-D nd-GIST index ──────────────────────────────────────────────────
    # gist_geometry_ops_nd indexes all N dimensions of the geometry's bounding
    # box, enabling the &&& (N-D overlap) operator for true 3-D spatial
    # queries such as:
    #   SELECT * FROM t WHERE bbox_geom &&& ST_GeomFromText('LINESTRING Z (...)', 0)
    cur.execute(
        f"""
        CREATE INDEX "{table}_bbox_geom_nd_idx"
        ON "{table}"
        USING GIST (bbox_geom gist_geometry_ops_nd);
        """
    )

    # ── 2-D GIST index ────────────────────────────────────────────────────
    # Standard 2-D GIST index keeps XY footprint queries fast (e.g. finding
    # all objects in a floor-plan region using &&).
    cur.execute(
        f"""
        CREATE INDEX "{table}_bbox_geom_2d_idx"
        ON "{table}"
        USING GIST (bbox_geom);
        """
    )

    # ── Caption lexical index ──────────────────────────────────────────────
    # Supports fast candidate prefiltering on caption text before BM25 scoring.
    cur.execute(
        f"""
        CREATE INDEX "{table}_caption_fts_idx"
        ON "{table}"
        USING GIN (to_tsvector('english', COALESCE(caption, '')));
        """
    )

    print(
        f"  Created table '{table}' with nd-GIST (3-D), 2-D GIST, and caption FTS indexes"
    )


def _insert_rows(
    cur: psycopg2.extensions.cursor, table: str, rows: list[tuple]
) -> None:
    execute_values(
        cur,
        f"""
        INSERT INTO "{table}"
            (component_id, caption, bbox_json, best_crop, bbox_geom)
        VALUES %s
        """,
        rows,
        # ST_GeomFromText parses the WKT and assigns SRID 0; NULL WKT → NULL geom.
        template="(%s, %s, %s, %s, ST_GeomFromText(%s, 0))",
    )
    print(f"  Inserted {len(rows)} row(s) into '{table}'")


# ── per-dataset orchestration ─────────────────────────────────────────────────


def process_dataset(con: psycopg2.extensions.connection, dataset_name: str) -> None:
    print(f"\nDataset: {dataset_name}")
    try:
        rows = _load_dataset(dataset_name)
    except FileNotFoundError as exc:
        print(f"  Skipped — {exc}")
        return

    table = _table_name(dataset_name)
    with con:  # transaction: commits on success, rolls back on error
        with con.cursor() as cur:
            _create_table(cur, table)
            _insert_rows(cur, table, rows)


# ── pipeline step entrypoint ──────────────────────────────────────────────────


def create_tables_from_config(config: dict, dataset_name: str) -> None:
    """Build the PostGIS spatial table for a single dataset (pipeline step).

    Reads the component outputs (captions, bbox, crops) for *dataset_name* and
    (re)creates its spatial table.  The *config* argument is accepted for
    interface uniformity with the other ``*_from_config`` steps; this step has
    no tunable parameters, so it is unused.
    """
    con = _pg_conn()
    try:
        process_dataset(con, dataset_name)
    finally:
        con.close()


# ── discovery & entrypoint ────────────────────────────────────────────────────


def discover_datasets() -> list[str]:
    """Return all dataset names that have a component_captions.json file."""
    if not OUTPUTS_DIR.exists():
        return []
    return sorted(
        d.name
        for d in OUTPUTS_DIR.iterdir()
        if d.is_dir() and (d / "component_captions.json").exists()
    )


def main() -> None:
    datasets = sys.argv[1:] if len(sys.argv) >= 2 else discover_datasets()

    if not datasets:
        print(f"No datasets found in {OUTPUTS_DIR}")
        sys.exit(0)

    print(f"Datasets to process: {', '.join(datasets)}")

    con = _pg_conn()
    try:
        for ds in datasets:
            process_dataset(con, ds)
    finally:
        con.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
