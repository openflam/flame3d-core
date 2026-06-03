"""dataindex.py – The ``dataindex`` table: one row per dataset + its status.

This is the durable registry of datasets known to the system.  Each row tracks
whether a dataset is currently being processed or has finished, plus the Celery
``job_id`` of its latest run (so the UI can attach to live progress).

The database is PostGIS-enabled because later milestones will add a separate
spatial table *per dataset* holding the processed components, indexed by 3D
coordinates and captions.  This module only manages the index table for now.

Schema
------
    dataindex(
        dataset_name TEXT PRIMARY KEY,
        data_source  TEXT,
        status       TEXT NOT NULL,   -- 'processing' | 'complete' | 'failed'
        job_id       TEXT,            -- latest Celery task id (NULL if unknown)
        error        TEXT,
        created_at   TIMESTAMPTZ,
        updated_at   TIMESTAMPTZ
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from psycopg2.extras import RealDictCursor

from server.database.connection import get_connection

# ── Status values ──────────────────────────────────────────────────────────

STATUS_PROCESSING = "processing"
STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"

# Process-local flag so we run the (idempotent) DDL only once per process.
_schema_ready = False


# ── Schema ─────────────────────────────────────────────────────────────────

_DDL = """
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS dataindex (
    dataset_name TEXT PRIMARY KEY,
    data_source  TEXT,
    status       TEXT NOT NULL DEFAULT 'processing',
    job_id       TEXT,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def init_db() -> None:
    """Create the PostGIS extension and ``dataindex`` table if missing."""
    global _schema_ready
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(_DDL)
    _schema_ready = True


def _ensure_schema() -> None:
    """Lazily run the DDL once per process before the first read/write."""
    if not _schema_ready:
        init_db()


def _row_to_dict(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Normalize a DB row: ISO-format timestamps for JSON serialization."""
    if row is None:
        return None
    out = dict(row)
    for key in ("created_at", "updated_at"):
        if out.get(key) is not None:
            out[key] = out[key].isoformat()
    return out


# ── Writes ─────────────────────────────────────────────────────────────────

def upsert_processing(
    dataset_name: str,
    data_source: Optional[str],
    job_id: str,
) -> None:
    """Mark *dataset_name* as ``processing`` for a freshly-started run.

    Inserts the dataset if new, or resets an existing one (clearing any prior
    error) so a re-run starts from a clean ``processing`` state.
    """
    _ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO dataindex (dataset_name, data_source, status, job_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (dataset_name) DO UPDATE SET
                    data_source = EXCLUDED.data_source,
                    status      = EXCLUDED.status,
                    job_id      = EXCLUDED.job_id,
                    error       = NULL,
                    updated_at  = now();
                """,
                (dataset_name, data_source, STATUS_PROCESSING, job_id),
            )


def set_status(
    dataset_name: str,
    status: str,
    error: Optional[str] = None,
) -> None:
    """Update a dataset's terminal status (``complete`` / ``failed``)."""
    _ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE dataindex
                   SET status = %s, error = %s, updated_at = now()
                 WHERE dataset_name = %s;
                """,
                (status, error, dataset_name),
            )


def reconcile_existing(dataset_names: List[str]) -> None:
    """Register pre-existing on-disk datasets that aren't in the index yet.

    Datasets that already exist on disk (e.g. processed before this index was
    introduced) are not actively running, so they are recorded as ``complete``.
    Existing rows are left untouched.
    """
    if not dataset_names:
        return
    _ensure_schema()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO dataindex (dataset_name, status)
                VALUES (%s, %s)
                ON CONFLICT (dataset_name) DO NOTHING;
                """,
                [(name, STATUS_COMPLETE) for name in dataset_names],
            )


# ── Reads ──────────────────────────────────────────────────────────────────

def list_datasets() -> List[Dict[str, Any]]:
    """Return all datasets, most recently updated first."""
    _ensure_schema()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM dataindex ORDER BY updated_at DESC;")
            return [_row_to_dict(r) for r in cur.fetchall()]


def get_dataset(dataset_name: str) -> Optional[Dict[str, Any]]:
    """Return a single dataset row, or ``None`` if unknown."""
    _ensure_schema()
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM dataindex WHERE dataset_name = %s;",
                (dataset_name,),
            )
            return _row_to_dict(cur.fetchone())
