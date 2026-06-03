"""connection.py – PostgreSQL/PostGIS connection helper.

A single place to obtain a psycopg2 connection from the ``DATABASE_URL``
environment variable.  Connections are short-lived: callers use the
:func:`get_connection` context manager, which commits on success, rolls back on
error, and always closes the connection.  Traffic to this DB is low (dataset
index reads/writes), so a per-operation connection is simpler and robust enough
— a pool can be added later if the per-dataset 3D query tables need it.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator

import psycopg2

# Default points at the ``postgis`` service on the docker-compose network.
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://flame:flame@postgis:5432/flame3d"
)


def _connect(retries: int = 10, delay: float = 1.5):
    """Open a psycopg2 connection, retrying while the DB is still booting."""
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return psycopg2.connect(DATABASE_URL)
        except psycopg2.OperationalError as exc:
            last_err = exc
            time.sleep(delay)
    raise RuntimeError(
        f"Could not connect to database at {DATABASE_URL!r} after "
        f"{retries} attempts"
    ) from last_err


@contextmanager
def get_connection() -> Iterator["psycopg2.extensions.connection"]:
    """Yield a connection, committing on success and rolling back on error."""
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
