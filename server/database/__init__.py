"""server.database – PostgreSQL/PostGIS interface layer.

Public helpers for the ``dataindex`` table (the per-dataset status registry).
Future milestones will add per-dataset spatial tables here.
"""

from server.database.dataindex import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_MARKED_DELETE,
    STATUS_PROCESSING,
    delete_row,
    get_dataset,
    init_db,
    list_datasets,
    list_marked_delete,
    mark_deleted,
    reconcile_existing,
    set_status,
    upsert_processing,
)

__all__ = [
    "STATUS_COMPLETE",
    "STATUS_FAILED",
    "STATUS_MARKED_DELETE",
    "STATUS_PROCESSING",
    "delete_row",
    "get_dataset",
    "init_db",
    "list_datasets",
    "list_marked_delete",
    "mark_deleted",
    "reconcile_existing",
    "set_status",
    "upsert_processing",
]
