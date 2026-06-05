"""Utility functions for tools."""

import os

def _get_dataset_name(dataset_name: str | None) -> str:
    """Resolve dataset from argument first, then from environment."""
    resolved = dataset_name or os.environ.get("DATASET_NAME")
    if not resolved:
        raise ValueError("dataset_name is required (arg or DATASET_NAME env var)")
    return resolved
