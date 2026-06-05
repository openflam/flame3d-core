"""BM25 search tool."""

from __future__ import annotations

from typing import Any

from server.database import spatial as database
from .utils import _get_dataset_name


def search_terms(
    query_terms: list[str],
    dataset_name: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Search component captions with BM25 and return the best matching components.

    Args:
            query_terms: List of term strings from the LLM (e.g. ["red chair", "desk"])
            dataset_name: Dataset to search; falls back to DATASET_NAME env var
            top_k: Max number of results to return

    Returns:
            Dict with ranked matches and metadata.
    """
    if not isinstance(query_terms, list):
        raise ValueError("query_terms must be a list of strings")

    resolved_dataset = _get_dataset_name(dataset_name)

    result = database.bm25_search(
        dataset_name=resolved_dataset,
        search_terms=query_terms,
        top_k=top_k,
        apply_elbow=False
    )
    
    return result


SEARCH_TERMS_TOOL = {
    "type": "function",
    "name": "search_terms",
    "description": (
        "Search component captions using BM25 over the current dataset table and "
        "return the most relevant components."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query_terms": {
                "type": "array",
                "description": "List of text terms to search for.",
                "items": {"type": "string"},
                "minItems": 1,
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of ranked components to return.",
                "default": 10,
                "minimum": 1,
            },
        },
        "required": ["query_terms"],
        "additionalProperties": False,
    },
}

SEARCH_THINKING_TEXT = "Searching for relevant objects..."

SEARCH_TERMS_DESCRIPTION = """- **Primary Search Tool:** Use `search_terms` to find objects in the dataset.
  - **BM25 Creative Expansion:** The `search_terms` tool uses BM25 keyword matching. To be effective, you must not simply repeat the user's query. Instead, creatively expand the `query_terms` list to include synonyms, related objects, or likely containers where an item might be found.
    - _Example:_ If asked "Where is my beverage?", search for `["beverage", "coffee machine", "tea", "vending machine", "cup", "mug", "soda"]`.
    - _Example:_ If asked "Find a place to sit," search for `["chair", "stool", "sofa", "armchair", "bench"]`."""
