"""BM25-based semantic search provider."""

import sys
from pathlib import Path
from typing import Any, Dict

from .base import SemanticSearchProvider

from server.database import spatial as database


class BM25Provider(SemanticSearchProvider):
    """Semantic search provider using BM25 algorithm."""

    def __init__(
        self,
        dataset_name: str,
        top_k: int = 10,
        gap_threshold: float = 1.0,
        ratio_threshold: float = 0.7,
    ):
        """
        Initialize the BM25 provider.

        Args:
            dataset_name: Dataset to search
            top_k: Number of top matching components to return
            gap_threshold: Minimum gap in scores to stop including results (elbow detection)
            ratio_threshold: Minimum ratio of current/previous score to continue (elbow detection)
        """
        super().__init__(dataset_name)
        self.top_k = top_k
        self.gap_threshold = gap_threshold
        self.ratio_threshold = ratio_threshold

    def match_components(self, query: str) -> Dict[str, Any]:
        """
        Match a search query to component descriptions using BM25.

        Args:
            query: The search query string

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice
        """
        # Call the unified bm25_search in database.py
        search_results = database.bm25_search(
            dataset_name=self.dataset_name,
            search_terms=query,
            top_k=self.top_k,
            apply_elbow=True,
            gap_threshold=self.gap_threshold,
            ratio_threshold=self.ratio_threshold
        )
        
        matched_component_ids = search_results["component_ids"]

        if len(matched_component_ids) > 0:
            reason = f"Found {len(matched_component_ids)} matching component(s) using BM25 ranked search."
        else:
            reason = "No matching components found."

        return {
            "component_ids": matched_component_ids,
            "reason": reason,
        }
