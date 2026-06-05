"""Base class for semantic search providers."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class SemanticSearchProvider(ABC):
    """Abstract base class for semantic search providers."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name

    @abstractmethod
    def match_components(self, query: str) -> Dict[str, Any]:
        """
        Match a search query to component descriptions.

        Args:
            query: The search query string

        Returns:
            A dictionary with:
                - "component_ids": list of matched component IDs
                - "reason": explanation for the choice
        """
        pass
