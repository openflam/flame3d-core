"""Semantic search module for component matching.

Only the BM25 provider is integrated into flame3d-core. The OpenAI (full),
OpenAI RAG, and CLIP providers from the original scan-to-map server are
intentionally omitted.
"""

from .base import SemanticSearchProvider
from .bm25_provider import BM25Provider

__all__ = [
    "SemanticSearchProvider",
    "BM25Provider",
]
