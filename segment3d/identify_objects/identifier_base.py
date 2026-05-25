"""
Base classes and interfaces for object identifier implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Protocol, runtime_checkable


@dataclass
class IdentificationResult:
    """Result of identifying objects in a single frame."""

    frame_name: str
    objects: List[str]
    image_path: str
    error: str | None = None


@runtime_checkable
class Identifier(Protocol):
    """
    Protocol for object identifier implementations.

    An identifier takes a batch of frames with their image paths and
    returns the list of tangible objects visible in each frame.
    """

    def identify_batch(
        self,
        batch_data: List[tuple[str, Path]],
    ) -> List[IdentificationResult]:
        """
        Identify objects for a batch of frames.

        Args:
            batch_data: List of (frame_name, image_path) tuples

        Returns:
            List of IdentificationResult objects
        """
        ...

    def cleanup(self) -> None:
        """
        Optional cleanup method called when identification is complete.
        Useful for releasing GPU memory, closing connections, etc.
        """
        ...


def create_identifier(
    identifier_type: str,
    model: str,
    device: int = 0,
    **kwargs,
) -> Identifier:
    """
    Factory function to create an identifier instance.

    Args:
        identifier_type: Type of identifier to create (e.g., "vllm")
        model: Model name/identifier
        device: GPU device ID to use
        **kwargs: Additional identifier-specific arguments

    Returns:
        Identifier instance

    Raises:
        ValueError: If identifier_type is not recognized
    """
    if identifier_type == "vllm":
        from .identifier_vllm import VLLMIdentifier

        return VLLMIdentifier(model=model, device=device, **kwargs)
    elif identifier_type == "openai":
        from .identifier_openai import OpenAIIdentifier

        return OpenAIIdentifier(model=model, **kwargs)
    else:
        raise ValueError(
            f"Unknown identifier type: {identifier_type}. "
            f"Supported types: vllm, openai"
        )
