"""
Base classes and interfaces for captioner implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, runtime_checkable


@dataclass
class CaptionResult:
    """Result of captioning a single component."""

    component_id: str | int
    caption: str
    image_paths: List[str]
    error: str | None = None


@runtime_checkable
class Captioner(Protocol):
    """
    Protocol for captioner implementations.
    
    A captioner takes a batch of components with their crop images and
    generates captions for them.
    """

    def caption_batch(
        self,
        batch_data: List[tuple[str | int, List[Dict[str, Any]]]],
        crops_dir: Path,
    ) -> List[CaptionResult]:
        """
        Generate captions for a batch of components.

        Args:
            batch_data: List of (component_id, top_images) tuples where
                       top_images is a list of crop info dicts from manifest
            crops_dir: Directory containing the cropped images

        Returns:
            List of CaptionResult objects
        """
        ...

    def cleanup(self) -> None:
        """
        Optional cleanup method called when captioning is complete.
        Useful for releasing GPU memory, closing connections, etc.
        """
        ...


def create_captioner(
    captioner_type: str,
    model: str,
    device: int = 0,
    **kwargs,
) -> Captioner:
    """
    Factory function to create a captioner instance.

    Args:
        captioner_type: Type of captioner to create (e.g., "vllm", "hf", "openai")
        model: Model name/identifier
        device: GPU device ID to use
        **kwargs: Additional captioner-specific arguments

    Returns:
        Captioner instance

    Raises:
        ValueError: If captioner_type is not recognized
    """
    if captioner_type == "vllm":
        from .captioner_vllm import VLLMCaptioner

        return VLLMCaptioner(model=model, device=device, **kwargs)
    else:
        raise ValueError(
            f"Unknown captioner type: {captioner_type}. "
            f"Supported types: vllm"
        )
