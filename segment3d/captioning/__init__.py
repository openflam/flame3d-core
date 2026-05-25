"""
Captioning package for generating descriptions of 3D scene components.
"""

from .captioner_base import Captioner, CaptionResult, create_captioner
from .orchestrator import caption_all_components_cli

__all__ = [
    "Captioner",
    "CaptionResult",
    "create_captioner",
    "caption_all_components_cli",
]
