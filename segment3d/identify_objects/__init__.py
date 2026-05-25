"""
Objects inventory package for identifying tangible objects in every frame of a 3D scene.
"""

from .identifier_base import Identifier, IdentificationResult, create_identifier
from .orchestrator import identify_all_frames_cli

__all__ = [
    "Identifier",
    "IdentificationResult",
    "create_identifier",
    "identify_all_frames_cli",
]
