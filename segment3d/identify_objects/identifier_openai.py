"""
OpenAI API-based object identifier implementation.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import List, Optional

# Load .env from the project root (scan-to-map/.env) if present
try:
    from dotenv import load_dotenv

    _ENV_PATH = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(dotenv_path=_ENV_PATH)
except ImportError:
    pass  # python-dotenv not installed; rely on the environment directly

from ..prompts import IDENTIFY_COMPONENT_PROMPT
from .identifier_base import IdentificationResult


def _encode_image_base64(image_path: Path) -> str:
    """Encode an image file to a base64 data URI string."""
    suffix = image_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/jpeg")
    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


class OpenAIIdentifier:
    """
    Object identifier implementation using the OpenAI Chat Completions API
    with vision support.

    For each frame, it sends the image along with IDENTIFY_COMPONENT_PROMPT
    and parses the comma-separated response into a list of objects.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_concurrent: int = 8,
    ):
        """
        Initialize the OpenAI identifier.

        Args:
            model: OpenAI model name (must support vision)
            api_key: OpenAI API key. Falls back to the OPENAI_API_KEY
                     environment variable if not provided.
            max_tokens: Maximum tokens to generate per response
            temperature: Sampling temperature (0.0 for deterministic)
            max_concurrent: Maximum number of concurrent API requests
        """
        from openai import OpenAI

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key must be provided via the `api_key` argument "
                "or the OPENAI_API_KEY environment variable."
            )

        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent = max_concurrent
        self.client = OpenAI(api_key=resolved_key)

        print(f"\nOpenAI identifier initialised (model={model})")

    def _call_single(self, frame_name: str, image_path: Path) -> IdentificationResult:
        """
        Call the OpenAI API for a single frame.

        Args:
            frame_name: Name/identifier of the frame
            image_path: Path to the image file

        Returns:
            IdentificationResult for this frame
        """
        image_path_str = str(image_path)

        if not image_path.exists():
            print(f"  Warning: Image not found: {image_path}, skipping")
            return IdentificationResult(
                frame_name=frame_name,
                objects=[],
                image_path=image_path_str,
                error=f"Image not found: {image_path}",
            )

        try:
            data_uri = _encode_image_base64(image_path)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": data_uri},
                            },
                            {
                                "type": "text",
                                "text": IDENTIFY_COMPONENT_PROMPT,
                            },
                        ],
                    }
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )

            raw_text = response.choices[0].message.content.strip()
            objects = self._parse_objects(raw_text)

            return IdentificationResult(
                frame_name=frame_name,
                objects=objects,
                image_path=image_path_str,
                error=None,
            )

        except Exception as e:
            return IdentificationResult(
                frame_name=frame_name,
                objects=[],
                image_path=image_path_str,
                error=str(e),
            )

    def _parse_objects(self, raw_text: str) -> List[str]:
        """
        Parse a comma-separated list of objects from model output.

        Args:
            raw_text: Raw text response from the model

        Returns:
            Deduplicated list of object name strings (stripped and non-empty)
        """
        objects = [obj.strip() for obj in raw_text.split(",")]
        objects = list(
            dict.fromkeys(obj for obj in objects if obj)
        )  # dedup, preserve order
        return objects

    def identify_batch(
        self,
        batch_data: List[tuple[str, Path]],
    ) -> List[IdentificationResult]:
        """
        Identify objects in a batch of frames by calling the OpenAI API
        concurrently (up to ``max_concurrent`` requests in flight at once).

        Args:
            batch_data: List of (frame_name, image_path) tuples

        Returns:
            List of IdentificationResult objects in the same order as the input
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        results: List[IdentificationResult] = [None] * len(batch_data)  # type: ignore[list-item]

        with ThreadPoolExecutor(max_workers=self.max_concurrent) as executor:
            future_to_idx = {
                executor.submit(self._call_single, frame_name, image_path): idx
                for idx, (frame_name, image_path) in enumerate(batch_data)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    frame_name, image_path = batch_data[idx]
                    results[idx] = IdentificationResult(
                        frame_name=frame_name,
                        objects=[],
                        image_path=str(image_path),
                        error=str(e),
                    )

        return results

    def cleanup(self) -> None:
        """No-op: the OpenAI client holds no persistent resources."""
        pass
