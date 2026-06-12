"""
LLM API-based object identifier implementation.

Uses the shared :class:`utils.llm_call.LLMCaller` (backed by LiteLLM) so any
provider/model supported by LiteLLM can be used, with credentials resolved from
the project-root ``.env``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import List, Optional

from utils.llm_call import LLMCaller

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


class LLMAPIIdentifier:
    """
    Object identifier implementation using an LLM API (via LiteLLM) with
    vision support.

    For each frame, it sends the image along with IDENTIFY_COMPONENT_PROMPT
    and parses the comma-separated response into a list of objects.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        max_tokens: int = 256,
        max_concurrent: int = 8,
        rate_limits: Optional[dict] = None,
    ):
        """
        Initialize the LLM API identifier.

        Args:
            model: Model name (must support vision). Any model routable by
                   LiteLLM is accepted.
            api_key: API key. If not provided, LiteLLM resolves credentials
                     from the environment (loaded from the project-root .env).
            max_tokens: Maximum tokens to generate per response
            max_concurrent: Maximum number of concurrent API requests
            rate_limits: Per-model RPM/TPM limits from the run config, handed to
                     the shared scheduler so this model is throttled. None means
                     no limiting. This step fans out across ``max_concurrent``
                     threads, so throttling here is what keeps it under provider
                     limits.
        """
        self.model = model
        self.max_concurrent = max_concurrent
        self.caller = LLMCaller(
            model=model,
            api_key=api_key,
            max_completion_tokens=max_tokens,
            rate_limits=rate_limits,
        )

        print(f"\nLLM API identifier initialised (model={model})")

    def _call_single(self, frame_name: str, image_path: Path) -> IdentificationResult:
        """
        Call the LLM API for a single frame.

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

            response = self.caller.stream_chat(
                input=[
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
            )

            raw_text = (response.get("content") or "").strip()
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
        Identify objects in a batch of frames by calling the LLM API
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
        """No-op: the LLM API client holds no persistent resources."""
        pass
