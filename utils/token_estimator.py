"""Token-count estimation for LLM requests, used to size rate-limit budgets.

The headline problem this solves: ``litellm.token_counter`` counts ``image_url``
parts only at the flat low-detail base (~85 tokens) because it treats the
default ``detail="auto"`` as low. The OpenAI API actually bills ``auto`` as
**high** detail (tiled) for non-tiny images, so a frame can really cost
hundreds–thousands of tokens. :func:`estimate_prompt_tokens` counts text with
``token_counter`` (images stripped out, to avoid the low-detail base) and adds
each image's high-detail cost back via litellm's own tiling math.

These estimates feed the tokens-per-minute bucket in :mod:`utils.rate_limiter`;
they only need to be a sane upper-ish bound, not exact accounting.
"""

from __future__ import annotations

from typing import Any

import litellm

# Conservative per-image fallback (tokens) when the image dimensions can't be
# read — e.g. a remote URL we won't fetch, or an unparseable payload.
DEFAULT_IMAGE_TOKENS = 1024


def estimate_image_tokens(image_url: Any) -> int:
    """Estimate the token cost of one ``image_url`` content part.

    Uses litellm's high-detail tiling math (matching how the API bills the
    default ``auto`` detail for non-tiny images). Only local ``data:`` URIs are
    sized — decoded in-process; remote URLs are not fetched in this hot path and
    fall back to :data:`DEFAULT_IMAGE_TOKENS`.

    Args:
        image_url: the ``image_url`` value of a content part — either a string
            URL or a ``{"url": ...}`` dict.
    """
    url = image_url.get("url") if isinstance(image_url, dict) else image_url
    if not isinstance(url, str) or not url.startswith("data:"):
        return DEFAULT_IMAGE_TOKENS
    try:
        from litellm.litellm_core_utils.token_counter import calculate_img_tokens

        return calculate_img_tokens(data=url, mode="high")
    except Exception:
        return DEFAULT_IMAGE_TOKENS


def estimate_prompt_tokens(model: str, messages: list[dict[str, Any]]) -> int:
    """Estimate prompt tokens for *messages*, counting images at high detail.

    Text/structure is counted by ``litellm.token_counter`` with image parts
    stripped out (so they aren't double-counted at the low-detail base), and
    each image's high-detail cost is added back via :func:`estimate_image_tokens`.
    """
    image_tokens = 0
    text_messages: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    image_tokens += estimate_image_tokens(part.get("image_url"))
                else:
                    text_parts.append(part)
            text_messages.append({**msg, "content": text_parts})
        else:
            text_messages.append(msg)

    try:
        text_tokens = litellm.token_counter(model=model, messages=text_messages)
    except Exception:
        text_tokens = 0
    return text_tokens + image_tokens
