"""Thin wrapper for LLM API calls using LiteLLM."""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from dotenv import load_dotenv
import litellm

load_dotenv()


DEFAULT_MODEL = "gpt-5.4"


class OutputItem:
    """Mock output item to maintain compatibility with legacy code expecting response items."""
    def __init__(self, type_: str, name: str, arguments: str, call_id: str):
        self.type = type_
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class LLMCaller:
    """Wrapper around LiteLLM streaming API."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        max_completion_tokens: int = 2000,
    ) -> None:
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.api_key = api_key

    def stream_chat(
        self,
        input: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Stream a litellm API request and emit incremental events.

        Returns the fully assembled response payload:
            {"content": str, "tool_calls": list[dict], "output_items": list}
        """
        request: dict[str, Any] = {
            "model": self.model,
            "messages": input,
            "max_tokens": self.max_completion_tokens,
            "stream": True,
        }
        if tools:
            formatted_tools = []
            for t in tools:
                if "function" not in t:
                    formatted_tools.append({
                        "type": "function",
                        "function": {
                            "name": t.get("name", ""),
                            "description": t.get("description", ""),
                            "parameters": t.get("parameters", {})
                        }
                    })
                else:
                    formatted_tools.append(t)
            request["tools"] = formatted_tools
        if self.api_key:
            request["api_key"] = self.api_key

        content_parts: list[str] = []
        # Track function calls being streamed: index -> dict of tool call data
        tool_calls_dict: dict[int, dict[str, Any]] = {}

        stream = litellm.completion(**request)
        for chunk in stream:
            if not getattr(chunk, "choices", None) or not chunk.choices:
                continue
            
            delta = chunk.choices[0].delta

            # Text content delta
            content_delta = getattr(delta, "content", None)
            if content_delta:
                content_parts.append(content_delta)
                if on_stream_event:
                    on_stream_event(
                        {"type": "assistant_text_delta", "delta": content_delta}
                    )
            
            # Reasoning / thinking delta
            reasoning_delta = getattr(delta, "reasoning_content", None)
            if reasoning_delta and on_stream_event:
                on_stream_event(
                    {"type": "assistant_reasoning_delta", "delta": reasoning_delta}
                )

            # Function call arguments streaming
            tool_calls_delta = getattr(delta, "tool_calls", None)
            if tool_calls_delta:
                for tc in tool_calls_delta:
                    idx = getattr(tc, "index", 0)
                    if idx not in tool_calls_dict:
                        # New tool call
                        name = tc.function.name if (getattr(tc, "function", None) and tc.function.name) else ""
                        args = tc.function.arguments if (getattr(tc, "function", None) and tc.function.arguments) else ""
                        tool_calls_dict[idx] = {
                            "id": getattr(tc, "id", "") or "",
                            "type": "function_call",
                            "name": name,
                            "arguments": args
                        }
                        if name and on_stream_event:
                            on_stream_event(
                                {
                                    "type": "tool_call_delta",
                                    "index": idx,
                                    "tool_call_id": tool_calls_dict[idx]["id"],
                                    "name_delta": name,
                                    "arguments_delta": "",
                                }
                            )
                        if args and on_stream_event:
                            on_stream_event(
                                {
                                    "type": "tool_call_delta",
                                    "index": idx,
                                    "tool_call_id": tool_calls_dict[idx]["id"],
                                    "name_delta": "",
                                    "arguments_delta": args,
                                }
                            )
                    else:
                        # Tool call arguments streaming
                        args_delta = tc.function.arguments if (getattr(tc, "function", None) and tc.function.arguments) else ""
                        if args_delta:
                            tool_calls_dict[idx]["arguments"] += args_delta
                            if on_stream_event:
                                on_stream_event(
                                    {
                                        "type": "tool_call_delta",
                                        "index": idx,
                                        "tool_call_id": tool_calls_dict[idx]["id"],
                                        "name_delta": "",
                                        "arguments_delta": args_delta,
                                    }
                                )

        output_items: list[Any] = []
        tool_calls: list[dict[str, Any]] = []

        for idx in sorted(tool_calls_dict.keys()):
            tc_data = tool_calls_dict[idx]
            tool_calls.append(tc_data)
            output_items.append(
                OutputItem(
                    type_="function_call",
                    name=tc_data["name"],
                    arguments=tc_data["arguments"],
                    call_id=tc_data["id"],
                )
            )

        if on_stream_event:
            on_stream_event(
                {
                    "type": "assistant_message_done",
                    "finish_reason": "stop",
                    "has_tool_calls": len(tool_calls) > 0,
                }
            )

        return {
            "content": "".join(content_parts),
            "tool_calls": tool_calls,
            "output_items": output_items,
        }
