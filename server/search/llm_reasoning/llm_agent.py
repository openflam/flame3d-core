"""Tool-calling LLM agent for dataset-grounded reasoning."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
from typing import Any, Callable, Sequence

from .llm_call import DEFAULT_MODEL, LLMCaller
from .tools import get_tools, get_tool_functions, get_thinking_texts

from server.search.prompts.tools_prompt import get_tools_prompt


def call_tool(tool_name: str, arguments: str | dict[str, Any], dataset_name: str | None = None, tool_functions: dict[str, Callable] | None = None) -> dict[str, Any]:
    """Execute a registered tool by name with the given JSON-encoded arguments."""
    _tool_functions = tool_functions if tool_functions is not None else get_tool_functions()
    if tool_name not in _tool_functions:
        return {"error": f"Unknown tool: {tool_name}"}

    tool_fn = _tool_functions[tool_name]

    if isinstance(arguments, str):
        try:
            tool_args = json.loads(arguments or "{}")
        except json.JSONDecodeError as exc:
            return {"error": f"Invalid tool arguments: {exc}"}
    else:
        tool_args = arguments

    if not isinstance(tool_args, dict):
        return {"error": "Tool arguments must decode to a JSON object"}

    if dataset_name is not None and "dataset_name" in inspect.signature(tool_fn).parameters:
        tool_args = tool_args.copy()
        tool_args["dataset_name"] = dataset_name

    try:
        return tool_fn(**tool_args)
    except Exception as exc:
        return {"error": str(exc)}


def _build_tool_output(tool_output: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Build the ``content`` value for a ``tool`` message and extract images.

    Returns:
        tuple containing:
        - text representation of the tool output
        - list of image dictionary parts to be appended as a user message
    """
    images: list[str] = tool_output.get("images") or []
    non_image_fields = {k: v for k, v in tool_output.items() if k != "images"}
    
    # Text content for the tool message itself
    text_content = json.dumps(non_image_fields) if non_image_fields else "Images successfully retrieved."

    image_parts: list[dict[str, Any]] = []
    for data_url in images:
        image_parts.append({
            "type": "image_url",
            "image_url": {"url": data_url},
        })

    return text_content, image_parts


class LLMAgent:
    """Runs a tool-calling loop with the configured LLM."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        max_tool_rounds: int = 8,
        api_key: str | None = None,
        allowed_tools: list[str] | None = None,
    ) -> None:
        self.model = model
        self.allowed_tools = allowed_tools
        self.system_prompt = system_prompt if system_prompt is not None else get_tools_prompt(allowed_tools)
        self.max_tool_rounds = max_tool_rounds
        self.caller = LLMCaller(model=model, api_key=api_key)
        self.tools = get_tools(allowed_tools)
        self.tool_functions = get_tool_functions(allowed_tools)
        self.thinking_texts = get_thinking_texts(allowed_tools)

    def answer_query(self, query: str, dataset_name: str) -> dict[str, Any]:
        return self.answer_query_stream(query=query, dataset_name=dataset_name)

    def answer_query_stream(
        self,
        query: str,
        dataset_name: str,
        on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """
        Answer a natural-language query for a given dataset using tool calls.

        Args:
            query: User query in natural language
            dataset_name: Dataset to run tools against (never exposed to model)
            on_stream_event: Callback receiving event dicts while model streams

        Returns:
            Dict containing final response text plus tool trace data.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if not dataset_name or not dataset_name.strip():
            raise ValueError("dataset_name must be a non-empty string")

        wrapped_on_stream_event = None
        if on_stream_event:
            def _wrapped_handler(event: dict[str, Any], redacted_thinking: bool = True) -> None:
                event_type = event.get("type")
                if event_type == "assistant_reasoning_delta":
                    on_stream_event({"type": "thinking", "content": event.get("delta", "")})
                elif event_type == "tool_call_delta":
                    name_delta = event.get("name_delta")

                    if not redacted_thinking:
                        if name_delta:
                            on_stream_event({"type": "thinking", "content": f"\n\n> Using tool: {name_delta}\n> Arguments: "})
                        args_delta = event.get("arguments_delta")
                        if args_delta:
                            on_stream_event({"type": "thinking", "content": args_delta})
                    else:
                        if name_delta and name_delta in self.thinking_texts:
                            on_stream_event({"type": "thinking", "content": f"\n\n> {self.thinking_texts[name_delta]}"})

            wrapped_on_stream_event = _wrapped_handler

        input_items: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"User query: {query}\n\n"
                    "Use tools as needed, then respond with a JSON object containing "
                    "component_ids (list of ints) and reason (string)."
                ),
            },
        ]
        tool_trace: list[dict[str, Any]] = []

        for _ in range(self.max_tool_rounds):
            message_payload = self.caller.stream_chat(
                input=input_items,
                tools=self.tools,
                on_stream_event=wrapped_on_stream_event,
            )

            assistant_content = (message_payload.get("content") or "").strip()
            tool_calls = message_payload.get("tool_calls", [])
            output_items = message_payload.get("output_items", [])

            if not tool_calls:
                component_ids, reason, custom_bboxes = _parse_final_response(assistant_content)
                return {
                    "dataset_name": dataset_name,
                    "query": query,
                    "model": self.model,
                    "response": assistant_content,
                    "component_ids": component_ids,
                    "custom_bboxes": custom_bboxes,
                    "reason": reason,
                    "tool_trace": tool_trace,
                }

            # Append the model's output items back to input for multi-turn
            if tool_calls:
                assistant_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": assistant_content if assistant_content else None,
                    "tool_calls": []
                }
                for tc in tool_calls:
                    assistant_message["tool_calls"].append({
                        "id": tc.get("id"),
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": tc.get("arguments")
                        }
                    })
                input_items.append(assistant_message)
            elif assistant_content:
                input_items.append({"role": "assistant", "content": assistant_content})

            tool_messages = []
            image_messages = []

            for call in tool_calls:
                tool_name = call.get("name", "")
                call_id = call.get("id", "")
                tool_output = call_tool(
                    tool_name=tool_name,
                    arguments=call.get("arguments", "{}"),
                    dataset_name=dataset_name,
                    tool_functions=self.tool_functions,
                )

                if on_stream_event:
                    on_stream_event(
                        {
                            "type": "tool_call_result",
                            "tool_name": tool_name,
                            "tool_call_id": call_id,
                            "arguments": call.get("arguments", "{}"),
                            "output": tool_output,
                        }
                    )

                tool_trace.append(
                    {
                        "tool_name": tool_name,
                        "arguments": call.get("arguments", "{}"),
                        "output": tool_output,
                    }
                )

                # Build the tool output message
                output_content, image_parts = _build_tool_output(tool_output)

                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": output_content,
                    }
                )

                if image_parts:
                    image_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Images from tool {tool_name}:"}
                            ] + image_parts
                        }
                    )

            input_items.extend(tool_messages)
            input_items.extend(image_messages)

        return {
            "dataset_name": dataset_name,
            "query": query,
            "model": self.model,
            "response": "Tool-calling loop stopped before a final answer was produced.",
            "component_ids": [],
            "custom_bboxes": [],
            "reason": "Tool-calling loop reached the maximum number of rounds without a final answer.",
            "tool_trace": tool_trace,
        }


def _parse_final_response(content: str) -> tuple[list[int], str | list[str], list[dict]]:
    """
    Extract component_ids, reason, and custom_bboxes from the model's final JSON response.

    Tries json.loads on the full content first; falls back to finding the
    outermost {...} block if the model included extra prose.

    reason may be a string (search_stream) or a list of strings (robot_steps).
    """
    text = content.strip()
    candidates: list[str] = [text]

    # Also try the last {...} block in case there is leading prose
    match = re.search(r"(\{[^{}]*\})", text, re.DOTALL)
    if match:
        candidates.append(match.group(1))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            raw_ids = parsed.get("component_ids") or []
            custom_bboxes = parsed.get("custom_bboxes") or []
            component_ids: list[int] = []
            for x in raw_ids:
                try:
                    component_ids.append(int(x))
                except (TypeError, ValueError):
                    pass
            raw_reason = parsed.get("reason")
            if isinstance(raw_reason, list):
                reason: str | list[str] = raw_reason
            else:
                reason = str(raw_reason or text)
            return component_ids, reason, custom_bboxes
        except (json.JSONDecodeError, AttributeError):
            continue

    return [], text, []


def answer_query(
    query: str,
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    on_stream_event: Callable[[dict[str, Any]], None] | None = None,
    allowed_tools: list[str] | None = None,
) -> dict[str, Any]:
    """Convenience function for single-shot use."""
    agent = LLMAgent(model=model, allowed_tools=allowed_tools)
    return agent.answer_query_stream(
        query=query,
        dataset_name=dataset_name,
        on_stream_event=on_stream_event,
    )


def _cli_stream_handler(
    show_reasoning: bool, show_tools: bool
) -> Callable[[dict[str, Any]], None]:
    def _handler(event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "assistant_text_delta":
            print(event.get("delta") or "", end="", flush=True)
        elif event_type == "assistant_reasoning_delta" and show_reasoning:
            print(event.get("delta") or "", end="", flush=True)
        elif event_type == "tool_call_result" and show_tools:
            try:
                args_str = json.dumps(
                    json.loads(event.get("arguments") or "{}"), ensure_ascii=False
                )
            except (json.JSONDecodeError, TypeError):
                args_str = event.get("arguments") or "{}"
            print(
                f"\n[Tool called: {event.get('tool_name')}\nArguments: {args_str}]",
                flush=True,
            )

    return _handler


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Test the LLM agent from the command line."
    )
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--query", required=True, help="Natural-language query")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model name")
    parser.add_argument(
        "--show-reasoning", action="store_true", help="Print reasoning tokens to stderr"
    )
    parser.add_argument(
        "--show-tools", action="store_true", help="Print tool results to stderr"
    )
    parser.add_argument(
        "--json", action="store_true", help="Print full result JSON after response"
    )
    args = parser.parse_args(argv)

    handler = _cli_stream_handler(
        show_reasoning=args.show_reasoning, show_tools=args.show_tools
    )
    result = answer_query(
        query=args.query,
        dataset_name=args.dataset,
        model=args.model,
        on_stream_event=handler,
    )
    print()  # newline after streamed text
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
