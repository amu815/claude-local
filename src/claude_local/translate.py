"""Translate between Anthropic Messages API and OpenAI Chat Completions API."""
from __future__ import annotations

import json
import uuid


def _gen_id(prefix: str = "msg") -> str:
    """Generate an Anthropic-style ID like ``msg_01XFDUDYJgAACzvnptvVoYEL``."""
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def _gen_tool_use_id() -> str:
    """Generate a tool-use block ID like ``toolu_01A09q90qw90lq917835lq9``."""
    return f"toolu_{uuid.uuid4().hex[:24]}"


# ---------------------------------------------------------------------------
# Anthropic request -> OpenAI request
# ---------------------------------------------------------------------------


def anthropic_to_openai(
    data: dict,
    model_name: str,
    thinking: bool = True,
) -> dict:
    """Convert an Anthropic Messages API request body to an OpenAI Chat
    Completions API request body.

    Parameters
    ----------
    data : dict
        The Anthropic Messages request JSON.
    model_name : str
        The model name to use in the OpenAI request.
    thinking : bool
        Whether the model has thinking/reasoning enabled.  When *False*,
        ``chat_template_kwargs`` is set to disable thinking on the server side
        (useful for Qwen3 served via vLLM/Ollama with a chat template that
        supports an ``enable_thinking`` flag).
    """
    messages: list[dict] = []

    # --- system prompt ---
    system = data.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            # Anthropic format: list of content blocks
            text_parts = []
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif isinstance(block, str):
                    text_parts.append(block)
            if text_parts:
                messages.append({"role": "system", "content": "\n".join(text_parts)})

    # --- conversation messages ---
    for msg in data.get("messages", []):
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content) if content else ""})
            continue

        # Content is a list of blocks.  We need to split assistant
        # messages that contain tool_use blocks, and convert tool_result
        # blocks into separate ``role: tool`` messages.

        if role == "user":
            _convert_user_blocks(content, messages)
        elif role == "assistant":
            _convert_assistant_blocks(content, messages)
        else:
            # Fallback: concatenate text blocks
            text = " ".join(
                b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
            )
            messages.append({"role": role, "content": text or ""})

    # --- tools ---
    oai: dict = {
        "model": model_name,
        "messages": messages,
    }

    if "max_tokens" in data:
        oai["max_tokens"] = data["max_tokens"]

    if data.get("stream"):
        oai["stream"] = True

    if data.get("temperature") is not None:
        oai["temperature"] = data["temperature"]

    if data.get("top_p") is not None:
        oai["top_p"] = data["top_p"]

    tools = data.get("tools")
    if tools:
        oai["tools"] = [_convert_tool_def(t) for t in tools]

    # tool_choice
    tc = data.get("tool_choice")
    if tc:
        oai["tool_choice"] = _convert_tool_choice(tc)

    # Thinking mode control
    if not thinking:
        oai["chat_template_kwargs"] = {"enable_thinking": False}

    return oai


def _convert_user_blocks(blocks: list, messages: list[dict]) -> None:
    """Convert Anthropic user content blocks to OpenAI messages.

    User messages can contain ``tool_result`` blocks (results of tool calls)
    which become separate ``role: tool`` messages in OpenAI format, plus
    regular text blocks.
    """
    text_parts: list[str] = []

    for block in blocks:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        btype = block.get("type", "")

        if btype == "tool_result":
            # Flush accumulated text first
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})
                text_parts = []

            tool_content = block.get("content", "")
            if isinstance(tool_content, list):
                # Extract text from content blocks
                parts = []
                for cb in tool_content:
                    if isinstance(cb, dict) and cb.get("type") == "text":
                        parts.append(cb.get("text", ""))
                tool_content = "\n".join(parts)
            elif not isinstance(tool_content, str):
                tool_content = json.dumps(tool_content) if tool_content else ""

            messages.append({
                "role": "tool",
                "tool_call_id": block.get("tool_use_id", ""),
                "content": tool_content,
            })

        elif btype == "text":
            text_parts.append(block.get("text", ""))

        elif btype == "image":
            # Skip images for now — local LLMs rarely handle them
            pass

        else:
            # Unknown block type — include as text if possible
            if "text" in block:
                text_parts.append(block["text"])

    if text_parts:
        messages.append({"role": "user", "content": "\n".join(text_parts)})


def _convert_assistant_blocks(blocks: list, messages: list[dict]) -> None:
    """Convert Anthropic assistant content blocks to a single OpenAI
    assistant message with optional ``tool_calls``.
    """
    text_parts: list[str] = []
    tool_calls: list[dict] = []

    for block in blocks:
        if not isinstance(block, dict):
            text_parts.append(str(block))
            continue

        btype = block.get("type", "")

        if btype == "text":
            text_parts.append(block.get("text", ""))

        elif btype == "thinking":
            # Anthropic extended thinking — include as reasoning text
            thinking_text = block.get("thinking", "")
            if thinking_text:
                text_parts.append(f"<think>\n{thinking_text}\n</think>")

        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", _gen_tool_use_id()),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {})),
                },
            })

    msg: dict = {"role": "assistant"}

    content_text = "\n".join(text_parts) if text_parts else None
    if tool_calls:
        # OpenAI requires content to be null or string when tool_calls present
        msg["content"] = content_text
        msg["tool_calls"] = tool_calls
    else:
        msg["content"] = content_text or ""

    messages.append(msg)


def _convert_tool_def(tool: dict) -> dict:
    """Convert an Anthropic tool definition to OpenAI format."""
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {}),
        },
    }


def _convert_tool_choice(tc: dict | str) -> str | dict:
    """Convert Anthropic ``tool_choice`` to OpenAI format."""
    if isinstance(tc, str):
        return tc

    tc_type = tc.get("type", "auto")
    if tc_type == "any":
        return "required"
    elif tc_type == "auto":
        return "auto"
    elif tc_type == "none":
        return "none"
    elif tc_type == "tool":
        # Specific tool requested
        return {"type": "function", "function": {"name": tc.get("name", "")}}
    return "auto"


# ---------------------------------------------------------------------------
# OpenAI response -> Anthropic response
# ---------------------------------------------------------------------------


def openai_to_anthropic(data: dict, model_name: str) -> dict:
    """Convert an OpenAI Chat Completions response to an Anthropic Messages
    API response.
    """
    choice = {}
    if data.get("choices"):
        choice = data["choices"][0]

    message = choice.get("message", {})
    finish = choice.get("finish_reason", "stop")

    content: list[dict] = []

    # Text content
    text = message.get("content")
    if text:
        content.append({"type": "text", "text": text})

    # Tool calls
    for tc in message.get("tool_calls", []):
        func = tc.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str)
        except (json.JSONDecodeError, TypeError):
            args = {"raw": args_str}

        content.append({
            "type": "tool_use",
            "id": tc.get("id", _gen_tool_use_id()),
            "name": func.get("name", ""),
            "input": args,
        })

    # Map stop reason
    if finish == "tool_calls":
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    # Usage
    usage_in = data.get("usage", {})
    usage = {
        "input_tokens": usage_in.get("prompt_tokens", 0),
        "output_tokens": usage_in.get("completion_tokens", 0),
    }

    return {
        "id": _gen_id("msg"),
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model_name,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Streaming: OpenAI SSE -> Anthropic SSE
# ---------------------------------------------------------------------------


class StreamTranslator:
    """Translates OpenAI streaming SSE chunks to Anthropic streaming SSE
    events.

    Usage::

        translator = StreamTranslator(model_name)
        for line in translator.header_events():
            wfile.write(line.encode())
        for raw_line in upstream_response:
            for line in translator.translate_chunk(raw_line):
                wfile.write(line.encode())
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.message_id = _gen_id("msg")
        self.input_tokens = 0
        self.output_tokens = 0
        self._content_index = 0
        self._current_tool_id: str | None = None
        self._current_tool_name: str | None = None
        self._tool_args_buffer: str = ""
        self._text_started = False
        self._saw_tool_calls = False
        # Track active tool call indices from OpenAI
        self._active_tool_calls: dict[int, dict] = {}

    def header_events(self) -> list[str]:
        """Return the initial ``message_start`` and ``content_block_start``
        events that Anthropic clients expect at the beginning of a stream.
        """
        msg_start = {
            "type": "message_start",
            "message": {
                "id": self.message_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self.model_name,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
            },
        }
        return [
            f"event: message_start\ndata: {json.dumps(msg_start)}\n\n",
        ]

    def translate_chunk(self, raw_line: str) -> list[str]:
        """Translate one SSE line from the OpenAI stream into zero or more
        Anthropic SSE event strings.

        Parameters
        ----------
        raw_line : str
            A single line from the upstream SSE stream (may or may not start
            with ``data: ``).

        Returns
        -------
        list[str]
            Zero or more SSE event strings ready to write to the client.
        """
        line = raw_line.strip()
        if not line:
            return []
        if not line.startswith("data: "):
            return []

        payload = line[6:]

        if payload == "[DONE]":
            return self._finish_events()

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            return []

        events: list[str] = []
        choices = chunk.get("choices", [])
        if not choices:
            # Usage-only chunk (some backends send this)
            usage = chunk.get("usage")
            if usage:
                self.input_tokens = usage.get("prompt_tokens", self.input_tokens)
                self.output_tokens = usage.get("completion_tokens", self.output_tokens)
            return []

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")

        # --- Text content ---
        text_content = delta.get("content")
        if text_content is not None:
            if not self._text_started:
                # Emit content_block_start for text
                events.append(self._event(
                    "content_block_start",
                    {"type": "content_block_start", "index": self._content_index, "content_block": {"type": "text", "text": ""}},
                ))
                self._text_started = True

            events.append(self._event(
                "content_block_delta",
                {"type": "content_block_delta", "index": self._content_index, "delta": {"type": "text_delta", "text": text_content}},
            ))

        # --- Tool calls ---
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                tc_index = tc.get("index", 0)
                func = tc.get("function", {})
                tc_id = tc.get("id")
                tc_name = func.get("name")
                tc_args = func.get("arguments", "")

                if tc_id is not None:
                    # New tool call starting
                    # Close text block if one was open
                    if self._text_started:
                        events.append(self._event(
                            "content_block_stop",
                            {"type": "content_block_stop", "index": self._content_index},
                        ))
                        self._content_index += 1
                        self._text_started = False

                    # Close previous tool call if any
                    if tc_index in self._active_tool_calls:
                        events.extend(self._close_tool_call(tc_index))

                    self._saw_tool_calls = True
                    tool_use_id = tc_id or _gen_tool_use_id()
                    self._active_tool_calls[tc_index] = {
                        "id": tool_use_id,
                        "name": tc_name or "",
                        "content_index": self._content_index,
                        "args_buffer": "",
                    }

                    events.append(self._event(
                        "content_block_start",
                        {
                            "type": "content_block_start",
                            "index": self._content_index,
                            "content_block": {
                                "type": "tool_use",
                                "id": tool_use_id,
                                "name": tc_name or "",
                                "input": {},
                            },
                        },
                    ))
                    self._content_index += 1

                if tc_args and tc_index in self._active_tool_calls:
                    self._active_tool_calls[tc_index]["args_buffer"] += tc_args
                    idx = self._active_tool_calls[tc_index]["content_index"]
                    events.append(self._event(
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": idx,
                            "delta": {
                                "type": "input_json_delta",
                                "partial_json": tc_args,
                            },
                        },
                    ))

        # --- Finish ---
        if finish_reason is not None:
            # Close any open blocks
            if self._text_started:
                events.append(self._event(
                    "content_block_stop",
                    {"type": "content_block_stop", "index": self._content_index},
                ))
                self._text_started = False

            for tc_idx in list(self._active_tool_calls.keys()):
                events.extend(self._close_tool_call(tc_idx))

        return events

    def _close_tool_call(self, tc_index: int) -> list[str]:
        """Close a tool call block and return the stop event."""
        info = self._active_tool_calls.pop(tc_index, None)
        if not info:
            return []
        return [
            self._event(
                "content_block_stop",
                {"type": "content_block_stop", "index": info["content_index"]},
            ),
        ]

    def _finish_events(self) -> list[str]:
        """Generate the closing ``message_delta`` and ``message_stop`` events."""
        events: list[str] = []

        # Close any remaining open blocks
        if self._text_started:
            events.append(self._event(
                "content_block_stop",
                {"type": "content_block_stop", "index": self._content_index},
            ))
            self._text_started = False

        for tc_idx in list(self._active_tool_calls.keys()):
            events.extend(self._close_tool_call(tc_idx))

        # Determine stop reason based on whether we saw tool calls
        stop_reason = "tool_use" if self._saw_tool_calls else "end_turn"

        events.append(self._event(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": {"output_tokens": self.output_tokens},
            },
        ))
        events.append("event: message_stop\ndata: {\"type\": \"message_stop\"}\n\n")

        return events

    def _event(self, event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
