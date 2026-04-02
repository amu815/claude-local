# Mac Studio Benchmark Learnings Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Apply 4 optimizations discovered during Mac Studio vs DGX Spark benchmarking to the claude-local repository.

**Architecture:** Three quick config/CLI changes (Tasks 1-3) plus one larger proxy enhancement (Task 4) that adds Anthropic→OpenAI API translation for MLX/Ollama backends.

**Tech Stack:** Python 3.10+, stdlib http.server, click CLI, YAML config

---

### Task 1: Add `--tools` flag to limit Claude Code tool count

The biggest optimization (54% speedup). Claude Code sends 22 tools by default (~70K chars, 79% of prompt). Limiting to essential tools dramatically reduces prefill time on backends without prefix caching.

**Files:**
- Modify: `src/claude_local/cli.py`

**Step 1: Add `--tools` option to the `start` command**

In `cli.py`, add a `--tools` option with a sensible default:

```python
@main.command()
@click.option("--no-claude", is_flag=True, help="Start backend and proxy only")
@click.option("--tools", default="Bash,Read,Write,Edit,Glob,Grep",
              help="Comma-separated list of Claude Code tools to enable (fewer = faster)")
def start(no_claude: bool, tools: str):
```

**Step 2: Pass `--tools` to Claude Code subprocess**

Change the `subprocess.run` call from:

```python
subprocess.run([claude_cmd, "--model", model_name], env=env)
```

to:

```python
claude_args = [claude_cmd, "--model", model_name]
if tools:
    claude_args.extend(["--tools", tools])
subprocess.run(claude_args, env=env)
```

**Step 3: Commit**

```bash
git add src/claude_local/cli.py
git commit -m "feat: add --tools flag to limit Claude Code tool count

Default to 6 essential tools instead of 22. Reduces prompt size
from ~70K chars to ~20K chars, cutting prefill time by ~54% on
backends without prefix caching (MLX, Ollama)."
```

---

### Task 2: Disable Qwen3.5 thinking mode in models.yaml

Qwen3.5's thinking mode wastes tokens on reasoning before responding (780→3 tokens for simple queries). Add a flag to models.yaml so the proxy can disable it.

**Files:**
- Modify: `configs/models.yaml`

**Step 1: Add `thinking: false` to qwen3.5 model entries**

```yaml
models:
  - id: qwen3.5-122b-int4
    name: Qwen3.5-122B-A10B-INT4
    # ... existing fields ...
    thinking: false  # Disable thinking mode for agent use (saves ~97% tokens)
```

Add `thinking: false` to both `qwen3.5-122b-int4` and `qwen3.5-32b`. Leave `qwen3-32b-int4` without it (Qwen3 doesn't have thinking mode).

**Step 2: Commit**

```bash
git add configs/models.yaml
git commit -m "config: disable thinking mode for Qwen3.5 models

Qwen3.5 thinking mode generates ~780 reasoning tokens before
responding, wasting max_tokens budget. With thinking disabled,
the same query uses ~3 tokens. Critical for agent use cases."
```

---

### Task 3: Expand Claude Code settings optimization

Add `CLAUDE_CODE_ENABLE_TELEMETRY` and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC` to the auto-applied settings (14% speedup from reduced traffic).

**Files:**
- Modify: `src/claude_local/cli.py`

**Step 1: Update `_LOCAL_ENV_DEFAULTS`**

```python
_LOCAL_ENV_DEFAULTS = {
    "CLAUDE_CODE_ATTRIBUTION_HEADER": "0",
    "CLAUDE_CODE_ENABLE_TELEMETRY": "0",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}
```

**Step 2: Commit**

```bash
git add src/claude_local/cli.py
git commit -m "feat: disable telemetry and nonessential traffic for local inference

Reduces background requests that add latency when using local
models. Combined with attribution header fix, yields ~14% speedup."
```

---

### Task 4: Add Anthropic→OpenAI API translation to proxy

The current proxy is a passthrough — it forwards requests as-is. This works for vLLM (which supports Anthropic format) but fails for MLX and Ollama (OpenAI format only). Add translation when the upstream is OpenAI-only.

**Files:**
- Create: `src/claude_local/translate.py`
- Modify: `src/claude_local/proxy.py`
- Modify: `configs/models.yaml` (add `native_anthropic` flag)

**Step 1: Add `native_anthropic` flag to backends in models.yaml**

```yaml
    backends:
      vllm-spark:
        native_anthropic: true   # vLLM supports /v1/messages natively
        recipe: qwen3.5-122b-int4-solo
        # ...
      mlx:
        native_anthropic: false  # MLX only supports /v1/chat/completions
        repo: mlx-community/Qwen3.5-122B-A10B-4bit
      ollama:
        native_anthropic: false
        tag: qwen3.5-coder:122b
```

**Step 2: Create `translate.py` with format conversion functions**

Create `src/claude_local/translate.py`:

```python
"""Anthropic Messages API ↔ OpenAI Chat Completions translation."""
from __future__ import annotations

import json
import uuid
from typing import Any


def anthropic_to_openai(data: dict, model_name: str, thinking: bool = True) -> dict:
    """Convert Anthropic Messages API request to OpenAI Chat Completions."""
    messages = []

    # System prompt: Anthropic has top-level 'system', OpenAI uses a system message
    system = data.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            text = "\n".join(
                b.get("text", "") for b in system if b.get("type") == "text"
            )
            if text:
                messages.append({"role": "system", "content": text})

    # Convert messages
    for msg in data.get("messages", []):
        role = msg.get("role")
        content = msg.get("content")

        if role == "assistant" and isinstance(content, list):
            # Assistant message with tool_use blocks
            text_parts = []
            tool_calls = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block["text"])
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(block.get("input", {})),
                        },
                    })
            oai_msg: dict[str, Any] = {"role": "assistant"}
            if text_parts:
                oai_msg["content"] = "\n".join(text_parts)
            else:
                oai_msg["content"] = None
            if tool_calls:
                oai_msg["tool_calls"] = tool_calls
            messages.append(oai_msg)

        elif role == "user" and isinstance(content, list):
            # User message with tool_result blocks
            text_parts = []
            for block in content:
                if block.get("type") == "tool_result":
                    tool_content = block.get("content", "")
                    if isinstance(tool_content, list):
                        tool_content = "\n".join(
                            b.get("text", "") for b in tool_content
                            if b.get("type") == "text"
                        )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": block["tool_use_id"],
                        "content": str(tool_content),
                    })
                elif block.get("type") == "text":
                    text_parts.append(block["text"])
            if text_parts:
                messages.append({"role": "user", "content": "\n".join(text_parts)})
        else:
            # Simple text message
            messages.append({"role": role, "content": content if content else ""})

    # Build OpenAI request
    oai: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "max_tokens": data.get("max_tokens", 4096),
        "stream": data.get("stream", False),
    }

    # Tools
    anthropic_tools = data.get("tools")
    if anthropic_tools:
        oai["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            }
            for t in anthropic_tools
        ]

    # Tool choice
    tc = data.get("tool_choice")
    if tc:
        tc_type = tc.get("type") if isinstance(tc, dict) else tc
        if tc_type == "any":
            oai["tool_choice"] = "required"
        elif tc_type == "auto":
            oai["tool_choice"] = "auto"
        elif tc_type == "tool" and isinstance(tc, dict):
            oai["tool_choice"] = {
                "type": "function",
                "function": {"name": tc["name"]},
            }

    # Temperature
    if "temperature" in data:
        oai["temperature"] = data["temperature"]

    # Disable thinking for Qwen3.5
    if not thinking:
        oai["chat_template_kwargs"] = {"enable_thinking": False}

    return oai


def openai_to_anthropic(data: dict, model_name: str) -> dict:
    """Convert OpenAI Chat Completions response to Anthropic Messages API."""
    choice = data.get("choices", [{}])[0]
    message = choice.get("message", {})

    content_blocks: list[dict] = []

    # Text content
    text = message.get("content")
    if text:
        content_blocks.append({"type": "text", "text": text})

    # Tool calls
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        func = tc.get("function", {})
        try:
            input_data = json.loads(func.get("arguments", "{}"))
        except json.JSONDecodeError:
            input_data = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}"),
            "name": func.get("name", ""),
            "input": input_data,
        })

    # Stop reason
    finish = choice.get("finish_reason", "end_turn")
    if finish == "tool_calls":
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"

    # Usage
    usage_in = data.get("usage", {})

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content_blocks if content_blocks else [{"type": "text", "text": ""}],
        "model": model_name,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": usage_in.get("prompt_tokens", 0),
            "output_tokens": usage_in.get("completion_tokens", 0),
        },
    }


def openai_stream_to_anthropic_events(model_name: str) -> "StreamTranslator":
    """Return a stateful translator for SSE stream chunks."""
    return StreamTranslator(model_name)


class StreamTranslator:
    """Translates OpenAI SSE streaming chunks to Anthropic SSE events."""

    def __init__(self, model_name: str) -> None:
        self._model = model_name
        self._started = False
        self._block_index = 0
        self._active_tool_ids: dict[int, str] = {}
        self._active_tool_names: dict[int, str] = {}

    def header_events(self) -> list[str]:
        """Return the initial message_start event."""
        msg_start = {
            "type": "message_start",
            "message": {
                "id": f"msg_{uuid.uuid4().hex[:24]}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": self._model,
                "stop_reason": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        return [f"event: message_start\ndata: {json.dumps(msg_start)}\n\n"]

    def translate_chunk(self, line: str) -> list[str]:
        """Translate one SSE data line. Returns list of Anthropic SSE events."""
        if not line.startswith("data: "):
            return []
        payload = line[6:].strip()
        if payload == "[DONE]":
            return self._finish_events()

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            return []

        events: list[str] = []
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        finish = choice.get("finish_reason")

        # Text content
        text = delta.get("content")
        if text:
            if not self._started:
                start = {"type": "content_block_start", "index": self._block_index,
                         "content_block": {"type": "text", "text": ""}}
                events.append(f"event: content_block_start\ndata: {json.dumps(start)}\n\n")
                self._started = True
            d = {"type": "content_block_delta", "index": self._block_index,
                 "delta": {"type": "text_delta", "text": text}}
            events.append(f"event: content_block_delta\ndata: {json.dumps(d)}\n\n")

        # Tool calls
        for tc in delta.get("tool_calls", []):
            tc_index = tc.get("index", 0)
            func = tc.get("function", {})

            if func.get("name"):
                # New tool call start
                if self._started:
                    stop = {"type": "content_block_stop", "index": self._block_index}
                    events.append(f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n")
                    self._block_index += 1

                tool_id = tc.get("id", f"toolu_{uuid.uuid4().hex[:24]}")
                self._active_tool_ids[tc_index] = tool_id
                self._active_tool_names[tc_index] = func["name"]

                start = {"type": "content_block_start", "index": self._block_index,
                         "content_block": {"type": "tool_use", "id": tool_id, "name": func["name"], "input": {}}}
                events.append(f"event: content_block_start\ndata: {json.dumps(start)}\n\n")
                self._started = True

            if func.get("arguments"):
                d = {"type": "content_block_delta", "index": self._block_index,
                     "delta": {"type": "input_json_delta", "partial_json": func["arguments"]}}
                events.append(f"event: content_block_delta\ndata: {json.dumps(d)}\n\n")

        # Finish
        if finish:
            events.extend(self._finish_events(finish))

        return events

    def _finish_events(self, finish_reason: str = "end_turn") -> list[str]:
        events = []
        if self._started:
            stop = {"type": "content_block_stop", "index": self._block_index}
            events.append(f"event: content_block_stop\ndata: {json.dumps(stop)}\n\n")

        stop_reason = "tool_use" if finish_reason == "tool_calls" else (
            "max_tokens" if finish_reason == "length" else "end_turn"
        )
        md = {"type": "message_delta", "delta": {"stop_reason": stop_reason},
              "usage": {"output_tokens": 0}}
        events.append(f"event: message_delta\ndata: {json.dumps(md)}\n\n")
        events.append(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n")
        return events
```

**Step 3: Update proxy.py to use translation when needed**

Add translation support to `ProxyServer` via a `translate` flag:

In `proxy.py`, update `ProxyServer.__init__` to accept `translate` and `model_name` and `thinking` parameters:

```python
class ProxyServer:
    def __init__(
        self,
        upstreams: list[str],
        host: str = "127.0.0.1",
        port: int = 8081,
        translate: bool = False,
        model_name: str = "",
        thinking: bool = True,
    ) -> None:
        handler_class = type(
            "_BoundProxyHandler",
            (_ProxyHandler,),
            {
                "upstreams": list(upstreams),
                "translate": translate,
                "model_name": model_name,
                "thinking": thinking,
            },
        )
        ...
```

Update `_ProxyHandler._proxy` to translate when `self.translate` is True and the path contains `/messages`:

```python
def _proxy(self, method: str) -> None:
    # ... existing body reading code ...

    if method == "POST" and body and self.translate and "/messages" in self.path:
        # Translate Anthropic → OpenAI
        from .translate import anthropic_to_openai, openai_to_anthropic, openai_stream_to_anthropic_events
        data = json.loads(body)
        is_stream = data.get("stream", False)
        oai_data = anthropic_to_openai(data, self.model_name, thinking=self.thinking)
        body = json.dumps(oai_data).encode()
        # Rewrite path to OpenAI endpoint
        original_path = self.path
        self.path = "/v1/chat/completions"
        # ... forward to upstream ...
        # ... translate response back ...
```

**Step 4: Update cli.py to enable translation for non-vLLM backends**

```python
# In start():
backend_config = model.get("backends", {}).get(backend_name, {})
needs_translation = not backend_config.get("native_anthropic", False)
thinking = model.get("thinking", True)

proxy = ProxyServer(
    upstreams=upstreams,
    host=proxy_host,
    port=proxy_port,
    translate=needs_translation,
    model_name=model_name,
    thinking=thinking,
)
```

**Step 5: Commit**

```bash
git add src/claude_local/translate.py src/claude_local/proxy.py src/claude_local/cli.py configs/models.yaml
git commit -m "feat: add Anthropic→OpenAI API translation for MLX/Ollama backends

vLLM supports Anthropic Messages API natively, but MLX and Ollama
only support OpenAI Chat Completions. The proxy now auto-detects
which format the backend needs and translates accordingly.

Includes full tool_use/tool_result translation and SSE streaming
conversion. Qwen3.5 thinking mode is auto-disabled based on
models.yaml config."
```

---

### Task 5: Update README with performance tips and MLX caveats

**Files:**
- Modify: `README.md`

**Step 1: Add Performance Tips section before Prerequisites**

```markdown
## Performance Tips / パフォーマンス最適化

### Tool count (`--tools`)

Claude Code sends 22 tools by default (~70K chars). Limiting to essential tools
dramatically reduces prefill time, especially on MLX (no prefix caching):

    claude-local start --tools "Bash,Read,Write,Edit,Glob,Grep"

### MLX prefix cache limitation

MLX (v0.31) only caches exact-match prompts. Unlike vLLM's block-level prefix
sharing, MLX cannot reuse KV cache for prompts that share a common prefix but
differ at the end. This means every request with a different user message
triggers a full prefill (~20K tokens).

Workaround: use `--tools` to reduce prompt size.
```

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add performance tips and MLX cache limitations"
```

---
