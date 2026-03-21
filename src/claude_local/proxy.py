"""OpenAI-compatible failover proxy for claude-local."""
from __future__ import annotations

import io
import json
import sys
import threading
import http.client
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

from claude_local.translate import anthropic_to_openai, openai_to_anthropic, StreamTranslator

MAX_OUTPUT_TOKENS = 16384
SAFETY_CHAR_LIMIT = 900_000  # ~225K tokens


def _estimate_chars(data: dict) -> int:
    """Count chars in system + tools + messages fields."""
    total = 0
    for key in ("system", "tools", "messages"):
        value = data.get(key)
        if value is None:
            continue
        total += len(json.dumps(value, ensure_ascii=False))
    return total


def _safety_compress(data: dict) -> dict:
    """Drop oldest non-system messages if over SAFETY_CHAR_LIMIT."""
    messages = data.get("messages")
    if not messages:
        return data

    while _estimate_chars(data) > SAFETY_CHAR_LIMIT and len(messages) > 1:
        # Keep first message if it's a system prompt, otherwise drop oldest
        if messages[0].get("role") == "system":
            if len(messages) > 2:
                messages.pop(1)
            else:
                break
        else:
            messages.pop(0)
    return data


def _try_upstream(
    upstream: str,
    method: str,
    path: str,
    body: bytes | None,
    headers: dict[str, str],
) -> tuple[bool, http.client.HTTPResponse | None, http.client.HTTPConnection | None]:
    """Attempt a request to one upstream. Returns (success, response, connection)."""
    parsed = urlparse(upstream)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    try:
        conn = http.client.HTTPConnection(host, port, timeout=30)
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        if resp.status == 503:
            conn.close()
            return False, None, None
        return True, resp, conn
    except (ConnectionError, OSError, http.client.HTTPException):
        return False, None, None


class _ProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that forwards requests with failover."""

    # Set by ProxyServer.__init__ via a dynamic type() call
    upstreams: list[str] = []
    translate: bool = False
    model_name: str = ""
    thinking: bool = True

    def do_GET(self) -> None:
        if self.path == "/health":
            self._health_check()
        else:
            self._proxy("GET")

    def do_POST(self) -> None:
        self._proxy("POST")

    def _health_check(self) -> None:
        """Check each upstream /v1/models and return JSON status."""
        results: list[dict] = []
        for upstream in self.upstreams:
            parsed = urlparse(upstream)
            host = parsed.hostname or "127.0.0.1"
            port = parsed.port or 80
            status = "unhealthy"
            try:
                conn = http.client.HTTPConnection(host, port, timeout=5)
                conn.request("GET", "/v1/models")
                resp = conn.getresponse()
                resp.read()  # drain
                if resp.status == 200:
                    status = "healthy"
                conn.close()
            except (ConnectionError, OSError, http.client.HTTPException):
                pass
            results.append({"url": upstream, "status": status})

        body = json.dumps({"upstreams": results}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy(self, method: str) -> None:
        """Read body, clamp max_tokens, apply safety compress, failover through upstreams.

        When ``self.translate`` is True and the request path contains
        ``/messages``, the Anthropic Messages API request is translated to
        OpenAI Chat Completions format before forwarding, and the response is
        translated back.
        """
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""

        # Determine whether this request needs translation
        needs_translation = (
            self.translate
            and method == "POST"
            and "/messages" in self.path
            and body
        )

        is_streaming_request = False
        anthropic_data: dict | None = None
        path = self.path

        # Parse and modify body for POST requests
        if method == "POST" and body:
            try:
                data = json.loads(body)

                if needs_translation:
                    # Save the original Anthropic data before translation
                    anthropic_data = data
                    is_streaming_request = data.get("stream", False)

                    # Safety compress before translation
                    data = _safety_compress(data)

                    # Translate Anthropic -> OpenAI
                    data = anthropic_to_openai(
                        data,
                        model_name=self.model_name,
                        thinking=self.thinking,
                    )

                    # Rewrite path to OpenAI endpoint
                    path = "/v1/chat/completions"
                else:
                    # Passthrough: just clamp and compress
                    pass

                # Clamp max_tokens
                if "max_tokens" in data:
                    data["max_tokens"] = min(data["max_tokens"], MAX_OUTPUT_TOKENS)
                else:
                    data["max_tokens"] = MAX_OUTPUT_TOKENS

                # Safety compress for non-translated messages endpoints
                if not needs_translation and (
                    "/messages" in self.path or "/chat/completions" in self.path
                ):
                    data = _safety_compress(data)

                body = json.dumps(data).encode()
            except (json.JSONDecodeError, TypeError):
                needs_translation = False

        # Build headers to forward
        fwd_headers: dict[str, str] = {
            "Content-Type": self.headers.get("Content-Type", "application/json"),
        }
        if body:
            fwd_headers["Content-Length"] = str(len(body))

        # Try upstreams in order
        for upstream in self.upstreams:
            ok, resp, conn = _try_upstream(
                upstream, method, path, body, fwd_headers
            )
            if not ok or resp is None or conn is None:
                continue

            # Detect streaming from upstream response
            is_streaming_response = False
            for header, value in resp.getheaders():
                if header.lower() == "content-type" and "text/event-stream" in value:
                    is_streaming_response = True

            if needs_translation:
                # Translate response back to Anthropic format
                try:
                    if is_streaming_response:
                        self._proxy_translate_stream(resp, conn)
                    else:
                        self._proxy_translate_body(resp, conn)
                except (BrokenPipeError, ConnectionError):
                    pass
                finally:
                    conn.close()
            else:
                # Passthrough: forward response as-is
                self.send_response(resp.status)
                is_streaming = False
                for header, value in resp.getheaders():
                    lower = header.lower()
                    if lower in ("transfer-encoding", "connection", "keep-alive"):
                        continue
                    if lower == "content-type" and "text/event-stream" in value:
                        is_streaming = True
                    self.send_header(header, value)
                self.end_headers()

                try:
                    if is_streaming:
                        while True:
                            chunk = resp.read(4096)
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            self.wfile.flush()
                    else:
                        data_bytes = resp.read()
                        self.wfile.write(data_bytes)
                except (BrokenPipeError, ConnectionError):
                    pass
                finally:
                    conn.close()
            return

        # All upstreams failed
        if needs_translation:
            # Return Anthropic-formatted error
            error_body = json.dumps({
                "type": "error",
                "error": {
                    "type": "proxy_error",
                    "message": "All upstreams unavailable",
                },
            }).encode()
        else:
            error_body = json.dumps(
                {"error": {"message": "All upstreams unavailable", "type": "proxy_error"}}
            ).encode()
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(error_body)))
        self.end_headers()
        self.wfile.write(error_body)

    def _proxy_translate_body(
        self,
        resp: http.client.HTTPResponse,
        conn: http.client.HTTPConnection,
    ) -> None:
        """Read a non-streaming OpenAI response, translate to Anthropic, and
        send to the client."""
        raw = resp.read()

        if resp.status >= 400:
            # Forward error as-is, wrapped in Anthropic error format
            try:
                err = json.loads(raw)
                error_body = json.dumps({
                    "type": "error",
                    "error": {
                        "type": "api_error",
                        "message": err.get("error", {}).get("message", raw.decode(errors="replace")),
                    },
                }).encode()
            except (json.JSONDecodeError, TypeError):
                error_body = json.dumps({
                    "type": "error",
                    "error": {"type": "api_error", "message": raw.decode(errors="replace")},
                }).encode()

            self.send_response(resp.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(error_body)))
            self.end_headers()
            self.wfile.write(error_body)
            return

        try:
            oai_data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self.send_response(502)
            err = json.dumps({
                "type": "error",
                "error": {"type": "proxy_error", "message": "Invalid JSON from upstream"},
            }).encode()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        anthropic_resp = openai_to_anthropic(oai_data, self.model_name)
        body = json.dumps(anthropic_resp).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _proxy_translate_stream(
        self,
        resp: http.client.HTTPResponse,
        conn: http.client.HTTPConnection,
    ) -> None:
        """Read a streaming OpenAI response, translate each SSE chunk to
        Anthropic format, and stream to the client."""
        translator = StreamTranslator(self.model_name)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        # Send header events (message_start)
        for event in translator.header_events():
            self.wfile.write(event.encode())
        self.wfile.flush()

        # Read the upstream SSE stream line by line
        # HTTPResponse doesn't have readline by default when reading raw,
        # so we wrap it in a buffered reader.
        buf = io.BufferedReader(resp, buffer_size=8192)  # type: ignore[arg-type]
        for raw_line in buf:
            line = raw_line.decode(errors="replace")
            events = translator.translate_chunk(line)
            for event in events:
                self.wfile.write(event.encode())
            if events:
                self.wfile.flush()

    def log_message(self, fmt: str, *args: object) -> None:
        """Suppress default stderr logging."""
        pass


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-per-request HTTP server."""

    daemon_threads = True
    allow_reuse_address = True


class ProxyServer:
    """Failover proxy that forwards to OpenAI-compatible upstreams.

    When *translate* is True, requests to ``/messages`` are translated from
    Anthropic Messages API to OpenAI Chat Completions API before forwarding,
    and the response is translated back.  This enables non-vLLM backends
    (MLX, Ollama) that only speak OpenAI format to work with Claude Code.
    """

    def __init__(
        self,
        upstreams: list[str],
        host: str = "127.0.0.1",
        port: int = 8081,
        translate: bool = False,
        model_name: str = "",
        thinking: bool = True,
    ) -> None:
        # Create a handler class with config bound via dynamic type()
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
        self._server = _ThreadingHTTPServer((host, port), handler_class)
        # Store the actual assigned port (important when port=0 for auto-assign)
        self.port: int = self._server.server_address[1]
        self._thread: threading.Thread | None = None

    def start_background(self) -> None:
        """Start server in a daemon thread."""
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def serve_forever(self) -> None:
        """Block in the main thread serving requests."""
        self._server.serve_forever()

    def stop(self) -> None:
        """Shutdown the server."""
        self._server.shutdown()
