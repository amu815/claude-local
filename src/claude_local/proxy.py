"""OpenAI-compatible failover proxy for claude-local."""
from __future__ import annotations

import json
import sys
import threading
import http.client
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

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

    # Set by ProxyServer.__init__ via a closure
    upstreams: list[str] = []

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
        """Read body, clamp max_tokens, apply safety compress, failover through upstreams."""
        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""

        # Parse and modify body for POST requests
        if method == "POST" and body:
            try:
                data = json.loads(body)
                # Clamp max_tokens
                if "max_tokens" in data:
                    data["max_tokens"] = min(data["max_tokens"], MAX_OUTPUT_TOKENS)
                else:
                    data["max_tokens"] = MAX_OUTPUT_TOKENS

                # Safety compress for messages endpoints
                if "/messages" in self.path or "/chat/completions" in self.path:
                    data = _safety_compress(data)

                body = json.dumps(data).encode()
            except (json.JSONDecodeError, TypeError):
                pass

        # Build headers to forward
        fwd_headers: dict[str, str] = {
            "Content-Type": self.headers.get("Content-Type", "application/json"),
        }
        if body:
            fwd_headers["Content-Length"] = str(len(body))

        # Try upstreams in order
        for upstream in self.upstreams:
            ok, resp, conn = _try_upstream(
                upstream, method, self.path, body, fwd_headers
            )
            if not ok or resp is None or conn is None:
                continue

            # Forward response
            self.send_response(resp.status)
            is_streaming = False
            for header, value in resp.getheaders():
                lower = header.lower()
                # Skip hop-by-hop headers
                if lower in ("transfer-encoding", "connection", "keep-alive"):
                    continue
                if lower == "content-type" and "text/event-stream" in value:
                    is_streaming = True
                self.send_header(header, value)
            self.end_headers()

            try:
                if is_streaming:
                    # Stream SSE data through
                    while True:
                        chunk = resp.read(4096)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    data = resp.read()
                    self.wfile.write(data)
            except (BrokenPipeError, ConnectionError):
                pass
            finally:
                conn.close()
            return

        # All upstreams failed
        error_body = json.dumps(
            {"error": {"message": "All upstreams unavailable", "type": "proxy_error"}}
        ).encode()
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(error_body)))
        self.end_headers()
        self.wfile.write(error_body)

    def log_message(self, fmt: str, *args: object) -> None:
        """Suppress default stderr logging."""
        pass


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Thread-per-request HTTP server."""

    daemon_threads = True
    allow_reuse_address = True


class ProxyServer:
    """Failover proxy that forwards to OpenAI-compatible upstreams."""

    def __init__(
        self,
        upstreams: list[str],
        host: str = "127.0.0.1",
        port: int = 8081,
    ) -> None:
        # Create a handler class with upstreams bound via closure
        handler_class = type(
            "_BoundProxyHandler",
            (_ProxyHandler,),
            {"upstreams": list(upstreams)},
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
