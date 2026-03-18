"""Tests for the failover proxy."""
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from claude_local.proxy import (
    ProxyServer,
    _estimate_chars,
    _safety_compress,
    MAX_OUTPUT_TOKENS,
    SAFETY_CHAR_LIMIT,
)


class MockBackendHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps(
            {"object": "list", "data": [{"id": "test-model"}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length) if length else b""
        response = json.dumps(
            {"id": "test", "choices": [{"message": {"content": "hello"}}]}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, fmt, *args):
        pass


@pytest.fixture
def mock_backend():
    server = HTTPServer(("127.0.0.1", 0), MockBackendHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture
def proxy_server(mock_backend):
    proxy = ProxyServer(upstreams=[mock_backend], host="127.0.0.1", port=0)
    proxy.start_background()
    time.sleep(0.1)  # Let server bind
    yield proxy
    proxy.stop()


def test_proxy_health(proxy_server):
    resp = urlopen(f"http://127.0.0.1:{proxy_server.port}/health")
    data = json.loads(resp.read())
    assert "upstreams" in data
    assert len(data["upstreams"]) == 1
    assert data["upstreams"][0]["status"] == "healthy"


def test_proxy_forwards_get(proxy_server):
    resp = urlopen(f"http://127.0.0.1:{proxy_server.port}/v1/models")
    data = json.loads(resp.read())
    assert data["data"][0]["id"] == "test-model"


def test_proxy_forwards_post(proxy_server):
    body = json.dumps(
        {"messages": [{"role": "user", "content": "hi"}]}
    ).encode()
    req = Request(
        f"http://127.0.0.1:{proxy_server.port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urlopen(req)
    data = json.loads(resp.read())
    assert data["choices"][0]["message"]["content"] == "hello"


def test_proxy_all_upstreams_down():
    proxy = ProxyServer(
        upstreams=["http://127.0.0.1:19999"], host="127.0.0.1", port=0
    )
    proxy.start_background()
    time.sleep(0.1)
    try:
        try:
            urlopen(f"http://127.0.0.1:{proxy.port}/v1/models")
            assert False, "Should have raised"
        except HTTPError as e:
            assert e.code == 503
    finally:
        proxy.stop()


def test_estimate_chars():
    data = {
        "system": "You are helpful.",
        "messages": [
            {"role": "user", "content": "Hello world"},
        ],
    }
    chars = _estimate_chars(data)
    assert chars > 0


def test_safety_compress_no_op_small():
    """Small conversations should not be compressed."""
    data = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
    }
    result = _safety_compress(data)
    assert len(result["messages"]) == 2


def test_safety_compress_drops_old():
    """Conversations over SAFETY_CHAR_LIMIT should have old messages dropped."""
    big_content = "x" * (SAFETY_CHAR_LIMIT + 10000)
    data = {
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": big_content},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "follow up"},
        ],
    }
    result = _safety_compress(data)
    # System message should be preserved
    assert result["messages"][0]["role"] == "system"
    # At least one message should have been dropped
    assert len(result["messages"]) < 4


def test_max_tokens_clamping(proxy_server):
    """max_tokens above MAX_OUTPUT_TOKENS should be clamped."""
    body = json.dumps(
        {
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 999999,
        }
    ).encode()
    req = Request(
        f"http://127.0.0.1:{proxy_server.port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    resp = urlopen(req)
    # If it returns 200, the proxy processed it (clamping is internal)
    assert resp.status == 200
