"""Unit tests for is_model_serving — best-effort loopback probe."""

from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from app.utils.server_probe import is_model_serving


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _MockHandler(BaseHTTPRequestHandler):
    response_body: bytes = b'{"data":[]}'
    response_code: int = 200

    def do_GET(self):
        self.send_response(self.response_code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, *args, **kwargs):
        pass  # silence default access log


@pytest.fixture
def mock_server():
    port = _free_port()
    server = HTTPServer(("127.0.0.1", port), _MockHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port, _MockHandler
    server.shutdown()


def test_returns_false_when_nothing_listening():
    """No server on the port → returns False, no exceptions."""
    assert is_model_serving("any/id", port=_free_port(), timeout=0.5) is False


def test_returns_true_when_id_in_data(mock_server):
    port, handler = mock_server
    handler.response_body = json.dumps({"data": [{"id": "mlx-community/Foo-4bit"}, {"id": "other:model"}]}).encode()
    handler.response_code = 200
    assert is_model_serving("mlx-community/Foo-4bit", port=port) is True


def test_returns_false_when_id_not_in_data(mock_server):
    port, handler = mock_server
    handler.response_body = json.dumps({"data": [{"id": "different/model"}]}).encode()
    handler.response_code = 200
    assert is_model_serving("mlx-community/Foo-4bit", port=port) is False


def test_returns_false_on_5xx(mock_server):
    port, handler = mock_server
    handler.response_code = 503
    handler.response_body = b"loading"
    assert is_model_serving("any/id", port=port) is False


def test_returns_false_on_malformed_json(mock_server):
    port, handler = mock_server
    handler.response_code = 200
    handler.response_body = b"not json"
    assert is_model_serving("any/id", port=port) is False


def test_serving_model_ids_batch(mock_server):
    """serving_model_ids fetches /v1/models once and returns the full set."""
    from app.utils.server_probe import serving_model_ids

    port, handler = mock_server
    handler.response_body = json.dumps({"data": [{"id": "a:1"}, {"id": "b:2"}, {"id": "c:3"}]}).encode()
    handler.response_code = 200
    got = serving_model_ids(port=port)
    assert got == {"a:1", "b:2", "c:3"}


def test_serving_model_ids_returns_empty_on_no_server():
    from app.utils.server_probe import serving_model_ids

    assert serving_model_ids(port=_free_port(), timeout=0.5) == set()
