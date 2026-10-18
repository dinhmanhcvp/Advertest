"""DataConnector: presigned-URL TTL handling, retries, cache hygiene."""

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from src.services.data_connector import DataConnector, PresignedURLExpired


@pytest.fixture
def server():
    state = {"hits": 0, "fail_first": 0, "deny_stale": False}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["hits"] += 1
            if "stale" in self.path and state["deny_stale"]:
                self.send_response(403)
                self.end_headers()
                return
            if state["hits"] <= state["fail_first"]:
                self.send_response(503)
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"video-bytes")

        def log_message(self, *args):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}", state
    srv.shutdown()


@pytest.fixture
def connector(tmp_path, monkeypatch):
    c = DataConnector()
    c.cache_dir = tmp_path
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return c


def test_downloads_and_cleans_up(server, connector):
    base, _ = server
    paths = connector.pull_video_batch([f"{base}/a.mp4?sig=x"])
    with open(paths[0], "rb") as f:
        assert f.read() == b"video-bytes"
    connector.clear_cache()
    assert not list(connector.cache_dir.iterdir())


def test_retries_on_5xx(server, connector):
    base, state = server
    state["fail_first"] = 2
    assert connector.pull_video_batch([f"{base}/a.mp4"])
    assert state["hits"] == 3


def test_expired_url_rejected_before_request(server, connector):
    base, state = server
    with pytest.raises(PresignedURLExpired):
        connector.pull_video_batch([f"{base}/a.mp4?Expires=1"])
    assert state["hits"] == 0


def test_expired_url_is_resigned(server, connector):
    base, _ = server
    paths = connector.pull_video_batch(
        [f"{base}/a.mp4?Expires=1"], refresh_url=lambda u: f"{base}/fresh.mp4"
    )
    assert len(paths) == 1


def test_403_midflight_resigns_once(server, connector):
    base, state = server
    state["deny_stale"] = True
    paths = connector.pull_video_batch([f"{base}/stale.mp4"], refresh_url=lambda u: f"{base}/ok.mp4")
    assert len(paths) == 1
    with pytest.raises(PresignedURLExpired):
        connector.pull_video_batch([f"{base}/stale2.mp4"])
