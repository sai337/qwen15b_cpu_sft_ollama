"""UI test against a fake Ollama server (no model needed)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
ROOT = Path(__file__).resolve().parents[1]


class FakeOllama(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"models": [{"name": "qwen2.5:1.5b-instruct"}, {"name": "qwen15b-cloudqa:latest"}]}).encode()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.seen.append(req)
        self.send_response(200)
        self.end_headers()
        for part in ["A DaemonSet ", "runs one pod ", "per node."]:
            self.wfile.write((json.dumps({"message": {"role": "assistant", "content": part}, "done": False}) + "\n").encode())
        self.wfile.write(b'{"done": true}\n')


@pytest.fixture()
def fake_ollama(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{srv.server_port}")
    monkeypatch.setenv("OLLAMA_MODEL", "qwen15b-cloudqa")
    yield
    srv.shutdown()


def test_chat_streams_answer_and_uses_finetuned_model(fake_ollama):
    at = st_testing.AppTest.from_file(str(ROOT / "ui/streamlit_app.py"), default_timeout=30).run()
    assert at.sidebar.selectbox[0].value == "qwen15b-cloudqa:latest"  # default picks the fine-tuned model
    at.chat_input[0].set_value("What is a DaemonSet?").run()
    assert "A DaemonSet runs one pod per node." in [m.markdown[0].value for m in at.chat_message]
    req = FakeOllama.seen[-1]
    assert req["model"] == "qwen15b-cloudqa:latest"
    assert all(m["role"] != "system" for m in req["messages"])  # Modelfile SYSTEM is used by default
