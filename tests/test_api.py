"""API tests via TestClient with fakes injected through dependency overrides.

No torch, no network: the model manager and synthesizer are faked, and a real
temp-dir AudioStore exercises the full synthesize -> store -> retrieve path.
"""

from __future__ import annotations

import pytest

from app.api.deps import get_manager, get_store, get_synthesizer
from app.audio.store import AudioStore
from app.emotion import pipeline
from app.emotion.weights import build_emotion_matrix
from app.main import app
from app.ml.manager import ModelStatus
from tests.conftest import LABELS, FakeEmotionModel, FakeSynthesizer, FakeVADModel


class FakeManager:
    def __init__(self, ready: bool = True) -> None:
        self.status = ModelStatus.READY if ready else ModelStatus.FAILED
        self.device = "cpu"
        self.detail = None
        self._w = build_emotion_matrix(LABELS)

    @property
    def ready(self) -> bool:
        return self.status is ModelStatus.READY

    async def analyze(self, text: str, expressiveness: float = 1.0):
        return pipeline.analyze(
            text, FakeEmotionModel(), FakeVADModel(), self._w, expressiveness
        )


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    store = AudioStore(tmp_path, ttl_seconds=3600)
    synth = FakeSynthesizer()
    app.dependency_overrides[get_manager] = lambda: FakeManager(ready=True)
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_synthesizer] = lambda: synth
    with TestClient(app) as c:
        c._synth = synth  # expose for assertions
        yield c
    app.dependency_overrides.clear()


def test_health_always_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_voices(client):
    r = client.get("/api/voices")
    assert r.status_code == 200
    body = r.json()
    assert body["default"]
    assert any(v["id"] == body["default"] for v in body["voices"])


def test_analyze_happy_path(client):
    r = client.post("/api/analyze", json={"text": "I am so happy right now"})
    assert r.status_code == 200
    body = r.json()
    assert body["top_emotions"][0]["label"] == "joy"
    assert set(body["prosody"]) == {"pitch_hz", "rate_pct", "energy_pct", "pause_level"}
    assert "model estimates" in body["explanation"]


def test_analyze_rejects_empty(client):
    r = client.post("/api/analyze", json={"text": "   "})
    assert r.status_code == 422


def test_analyze_rejects_too_long(client):
    r = client.post("/api/analyze", json={"text": "a" * 5000})
    assert r.status_code == 422


def test_synthesize_and_retrieve_audio(client):
    r = client.post("/api/synthesize", json={"text": "hello there friend"})
    assert r.status_code == 200
    body = r.json()
    assert body["audio_url"].startswith("/api/audio/")
    assert body["expires_in_seconds"] == 3600
    assert body["analysis"]["top_emotions"]

    # The generated audio is retrievable at the returned url.
    audio = client.get(body["audio_url"])
    assert audio.status_code == 200
    assert audio.headers["content-type"] == "audio/mpeg"
    assert audio.content == b"ID3\x04fake-mp3-bytes"


def test_synthesize_rejects_unknown_voice(client):
    r = client.post("/api/synthesize", json={"text": "hi", "voice": "xx-ZZ-NopeNeural"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"


def test_audio_404_for_unknown_id(client):
    r = client.get("/api/audio/deadbeefdeadbeefdeadbeefdeadbeef")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "audio_not_found"


def test_not_ready_returns_503():
    from fastapi.testclient import TestClient

    app.dependency_overrides[get_manager] = lambda: FakeManager(ready=False)
    try:
        with TestClient(app) as c:
            r = c.post("/api/analyze", json={"text": "hello"})
            assert r.status_code == 503
            assert r.json()["error"]["code"] == "model_not_ready"
    finally:
        app.dependency_overrides.clear()
