"""Optional integration tests that exercise the real models and real TTS.

These are deselected by default (see pyproject addopts ``-m 'not integration'``)
because they download pretrained models and call the live edge-tts service.

Run explicitly with:
    pytest -m integration

They skip cleanly if torch / edge-tts are not installed, so the command is safe
to run anywhere.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _load_manager():
    from app.core.config import Settings
    from app.ml.manager import ModelManager

    mgr = ModelManager(Settings())
    mgr._load_blocking()  # synchronous load for the test
    return mgr


def test_real_models_load_and_analyze():
    mgr = _load_manager()
    assert mgr.device in ("cpu", "cuda")

    import asyncio

    joy = asyncio.run(mgr.analyze("I am absolutely thrilled and overjoyed today!"))
    sad = asyncio.run(mgr.analyze("I feel so empty and heartbroken, nothing matters."))

    # Qualitative, not accuracy claims: happy text should not be gloomier than sad.
    assert joy.prosody.pitch_hz >= sad.prosody.pitch_hz
    assert joy.vad.arousal >= 0.0 and sad.vad.valence <= 1.0
    assert len(joy.top_emotions) == 5


@pytest.mark.asyncio
async def test_real_tts_produces_audio():
    pytest.importorskip("edge_tts")
    from app.tts.synth import EdgeTTSSynthesizer

    audio = await EdgeTTSSynthesizer().synthesize(
        text="This is a short test.",
        voice="en-US-AriaNeural",
        rate="+0%",
        pitch="+0Hz",
        volume="+0%",
    )
    assert isinstance(audio, bytes) and len(audio) > 1000  # non-trivial mp3
