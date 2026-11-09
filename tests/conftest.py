"""Shared fixtures and lightweight fakes.

Fakes stand in for the heavyweight models and the external TTS service so the
whole API and pipeline can be tested with no downloads and no network.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.emotion.weights import EMOTION_WEIGHTS, build_emotion_matrix

LABELS = list(EMOTION_WEIGHTS.keys())  # includes "neutral" last, mirrors model order


class FakeEmotionModel:
    """Emotion model returning a controllable score vector."""

    labels = LABELS

    def __init__(self, dominant: str = "joy", score: float = 0.9) -> None:
        self._dominant = dominant
        self._score = score

    def infer(self, text: str) -> np.ndarray:
        v = np.full(len(LABELS), 0.05)
        v[LABELS.index(self._dominant)] = self._score
        return v


class FakeVADModel:
    def __init__(self, vad=(0.8, 0.7, 0.6)) -> None:
        self._vad = np.array(vad, dtype=float)

    def infer(self, text: str) -> np.ndarray:
        return self._vad


class FakeSynthesizer:
    """Returns deterministic dummy bytes; records the last call for assertions."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def synthesize(self, text, voice, rate, pitch, volume) -> bytes:
        self.calls.append(
            {"text": text, "voice": voice, "rate": rate, "pitch": pitch, "volume": volume}
        )
        return b"ID3\x04fake-mp3-bytes"


@pytest.fixture
def weight_matrix() -> np.ndarray:
    return build_emotion_matrix(LABELS)


@pytest.fixture
def fake_emotion() -> FakeEmotionModel:
    return FakeEmotionModel()


@pytest.fixture
def fake_vad() -> FakeVADModel:
    return FakeVADModel()
