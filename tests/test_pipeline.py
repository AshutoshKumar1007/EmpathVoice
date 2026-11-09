"""Integration tests for the pure pipeline using fake models."""

from __future__ import annotations

import numpy as np

from app.emotion import pipeline
from tests.conftest import LABELS, FakeEmotionModel, FakeVADModel


def test_pipeline_returns_full_analysis(weight_matrix, fake_emotion, fake_vad):
    resp = pipeline.analyze("so happy today", fake_emotion, fake_vad, weight_matrix)
    assert resp.top_emotions[0].label == "joy"
    assert len(resp.top_emotions) == 5
    assert 0.0 <= resp.vad.valence <= 1.0
    assert "model estimates" in resp.explanation


def test_positive_emotion_raises_pitch_and_rate(weight_matrix):
    joy = pipeline.analyze(
        "wonderful", FakeEmotionModel("joy", 0.95), FakeVADModel((0.9, 0.8, 0.6)), weight_matrix
    )
    assert joy.prosody.pitch_hz > 0
    assert joy.prosody.rate_pct > 0


def test_grief_lowers_pitch_and_adds_pauses(weight_matrix):
    grief = pipeline.analyze(
        "I miss them so much",
        FakeEmotionModel("grief", 0.95),
        FakeVADModel((0.1, 0.2, 0.3)),
        weight_matrix,
    )
    assert grief.prosody.pitch_hz < 0
    assert grief.prosody.pause_level > 0.5  # more pauses for somber delivery


def test_neutral_text_leans_on_vad(weight_matrix):
    """With neutral dominant, confidence excludes neutral, so VAD still shapes output."""
    probs = np.full(len(LABELS), 0.02)
    probs[LABELS.index("neutral")] = 0.95

    class NeutralEmotion:
        labels = LABELS

        def infer(self, text):
            return probs

    # High-arousal VAD should still push rate up despite neutral dominance.
    excited_vad = FakeVADModel((0.6, 0.95, 0.6))
    calm_vad = FakeVADModel((0.5, 0.05, 0.5))
    hot = pipeline.analyze("the meeting is at noon", NeutralEmotion(), excited_vad, weight_matrix)
    cool = pipeline.analyze("the meeting is at noon", NeutralEmotion(), calm_vad, weight_matrix)
    assert hot.prosody.rate_pct > cool.prosody.rate_pct
