"""End-to-end emotion -> prosody orchestration (pure, synchronous).

This is the heart of the system and is deliberately free of torch, FastAPI, and
I/O: it takes already-loaded model objects (anything exposing ``infer`` / an
``EmotionModel`` with ``.labels``) plus the prebuilt weight matrix, and returns
the full analysis. That makes it unit-testable with lightweight fakes and no
model downloads.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from app.emotion.fusion import fuse
from app.emotion.prosody import process_prosody
from app.emotion.vad_mapping import compute_continuous_shift
from app.emotion.weights import compute_discrete_shift
from app.schemas import AnalyzeResponse, EmotionEstimate, VADEstimate


class _Infers(Protocol):
    def infer(self, text: str) -> np.ndarray: ...


def _build_explanation(emotions: list[EmotionEstimate], prosody) -> str:
    top = emotions[0].label if emotions else "neutral"

    def describe(value: float, up: str, down: str, unit: str) -> str | None:
        if abs(value) < 1e-6:
            return None
        direction = up if value > 0 else down
        return f"{direction} {abs(value):.0f}{unit}"

    parts = [
        p for p in (
            describe(prosody.pitch_hz, "higher pitch", "lower pitch", " Hz"),
            describe(prosody.rate_pct, "faster rate", "slower rate", "%"),
            describe(prosody.energy_pct, "more energy", "less energy", "%"),
        ) if p
    ]
    pace = "more pauses" if prosody.pause_level > 0.55 else (
        "fewer pauses" if prosody.pause_level < 0.45 else "balanced pauses"
    )
    parts.append(pace)
    detail = ", ".join(parts) if parts else "near-neutral delivery"
    return (
        f"Estimated dominant tone '{top}'. Delivery adjusts toward {detail}. "
        "These are model estimates, not a definitive reading of emotion."
    )


def analyze(
    text: str,
    emotion_model,
    vad_model: _Infers,
    weight_matrix: np.ndarray,
    expressiveness: float = 1.0,
    top_k: int = 5,
) -> AnalyzeResponse:
    probs = emotion_model.infer(text)
    vad = vad_model.infer(text)

    labels = emotion_model.labels
    neutral_index = labels.index("neutral") if "neutral" in labels else None

    s_disc = compute_discrete_shift(probs, weight_matrix)
    s_cont = compute_continuous_shift(vad)
    s_final = fuse(s_disc, s_cont, probs, neutral_index=neutral_index)
    prosody = process_prosody(s_final, expressiveness=expressiveness)

    order = np.argsort(probs)[::-1][:top_k]
    top_emotions = [
        EmotionEstimate(label=emotion_model.labels[i], score=round(float(probs[i]), 4))
        for i in order
    ]

    return AnalyzeResponse(
        text=text,
        top_emotions=top_emotions,
        vad=VADEstimate(
            valence=round(float(vad[0]), 4),
            arousal=round(float(vad[1]), 4),
            dominance=round(float(vad[2]), 4),
        ),
        prosody=prosody,
        explanation=_build_explanation(top_emotions, prosody),
    )
