"""Discrete emotion -> prosody mapping (GoEmotions branch).

Handcrafted weights mapping each of the 28 GoEmotions labels to a 4-D prosody
shift [pitch, rate, energy, pauses]. Values are unchanged from the original
implementation; only the matrix construction has been decoupled from model
import so this module can be loaded and tested without downloading a model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Format: [pitch, rate, energy, pauses]
EMOTION_WEIGHTS: dict[str, list[float]] = {
    "admiration":      [0.15,  0.08,  0.18, -0.10],
    "amusement":       [0.25,  0.22,  0.20, -0.20],
    "excitement":      [0.45,  0.42,  0.45, -0.45],
    "joy":             [0.38,  0.30,  0.35, -0.32],
    "pride":           [0.12, -0.08,  0.30, -0.05],
    "love":            [0.15, -0.12,  0.08,  0.10],
    "approval":        [0.08,  0.05,  0.10, -0.05],
    "caring":          [0.05, -0.15,  0.05,  0.15],
    "gratitude":       [0.18,  0.05,  0.15, -0.08],
    "optimism":        [0.20,  0.12,  0.18, -0.12],
    "relief":          [0.05, -0.20, -0.10,  0.18],
    "anger":           [0.15,  0.32,  0.50, -0.38],
    "annoyance":       [0.05,  0.15,  0.25, -0.12],
    "disgust":         [-0.12, -0.10,  0.18,  0.20],
    "fear":            [0.25,  0.28,  0.10,  0.18],
    "nervousness":     [0.15,  0.18, -0.08,  0.25],
    "disappointment":  [-0.22, -0.20, -0.18,  0.28],
    "disapproval":     [-0.08,  0.05,  0.18,  0.10],
    "embarrassment":   [-0.18, -0.15, -0.22,  0.25],
    "grief":           [-0.42, -0.45, -0.40,  0.50],
    "remorse":         [-0.25, -0.22, -0.20,  0.32],
    "sadness":         [-0.32, -0.30, -0.28,  0.38],
    "confusion":       [0.12, -0.15, -0.08,  0.25],
    "curiosity":       [0.18,  0.10,  0.08,  0.05],
    "desire":          [0.12, -0.05,  0.10, -0.05],
    "realization":     [0.10, -0.08,  0.05,  0.15],
    "surprise":        [0.35,  0.12,  0.28,  0.08],
    "neutral":         [0.0,  0.0,  0.0,  0.0],
}


def build_emotion_matrix(labels: Sequence[str]) -> np.ndarray:
    """Build the (num_labels x 4) weight matrix aligned to the model's label order.

    Aligning by label *name* (not index) keeps the mapping correct regardless of
    how the model orders its labels. Raises if a label has no defined mapping.
    """
    try:
        rows = [EMOTION_WEIGHTS[label.lower()] for label in labels]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"No prosody weights defined for emotion label: {exc}") from exc
    return np.array(rows, dtype=np.float64)


def compute_discrete_shift(probs: np.ndarray, weight_matrix: np.ndarray) -> np.ndarray:
    """Discrete prosody shift: S_disc = probs @ W_emo."""
    return probs @ weight_matrix
