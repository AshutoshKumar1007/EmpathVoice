"""Continuous affect (VAD) -> prosody mapping.

Maps valence/arousal/dominance in [0, 1] to the same 4-D prosody space. Values
and transform (center to [-1, 1], tanh nonlinearity) are unchanged from the
original implementation.
"""

from __future__ import annotations

import numpy as np

# Columns: [Valence, Arousal, Dominance]; rows: [pitch, rate, energy, pauses]
W_VAD = np.array(
    [
        [0.25,  0.38, -0.18],   # pitch
        [0.05,  0.58, -0.08],   # rate
        [0.15,  0.48,  0.22],   # energy
        [-0.10, -0.55, -0.10],  # pauses
    ],
    dtype=np.float64,
)


def compute_continuous_shift(vad_01: np.ndarray, beta: float = 1.35) -> np.ndarray:
    """Continuous prosody shift from VAD coordinates in [0, 1]."""
    vad_01 = np.clip(np.asarray(vad_01, dtype=np.float64), 0.0, 1.0)
    vad_centered = 2.0 * vad_01 - 1.0
    vad_nonlinear = np.tanh(beta * vad_centered)
    return W_VAD @ vad_nonlinear
