"""Post-processing: fused prosody vector -> bounded, TTS-ready parameters.

Scales each fused dimension and clamps it to a safe range. Values are unchanged
from the original implementation. An optional ``expressiveness`` scalar applies a
single global gain to the (signed) deltas *before* clamping, so the emotion
mappings and fusion are untouched — it only dials the overall strength.
"""

from __future__ import annotations

import numpy as np

from app.schemas import ProsodyParameters

# Clamp bounds (previously in config.py PROSODY_CLAMPS; kept with the logic they
# govern so the mapping is readable in one place).
PITCH_HZ_CLAMP = 40.0
RATE_PCT_CLAMP = 35.0
ENERGY_PCT_CLAMP = 30.0
PAUSE_GAIN = 0.45

# Per-dimension pre-scale (unchanged original factors).
PITCH_SCALE = 0.60
RATE_SCALE = 0.85
ENERGY_SCALE = 1.00


def process_prosody(fused: np.ndarray, expressiveness: float = 1.0) -> ProsodyParameters:
    pitch = np.clip(
        expressiveness * PITCH_SCALE * PITCH_HZ_CLAMP * fused[0],
        -PITCH_HZ_CLAMP, PITCH_HZ_CLAMP,
    )
    rate = np.clip(
        expressiveness * RATE_SCALE * RATE_PCT_CLAMP * fused[1],
        -RATE_PCT_CLAMP, RATE_PCT_CLAMP,
    )
    energy = np.clip(
        expressiveness * ENERGY_SCALE * ENERGY_PCT_CLAMP * fused[2],
        -ENERGY_PCT_CLAMP, ENERGY_PCT_CLAMP,
    )
    # Pause level is centered at 0.5; expressiveness scales the deviation from it.
    pause = np.clip(0.5 + expressiveness * PAUSE_GAIN * fused[3], 0.10, 0.92)

    return ProsodyParameters(
        pitch_hz=round(float(pitch), 2),
        rate_pct=round(float(rate), 2),
        energy_pct=round(float(energy), 2),
        pause_level=round(float(pause), 3),
    )
