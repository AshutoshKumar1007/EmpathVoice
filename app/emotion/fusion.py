"""Confidence-adaptive fusion of the discrete and continuous prosody shifts.

The blend is unchanged from the original (a per-dimension weighted *arithmetic*
combination — the original docstring mislabelled it a "geometric mean"; the
README's math is the correct description).

One technically justified refinement (Stage 2, #1): the confidence term now
ignores the ``neutral`` label. GoEmotions assigns neutral text a high
``neutral`` probability, whose prosody row is all zeros; using it as confidence
made the (zero) discrete branch dominate and suppressed the VAD branch, flat-
tening neutral/ambiguous delivery. Excluding neutral lets such text lean on the
continuous branch as the hybrid design intends. When ``neutral_index`` is None
the behaviour is identical to the original.

    c        = (max_i, i != neutral, p_i) ** kappa   # confidence (neutral excluded)
    w_disc   = c * alpha_priors                        # per-dimension discrete weight
    w_cont   = 1 - w_disc
    S_final  = w_disc * (G_disc * S_disc) + w_cont * (G_cont * S_cont)
"""

from __future__ import annotations

import numpy as np

ALPHA_PRIORS = np.array([0.62, 0.50, 0.52, 0.55])
DISC_GAIN = np.array([1.35, 1.35, 1.35, 1.20])
CONT_GAIN = np.array([0.80, 0.80, 0.80, 0.75])


def fuse(
    s_disc: np.ndarray,
    s_cont: np.ndarray,
    emotion_probs: np.ndarray,
    kappa: float = 1.5,
    neutral_index: int | None = None,
) -> np.ndarray:
    """Fuse discrete and continuous shifts with a confidence-adaptive blend.

    If ``neutral_index`` is given, that label is excluded from the confidence
    term (see module docstring). Falls back to the plain max if neutral is the
    only label present.
    """
    if neutral_index is None:
        top_prob = float(np.max(emotion_probs))
    else:
        mask = np.ones(len(emotion_probs), dtype=bool)
        mask[neutral_index] = False
        non_neutral = emotion_probs[mask]
        top_prob = float(np.max(non_neutral)) if non_neutral.size else float(np.max(emotion_probs))

    confidence = top_prob**kappa

    w_disc = confidence * ALPHA_PRIORS
    w_cont = 1.0 - w_disc

    s_disc_eff = DISC_GAIN * s_disc
    s_cont_eff = CONT_GAIN * s_cont

    return w_disc * s_disc_eff + w_cont * s_cont_eff
