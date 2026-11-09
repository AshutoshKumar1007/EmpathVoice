"""Unit tests for the emotion/VAD -> prosody math."""

from __future__ import annotations

import numpy as np

from app.emotion.fusion import ALPHA_PRIORS, fuse
from app.emotion.prosody import (
    ENERGY_PCT_CLAMP,
    PITCH_HZ_CLAMP,
    RATE_PCT_CLAMP,
    process_prosody,
)
from app.emotion.vad_mapping import compute_continuous_shift
from app.emotion.weights import (
    EMOTION_WEIGHTS,
    build_emotion_matrix,
    compute_discrete_shift,
)

LABELS = list(EMOTION_WEIGHTS.keys())


def test_emotion_matrix_shape_and_alignment():
    W = build_emotion_matrix(LABELS)
    assert W.shape == (28, 4)
    # A one-hot on a label reproduces that label's weights exactly.
    probs = np.zeros(28)
    probs[LABELS.index("grief")] = 1.0
    assert np.allclose(compute_discrete_shift(probs, W), EMOTION_WEIGHTS["grief"])


def test_matrix_alignment_is_by_name_not_index():
    # Reverse the label order; weights must still match each label.
    reversed_labels = list(reversed(LABELS))
    W = build_emotion_matrix(reversed_labels)
    i = reversed_labels.index("anger")
    assert np.allclose(W[i], EMOTION_WEIGHTS["anger"])


def test_vad_mapping_monotonic_arousal_increases_rate():
    low = compute_continuous_shift(np.array([0.5, 0.1, 0.5]))
    high = compute_continuous_shift(np.array([0.5, 0.9, 0.5]))
    assert high[1] > low[1]  # rate dimension


def test_vad_mapping_clips_out_of_range_inputs():
    a = compute_continuous_shift(np.array([5.0, -5.0, 0.5]))
    b = compute_continuous_shift(np.array([1.0, 0.0, 0.5]))
    assert np.allclose(a, b)


def test_fusion_confidence_adaptive():
    s_disc = np.array([1.0, 0, 0, 0])
    s_cont = np.array([-1.0, 0, 0, 0])
    hi = fuse(s_disc, s_cont, np.array([0.99]))
    lo = fuse(s_disc, s_cont, np.array([0.30]))
    assert hi[0] > lo[0]  # higher discrete confidence -> more discrete influence


def test_fusion_neutral_exclusion_changes_confidence():
    # Build a 28-vector: high neutral prob, modest joy prob.
    probs = np.full(28, 0.02)
    probs[LABELS.index("neutral")] = 0.95
    probs[LABELS.index("joy")] = 0.40
    s_disc = np.array([1.0, 0, 0, 0])
    s_cont = np.array([0.0, 0, 0, 0])

    neutral_idx = LABELS.index("neutral")
    without = fuse(s_disc, s_cont, probs)  # uses 0.95 as confidence
    with_excl = fuse(s_disc, s_cont, probs, neutral_index=neutral_idx)  # uses 0.40

    # Excluding neutral lowers confidence -> lowers the discrete weight.
    w_without = (0.95**1.5) * ALPHA_PRIORS[0]
    w_with = (0.40**1.5) * ALPHA_PRIORS[0]
    assert with_excl[0] == w_with * 1.35  # DISC_GAIN[0]
    assert without[0] == w_without * 1.35
    assert with_excl[0] < without[0]


def test_fusion_neutral_exclusion_noop_when_neutral_only():
    probs = np.zeros(28)
    probs[LABELS.index("neutral")] = 0.9
    s_disc = np.array([0.3, 0, 0, 0])
    s_cont = np.array([0.1, 0, 0, 0])
    # Falls back to plain max; should not raise or produce nan.
    out = fuse(s_disc, s_cont, probs, neutral_index=LABELS.index("neutral"))
    assert np.all(np.isfinite(out))


def test_prosody_saturation_and_bounds():
    p = process_prosody(np.array([10.0, 10.0, 10.0, 10.0]))
    assert p.pitch_hz == PITCH_HZ_CLAMP
    assert p.rate_pct == RATE_PCT_CLAMP
    assert p.energy_pct == ENERGY_PCT_CLAMP
    assert 0.10 <= p.pause_level <= 0.92

    n = process_prosody(np.array([-10.0, -10.0, -10.0, -10.0]))
    assert n.pitch_hz == -PITCH_HZ_CLAMP


def test_prosody_expressiveness_zero_is_flat_pitch_rate_energy():
    p = process_prosody(np.array([0.5, 0.5, 0.5, 0.5]), expressiveness=0.0)
    assert p.pitch_hz == 0.0
    assert p.rate_pct == 0.0
    assert p.energy_pct == 0.0
    assert p.pause_level == 0.5  # centered when no deviation
