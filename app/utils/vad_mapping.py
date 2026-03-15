import numpy as np

# Columns: [Valence, Arousal, Dominance]

W_vad = np.array([
    [ 0.25,  0.38, -0.18],   # pitch
    [ 0.05,  0.58, -0.08],   # rate
    [ 0.15,  0.48,  0.22],   # energy
    [-0.10, -0.55, -0.10],   # pauses
], dtype=np.float64)


def compute_S_cont(vad_01, beta=1.35):
    """
    Continuous prosody shift from VAD coordinates.
    """

    vad_01 = np.clip(np.asarray(vad_01), 0.0, 1.0)

    vad_centered = 2 * vad_01 - 1

    vad_nonlinear = np.tanh(beta * vad_centered)

    return W_vad @ vad_nonlinear