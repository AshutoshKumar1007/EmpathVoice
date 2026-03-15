import numpy as np

ALPHA_PRIORS = np.array([0.62, 0.50, 0.52, 0.55])

DISC_GAIN = np.array([1.35,1.35,1.35,1.20])
CONT_GAIN = np.array([0.80,0.80,0.80,0.75])

def fuse(S_disc: np.ndarray, S_cont: np.ndarray, emotion_probs: np.ndarray, kappa: float = 1.5) -> np.ndarray:
    """
    Fuses discrete and continuous shift outputs using a weighted geometric mean.
    """
    max_prob = np.max(emotion_probs)

    c = max_prob ** kappa

    w_disc = c * ALPHA_PRIORS
    w_cont = 1 - w_disc

    S_disc_eff = DISC_GAIN * S_disc
    S_cont_eff = CONT_GAIN * S_cont

    S_final = w_disc * S_disc_eff + w_cont * S_cont_eff
    return S_final