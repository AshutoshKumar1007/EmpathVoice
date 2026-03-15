import numpy as np
from app.models.emotion_model import LABELS_GO

# Format: [pitch, rate, energy, pauses]
weights_emotions = {

    
    "admiration":      [ 0.15,  0.08,  0.18, -0.10],
    "amusement":       [ 0.25,  0.22,  0.20, -0.20],
    "excitement":      [ 0.45,  0.42,  0.45, -0.45],
    "joy":             [ 0.38,  0.30,  0.35, -0.32],
    "pride":           [ 0.12, -0.08,  0.30, -0.05],
    "love":            [ 0.15, -0.12,  0.08,  0.10],

    
    "approval":        [ 0.08,  0.05,  0.10, -0.05],
    "caring":          [ 0.05, -0.15,  0.05,  0.15],
    "gratitude":       [ 0.18,  0.05,  0.15, -0.08],
    "optimism":        [ 0.20,  0.12,  0.18, -0.12],
    "relief":          [ 0.05, -0.20, -0.10,  0.18],

    
    "anger":           [ 0.15,  0.32,  0.50, -0.38],
    "annoyance":       [ 0.05,  0.15,  0.25, -0.12],
    "disgust":         [-0.12, -0.10,  0.18,  0.20],
    "fear":            [ 0.25,  0.28,  0.10,  0.18],
    "nervousness":     [ 0.15,  0.18, -0.08,  0.25],

    
    "disappointment":  [-0.22, -0.20, -0.18,  0.28],
    "disapproval":     [-0.08,  0.05,  0.18,  0.10],
    "embarrassment":   [-0.18, -0.15, -0.22,  0.25],
    "grief":           [-0.42, -0.45, -0.40,  0.50],
    "remorse":         [-0.25, -0.22, -0.20,  0.32],
    "sadness":         [-0.32, -0.30, -0.28,  0.38],

    
    "confusion":       [ 0.12, -0.15, -0.08,  0.25],
    "curiosity":       [ 0.18,  0.10,  0.08,  0.05],
    "desire":          [ 0.12, -0.05,  0.10, -0.05],
    "realization":     [ 0.10, -0.08,  0.05,  0.15],
    "surprise":        [ 0.35,  0.12,  0.28,  0.08],

    "neutral":         [0.0, 0.0, 0.0, 0.0],
}

# Build weight matrix aligned with model label order
W_emo = np.array(
    [weights_emotions[label.lower()] for label in LABELS_GO],
    dtype=np.float64
)


def compute_S_disc(probs):
    """
    Discrete prosody shift.

    S_disc = probs^T @ W_emo
    """
    return probs @ W_emo