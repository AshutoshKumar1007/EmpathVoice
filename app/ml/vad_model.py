"""VAD-BERT continuous-affect model wrapper.

Fixes two correctness issues from the original:
  * the model is now moved to the target device (the original left it on CPU
    while sending inputs to the GPU — a crash waiting for any CUDA box);
  * the model is switched to eval mode so dropout no longer randomizes output.

The output head is validated to be 3-dimensional (valence, arousal, dominance)
so a mismatched checkpoint fails loudly at load instead of deep inside NumPy.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


class VADModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = "cpu"

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading VAD model %s on %s", self.model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        n_out = model.config.num_labels
        if n_out != 3:
            raise ValueError(
                f"VAD model '{self.model_name}' has {n_out} outputs; expected 3 "
                "(valence, arousal, dominance)."
            )
        model.to(device)
        model.eval()
        self._model = model
        self._device = device
        self._torch = torch

    def infer(self, text: str, temperature: float = 1.2) -> np.ndarray:
        """Return [valence, arousal, dominance], each in [0, 1]."""
        if self._model is None:
            raise RuntimeError("VADModel.infer called before load()")
        torch = self._torch
        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)
        with torch.no_grad():
            raw = self._model(**enc).logits.squeeze(0).cpu().numpy()
        return _sigmoid(raw / temperature)
