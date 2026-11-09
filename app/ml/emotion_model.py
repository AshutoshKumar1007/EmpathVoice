"""GoEmotions discrete-emotion model wrapper.

Heavy imports (torch/transformers) happen inside ``load`` so this module — and
anything that imports it — stays importable without the ML stack installed.
The model is placed on the target device and switched to eval mode (dropout
off) so inference is deterministic.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))


class EmotionModel:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._device = "cpu"
        self.labels: list[str] = []

    def load(self, device: str) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("Loading emotion model %s on %s", self.model_name, device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.to(device)
        model.eval()  # disable dropout -> deterministic inference
        self._model = model
        self._device = device
        self.labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
        self._torch = torch

    def infer(self, text: str, temperature: float = 1.1) -> np.ndarray:
        """Return per-label sigmoid scores (multi-label, not a distribution)."""
        if self._model is None:
            raise RuntimeError("EmotionModel.infer called before load()")
        torch = self._torch
        enc = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self._device)
        with torch.no_grad():
            logits = self._model(**enc).logits.squeeze(0).cpu().numpy()
        return _sigmoid(logits / temperature)
