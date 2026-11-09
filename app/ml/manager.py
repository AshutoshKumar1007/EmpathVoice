"""Model lifecycle, device handling, and inference concurrency.

Responsibilities, deliberately in one place:
  * detect the device once (auto: cuda if available else cpu; overridable);
  * load both models exactly once per process, in eval mode, on that device;
  * expose a three-state readiness (LOADING -> READY | FAILED) that is distinct
    from "the process is up" (health) and from "a request succeeded";
  * run blocking inference off the event loop, with a semaphore bounding how
    many inferences touch the shared model instance at once.

Loading runs in a background thread kicked off at startup, so the server accepts
connections immediately and ``/ready`` flips to true when the models finish.
"""

from __future__ import annotations

import asyncio
import enum
import logging

from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.emotion import pipeline
from app.emotion.weights import build_emotion_matrix
from app.ml.emotion_model import EmotionModel
from app.ml.vad_model import VADModel
from app.schemas import AnalyzeResponse

logger = logging.getLogger(__name__)


class ModelStatus(str, enum.Enum):
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


def _resolve_device(override: str | None) -> str:
    if override:
        return override
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # torch missing or broken -> degrade to cpu label
        return "cpu"


class ModelManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.status = ModelStatus.LOADING
        self.device: str | None = None
        self.detail: str | None = None
        self._emotion: EmotionModel | None = None
        self._vad: VADModel | None = None
        self._weight_matrix = None
        self._semaphore = asyncio.Semaphore(settings.inference_concurrency)

    @property
    def ready(self) -> bool:
        return self.status is ModelStatus.READY

    async def start_loading(self) -> None:
        """Kick off model loading in the background; returns immediately."""
        asyncio.create_task(self._load_async())

    async def _load_async(self) -> None:
        try:
            await run_in_threadpool(self._load_blocking)
            self.status = ModelStatus.READY
            logger.info("Models ready on %s", self.device)
        except Exception as exc:  # noqa: BLE001 - record, don't crash the app
            self.status = ModelStatus.FAILED
            self.detail = str(exc)
            logger.exception("Model loading failed: %s", exc)

    def _load_blocking(self) -> None:
        self.device = _resolve_device(self._settings.device)
        emotion = EmotionModel(self._settings.emotion_model)
        emotion.load(self.device)
        vad = VADModel(self._settings.vad_model)
        vad.load(self.device)
        self._weight_matrix = build_emotion_matrix(emotion.labels)
        self._emotion = emotion
        self._vad = vad

    async def analyze(self, text: str, expressiveness: float = 1.0) -> AnalyzeResponse:
        if not self.ready:
            raise RuntimeError("Models are not ready")
        async with self._semaphore:
            return await run_in_threadpool(
                pipeline.analyze,
                text,
                self._emotion,
                self._vad,
                self._weight_matrix,
                expressiveness,
            )
