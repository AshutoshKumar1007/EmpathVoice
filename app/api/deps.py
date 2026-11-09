"""FastAPI dependency accessors for process-wide singletons.

The model manager, audio store, and synthesizer are created once in the app
lifespan and stashed on ``app.state``; these helpers expose them to routes.
"""

from __future__ import annotations

from fastapi import Request

from app.audio.store import AudioStore
from app.ml.manager import ModelManager
from app.tts.synth import Synthesizer


def get_manager(request: Request) -> ModelManager:
    return request.app.state.manager


def get_store(request: Request) -> AudioStore:
    return request.app.state.audio_store


def get_synthesizer(request: Request) -> Synthesizer:
    return request.app.state.synthesizer
