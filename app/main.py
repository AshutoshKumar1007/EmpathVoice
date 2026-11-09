"""Application factory and lifespan wiring.

Startup sequence (distinct, observable stages):
  1. process starts, logging configured, singletons created, audio dir purged;
  2. model loading kicked off in the background (server is already accepting
     connections and /health is green);
  3. /ready flips to true once both models finish loading.

Shutdown performs a final audio cleanup.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api import analyze, audio, health, synthesize, voices
from app.audio.store import AudioStore
from app.core.config import get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging
from app.ml.manager import ModelManager
from app.tts.synth import EdgeTTSSynthesizer

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting EmpathVoice")

    app.state.audio_store = AudioStore(settings.audio_dir, settings.audio_ttl_seconds)
    app.state.audio_store.purge_all()  # startup cleanup; nothing persists across runs
    app.state.synthesizer = EdgeTTSSynthesizer()
    app.state.manager = ModelManager(settings)
    await app.state.manager.start_loading()  # non-blocking; readiness flips later

    yield

    app.state.audio_store.purge_all()
    logger.info("EmpathVoice stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="EmpathVoice",
        version="0.2.0",
        summary="Emotion-aware text-to-speech with a transparent prosody pipeline.",
        lifespan=lifespan,
    )

    # Dev convenience: allow the Vite dev server origin. Same-origin in prod.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    register_exception_handlers(app)

    # Liveness/readiness at root (conventional for probes); everything else under /api.
    app.include_router(health.router)
    for module in (analyze, synthesize, audio, voices):
        app.include_router(module.router, prefix="/api")

    # Serve the built React app when present (frontend/dist). API routes and
    # /docs are registered first, so they take precedence over this catch-all
    # mount. If the frontend isn't built, fall back to the API docs.
    dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
    else:
        @app.get("/", include_in_schema=False)
        async def root() -> RedirectResponse:
            return RedirectResponse(url="/docs")

    return app


app = create_app()
