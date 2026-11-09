"""Controlled audio retrieval.

Serves generated audio by opaque id (never by filesystem path) and 404s once a
file has expired or if the id is malformed. ``Content-Disposition: attachment``
offers a clean download filename while the same url still streams for replay.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse

from app.api.deps import get_store
from app.audio.store import AudioStore
from app.core.errors import AudioNotFoundError

router = APIRouter(tags=["audio"])


@router.get("/audio/{audio_id}")
async def get_audio(audio_id: str, store: AudioStore = Depends(get_store)) -> FileResponse:
    path = store.path_for(audio_id)
    if path is None:
        raise AudioNotFoundError("Audio not found or expired.")
    return FileResponse(
        path,
        media_type="audio/mpeg",
        filename=f"empathvoice-{audio_id}.mp3",
        headers={"Cache-Control": "no-store"},
    )
