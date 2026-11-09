"""Speech-synthesis endpoint.

Runs the same analysis as /analyze, converts the prosody parameters into
edge-tts controls, injects clause pauses, synthesizes audio, and stores it under
a temporary id. Returns the audio url plus the analysis that produced it, so the
client never has to call /analyze separately for a one-shot generation.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_manager, get_store, get_synthesizer
from app.audio.store import AudioStore
from app.core.config import get_settings
from app.core.errors import InvalidRequestError, ModelNotReadyError
from app.ml.manager import ModelManager
from app.schemas import SynthesizeRequest, SynthesizeResponse
from app.tts.pause_injection import inject_pauses
from app.tts.synth import Synthesizer
from app.tts.voices import VALID_VOICE_IDS

router = APIRouter(tags=["synthesis"])


@router.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(
    payload: SynthesizeRequest,
    manager: ModelManager = Depends(get_manager),
    store: AudioStore = Depends(get_store),
    synthesizer: Synthesizer = Depends(get_synthesizer),
) -> SynthesizeResponse:
    settings = get_settings()

    if not manager.ready:
        raise ModelNotReadyError(f"Models are not ready (status: {manager.status.value}).")

    voice = payload.voice or settings.default_voice
    if voice not in VALID_VOICE_IDS:
        raise InvalidRequestError(f"Unknown voice '{voice}'.")

    analysis = await manager.analyze(payload.text, expressiveness=payload.expressiveness)
    prosody = analysis.prosody

    text_for_tts = inject_pauses(payload.text, prosody.pause_level)
    audio = await synthesizer.synthesize(
        text=text_for_tts,
        voice=voice,
        rate=f"{prosody.rate_pct:+.0f}%",
        pitch=f"{prosody.pitch_hz:+.0f}Hz",
        volume=f"{prosody.energy_pct:+.0f}%",
    )

    audio_id = store.save(audio)
    return SynthesizeResponse(
        audio_id=audio_id,
        audio_url=f"/api/audio/{audio_id}",
        expires_in_seconds=settings.audio_ttl_seconds,
        analysis=analysis,
    )
