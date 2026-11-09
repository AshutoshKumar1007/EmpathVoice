"""Request/response contracts for the public API.

Kept in one module because the surface is small. These models define the
OpenAPI schema, validate input, and give the frontend typed shapes to mirror.

Terminology note: emotion/VAD numbers are *model estimates*, not calibrated
probabilities or ground-truth assessments of anyone's feelings. Field names and
descriptions reflect that deliberately.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.core.config import get_settings


def _validate_text(value: str) -> str:
    text = value.strip()
    if not text:
        raise ValueError("text must not be empty")
    limit = get_settings().max_text_chars
    if len(text) > limit:
        raise ValueError(f"text must be at most {limit} characters")
    return text


# --- Shared sub-models -------------------------------------------------------

class EmotionEstimate(BaseModel):
    """One discrete emotion label and its (uncalibrated) model score."""

    label: str
    score: float = Field(description="Sigmoid score in [0, 1]; not a calibrated probability.")


class VADEstimate(BaseModel):
    """Continuous affect estimate, each component in [0, 1]."""

    valence: float
    arousal: float
    dominance: float


class ProsodyParameters(BaseModel):
    """Final speech-delivery adjustments applied to the TTS voice."""

    pitch_hz: float = Field(description="Pitch shift in Hz (signed).")
    rate_pct: float = Field(description="Speaking-rate change in percent (signed).")
    energy_pct: float = Field(description="Loudness/energy change in percent (signed).")
    pause_level: float = Field(description="Pause strength scalar in [0, 1].")


# --- /analyze ----------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str

    _check_text = field_validator("text")(_validate_text)


class AnalyzeResponse(BaseModel):
    text: str
    top_emotions: list[EmotionEstimate]
    vad: VADEstimate
    prosody: ProsodyParameters
    explanation: str = Field(
        description="Plain-language summary of how the estimates shaped delivery."
    )


# --- /synthesize -------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    text: str
    voice: str | None = Field(default=None, description="Voice id; defaults to server default.")
    expressiveness: float = Field(
        default=1.0,
        ge=0.0,
        le=1.5,
        description="Global scale on prosody deltas before clamping (0 = flat, 1 = normal).",
    )

    _check_text = field_validator("text")(_validate_text)


class SynthesizeResponse(BaseModel):
    audio_id: str
    audio_url: str
    expires_in_seconds: int
    analysis: AnalyzeResponse


# --- Health / readiness / voices --------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"


class ReadinessResponse(BaseModel):
    ready: bool
    model_status: str = Field(description="loading | ready | failed")
    device: str | None = None
    detail: str | None = None


class Voice(BaseModel):
    id: str
    label: str
    locale: str
    gender: str


class VoicesResponse(BaseModel):
    default: str
    voices: list[Voice]
