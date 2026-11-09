"""Available voices endpoint for the frontend voice picker."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import get_settings
from app.schemas import VoicesResponse
from app.tts.voices import VOICES

router = APIRouter(tags=["voices"])


@router.get("/voices", response_model=VoicesResponse)
async def list_voices() -> VoicesResponse:
    return VoicesResponse(default=get_settings().default_voice, voices=VOICES)
