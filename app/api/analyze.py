"""Emotion-analysis endpoint.

Returns the full emotion -> prosody analysis without synthesizing audio, so the
frontend can preview *why* the voice will sound a certain way before committing
to a (slower) synthesis call.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_manager
from app.core.errors import ModelNotReadyError
from app.ml.manager import ModelManager
from app.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    payload: AnalyzeRequest, manager: ModelManager = Depends(get_manager)
) -> AnalyzeResponse:
    if not manager.ready:
        raise ModelNotReadyError(
            f"Models are not ready (status: {manager.status.value})."
        )
    return await manager.analyze(payload.text)
