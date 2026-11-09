"""Liveness and readiness endpoints.

  * /health   — process is up and serving. Always 200. For container liveness.
  * /ready    — models are loaded and inference can run. 200 when ready, else
                503 with the current model status. For readiness gating.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response

from app.api.deps import get_manager
from app.ml.manager import ModelManager
from app.schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/ready", response_model=ReadinessResponse)
async def ready(
    response: Response, manager: ModelManager = Depends(get_manager)
) -> ReadinessResponse:
    if not manager.ready:
        response.status_code = 503
    return ReadinessResponse(
        ready=manager.ready,
        model_status=manager.status.value,
        device=manager.device,
        detail=manager.detail,
    )
