"""Application error types and their HTTP translation.

Routes raise these semantic errors; a single set of handlers turns them into
consistent JSON responses ``{"error": {"code", "message"}}`` with the right
status code. This keeps raw tracebacks out of responses and gives the frontend
a stable error shape to render.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Base class for expected, client-facing errors."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidRequestError(AppError):
    status_code = 400
    code = "invalid_request"


class ModelNotReadyError(AppError):
    status_code = 503
    code = "model_not_ready"


class SynthesisError(AppError):
    status_code = 502
    code = "synthesis_failed"


class AudioNotFoundError(AppError):
    status_code = 404
    code = "audio_not_found"


def _error_response(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message}},
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return _error_response(exc.status_code, exc.code, exc.message)

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # Last resort: never leak a traceback to the client.
        return _error_response(500, "internal_error", "An unexpected error occurred.")
