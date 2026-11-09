"""Development entry point: `python run.py`.

Reads host/port from settings. ``reload`` is on for local development; disable it
in production (or run uvicorn/gunicorn directly) so models load once.
"""

from __future__ import annotations

import uvicorn

from app.core.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )


if __name__ == "__main__":
    main()
