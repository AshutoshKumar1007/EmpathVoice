"""Minimal, dependency-free logging setup.

One place configures the root logger so every module can simply call
``logging.getLogger(__name__)`` and get consistent, timestamped output.
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # edge-tts / urllib3 are chatty at DEBUG; keep them at WARNING.
    logging.getLogger("edge_tts").setLevel(logging.WARNING)
