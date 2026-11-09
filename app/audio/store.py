"""Temporary, TTL-based audio storage.

Design goals (no Redis/Celery, nothing to persist across restarts):
  * files are addressed by an opaque uuid id, never by caller-supplied paths;
  * a file stays retrievable for ``ttl_seconds`` (replay + repeated downloads),
    i.e. it is NOT deleted on first download;
  * expiry is derived from file mtime, so no in-memory index must survive;
  * cleanup happens (a) once at startup — wiping any leftovers from a previous
    run — and (b) opportunistically, rate-limited, during normal requests.

This is intentionally simple and self-contained; it is the entire "background
processing" footprint of the app.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

_UUID_HEX = frozenset("0123456789abcdef-")
_SWEEP_MIN_INTERVAL_S = 30.0  # don't scan the dir more often than this


class AudioStore:
    def __init__(self, directory: Path, ttl_seconds: int) -> None:
        self._dir = Path(directory)
        self._ttl = ttl_seconds
        self._last_sweep = 0.0
        self._dir.mkdir(parents=True, exist_ok=True)

    # --- lifecycle ----------------------------------------------------------

    def purge_all(self) -> int:
        """Delete every stored file. Called at startup (no cross-run persistence)."""
        removed = 0
        for f in self._dir.glob("*.mp3"):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            logger.info("Startup cleanup removed %d leftover audio file(s)", removed)
        return removed

    # --- write / read -------------------------------------------------------

    def save(self, data: bytes) -> str:
        self._maybe_sweep()
        audio_id = uuid.uuid4().hex
        (self._dir / f"{audio_id}.mp3").write_bytes(data)
        return audio_id

    def path_for(self, audio_id: str) -> Path | None:
        """Return the file path if the id is valid, present, and unexpired."""
        self._maybe_sweep()
        if not self._is_valid_id(audio_id):
            return None
        path = self._dir / f"{audio_id}.mp3"
        if not path.is_file():
            return None
        if self._is_expired(path):
            self._safe_unlink(path)
            return None
        return path

    # --- cleanup ------------------------------------------------------------

    def _maybe_sweep(self) -> None:
        now = time.time()
        if now - self._last_sweep < _SWEEP_MIN_INTERVAL_S:
            return
        self._last_sweep = now
        for f in self._dir.glob("*.mp3"):
            if self._is_expired(f):
                self._safe_unlink(f)

    def _is_expired(self, path: Path) -> bool:
        try:
            return (time.time() - path.stat().st_mtime) > self._ttl
        except OSError:
            return False

    @staticmethod
    def _safe_unlink(path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            pass

    @staticmethod
    def _is_valid_id(audio_id: str) -> bool:
        # Reject anything that isn't a bare uuid hex -> no path traversal.
        if not audio_id or not set(audio_id) <= _UUID_HEX:
            return False
        try:
            uuid.UUID(hex=audio_id)
            return True
        except ValueError:
            return False
