"""Text-to-speech synthesis behind a small interface.

The ``Synthesizer`` protocol lets tests inject a fake that returns dummy bytes,
so the API and audio-store logic can be exercised without network access or the
edge-tts dependency. ``EdgeTTSSynthesizer`` is the real implementation; it
streams audio chunks and returns the full MP3 as bytes (decoupled from storage).
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.core.errors import SynthesisError

logger = logging.getLogger(__name__)


class Synthesizer(Protocol):
    async def synthesize(
        self, text: str, voice: str, rate: str, pitch: str, volume: str
    ) -> bytes: ...


class EdgeTTSSynthesizer:
    async def synthesize(
        self, text: str, voice: str, rate: str, pitch: str, volume: str
    ) -> bytes:
        import edge_tts

        try:
            communicate = edge_tts.Communicate(
                text=text, voice=voice, rate=rate, pitch=pitch, volume=volume
            )
            chunks = bytearray()
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    chunks.extend(chunk["data"])
        except Exception as exc:  # noqa: BLE001 - surface as a clean 502
            logger.warning("edge-tts synthesis failed: %s", exc)
            raise SynthesisError("Speech synthesis failed. Please try again.") from exc

        if not chunks:
            raise SynthesisError("Speech synthesis produced no audio.")
        return bytes(chunks)
