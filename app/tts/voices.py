"""Curated set of edge-tts voices offered by the UI.

A small, hand-checked list keeps the voice picker predictable and avoids a slow
network call to enumerate the full edge-tts catalogue on every request. Extend
as needed; ids must be valid edge-tts voice names.
"""

from __future__ import annotations

from app.schemas import Voice

VOICES: list[Voice] = [
    Voice(id="en-IN-NeerjaNeural", label="Neerja (Indian English)", locale="en-IN", gender="Female"),
    Voice(id="en-IN-PrabhatNeural", label="Prabhat (Indian English)", locale="en-IN", gender="Male"),
    Voice(id="en-US-AriaNeural", label="Aria (US English)", locale="en-US", gender="Female"),
    Voice(id="en-US-GuyNeural", label="Guy (US English)", locale="en-US", gender="Male"),
    Voice(id="en-GB-SoniaNeural", label="Sonia (British English)", locale="en-GB", gender="Female"),
    Voice(id="en-GB-RyanNeural", label="Ryan (British English)", locale="en-GB", gender="Male"),
    Voice(id="en-AU-NatashaNeural", label="Natasha (Australian English)", locale="en-AU", gender="Female"),
]

VALID_VOICE_IDS = frozenset(v.id for v in VOICES)
