"""Centralized, environment-driven configuration.

Every tunable the app needs lives here as a single `Settings` object, read once
at import and injected where needed. Values come from environment variables
(prefix ``EMPATHVOICE_``) or an optional ``.env`` file; all have defaults, so
the app runs with zero configuration.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EMPATHVOICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Server ---
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Models ---
    emotion_model: str = "SamLowe/roberta-base-go_emotions"
    vad_model: str = "RobroKools/vad-bert"
    # None => auto-detect (cuda if available else cpu). Set "cpu"/"cuda" to force.
    device: str | None = None
    inference_concurrency: int = Field(default=1, ge=1, le=8)

    # --- Audio ---
    audio_dir: Path = Path("audio_outputs")
    audio_ttl_seconds: int = Field(default=3600, ge=1)
    default_voice: str = "en-IN-NeerjaNeural"

    # --- Limits ---
    max_text_chars: int = Field(default=2000, ge=1)

    # --- Logging ---
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
