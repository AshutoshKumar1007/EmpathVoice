from pathlib import Path

VOICE = "en-IN-NeerjaNeural"

OUT_DIR = Path("audio_outputs")
OUT_DIR.mkdir(exist_ok=True)

PROSODY_KEYS = ["pitch", "rate", "energy", "pauses"]

PROSODY_CLAMPS = {
    "pitch_hz": 40.0,
    "rate_pct": 35.0,
    "energy_pct": 30.0,
    "pause_gain": 0.45,
}