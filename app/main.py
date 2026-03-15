from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.services.engine import run_empathy_engine
from app.tts.pause_injection import inject_pauses

import uuid
from pathlib import Path
from app.tts.tts_engine import synthesize

app = FastAPI()

audio_dir = Path("audio_outputs")
audio_dir.mkdir(exist_ok=True)

@app.post("/speak")
async def speak(text:str):

    pitch,rate,energy,pauses = run_empathy_engine(text)
    text_for_tts = inject_pauses(text, pauses)

    file=f"audio_outputs/{uuid.uuid4()}.mp3"

    await synthesize(
        text_for_tts,
        file,
        rate=f"{rate:+.0f}%",
        pitch=f"{pitch:+.0f}Hz",
        volume=f"{energy:+.0f}%"
    )

    return {"audio":file}


app.mount("/audio_outputs", StaticFiles(directory="audio_outputs"), name="audio_outputs")
app.mount("/", StaticFiles(directory="app/static", html=True), name="static")