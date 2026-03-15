import edge_tts
from app.config import VOICE

async def synthesize(text,path,rate="+0%",pitch="+0Hz",volume="+0%"):

    comm = edge_tts.Communicate(
        text=text,
        voice=VOICE,
        rate=rate,
        pitch=pitch,
        volume=volume
    )

    await comm.save(path)