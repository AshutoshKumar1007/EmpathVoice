from app.models.emotion_model import infer_emotions
from app.models.vad_model import infer_vad

from app.utils.emotion_weights import compute_S_disc
from app.utils.vad_mapping import compute_S_cont
from app.utils.fusion import fuse

from app.utils.processing import process_prosody


def run_empathy_engine(text):

    probs = infer_emotions(text)

    vad = infer_vad(text)

    S_disc = compute_S_disc(probs)

    S_cont = compute_S_cont(vad)

    S_final = fuse(S_disc,S_cont,probs)

    pitch,rate,energy,pauses = process_prosody(S_final)

    return pitch,rate,energy,pauses