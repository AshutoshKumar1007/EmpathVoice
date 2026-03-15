import numpy as np
from app.config import PROSODY_CLAMPS

def process_prosody(p):

    pitch = np.clip(0.60*PROSODY_CLAMPS["pitch_hz"]*p[0],
                    -PROSODY_CLAMPS["pitch_hz"],
                    PROSODY_CLAMPS["pitch_hz"])

    rate = np.clip(0.85*PROSODY_CLAMPS["rate_pct"]*p[1],
                   -PROSODY_CLAMPS["rate_pct"],
                   PROSODY_CLAMPS["rate_pct"])

    energy = np.clip(PROSODY_CLAMPS["energy_pct"]*p[2],
                     -PROSODY_CLAMPS["energy_pct"],
                     PROSODY_CLAMPS["energy_pct"])

    pauses = np.clip(0.5 + PROSODY_CLAMPS["pause_gain"]*p[3],
                     0.10,0.92)

    return pitch,rate,energy,pauses