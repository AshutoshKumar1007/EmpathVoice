import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

VAD_MODEL = "RobroKools/vad-bert"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok_vad = AutoTokenizer.from_pretrained(VAD_MODEL)
mdl_vad = AutoModelForSequenceClassification.from_pretrained(VAD_MODEL)

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

@torch.no_grad()
def infer_vad(text, temperature=1.2):

    enc = tok_vad(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    raw = mdl_vad(**enc).logits.squeeze(0).cpu().numpy()

    vad = _sigmoid(raw / temperature)

    return vad

