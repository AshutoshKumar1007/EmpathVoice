import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

GO_MODEL = "SamLowe/roberta-base-go_emotions"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tok_go = AutoTokenizer.from_pretrained(GO_MODEL)
mdl_go = AutoModelForSequenceClassification.from_pretrained(GO_MODEL).to(device)

LABELS_GO = [mdl_go.config.id2label[i] for i in range(mdl_go.config.num_labels)]

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

@torch.no_grad()
def infer_emotions(text, temperature=1.1):

    enc = tok_go(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    logits = mdl_go(**enc).logits.squeeze(0).cpu().numpy()

    return _sigmoid(logits / temperature)