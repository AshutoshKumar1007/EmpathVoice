# Darvix Empathy Engine

Darvix Empathy Engine is a FastAPI-based text-to-speech service that tries to make generated voice sound emotionally appropriate.

In simple words:

- You give text.
- The system detects emotion.
- It changes vocal controls (pitch, rate, energy, pauses).
- It returns an MP3 with emotion-aware delivery.

## Core Contribution (Quick Overview)

The main idea is a deterministic hybrid emotion-to-prosody controller.

Instead of a black-box end-to-end prosody model, we explicitly combine two emotional views:

1. Discrete emotion space (28-label GoEmotions probabilities)
2. Continuous affective space (Valence-Arousal-Dominance)

These two state spaces are mapped to the same 4 vocal parameters and fused using confidence-adaptive weights.

This keeps the system interpretable, controllable, and easy to debug.

## Demo (Frontend Preview)

<!-- Replace with your demo screenshot/svg -->
![Demo Placeholder](frontend.png)

## High-Level Methodology

### 1) Emotion inference from two state spaces

- Discrete branch: GoEmotions gives a 28-dimensional multi-label probability vector.
- Continuous branch: VAD model gives valence, arousal, dominance in [0, 1].

### 2) Deterministic mappings to vocal parameter space

Both branches are mapped to the same prosody state:

$$
\mathbf{s} = [\text{pitch},\,\text{rate},\,\text{energy},\,\text{pauses}]
$$

- Discrete mapping: handcrafted matrix from 28 emotions to 4 prosody controls.
- Continuous mapping: VAD transform (with nonlinearity) to the same 4 controls.

This deterministic mapping layer is the central design choice of this project.

### 3) Confidence-adaptive fusion (key integration logic)

Let:

- $\mathbf{s}_{disc}$ = prosody from discrete branch
- $\mathbf{s}_{cont}$ = prosody from VAD branch
- $c = (\max_i p_i)^\kappa$ = confidence from top discrete probability

Fusion is:

$$
\mathbf{s}_{final} = \mathbf{w}_d \odot (\mathbf{G}_d \odot \mathbf{s}_{disc}) + \mathbf{w}_c \odot (\mathbf{G}_c \odot \mathbf{s}_{cont})
$$

with

$$
\mathbf{w}_d = c\,\boldsymbol{\alpha}, \quad \mathbf{w}_c = 1-\mathbf{w}_d
$$

Why this matters:

- High discrete confidence -> stronger categorical influence.
- Lower confidence -> smoother fallback from VAD branch.
- Per-dimension priors allow different trust levels for pitch/rate/energy/pauses.

## Post-Processing for TTS Control

The fused prosody vector is converted into bounded TTS controls for edge-tts:

- pitch shift (Hz)
- speaking rate (%)
- energy/volume (%)
- pause control scalar

Although SSML-level fine control is not integrated, the project still applies prosody modulation and includes an intra-sentence pause strategy module.

## Project Structure

```text
DarvixProject/
├── README.md
├── readme1.md
├── requirements.txt
├── run.py
├── app/
│   ├── config.py
│   ├── main.py
│   ├── models/
│   │   ├── emotion_model.py
│   │   └── vad_model.py
│   ├── services/
│   │   └── engine.py
│   ├── static/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── script.js
│   ├── tts/
│   │   ├── pause_injection.py
│   │   └── tts_engine.py
│   └── utils/
│       ├── emotion_weights.py
│       ├── fusion.py
│       ├── processing.py
│       └── vad_mapping.py
└── audio_outputs/
```

## Setup

```bash
cd DarvixProject
pip install -r requirements.txt
```

Recommended:

- Python 3.9+
- CUDA-enabled PyTorch (optional, for faster inference)

## Run

```bash
cd DarvixProject
python run.py
```

Service URL:

- http://127.0.0.1:8000

## API

### POST /speak

Input:

- query parameter: text

Example:

```bash
curl -X POST "http://127.0.0.1:8000/speak?text=I%20am%20proud%20of%20you"
```

Response:

```json
{
  "audio": "audio_out/<uuid>.mp3"
}
```

## Design Rationale

1. Hybrid emotion modeling
   - Categorical labels capture semantic emotion type.
   - VAD captures smooth affective intensity.

2. Deterministic mapping instead of black-box prosody generation
   - Better interpretability and direct control over behavior.

3. Confidence-adaptive fusion
   - Robust under ambiguity; branch contribution is input-dependent.

4. Practical deployability
   - FastAPI + edge-tts pipeline is simple and easy to demo.

