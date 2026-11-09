"""Hugging Face Spaces entry point using Gradio.

Exposes the EmpathVoice emotion-to-prosody pipeline and edge-tts synthesis
through an interactive, memory-efficient studio interface.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

# Disable experimental Gradio 5 Node.js SSR server to ensure stable container runtime
os.environ["GRADIO_SSR_MODE"] = "False"

import gradio as gr
import torch

from app.core.config import get_settings
from app.ml.manager import ModelManager, ModelStatus
from app.tts.pause_injection import inject_pauses
from app.tts.synth import EdgeTTSSynthesizer
from app.tts.voices import VOICES

# Initialize settings and synthesizer (singletons)
settings = get_settings()
synthesizer = EdgeTTSSynthesizer()

# Initialize and load model manager once in eval mode
manager = ModelManager(settings)
manager._load_blocking()
manager.status = ModelStatus.READY

# Build voice choices map: "Label" -> "id"
VOICE_MAP = {v.label: v.id for v in VOICES}
VOICE_LABELS = list(VOICE_MAP.keys())
DEFAULT_VOICE_LABEL = next(
    (v.label for v in VOICES if v.id == settings.default_voice),
    VOICE_LABELS[0],
)


def run_pipeline(text: str, voice_label: str) -> tuple[dict[str, float], str, str, str]:
    """Run emotion analysis and speech synthesis with minimal memory footprint."""
    clean_text = text.strip()
    if not clean_text:
        raise gr.Error("Please enter some text to analyze and synthesize.")
    if len(clean_text) > settings.max_text_chars:
        raise gr.Error(f"Text exceeds maximum length of {settings.max_text_chars} characters.")

    voice_id = VOICE_MAP.get(voice_label, settings.default_voice)

    # 1. Inference with no gradients to conserve memory
    with torch.inference_mode():
        analysis = asyncio.run(manager.analyze(clean_text))

    prosody = analysis.prosody
    vad = analysis.vad

    # 2. Audio Synthesis via edge-tts
    text_for_tts = inject_pauses(clean_text, prosody.pause_level)
    audio_bytes = asyncio.run(
        synthesizer.synthesize(
            text=text_for_tts,
            voice=voice_id,
            rate=f"{prosody.rate_pct:+.0f}%",
            pitch=f"{prosody.pitch_hz:+.0f}Hz",
            volume=f"{prosody.energy_pct:+.0f}%",
        )
    )

    # 3. Write audio to a temporary file for the Gradio audio component
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp.write(audio_bytes)
    tmp.close()

    # Format top emotions for gr.Label
    top_emotions_dict = {
        item.label.capitalize(): round(float(item.score), 4)
        for item in analysis.top_emotions[:5]
    }

    # Format VAD & Prosody metrics as Markdown cards
    metrics_md = f"""
### 📊 Affect & Delivery Estimates

| Metric | Estimated Value | Context / Direction |
| :--- | :--- | :--- |
| **Valence** | `{vad.valence:.2f}` | *Pleasantness (0 = negative, 1 = positive)* |
| **Arousal** | `{vad.arousal:.2f}` | *Energy / Intensity (0 = calm, 1 = excited)* |
| **Dominance** | `{vad.dominance:.2f}` | *Assertiveness (0 = submissive, 1 = dominant)* |
| **Pitch Shift** | `{prosody.pitch_hz:+.1f} Hz` | *Adjusts vocal pitch* |
| **Speech Rate** | `{prosody.rate_pct:+.1f}%` | *Adjusts speaking pace* |
| **Vocal Energy** | `{prosody.energy_pct:+.1f}%` | *Adjusts volume / dynamics* |
| **Pause Level** | `{prosody.pause_level:.2f}` | *Approximate clause pause density* |
"""

    return top_emotions_dict, metrics_md, analysis.explanation, tmp.name


# Build Gradio UI Blocks
custom_css = """
.container { max-width: 900px; margin: auto; }
.output-audio { margin-top: 10px; }
"""

with gr.Blocks(title="EmpathVoice Studio", theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.Markdown(
        """
# 🎙️ EmpathVoice
### Emotion-Aware Text-to-Speech with a Transparent Prosody Pipeline

EmpathVoice estimates emotional tone using **GoEmotions** and **VAD-BERT**, transparently derives speech parameters (pitch, rate, energy, pauses), and synthesizes expressive audio via Edge-TTS.
        """
    )

    with gr.Row():
        with gr.Column(scale=5):
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Type anything here...",
                value="I can't believe it — we actually won! This is the best day ever!",
                lines=4,
                max_lines=8,
            )
            with gr.Row():
                voice_dropdown = gr.Dropdown(
                    label="Voice",
                    choices=VOICE_LABELS,
                    value=DEFAULT_VOICE_LABEL,
                )
            submit_btn = gr.Button("Generate Voice", variant="primary", size="lg")

        with gr.Column(scale=5):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath",
                interactive=False,
                autoplay=True,
            )
            explanation_box = gr.Markdown(label="Pipeline Explanation")

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 🎭 Top Estimated Emotions")
            emotions_label = gr.Label(num_top_classes=5, label="Emotions")
        with gr.Column(scale=5):
            metrics_display = gr.Markdown()

    # Presets / Examples
    gr.Examples(
        examples=[
            ["I can't believe it — we actually won! This is the best day ever!", "Neerja (Indian English)"],
            ["I was really sorry to hear the difficult news yesterday. Please take all the time you need.", "Aria (US English)"],
            ["Wait, did you hear that sudden noise? Something is not right, and we need to check the servers immediately!", "Guy (US English)"],
            ["The system architecture consists of a FastAPI backend service and a Vite frontend running on local ports.", "Ryan (British English)"],
        ],
        inputs=[input_text, voice_dropdown],
    )

    submit_btn.click(
        fn=run_pipeline,
        inputs=[input_text, voice_dropdown],
        outputs=[emotions_label, metrics_display, explanation_box, audio_output],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
    )
