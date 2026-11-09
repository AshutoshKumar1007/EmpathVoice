// Types mirror the backend Pydantic schemas (app/schemas.py). Keep in sync.

export interface EmotionEstimate {
  label: string;
  score: number; // 0..1, uncalibrated
}

export interface VADEstimate {
  valence: number;
  arousal: number;
  dominance: number;
}

export interface ProsodyParameters {
  pitch_hz: number;
  rate_pct: number;
  energy_pct: number;
  pause_level: number;
}

export interface AnalyzeResponse {
  text: string;
  top_emotions: EmotionEstimate[];
  vad: VADEstimate;
  prosody: ProsodyParameters;
  explanation: string;
}

export interface SynthesizeResponse {
  audio_id: string;
  audio_url: string;
  expires_in_seconds: number;
  analysis: AnalyzeResponse;
}

export interface Voice {
  id: string;
  label: string;
  locale: string;
  gender: string;
}

export interface VoicesResponse {
  default: string;
  voices: Voice[];
}

export interface ApiError {
  error: { code: string; message: string };
}
