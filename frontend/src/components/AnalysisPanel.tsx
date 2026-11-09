import type { AnalyzeResponse } from "../types";
import { ENERGY_PCT_MAX, PITCH_HZ_MAX, RATE_PCT_MAX } from "../constants";
import { Meter } from "./Meter";

interface Props {
  analysis: AnalyzeResponse;
  stale?: boolean; // dimmed while a newer analysis is loading
}

const signed = (v: number, unit: string) => `${v >= 0 ? "+" : ""}${v}${unit}`;

export function AnalysisPanel({ analysis, stale }: Props) {
  const { top_emotions, vad, prosody, explanation } = analysis;

  return (
    <section className={`analysis ${stale ? "stale" : ""}`} aria-label="Emotion analysis">
      <p className="analysis-explain">{explanation}</p>

      <div className="analysis-grid">
        <div className="analysis-block">
          <h3>Estimated tone</h3>
          {top_emotions.map((e) => (
            <Meter key={e.label} label={e.label} fraction={e.score} value={e.score.toFixed(2)} />
          ))}
        </div>

        <div className="analysis-block">
          <h3>Affect (VAD)</h3>
          <Meter label="Valence" fraction={vad.valence} value={vad.valence.toFixed(2)} />
          <Meter label="Arousal" fraction={vad.arousal} value={vad.arousal.toFixed(2)} />
          <Meter label="Dominance" fraction={vad.dominance} value={vad.dominance.toFixed(2)} />
        </div>

        <div className="analysis-block">
          <h3>Delivery</h3>
          <Meter
            label="Pitch"
            fraction={Math.abs(prosody.pitch_hz) / PITCH_HZ_MAX}
            value={signed(prosody.pitch_hz, " Hz")}
            signed
            positive={prosody.pitch_hz >= 0}
          />
          <Meter
            label="Rate"
            fraction={Math.abs(prosody.rate_pct) / RATE_PCT_MAX}
            value={signed(prosody.rate_pct, "%")}
            signed
            positive={prosody.rate_pct >= 0}
          />
          <Meter
            label="Energy"
            fraction={Math.abs(prosody.energy_pct) / ENERGY_PCT_MAX}
            value={signed(prosody.energy_pct, "%")}
            signed
            positive={prosody.energy_pct >= 0}
          />
          <Meter
            label="Pauses"
            fraction={prosody.pause_level}
            value={prosody.pause_level.toFixed(2)}
          />
        </div>
      </div>
    </section>
  );
}
