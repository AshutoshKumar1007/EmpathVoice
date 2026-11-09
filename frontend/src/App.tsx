import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiRequestError } from "./api";
import { MAX_CHARS } from "./constants";
import { useDebounced } from "./hooks";
import type { AnalyzeResponse, Voice } from "./types";
import { TextComposer } from "./components/TextComposer";
import { VoicePicker } from "./components/VoicePicker";
import { AnalysisPanel } from "./components/AnalysisPanel";
import { AudioPlayer } from "./components/AudioPlayer";
import { Banner } from "./components/Banner";

type ModelState = "loading" | "ready" | "failed";

async function fetchReady(): Promise<ModelState> {
  try {
    const r = await fetch("/ready");
    const j = await r.json();
    return (j.model_status as ModelState) ?? "loading";
  } catch {
    return "loading";
  }
}

export default function App() {
  const [text, setText] = useState("");
  const [voices, setVoices] = useState<Voice[]>([]);
  const [voice, setVoice] = useState("");
  const [modelState, setModelState] = useState<ModelState>("loading");

  const [analysis, setAnalysis] = useState<AnalyzeResponse | null>(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [audio, setAudio] = useState<{ url: string; expiresIn: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const debouncedText = useDebounced(text.trim(), 500);
  const analyzeSeq = useRef(0); // guards against out-of-order responses

  const ready = modelState === "ready";
  const trimmed = text.trim();
  const valid = trimmed.length > 0 && trimmed.length <= MAX_CHARS;

  // --- Load voices + poll readiness on mount ---
  useEffect(() => {
    api
      .getVoices()
      .then((v) => {
        setVoices(v.voices);
        setVoice(v.default);
      })
      .catch(() => setError("Could not load voices."));

    let active = true;
    const poll = async () => {
      const s = await fetchReady();
      if (!active) return;
      setModelState(s);
      if (s === "loading") setTimeout(poll, 2000);
    };
    poll();
    return () => {
      active = false;
    };
  }, []);

  // --- Debounced live analysis preview ---
  useEffect(() => {
    if (!ready || !debouncedText || debouncedText.length > MAX_CHARS) {
      if (!debouncedText) setAnalysis(null);
      return;
    }
    const seq = ++analyzeSeq.current;
    setAnalyzing(true);
    api
      .analyze(debouncedText)
      .then((res) => {
        if (seq === analyzeSeq.current) setAnalysis(res);
      })
      .catch((e: ApiRequestError) => {
        if (seq === analyzeSeq.current) setError(e.message);
      })
      .finally(() => {
        if (seq === analyzeSeq.current) setAnalyzing(false);
      });
  }, [debouncedText, ready]);

  // --- Generate speech ---
  const generate = useCallback(async () => {
    if (!valid || !ready) return;
    setGenerating(true);
    setError(null);
    try {
      const res = await api.synthesize(trimmed, voice);
      setAnalysis(res.analysis);
      setAudio({ url: res.audio_url, expiresIn: res.expires_in_seconds });
    } catch (e) {
      const err = e as ApiRequestError;
      setError(err.message);
    } finally {
      setGenerating(false);
    }
  }, [valid, ready, trimmed, voice]);

  const busy = analyzing || generating;

  return (
    <div className="page">
      <main className="card">
        <header className="card-header">
          <div className="icon-wrap" aria-hidden>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z" />
              <path d="M9 15.5c.8.8 1.9 1.3 3 1.3s2.2-.5 3-1.3" />
              <circle cx="9" cy="10" r="0.9" fill="currentColor" />
              <circle cx="15" cy="10" r="0.9" fill="currentColor" />
            </svg>
          </div>
          <div>
            <h1>EmpathVoice</h1>
            <p className="subtitle">Emotion-aware text to speech</p>
          </div>
        </header>

        {modelState === "loading" && (
          <Banner kind="info" message="Models are warming up — this can take a moment on first run." />
        )}
        {modelState === "failed" && (
          <Banner
            kind="error"
            message="The models failed to load. Check the server logs; analysis and synthesis are unavailable."
          />
        )}
        {error && <Banner kind="error" message={error} onDismiss={() => setError(null)} />}

        <TextComposer value={text} onChange={setText} disabled={generating} />

        <div className="controls">
          <VoicePicker voices={voices} value={voice} onChange={setVoice} disabled={generating} />
          <button
            type="button"
            className="generate-btn"
            onClick={generate}
            disabled={!valid || !ready || generating}
          >
            {generating ? "Generating…" : "Generate Voice"}
          </button>
        </div>

        {/* States: empty / analyzing / analysis preview / audio */}
        {!trimmed && !analysis && (
          <p className="empty-hint">
            Enter some text to see its estimated emotional delivery, then generate speech.
          </p>
        )}

        {analyzing && !analysis && <p className="empty-hint">Analyzing…</p>}

        {analysis && <AnalysisPanel analysis={analysis} stale={busy} />}

        {audio && <AudioPlayer url={audio.url} expiresIn={audio.expiresIn} />}
      </main>

      <footer className="page-footer">
        Emotion and VAD values are model estimates, not definitive readings of feeling.
      </footer>
    </div>
  );
}
