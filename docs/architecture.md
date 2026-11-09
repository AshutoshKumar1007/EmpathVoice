# EmpathVoice — Architecture

This document explains how EmpathVoice is put together: the request lifecycle,
the component boundaries, the model lifecycle, the audio strategy, and the main
design decisions and their trade-offs.

## Goals that shaped the design

1. **Interpretability first.** The emotion→prosody mapping is explicit and
   deterministic, so every delivery decision can be traced to model outputs.
2. **Separation of concerns.** HTTP handling, schemas, configuration, model
   lifecycle, the analysis pipeline, TTS, and audio storage are independent and
   individually testable.
3. **No unnecessary infrastructure.** No database, no task queue, no cache
   server. The only background work is a self-contained audio TTL sweep.
4. **Testable without the heavy stack.** The pure pipeline and the API can be
   tested with fakes — no model downloads, no network.

## Component map

```
app/
  main.py            app factory, lifespan, routing, SPA serving
  core/
    config.py        Settings (env-driven, one source of truth)
    logging.py       logging setup
    errors.py        AppError hierarchy + JSON exception handlers
  schemas.py         Pydantic request/response contracts (the API surface)
  ml/
    emotion_model.py GoEmotions wrapper (lazy heavy imports, eval, device)
    vad_model.py     VAD-BERT wrapper (device fix, eval, 3-D head check)
    manager.py       lifecycle: load once, device, readiness, concurrency
  emotion/
    weights.py       discrete 28×4 mapping + matrix builder
    vad_mapping.py   continuous VAD→prosody mapping
    fusion.py        confidence-adaptive fusion (neutral-excluded)
    prosody.py       scale + clamp → bounded TTS parameters
    pipeline.py      pure orchestration: text → full analysis
  tts/
    pause_injection.py   clause/interval comma insertion
    synth.py             Synthesizer protocol + EdgeTTSSynthesizer
    voices.py            curated voice list
  audio/
    store.py         TTL, id-based storage with cleanup
  api/
    deps.py          accessors for the lifespan singletons
    health.py analyze.py synthesize.py audio.py voices.py
```

The dependency direction is one-way: `api` → `ml`/`emotion`/`tts`/`audio` →
`core`/`schemas`. The `emotion/pipeline.py` core imports none of torch, FastAPI,
or I/O — it takes already-loaded model objects, which is what makes it unit-
testable with lightweight fakes.

## Request lifecycle

### `POST /api/synthesize`

```
client
  → FastAPI route (validate SynthesizeRequest: non-empty, ≤ max chars)
  → check readiness (503 model_not_ready if models aren't loaded)
  → validate voice against the curated list (400 invalid_request)
  → ModelManager.analyze(text):
        acquire inference semaphore
        run_in_threadpool(pipeline.analyze):          # off the event loop
            probs = EmotionModel.infer(text)          # torch, no_grad, eval
            vad   = VADModel.infer(text)
            s_disc = probs @ W_emo
            s_cont = W_vad @ tanh(β·(2·vad−1))
            s_final = fuse(s_disc, s_cont, probs, neutral_index)
            prosody = clamp(scale(s_final))
            → AnalyzeResponse (emotions, vad, prosody, explanation)
  → inject_pauses(text, prosody.pause_level)
  → EdgeTTSSynthesizer.synthesize(...) → mp3 bytes     # async, external service
  → AudioStore.save(bytes) → audio_id
  → SynthesizeResponse(audio_id, audio_url=/api/audio/<id>, ttl, analysis)
```

`/api/analyze` is the same up to the `AnalyzeResponse` and returns there, so the
frontend can preview the analysis without paying for synthesis.

### Why inference runs in a threadpool

Model inference is synchronous, CPU/GPU-bound work. Running it directly in an
`async` endpoint would block the event loop and serialize all requests behind
each inference. It is dispatched with `run_in_threadpool`, and an
`asyncio.Semaphore` (size `INFERENCE_CONCURRENCY`, default 1) bounds how many
inferences touch the shared model instance at once — keeping the event loop
responsive without oversubscribing a single shared model.

## Model lifecycle and readiness

Three states are kept deliberately distinct:

- **Process health** (`/health`) — the server is up. Always 200. Suitable for a
  liveness probe.
- **Model readiness** (`/ready`) — the models have finished loading. Returns 200
  when ready, otherwise 503 with `model_status` ∈ {`loading`, `failed`} plus the
  resolved device and any error detail. Suitable for a readiness probe.
- **Request success** — an individual `/api/analyze` or `/api/synthesize` call.

At startup, the lifespan creates the singletons and kicks off model loading in a
**background thread**, so the server accepts connections immediately and
`/ready` flips to `ready` once loading completes. If loading fails (e.g. no
network to download weights), the status becomes `failed` with a detail message
rather than crashing the process; the UI surfaces this.

The models are loaded **once per process**, placed on the resolved device
(`cuda` if available else `cpu`, overridable), and switched to `eval()` so
dropout is disabled and inference is deterministic. The VAD model's output head
is validated to be 3-dimensional at load, turning a would-be deep NumPy error
into a clear startup failure.

## The fusion math

Given the discrete shift `s_disc`, continuous shift `s_cont`, and discrete
scores `p`:

```
c        = (max_{i ≠ neutral} p_i) ^ κ        # confidence, neutral excluded
w_disc   = c · α                               # per-dimension discrete weight
w_cont   = 1 − w_disc
s_final  = w_disc · (G_disc · s_disc) + w_cont · (G_cont · s_cont)
```

`α` (per-dimension priors), `G_disc`/`G_cont` (per-dimension gains), and `κ` are
fixed constants. High discrete confidence weights the categorical branch; low
confidence falls back toward the smoother VAD branch. Excluding `neutral` from
`c` prevents confidently-neutral text from suppressing the VAD branch (its
discrete prosody row is all zeros).

The final vector is scaled per dimension and clamped to `±40 Hz` (pitch),
`±35 %` (rate), `±30 %` (energy), and `[0.10, 0.92]` (pause level) before being
handed to edge-tts.

## Audio storage strategy

Generated audio is temporary and self-managing — no database, no object store:

- Files are addressed by an **opaque uuid id**, never by caller-supplied paths;
  the retrieval endpoint rejects anything that isn't a bare uuid, so there is no
  path traversal.
- A file remains retrievable for `AUDIO_TTL_SECONDS` (default 1 hour), supporting
  replay and repeated downloads — it is **not** deleted on first download.
- Expiry is derived from file mtime, so no in-memory index needs to survive.
- Cleanup happens at **startup** (wiping any leftovers from a previous run, since
  nothing is meant to persist) and **opportunistically** during requests,
  rate-limited so the directory isn't scanned on every call.

This is the entire "background processing" footprint of the app.

## Frontend

A React + TypeScript + Vite single-page app. State is handled with plain hooks
(no external state library). The API client is a thin typed wrapper whose types
mirror the backend schemas. Analysis is previewed live via a debounced
`/api/analyze` call as the user types; synthesis is an explicit action that
returns both the audio and the analysis that produced it. In development, Vite
proxies `/api` and the probe routes to the backend; in production, FastAPI serves
the built assets, so the app is same-origin in both cases.

## Notable design decisions and trade-offs

- **FastAPI retained, reorganized.** The original stack was sound; the work was
  boundaries and lifecycle, not a rewrite.
- **`/api` prefix + SPA at `/`.** Clean separation between the API and the
  single-page app, with probes at the root.
- **Pure pipeline core.** Keeping `pipeline.analyze` free of torch/HTTP is what
  makes the whole thing testable without model downloads; the cost is passing
  model objects in explicitly rather than importing singletons.
- **Heuristic prosody kept.** The hand-tuned mapping is the project's
  interpretable contribution. It is refined (neutral-aware fusion, pause
  robustness) but not replaced by a learned model, which would change the
  project's nature and require data and evaluation.
