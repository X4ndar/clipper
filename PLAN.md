# Plan

## Architecture

One Python package (`clipper`), one pipeline core, thin front ends (CLI now; HTTP API, web UI,
MCP server and agent skill in M6). Every stage reads and writes files in a job folder, so any
stage can be re-run alone. SQLite (`$CLIPPER_HOME/clipper.db`) holds job history, stage runs,
every LLM call (prompt version, model, params, output, usage, cost), ASR usage and eval runs.

```
ingest -> transcribe -> quality -> candidates -> judge -> snap -> reframe -> captions -> render -> metadata -> report
          (cached by    (blocks    (pass 1,      (pass 2,  (word    (faces,    (ASS per   (ffmpeg   (title,     (HTML,
           audio hash)   before     cheap         strong    bounds,  cuts,      clip +     9:16,     tags,       all
                         LLM $)     model)        model)    RMS)     crop path) hook)      burn-in)  hook)       candidates)
```

Backends sit behind small interfaces so they can be swapped and tested offline:

- `Transcriber`: `assemblyai` (v1), `fixture` (read a word-level JSON; used by tests and for
  bring-your-own transcripts). Local backends (WhisperX, Qwen3-ASR, Nemotron) come later.
- `LLMClient`: `anthropic` (structured outputs via `messages.parse`), `fake` (deterministic
  heuristics, offline tests). Every call is logged; identical calls are served from the log.

## Repo layout

```
src/clipper/
  cli.py            Typer CLI (run, stage, jobs, report, rerender, preview, usage, config, eval)
  config.py         Pydantic config, precedence: CLI > UI settings.json > config.yaml > defaults
  pipeline.py       stage registry, job runner, --stop-at / --from-stage
  jobs.py           job folder layout and job.json
  db.py             SQLite schema and helpers
  media.py          ffprobe, audio extraction, hashing, WAV access, RMS
  transcript.py     data model, sentence segmentation, quality check, SRT export
  transcribe/       base.py, assemblyai.py, fixture.py
  llm/              client.py (anthropic + fake), schemas.py, prompts/*.md (versioned)
  select.py         pass 1 candidates, pass 2 judging, gates, weights, dedupe, final pick
  boundaries.py     word-range mapping, clause-offset validation, filler drop, snapping
  captions.py       caption layout, ASS generation (word highlight / karaoke), hook overlay
  reframe.py        shot cuts, face detection (MediaPipe), smoothed crop path, framing metrics
  render.py         ffmpeg cut, fixed or moving crop, fades, burn-in, single-frame preview
  report.py         HTML review report
  evaluation.py     gold set format, label sheet, precision@k and other metrics
tests/              unit tests + integration test on a generated sample video
scripts/            make_sample.py (synthetic talking video with flite TTS)
assets/fonts/       OFL fonts (Noto Sans, Noto Sans Arabic) + license
```

## Milestones

M1 (this delivery): CLI, AssemblyAI transcription with cache and credit tracking, transcript
quality gate, pass 1 + pass 2 with gates and dedupe, boundary snapping, center-crop render,
review report, eval harness (gold format, label sheet, precision@k, run log), AGENTS.md.
M2 (this delivery, basic): word-highlight ASS captions burned in, hook overlay, editable captions
JSON + re-render, single-frame style preview, RTL font.
M3: reframing. Done: sampled face detection (MediaPipe full-range, CPU), shot-cut detection, a
hold-still / smooth-pan crop path per shot, framing metrics vs center crop. mouth-movement speaker pick, camera-over-gameplay
stack layout for streams. Next: measure the speaker pick (or move to LR-ASD, MIT), two-person layouts.
M4 (started): audio loudness and chat-rate signals, peak-first selection for streams, `--direction`,
prompts chosen by content type. Remaining: visual check, pairwise ranking, A/B, batch mode, job history UI.
M5: YouTube upload (private, human approval), then performance loop.
M6: FastAPI + web UI + SSE progress, API keys, MCP server, SKILL.md, packaging.

## Assumptions

AssemblyAI's REST API is used directly through `httpx` (no SDK) so `speech_models` is always
sent explicitly; word times arrive in milliseconds and are stored in seconds. Speaker labels
default to on for `podcast`/`interview` content or `--speakers > 1`, off otherwise. Current
Claude models (`claude-haiku-5-5`, `claude-sonnet-5-5`) reject custom `temperature`, so the
brief's "low temperature" is replaced by a low `effort` setting, a response cache, and logging;
run-to-run variance is measured by the eval harness instead. Pass 2 also writes the title,
description and tags, so metadata costs no extra call. Clips longer than `max_duration` (60 s)
are allowed up to `extended_max_duration` (90 s) only with a higher score bar. The definition-of-
done numbers (15 minutes, under $1 for a one-hour podcast) are estimated from published prices
and local render speed; they must be confirmed with real API keys on a real podcast.
