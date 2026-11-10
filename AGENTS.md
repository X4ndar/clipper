# AGENTS.md

Guide for coding agents working in this repo. Keep it current when you change layout or commands.

## What this is

A self-hosted tool that turns a long video the user owns into short vertical clips. One Python
package, SQLite for state, ffmpeg for media. Transcript-first: ASR -> two LLM passes -> boundary
snapping -> face-following 9:16 crop -> render with burned-in captions -> HTML review report. See `PLAN.md`.

## Layout

```
src/clipper/
  cli.py          Typer CLI entry point (`clipper`)
  config.py       Pydantic config + precedence (CLI > settings.json > config.yaml > defaults)
  pipeline.py     stage functions st_* and run_job(); all front ends must call run_job()
  jobs.py         job folder + job.json, stage names and aliases
  db.py           SQLite: jobs, stage_runs, llm_runs, asr_usage, eval_runs
  media.py        ffprobe/ffmpeg helpers, hashing, WAV RMS access
  transcript.py   Word/Sentence/Transcript, segmentation, quality check, SRT
  transcribe/     Transcriber backends: assemblyai (REST via httpx), fixture (JSON on disk)
  llm/            client.py (Anthropic API, Claude Code CLI (subscription) and fake backends, cache, run log, budget), schemas.py,
                  prompts/<name>_vN.md (versioned; never edit a version in place, add v2)
  signals.py      audio loudness + chat-rate z-scores, peaks (signals.json), chat file parsers
  select.py       peak candidates, selection mode, prompt auto-choice, pass 1, pass 2, hard gates, weights, dedupe, final pick
  boundaries.py   sentence->word mapping, clause offsets, filler drop, quiet-frame snapping
  reframe.py      shot cuts, MediaPipe face detection, mouth-movement speaker pick, crop path (sendcmd),
                  facecam detection + stack geometry, framing metrics
  captions.py     ASS generation (word highlight, karaoke, plain) + hook overlay
  render.py       ffmpeg cut/crop (fixed or moving)/burn-in, single-frame preview, font folder merge
  report.py       report.html
  evaluation.py   gold format, label sheet, metrics
  picks.py        hand-picked clips (JSON) -> accepted candidates; `clipper picks JOB FILE`
scripts/make_sample.py   synthetic two-speaker sample (flite TTS + test pattern)
tests/                   pytest; test_integration.py needs ffmpeg with libass and flite
assets/fonts/            OFL fonts only; record any new asset in LICENSES.md
```

## Commands

```
uv sync --frozen --extra dev --extra reframe   # install (or: make install); reframe = MediaPipe
make test                           # all tests (unit + integration), about 1 minute
make test-unit                      # skip the ffmpeg integration tests
uv run clipper run VIDEO --i-own-rights --set transcribe.backend=fixture --set llm.backend=fake
                                    # offline end-to-end run (needs VIDEO.words.json next to VIDEO)
make sample                         # writes sample/podcast.mp4 + sample/podcast.words.json
```

## Conventions

- Every stage reads and writes files in the job folder; a stage must be re-runnable alone.
- Never re-transcribe: transcripts are cached by audio SHA-256 + backend settings in `$CLIPPER_HOME/cache`.
- LLM calls go through `LLMService` only (logging, cache, budget). Do not call the SDK elsewhere.
- Candidates are sentence IDs, never raw seconds. Seconds come from word timestamps in code.
- The LLM never invents or reorders speech; code clamps any boundary move it proposes.
- Editorial score, confidence and technical checks stay separate fields. Low confidence or a failed
  technical check flags `needs_review`; it does not reject.
- Zero clips is a valid result. Do not pad.
- No secrets in code or config files. Keys come from env / `.env`.
- Pin dependency versions; regenerate `uv.lock` with `uv lock`; check licenses before adding one.
- Do not copy code, prompts or UI from AGPL projects (SupoClip).
- Face detection goes through the `FaceDetector` interface in reframe.py; tests inject a fake one
  (`run_job(..., face_detector=...)`). Integration tests pin `render.reframe=center` unless they test reframing.
- Windows: no `make`; run the uv commands above. Read and write text files with `encoding="utf-8"`.
- Current Claude models reject `temperature`; use `output_config.effort`.
- Change prompts by adding a new version file and pointing `llm.candidate_prompt` / `llm.judge_prompt`
  at it, then compare on the gold set with `clipper eval run` / `clipper eval summary`.
