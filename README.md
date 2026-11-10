# clipper

Self-hosted tool that turns a long video you own (podcast, interview, talk) into short vertical
clips with burned-in captions. Clip selection is transcript-first: AssemblyAI transcribes, a cheap
model proposes candidates, a stronger model judges them against hard gates (standalone sense,
payoff), and code snaps every cut to word boundaries and quiet audio. Every candidate, including
the rejected ones and why, ends up in an HTML review report. Nothing is published.

Status: milestone M1 (CLI, selection, center-crop render, report, eval harness), a basic M2
(captions) and the first part of M3 (face-following crop) are built. Real AssemblyAI transcription
has been run; real Claude calls have not. See [PLAN.md](PLAN.md) for the roadmap and
[What is verified](#what-is-verified) before relying on the numbers below.

## Rules

- Only process videos you own or have rights to. `clipper run` refuses to start without `--i-own-rights`.
- Cloud ASR uploads the audio to AssemblyAI. Use it only for content you are comfortable uploading.
- Review every clip before publishing. Upload (M5) will default to private.

## Setup

Needs Python 3.11+, [uv](https://docs.astral.sh/uv/), and ffmpeg built with libass (Debian and
Ubuntu packages are). On Windows, the gyan.dev "full" build has libass and flite:
`winget install --id Gyan.FFmpeg -e`, then open a new terminal so `ffmpeg` is on PATH.

```bash
uv sync --frozen --extra dev --extra reframe
```

The `reframe` extra (MediaPipe, about 300 MB with its dependencies) enables the face-following crop.
Without it, clips use a center crop and the job records a warning.

```bash
cp .env.example .env   # then fill in ASSEMBLYAI_API_KEY and ANTHROPIC_API_KEY
```

AssemblyAI's free tier gives $50 of credit with no card. At $0.21/hour for Universal-3.5 Pro plus
$0.02/hour for speaker labels that is about 215 hours of audio. Test on 20 minutes first (under $0.10).

Docker (CPU):

```bash
docker build -t clipper .
```

```bash
docker run --rm --env-file .env -v "$PWD/media:/media" -v clipper-home:/data clipper run /media/episode.mp4 --i-own-rights
```

## Use

```bash
uv run clipper run episode.mp4 --i-own-rights --content-type podcast --max-clips 8
```

Options: `--language` (default auto-detect), `--content-type` (podcast, interview, talking_head,
tutorial, other), `--max-clips`, `--min-duration` / `--max-duration` (default 20-60 s; clips up to
90 s pass only with a higher score), `--speakers N`, `--chat FILE`, `--direction TEXT`, `--stop-at STAGE`, `--from STAGE`,
`--set key=value` for any config key.

Output lands in `$CLIPPER_HOME/jobs/<name>-<hash>/` (default `~/.clipper`):

| File | What |
|---|---|
| `clips/clip_01.mp4` … | 1080x1920 H.264/AAC, burned-in captions and hook overlay, best first |
| `clips.json` | start, end, scores, confidence, technical checks, title, description, tags, hook text, rationale |
| `report.html` | review page: selected clips with players, every rejected candidate with the reason |
| `transcript.json` / `.srt` / `.ass` | word-level transcript (`{word, start, end, speaker, score}`) and subtitle files |
| `transcript.raw.json` | the full AssemblyAI response |
| `clips/clip_01.captions.json` | editable captions: fix ASR errors, then `clipper rerender JOB --clip 1` |
| `candidates.json`, `judged.json`, `selection.json`, `quality.json` | every stage's artifacts |

### Stages

`ingest → transcribe → quality → signals → candidates → judge → snap → reframe → captions → render → metadata → report`

Each stage reads and writes files in the job folder, so any stage can be re-run alone. Aliases for
`--stop-at`: `transcript` (= quality), `selection`, `clips` (= render).

```bash
uv run clipper run episode.mp4 --i-own-rights --stop-at transcript   # transcribe and quality-check only
```

```bash
uv run clipper resume episode-1a2b3c4d                                # continue where it stopped
```

```bash
uv run clipper resume episode-1a2b3c4d --from judge --set llm.judge_effort=high   # re-judge, cut again
```

```bash
uv run clipper stage episode-1a2b3c4d captions                       # one stage alone
```

The same audio is never transcribed twice: transcripts are cached by the audio file's SHA-256 (plus
model and speaker settings) under `$CLIPPER_HOME/cache/transcripts/`. Identical LLM calls are also
served from the run log (`llm.cache_responses`).

The quality stage flags a transcript before any LLM tokens are spent: share of low-confidence words,
repeated n-gram loops, repeated sentences, implausible speech rate, and languages with weak ASR
support (with an explicit warning for Arabic dialects such as Darija).

### Framing

`render.reframe: face` (default) keeps faces in the 9:16 crop. Per clip, the `reframe` stage finds
shot cuts on every frame, detects faces on 4 frames per second (MediaPipe BlazeFace full-range, CPU;
the model is downloaded once to `$CLIPPER_HOME/models/` and checked by SHA-256), and plans a crop path:
within a shot the crop holds still while the subject stays inside a dead zone and pans smoothly when
it leaves; at a cut it jumps. If everyone fits, it frames the group; if not, the largest face, with
near-ties going to the face it is already on or, at a new shot, the one nearest the middle.

`clips/clip_NN.crop.json` holds the path and its framing numbers: how often a face is fully in frame
and how often a face is sliced by the crop edge, each also computed for a plain center crop. They
show in the report. A clip below `reframe.min_face_in_frame` (0.8) is flagged for review.
Detections are cached in `clip_NN.faces.json`, so re-planning after a config change takes seconds:

```bash
uv run clipper resume episode-1a2b3c4d --from reframe --set reframe.dead_zone=0.1
```

Not done yet (rest of M3): following the active speaker, and split layouts for two people.
`render.reframe: center` turns face following off.

On Windows, a clip that is open in a video player or a browser tab (the report) cannot be replaced:
close it before re-rendering.

### Captions

Word-by-word highlight by default (`captions.mode`: `word`, `karaoke`, `plain`), 2 lines of about
22 characters, placed above the Shorts UI zone. Font, size, colors, outline, background box,
position and the hook overlay are all config keys. Tune a style on one frame in about a second:

```bash
uv run clipper preview episode-1a2b3c4d --clip 1 --at 2.0 --set captions.size=90 --set captions.highlight_color=#00E5FF
```

Fonts: Noto Sans and Noto Sans Arabic (OFL) are bundled. Arabic and other RTL text is shaped by
libass (tested with an Arabic sample). Add your own fonts to `$CLIPPER_HOME/fonts/`.

### Hand-picked clips

To choose clips yourself (or override the models), list them in a JSON file, best first, by time or
by sentence ID from `transcript.json`:

```json
[{"start": "19:35", "end": "20:30", "title": "...", "hook_text": "..."},
 {"start_sentence": 679, "end_sentence": 704}]
```

```bash
uv run clipper picks episode-1a2b3c4d picks.json
```

To tighten a clip, give `segments` instead of one span. The pieces are joined with jump cuts (each
with a short audio fade) and play in the order listed, so a clip can also open on its best line:

```json
{"segments": [{"start_sentence": 645, "end_sentence": 648}, {"start": "19:52.5", "end": "20:30"}],
 "title": "..."}
```

Times snap to whole sentences, then the usual word-boundary snapping, reframing, captions and report
run. Captions and the crop path follow the joins. Hand picks skip the length, score and clip-count limits (the 180 s hard cap still applies) and
show as "hand-picked" in the report. Running `--from candidates` again lets the models select instead.

### Other commands

```bash
uv run clipper jobs                 # job history
```

```bash
uv run clipper usage                # ASR credit used and remaining, LLM spend
```

```bash
uv run clipper config show          # effective config and which files contributed
```

```bash
uv run clipper config export settings.json    # API keys left out unless --include-secrets
```

## Configuration

Precedence, highest first:

1. CLI flags and `--set key=value`
2. `$CLIPPER_HOME/settings.json` (settings saved from the web UI, or `clipper config import`)
3. `--config path`, else `$CLIPPER_HOME/config.yaml`, else `./config.yaml`
4. built-in defaults

See [config.example.yaml](config.example.yaml) for every key. Secrets are never read from config
files, only from the environment or `.env`. `tests/test_config.py` covers the precedence.

Models are config values: `llm.candidate_model` (default `claude-haiku-5-5`) and `llm.judge_model`
(default `claude-sonnet-5-5`). Current Claude models reject custom `temperature`, so the brief's
"low temperature" is replaced by `llm.*_effort` plus logging every call (prompt version, model,
parameters, output, tokens, cost) in SQLite. The judge call opts into Anthropic's server-side
refusal fallback (`llm.judge_fallbacks: default`; set to `null` to turn it off). Transcription
always sends `speech_models: ["universal-3-5-pro"]` explicitly (`transcribe.speech_models`).

### Claude subscription instead of an API key

`--set llm.backend=claude_code` (or `llm: {backend: claude_code}` in config.yaml) sends the two LLM
passes through the Claude Code CLI, which uses the Claude subscription you are logged in with. It
needs `claude` on PATH and a login (`claude`, then `/login`); no `ANTHROPIC_API_KEY`. Each call runs
with no tools, no project settings and no saved session. Calls count against your subscription's
usage limits, so cost is logged as $0; the CLI's API-equivalent estimate is kept in the run log.
There is no refusal fallback on this backend.

### Streams, challenges and other event-driven videos

```bash
uv run clipper run stream.mp4 --i-own-rights --content-type stream --chat stream_chat.json --direction "funny fails and clutch wins"
```

`--content-type stream` (or `challenge`) changes four things:

- **Prompts.** `llm.candidate_prompt` / `llm.judge_prompt` default to `auto`: `candidates_v2` and
  `judge_v2` for these types (they look for an arc: setup, attempt, outcome, reaction), v1 for the rest.
- **Signals.** The free `signals` stage scores every second of the video against its own rolling
  baseline: audio loudness, and chat rate when `--chat` is given. Chat files: TwitchDownloader JSON,
  yt-dlp `.live_chat.json`, or text lines `[HH:MM:SS] user: message`. Messages with emotes or hype
  words count extra, and chat is shifted 5 s earlier because viewers type after the moment. Agreement
  between audio and chat scores higher than either alone. Output: `signals.json`.
- **Peak-first selection** (`selection.mode`, default `auto`). Each of the strongest peaks becomes a
  candidate window (40 s before, 15 s after) that the judge trims to the real setup and reaction.
  Videos up to 45 minutes use `both` (LLM pass 1 plus peaks); longer ones use `peaks` only, so a
  4-hour stream costs about `max_clips x 3` judge calls and no pass over the whole transcript.
  A peak with nobody talking around it is skipped.
- **Layout** (`render.layout`, default `auto`). For `stream`, if a small face sits in the same place
  across the clips (a camera overlay), clips are stacked: camera on top, gameplay below. Set the
  camera box yourself with `--set reframe.facecam_region=0.75,0.70,0.25,0.30` (x, y, w, h as shares of
  the frame), or force it on any video with `--set render.layout=stack`.

For every content type, the loudness and chat around a clip now fill the `speaker_energy` part of the
editorial score, and `--direction "..."` steers both LLM passes in plain words.

When several people are on screen and do not all fit, the crop prefers the face whose mouth is moving
(`reframe.follow_speaker`). This is a pixel-change heuristic on 4 frames per second, not a speaker
model; it has unit tests but no accuracy measurement yet.

Measured on one 22-minute challenge video (Claude subscription backend): v1 prompts selected 0 clips;
v2 with transcript only selected 4; v2 with audio peaks judged 19 candidates and selected 4, two of
them found only by the peaks. None of this is measured on a gold set yet, and chat scoring has only
been tested on synthetic chat.

## Selection, in short

- Pass 1 sees the whole transcript as `[sentence_id] [mm:ss] SPEAKER: text` and returns about 3x the
  wanted clip count as sentence IDs, never seconds. Transcripts above `llm.chunk_token_limit` are
  split with overlap.
- Pass 2 sees each candidate plus 30 s of context on each side and scores hook, standalone sense,
  payoff, peak, quotability and clean ending. It may move boundaries a few sentences (clamped in
  code), trim at clause boundaries (validated in code, never leaving a fragment) and propose an
  8-word hook overlay, title, description and tags.
- Hard gates: standalone sense < 6 or payoff < 6 is rejected whatever the total.
- Editorial score = weighted mean (hook 30, peak 25, standalone 20, speaker energy 15, quotability
  10; speaker energy has no signal yet and is skipped). Confidence (judge confidence, mean ASR word
  confidence, boundary check) and technical checks (crop, captions, audio) are kept separate. Low
  confidence or a failed check sends the clip to manual review instead of rejecting it.
- Overlaps are deduplicated (time overlap and word overlap), then the top `max_clips` are kept.
  Zero clips is a valid result.
- Cuts start about 80 ms before the first word and end about 150 ms after the last, each moved to the
  quietest 10 ms frame within 150 ms, with 25 ms audio fades. Leading "um/uh" are dropped.

## Evaluation

Build the gold set before tuning any prompt (target: 5-10 videos, 50+ labeled clips).

```bash
uv run clipper run ep12.mp4 --i-own-rights --stop-at selection
```

```bash
uv run clipper eval label-sheet ep12-1a2b3c4d --out labels/ep12.csv
```

Fill in `label` (good, ok, bad), `needs_trim` and `reason` for each row, and add rows for clips you
would have picked that the system missed (`candidate_id` = human, with start and end).

```bash
uv run clipper eval import labels/ep12.csv --job ep12-1a2b3c4d --out gold/ep12.json
```

```bash
uv run clipper eval score ep12-1a2b3c4d --gold gold/ep12.json
```

```bash
uv run clipper eval run gold/ --i-own-rights --set llm.judge_prompt=judge_v2
```

```bash
uv run clipper eval summary
```

Metrics: precision at k (strict and with ok = 0.5), bad clips in the top k, share that start and end
on complete thoughts, share needing manual trim, recall of human "good" picks, cost and time per
hour of video. Every eval run is logged with its prompt versions and models, so `eval summary`
compares prompt and model changes. Prompts are versioned files in `src/clipper/llm/prompts/`; add
`judge_v2.md` rather than editing `judge_v1.md`.

## Cost and speed

Measured in this repo's sandbox (4 vCPU, 15 GB RAM, no GPU), offline backends:

| Step | Measured |
|---|---|
| Ingest (hash + mono 16 kHz audio) of a 10-minute 1080p30 file | 3.2 s (about 20 s per hour, extrapolated) |
| Render 8 clips, 340 s of 1080x1920 output, libx264 `veryfast` CRF 20, with captions | 107 s (3.2x faster than real time) |
| Snapping, selection, captions, report | under 1 s |
| Face reframing, 6 clips / 248 s of 720p source (laptop i7-1355U) | 19.5 s; moving crop adds about 20% to render time |

Estimated for a one-hour podcast with real APIs (not measured here, no keys in the sandbox):

| Item | Estimate |
|---|---|
| AssemblyAI, 1 h with speaker labels | $0.23 (published price); turnaround not measured |
| Pass 1, Haiku 5.5, ~15k input tokens | under $0.01 |
| Pass 2, Sonnet 5.5, 24 calls of ~2.5k in / ~1.5k out (incl. thinking) | about $0.48 |
| Total | about $0.71 per hour |
| Render 8 x 60 s clips | about 2.5 minutes on 4 cores |

Pass 2 dominates cost. To spend less: `--set llm.judge_effort=low`, a lower
`llm.candidate_multiplier`, or fewer `--max-clips`. A per-job guard stops LLM calls at
`max(llm.min_budget_usd, llm.max_cost_usd_per_hour x hours)` (default $0.75 per hour). The run log
records real token counts and cost, so `clipper usage` and `clips.json` show actual spend.

### Hardware

| | CPU | RAM | GPU | Notes |
|---|---|---|---|---|
| Minimum | 2 cores | 2 GB | none | renders at roughly 1.5x real time (estimated from the 4-core figure) |
| Recommended | 4 cores | 4 GB | none | measured: 3.2x real time render |
| Ideal | 8+ cores | 8 GB | optional | GPU only matters for later local ASR and YOLO face tracking |

With AssemblyAI doing transcription, v1 needs no GPU. Disk: the job keeps a mono WAV (about 115 MB
per hour) and a FLAC copy for upload (about half that).

## What is verified

Tested (68 tests, `make test`, on Windows 11; the first 55 also passed on Linux): boundary snapping on synthetic audio, clause-offset rules,
JSON schema validation and retries, the LLM cache, budget guard and refusal handling, the Anthropic
request shape (with a mock client), the AssemblyAI request and polling flow (with a mock HTTP
server), ASS generation, config precedence, selection gates, dedupe and quota, the eval metrics,
and a full end-to-end run on a generated two-speaker video (real ffmpeg, real speech audio,
fixture transcript, deterministic fake LLM), crop-path planning on synthetic face tracks, and a
render with a moving crop.

Run for real (Windows 11, i7-1355U, 2026-11-07): AssemblyAI Universal-3.5 Pro on a 22.5-minute
video, $0.079, 22 s turnaround, word timestamps checked against the raw response; face reframing on
6 clips of that video (about 4 minutes): planning took 19.5 s, and a face was fully in frame in
85-98% of frames with faces, against 50-86% for a center crop. On fast-cut videos with background
music, AssemblyAI squeezed about 4% of word timestamps into 10-20 ms runs; the quality check flags
this as fast speech.

Not verified yet: real Claude calls, clip quality on real podcasts, and the definition-of-done numbers (under 15 minutes and under $1 for one hour). The
Docker image has not been built (no Docker daemon in the build sandbox).

## Development

See [AGENTS.md](AGENTS.md). Licenses of bundled assets and dependencies: [LICENSES.md](LICENSES.md).
