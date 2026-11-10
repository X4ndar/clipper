# Licenses of bundled assets and dependencies

The project's own license is not chosen yet (owner's decision). Everything bundled or depended on
is listed here. Add a row before adding any asset or dependency.

## Bundled assets

| File | Source | License |
|---|---|---|
| `assets/fonts/NotoSans-Bold.ttf` | github.com/notofonts/notofonts.github.io (hinted TTF) | SIL Open Font License 1.1 (`assets/fonts/OFL.txt`) |
| `assets/fonts/NotoSansArabic-Bold.ttf` | github.com/notofonts/notofonts.github.io (hinted TTF) | SIL Open Font License 1.1 (`assets/fonts/OFL.txt`) |

No music, stock footage or sample media is bundled. `scripts/make_sample.py` generates test media
at run time from ffmpeg's built-in test pattern and flite speech synthesizer.

User fonts go in `$CLIPPER_HOME/fonts/` (or `captions.fonts_dir`). Check their licenses yourself.

## Python dependencies (resolved in `uv.lock`, checked 2026-11-04)

| Package | Version | License |
|---|---|---|
| anthropic | 2.4.0 | Apache-2.0 |
| httpx | 0.28.1 | BSD-3-Clause |
| pydantic / pydantic-core | 2.14.2 / 2.50.2 | MIT |
| typer | 0.27.3 | MIT |
| PyYAML | 6.0.3 | MIT |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| numpy | 2.4.4 | BSD-3-Clause (plus 0BSD, MIT, Zlib, CC0 parts) |
| pytest (dev) | 9.2.0 | MIT |
| hatchling (build) | 1.27.0 | MIT |

Transitive: annotated-doc, annotated-types, anyio, docstring_parser, h11, jiter, markdown-it-py, mdurl,
pluggy, iniconfig, rich, truststore, typing-inspection (MIT); httpcore, httpcore2, httpx2, idna
(BSD-3-Clause); Pygments (BSD-2-Clause); packaging (Apache-2.0 OR BSD-2-Clause); opentelemetry-api
(Apache-2.0); shellingham (ISC); sniffio (MIT OR Apache-2.0); typing_extensions (PSF-2.0);
certifi (MPL-2.0, used unmodified).

### Optional extra `reframe` (face-following crop, checked 2026-11-07)

| Package | Version | License |
|---|---|---|
| mediapipe | 1.2.2 | Apache-2.0 |
| opencv-contrib-python | 5.0.0.93 | Apache-2.0 |
| matplotlib | 3.11.2 | Matplotlib License (PSF-based) |
| absl-py / flatbuffers | 2.5.1 / 25.12.19 | Apache-2.0 |
| sounddevice | 0.5.6 | MIT |
| pillow | 13.0.0 | MIT-CMU |

Transitive: cffi (MIT-0); fonttools, pyparsing (MIT); pycparser, contourpy, python-dateutil
(BSD-3-Clause); cycler, kiwisolver (BSD); tzdata (Apache-2.0). MediaPipe imports OpenCV,
matplotlib and sounddevice at load time; only its face detector is used.

## Downloaded model

| File | Source | License |
|---|---|---|
| `blaze_face_full_range.tflite` (BlazeFace full-range, float16 v1) | storage.googleapis.com/mediapipe-models, pinned by SHA-256 in `reframe.model_sha256` | Apache-2.0 (MediaPipe model card) |

Downloaded on first use to `$CLIPPER_HOME/models/`, not bundled in the repo.

## External programs

ffmpeg/ffprobe are called as separate programs, not linked. Distribution builds (Debian, Ubuntu)
are GPL-licensed because of enabled components such as libx264. They must be built with libass for
captions.

## Projects read for ideas only

SupoClip is AGPL-3.0: nothing (code, prompts, UI) is copied from it. MoneyPrinterTurbo was read for
structure only; none of its bundled music or code is used.
