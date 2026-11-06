"""Generate a short synthetic two-speaker "podcast" for tests and demos.

Speech comes from ffmpeg's built-in flite synthesizer, the picture from ffmpeg's test pattern, so
no third-party media is involved. Writes <out>.mp4 and <out>.words.json (word timings estimated
from each synthesized sentence's speech span, split by character length).

usage: python scripts/make_sample.py OUT_STEM
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

SR = 16000
SCRIPT = [
    ("A", "Hey everyone, welcome back to the show."),
    ("A", "Before we start, thanks to our sponsor for supporting this episode."),
    ("B", "Thanks for having me."),
    ("A", "So why do most productivity systems fail?"),
    ("B", "Most people think productivity is about doing more."),
    ("B", "But the real problem is that they never decide what not to do."),
    ("B", "I tracked my own time for ninety days, and forty percent of my week went to tasks nobody asked for."),
    ("B", "That's why the first rule is simple."),
    ("B", "Cut one recurring task every Friday, because the result compounds."),
    ("B", "After three months I had eleven free hours a week."),
    ("A", "That is a huge number."),
    ("A", "Let me ask you about mistakes."),
    ("B", "The biggest mistake I ever made was hiring too fast."),
    ("B", "We doubled the team in six months, and nobody knew who owned what."),
    ("B", "It turns out that speed without ownership just creates meetings."),
    ("B", "So the answer was to give every project exactly one owner."),
    ("B", "Within a quarter, our shipping speed went back up by half."),
    ("A", "Okay, that's all the time we have."),
    ("A", "Thanks for listening, and see you next week."),
]
VOICES = {"A": "slt", "B": "kal"}


def synth(text: str, voice: str, path: Path) -> np.ndarray:
    safe = text.replace("'", "").replace(":", " ")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
         "-i", f"flite=text='{safe}':voice={voice}", "-ar", str(SR), "-ac", "1", str(path)],
        check=True,
    )
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()


def speech_span(x: np.ndarray) -> tuple[int, int]:
    hop = SR // 100
    n = len(x) // hop
    rms = np.sqrt(np.mean((x[: n * hop].astype(np.float32).reshape(n, hop) / 32768) ** 2, axis=1))
    on = np.where(rms > 0.01)[0]
    if len(on) == 0:
        return 0, len(x)
    return int(on[0] * hop), int((on[-1] + 1) * hop)


def main(stem: str) -> None:
    out = Path(stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    audio: list[np.ndarray] = [np.zeros(int(0.5 * SR), dtype="<i2")]
    t = 0.5
    words = []
    prev_spk = None
    with tempfile.TemporaryDirectory() as td:
        for i, (spk, text) in enumerate(SCRIPT):
            gap = 0.6 if prev_spk not in (None, spk) else 0.3
            if i:
                audio.append(np.zeros(int(gap * SR), dtype="<i2"))
                t += gap
            x = synth(text, VOICES[spk], Path(td) / f"s{i}.wav")
            a, b = speech_span(x)
            toks = text.split()
            total = sum(len(w) for w in toks) + 0.5 * (len(toks) - 1)
            cur = t + a / SR
            span = (b - a) / SR
            for w in toks:
                d = span * len(w) / total
                words.append({"word": w, "start": round(cur, 3), "end": round(cur + d, 3), "speaker": spk,
                              "score": 0.95})
                cur += d + span * 0.5 / total
            audio.append(x)
            t += len(x) / SR
            prev_spk = spk
        audio.append(np.zeros(int(1.0 * SR), dtype="<i2"))
        t += 1.0
        pcm = np.concatenate(audio)
        wav = Path(td) / "all.wav"
        with wave.open(str(wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        dur = len(pcm) / SR
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=25:duration={dur:.2f}",
             "-i", str(wav), "-shortest", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-ac", "2", "-b:a", "96k", str(out.with_suffix(".mp4"))],
            check=True,
        )
    out.with_suffix(".words.json").write_text(
        json.dumps({"language": "en", "duration": round(dur, 3), "words": words}, indent=1), encoding="utf-8"
    )
    print(out.with_suffix(".mp4"), f"{dur:.1f}s", len(words), "words")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "sample/podcast")
