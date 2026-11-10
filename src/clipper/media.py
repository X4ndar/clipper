"""ffmpeg/ffprobe helpers, hashing and WAV access."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

AUDIO_SR = 16000


class MediaError(RuntimeError):
    pass


def require_tools() -> None:
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise MediaError(f"{tool} not found on PATH. Install ffmpeg (with libass) first.")


def run(cmd: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    # ffmpeg/ffprobe write UTF-8; don't rely on the locale codec (cp1252 on Windows).
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise MediaError(f"command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    return proc


def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


@dataclass
class Probe:
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool
    audio_channels: int
    raw: dict


def probe(path: Path) -> Probe:
    proc = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)])
    data = json.loads(proc.stdout)
    v = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
    if v is None:
        raise MediaError(f"{path}: no video stream found")
    fps = 0.0
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    try:
        n, d = rate.split("/")
        fps = float(n) / float(d) if float(d) else 0.0
    except (ValueError, ZeroDivisionError):
        pass
    duration = float(data.get("format", {}).get("duration") or v.get("duration") or 0)
    # Respect rotation metadata so portrait phone videos are handled correctly.
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) in (90, 270):
        w, h = h, w
    return Probe(
        duration=duration,
        width=w,
        height=h,
        fps=fps,
        has_audio=a is not None,
        audio_channels=int(a.get("channels", 0)) if a else 0,
        raw=data,
    )


def extract_audio(src: Path, dst: Path) -> None:
    """Mono 16 kHz PCM. Mono matters: multichannel audio is billed per channel."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.wav")
    run([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(AUDIO_SR), "-c:a", "pcm_s16le", "-map_metadata", "-1",
        "-fflags", "+bitexact", "-flags:a", "+bitexact", str(tmp),
    ])
    tmp.replace(dst)


def encode_flac(src_wav: Path, dst: Path) -> None:
    """Lossless and about half the size of the WAV: a faster upload with no accuracy cost."""
    tmp = dst.with_suffix(".tmp.flac")
    run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src_wav),
         "-c:a", "flac", "-compression_level", "5", str(tmp)])
    tmp.replace(dst)


class Wav:
    """Random access to a mono 16-bit WAV, for RMS around cut points."""

    def __init__(self, path: Path):
        self.path = path
        with wave.open(str(path), "rb") as w:
            if w.getnchannels() != 1 or w.getsampwidth() != 2:
                raise MediaError(f"{path}: expected mono 16-bit WAV")
            self.sr = w.getframerate()
            self.n = w.getnframes()

    @property
    def duration(self) -> float:
        return self.n / self.sr

    def read(self, t0: float, t1: float) -> np.ndarray:
        a = max(0, int(t0 * self.sr))
        b = min(self.n, int(t1 * self.sr))
        if b <= a:
            return np.zeros(0, dtype=np.float32)
        with wave.open(str(self.path), "rb") as w:
            w.setpos(a)
            raw = w.readframes(b - a)
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

    def rms_frames(self, t0: float, t1: float, frame_s: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (frame_center_times, rms) for frames of `frame_s` covering [t0, t1]."""
        t0 = max(0.0, t0)
        t1 = min(self.duration, t1)
        x = self.read(t0, t1)
        hop = max(1, int(frame_s * self.sr))
        n = len(x) // hop
        if n == 0:
            return np.zeros(0), np.zeros(0)
        frames = x[: n * hop].reshape(n, hop)
        rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
        times = t0 + (np.arange(n) + 0.5) * frame_s
        return times, rms

    def rms(self, t0: float, t1: float) -> float:
        x = self.read(t0, t1)
        return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0
