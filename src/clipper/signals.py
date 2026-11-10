"""Highlight signals that do not depend on the words: audio loudness spikes and chat bursts.

Both become a per-second z-score against a rolling baseline of the same video, so a quiet stream and a
loud one are judged on their own terms. Peaks of the combined score seed clip candidates (select.py)
and feed the `speaker_energy` part of the editorial score.

Chat logs are read from files saved next to a VOD:
  - TwitchDownloader JSON ({"comments": [{"content_offset_seconds": ..., "message": {"body": ...}}]})
  - yt-dlp live chat (.live_chat.json, one JSON object per line with videoOffsetTimeMsec)
  - plain text, one message per line: `[HH:MM:SS] user: message`
"""

from __future__ import annotations

import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

STEP = 1.0  # seconds per sample of every series


class SignalError(RuntimeError):
    pass


# ---------------- audio ----------------


def loudness_db(wav_path: Path) -> np.ndarray:
    """RMS level per second, in dB, of a mono 16-bit WAV. Read in blocks so hours of audio are fine."""
    out = []
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        hop = int(sr * STEP)
        while True:
            raw = w.readframes(hop * 600)
            if not raw:
                break
            x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            n = len(x) // hop
            if n == 0:
                break
            rms = np.sqrt(np.mean(x[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-10)
            out.append(20 * np.log10(rms))
    return np.concatenate(out) if out else np.zeros(0)


def smooth(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(x) < 2:
        return x
    r = k // 2
    return np.convolve(np.pad(x, r, mode="edge"), np.ones(2 * r + 1) / (2 * r + 1), mode="valid")


def rolling_z(x: np.ndarray, window_s: float, floor: float) -> np.ndarray:
    """(x - rolling median) / rolling spread, the spread being MAD scaled to a standard deviation and
    never below `floor` (so flat stretches do not turn tiny wiggles into spikes)."""
    n = len(x)
    if n == 0:
        return x
    r = max(2, int(window_s / STEP / 2))
    pad = np.pad(x, r, mode="reflect") if n > r else np.pad(x, r, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * r + 1)
    med = np.median(win, axis=1)
    mad = np.median(np.abs(win - med[:, None]), axis=1) * 1.4826
    return (x - med) / np.maximum(mad, floor)


# ---------------- chat ----------------


@dataclass
class ChatMessage:
    t: float
    text: str
    emotes: int = 0


_TXT = re.compile(r"^\[(\d+):(\d\d):(\d\d)(?:\.\d+)?\]\s*(?:[^:]{1,40}:\s*)?(.*)$")


def _parse_twitch_downloader(data: dict) -> list[ChatMessage]:
    out = []
    for c in data.get("comments", []):
        m = c.get("message") or {}
        emotes = sum(1 for f in (m.get("fragments") or []) if f.get("emoticon"))
        out.append(ChatMessage(float(c.get("content_offset_seconds", 0)), str(m.get("body") or ""), emotes))
    return out


def _parse_ytdlp_line(obj: dict) -> Optional[ChatMessage]:
    rep = obj.get("replayChatItemAction") or {}
    off = rep.get("videoOffsetTimeMsec")
    if off is None:
        return None
    for act in rep.get("actions", []):
        item = (act.get("addChatItemAction") or {}).get("item") or {}
        r = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
        if not r:
            continue
        runs = (r.get("message") or {}).get("runs") or []
        text = "".join(x.get("text", "") for x in runs)
        return ChatMessage(int(off) / 1000.0, text, sum(1 for x in runs if "emoji" in x))
    return None


def load_chat(path: Path) -> list[ChatMessage]:
    if not path.is_file():
        raise SignalError(f"chat file not found: {path}")
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    msgs: list[ChatMessage] = []
    stripped = text.lstrip()
    if stripped.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "comments" in data:
                msgs = _parse_twitch_downloader(data)
        except json.JSONDecodeError:
            pass
        if not msgs:  # JSON lines (yt-dlp live chat)
            for line in text.splitlines():
                try:
                    m = _parse_ytdlp_line(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if m:
                    msgs.append(m)
    else:
        for line in text.splitlines():
            m = _TXT.match(line.strip())
            if m:
                h, mi, s, body = m.groups()
                msgs.append(ChatMessage(int(h) * 3600 + int(mi) * 60 + int(s), body))
    msgs = [m for m in msgs if m.t >= 0]
    if not msgs:
        raise SignalError(f"{path}: no chat messages recognised (TwitchDownloader JSON, yt-dlp live chat "
                          "or '[HH:MM:SS] user: message' lines)")
    return sorted(msgs, key=lambda m: m.t)


def chat_rate(msgs: list[ChatMessage], n: int, cfg) -> np.ndarray:
    """Weighted messages per second. Emotes and hype words count extra; the series is shifted earlier
    by `chat_lag_s` because viewers type after the moment."""
    x = np.zeros(n)
    hype = [w.lower() for w in cfg.hype_words]
    for m in msgs:
        i = int((m.t - cfg.chat_lag_s) / STEP)
        if 0 <= i < n:
            low = m.text.lower()
            x[i] += 1.0 + (cfg.hype_bonus if (m.emotes or any(w in low for w in hype)) else 0.0)
    return x


# ---------------- combine ----------------


def find_peaks(score: np.ndarray, threshold: float, min_gap_s: float, limit: int) -> list[int]:
    """Indices of the strongest local maxima above `threshold`, at least `min_gap_s` apart."""
    cand = [i for i in range(len(score)) if score[i] >= threshold
            and score[i] >= score[max(0, i - 2): i + 3].max()]
    cand.sort(key=lambda i: -score[i])
    kept: list[int] = []
    for i in cand:
        if all(abs(i - k) * STEP >= min_gap_s for k in kept):
            kept.append(i)
        if len(kept) >= limit:
            break
    return sorted(kept)


def compute_signals(wav_path: Path, chat_path: Optional[Path], cfg, max_peaks: int) -> dict:
    db = loudness_db(wav_path)
    n = len(db)
    audio_z = rolling_z(smooth(db, 3), cfg.audio_window_s, floor=1.5)
    chat_z = None
    n_msgs = 0
    if chat_path is not None:
        msgs = load_chat(chat_path)
        n_msgs = len(msgs)
        rate = smooth(chat_rate(msgs, n, cfg), int(cfg.chat_smooth_s))
        chat_z = rolling_z(rate, cfg.chat_window_s, floor=max(0.05, float(np.median(rate)) * 0.25))
    a = np.clip(audio_z, 0, None) * cfg.audio_weight
    if chat_z is None:
        combined = a
    else:
        c = np.clip(chat_z, 0, None) * cfg.chat_weight
        # Agreement between the two is worth more than either alone.
        combined = np.maximum(a, c) + 0.5 * np.minimum(a, c)
    combined = smooth(combined, 5)
    peaks = find_peaks(combined, cfg.threshold, cfg.min_gap_s, max_peaks)
    return {
        "step": STEP, "seconds": n, "chat_messages": n_msgs, "has_chat": chat_z is not None,
        "audio_z": [round(float(v), 2) for v in audio_z],
        "chat_z": None if chat_z is None else [round(float(v), 2) for v in chat_z],
        "combined": [round(float(v), 2) for v in combined],
        "peaks": [{"t": i * STEP, "score": round(float(combined[i]), 2),
                   "audio_z": round(float(audio_z[max(0, i - 3): i + 4].max()), 2),
                   "chat_z": None if chat_z is None else round(float(chat_z[max(0, i - 3): i + 4].max()), 2)}
                  for i in peaks],
    }


def energy_score(sig: dict, start: float, end: float) -> Optional[float]:
    """0-10 from the strongest combined z-score inside [start, end]: 0 sigma -> 4, 4 sigma -> 10."""
    c = sig.get("combined") or []
    a, b = max(0, int(start / STEP)), min(len(c), int(end / STEP) + 1)
    if b <= a:
        return None
    return round(float(min(10.0, max(0.0, 4.0 + 1.5 * max(c[a:b])))), 2)
