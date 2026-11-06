"""Map sentence IDs to word ranges and snap cut points to quiet audio.

Rules:
- Start on a complete sentence (or a validated clause boundary), ~80 ms before the first word.
- End after the last word with ~150 ms of tail. Never cut mid-word.
- Each cut moves to the quietest 10 ms frame within ~150 ms of the word boundary, because cloud
  word timings are estimates and can clip a plosive or a breath.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .media import Wav
from .transcript import Transcript, Word, ends_clause, ends_sentence, normalize

CONJUNCTIONS = {
    "but", "because", "so", "and", "which", "then", "yet", "however", "although", "though", "or",
    "cause", "'cause", "since", "while", "whereas",
}
MIN_FRAGMENT_WORDS = 3


@dataclass
class Bounds:
    w0: int
    w1: int
    start: float
    end: float
    notes: list[str] = field(default_factory=list)
    start_offset: int = 0
    end_offset: int = 0
    fillers_dropped: int = 0

    @property
    def duration(self) -> float:
        return self.end - self.start


def sentence_word_range(tr: Transcript, s0: int, s1: int) -> tuple[int, int]:
    return tr.sentences[s0].start_word, tr.sentences[s1].end_word


def valid_start_cut(words: list[Word], new_first: int, sent_end: int) -> bool:
    """Cutting before words[new_first]: previous word must close a clause or be a conjunction."""
    if new_first <= 0 or sent_end - new_first + 1 < MIN_FRAGMENT_WORDS:
        return False
    prev = words[new_first - 1].word
    return ends_clause(prev) or normalize(prev) in CONJUNCTIONS


def valid_end_cut(words: list[Word], new_last: int, sent_start: int) -> bool:
    """Cutting after words[new_last]: it must close a clause, or the next word starts one."""
    if new_last + 1 >= len(words) or new_last - sent_start + 1 < MIN_FRAGMENT_WORDS:
        return False
    return ends_clause(words[new_last].word) or normalize(words[new_last + 1].word) in CONJUNCTIONS


def apply_offsets(tr: Transcript, s0: int, s1: int, start_off: int, end_off: int) -> tuple[int, int, int, int, list[str]]:
    notes: list[str] = []
    w0, w1 = sentence_word_range(tr, s0, s1)
    so = eo = 0
    if start_off > 0:
        cand = w0 + start_off
        sent_end = tr.sentences[s0].end_word
        if cand <= sent_end and valid_start_cut(tr.words, cand, sent_end):
            w0, so = cand, start_off
        else:
            notes.append(f"start_word_offset={start_off} ignored: not a clause boundary or leaves a fragment")
    if end_off > 0:
        cand = w1 - end_off
        sent_start = tr.sentences[s1].start_word
        if cand >= sent_start and cand >= w0 and valid_end_cut(tr.words, cand, sent_start):
            w1, eo = cand, end_off
        else:
            notes.append(f"end_word_offset={end_off} ignored: not a clause boundary or leaves a fragment")
    if w1 - w0 + 1 < MIN_FRAGMENT_WORDS:
        w0, w1 = sentence_word_range(tr, s0, s1)
        so = eo = 0
        notes.append("offsets dropped: clip would be too short")
    return w0, w1, so, eo, notes


def drop_leading_fillers(words: list[Word], w0: int, w1: int, fillers: list[str], max_drop: int = 3) -> tuple[int, int]:
    fs = {f.lower() for f in fillers}
    dropped = 0
    while dropped < max_drop and w1 - w0 + 1 > MIN_FRAGMENT_WORDS and normalize(words[w0].word) in fs:
        w0 += 1
        dropped += 1
    return w0, dropped


def _quietest(wav: Wav, lo: float, hi: float, target: float, frame_s: float) -> Optional[float]:
    """Start time of the quietest frame in [lo, hi], with a mild pull toward `target`."""
    if hi - lo < frame_s:
        return None
    times, rms = wav.rms_frames(lo, hi, frame_s)
    if len(rms) == 0:
        return None
    norm = rms / (rms.max() + 1e-9)
    span = max(hi - lo, 1e-6)
    cost = norm + 0.25 * np.abs(times - target) / span
    i = int(np.argmin(cost))
    return float(times[i] - frame_s / 2)


def snap(tr: Transcript, w0: int, w1: int, wav: Optional[Wav], cfg) -> tuple[float, float, list[str]]:
    """Return (start, end) seconds for words w0..w1. `cfg` is a SnapConfig."""
    words = tr.words
    notes: list[str] = []
    pre, tail, search, frame = cfg.pre_roll_ms / 1000, cfg.tail_ms / 1000, cfg.search_ms / 1000, cfg.frame_ms / 1000
    total = wav.duration if wav else (tr.duration or words[-1].end + 1)
    first, last = words[w0], words[w1]
    prev_end = words[w0 - 1].end if w0 > 0 else 0.0
    next_start = words[w1 + 1].start if w1 + 1 < len(words) else total

    # Start: somewhere in [first.start - search, first.start], never inside the previous word.
    lo = max(0.0, first.start - search, min(prev_end, first.start))
    start = max(lo, first.start - pre)
    if wav is not None:
        q = _quietest(wav, lo, first.start, first.start - pre, frame)
        if q is not None:
            start = q
    start = min(max(start, lo), first.start)

    # End: somewhere in [last.end, last.end + tail + search], never inside the next word.
    hi = min(total, last.end + tail + search, max(next_start, last.end))
    end = min(hi, last.end + tail)
    if wav is not None:
        q = _quietest(wav, last.end, hi, last.end + tail, frame)
        if q is not None:
            end = q + frame  # end of the quiet frame
    end = max(min(end, hi), last.end)
    if next_start - last.end < 0.05:
        notes.append("next word starts right after the last word; tail is very short")
    return round(start, 3), round(end, 3), notes


def compute_bounds(
    tr: Transcript, s0: int, s1: int, start_off: int, end_off: int, wav: Optional[Wav], cfg
) -> Bounds:
    w0, w1, so, eo, notes = apply_offsets(tr, s0, s1, start_off, end_off)
    w0, dropped = drop_leading_fillers(tr.words, w0, w1, cfg.leading_fillers)
    if dropped:
        notes.append(f"dropped {dropped} leading filler word(s)")
    start, end, snap_notes = snap(tr, w0, w1, wav, cfg)
    return Bounds(w0, w1, start, end, notes + snap_notes, so, eo, dropped)


def boundary_ok(tr: Transcript, b: Bounds) -> bool:
    """Starts on a complete thought and ends on one."""
    words = tr.words
    starts_ok = b.w0 == 0 or ends_sentence(words[b.w0 - 1].word) or b.start_offset > 0 or b.fillers_dropped > 0
    if not starts_ok:
        # sentence boundaries can also come from a speaker change or pause
        starts_ok = any(s.start_word == b.w0 for s in tr.sentences)
    ends_ok = ends_sentence(words[b.w1].word) or (b.end_offset > 0 and ends_clause(words[b.w1].word))
    if not ends_ok:
        ends_ok = b.end_offset > 0 or any(s.end_word == b.w1 and ends_clause(words[b.w1].word) for s in tr.sentences)
    return bool(starts_ok and ends_ok)
