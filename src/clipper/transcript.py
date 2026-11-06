"""Transcript data model, sentence segmentation, quality check and SRT export."""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

from pydantic import BaseModel

TERMINAL = (".", "?", "!", "…", "؟", "۔", "。", "？", "！")
CLAUSE_PUNCT = (",", ";", ":", "—", "–", "،", "؛")
ABBREVIATIONS = {
    "mr.", "mrs.", "ms.", "dr.", "prof.", "st.", "vs.", "etc.", "e.g.", "i.e.", "jr.", "sr.", "inc.",
    "u.s.", "u.k.", "a.m.", "p.m.", "no.", "approx.",
}
RTL_LANGS = {"ar", "fa", "he", "ur", "ps", "yi", "dv", "ckb"}


class Word(BaseModel):
    word: str
    start: float  # seconds
    end: float
    speaker: Optional[str] = None
    score: Optional[float] = None  # ASR confidence, 0..1


class Sentence(BaseModel):
    id: int
    start_word: int  # index into Transcript.words
    end_word: int  # inclusive
    start: float
    end: float
    speaker: Optional[str] = None
    text: str


class Transcript(BaseModel):
    language: str = "en"
    language_confidence: Optional[float] = None
    duration: float = 0.0
    backend: str = ""
    model: str = ""
    words: list[Word]
    sentences: list[Sentence] = []

    def text_range(self, w0: int, w1: int) -> str:
        return join_words([w.word for w in self.words[w0 : w1 + 1]])

    def sentence(self, sid: int) -> Sentence:
        return self.sentences[sid]


def base_lang(code: Optional[str]) -> str:
    return (code or "").lower().replace("-", "_").split("_")[0]


def is_rtl(code: Optional[str]) -> bool:
    return base_lang(code) in RTL_LANGS


def join_words(words: list[str]) -> str:
    return " ".join(w.strip() for w in words if w.strip())


def ends_sentence(token: str) -> bool:
    t = token.strip().rstrip("\"'”’)»]")
    if not t:
        return False
    if t.lower() in ABBREVIATIONS:
        return False
    return t.endswith(TERMINAL)


def ends_clause(token: str) -> bool:
    t = token.strip().rstrip("\"'”’)»]")
    return t.endswith(CLAUSE_PUNCT) or ends_sentence(token)


def normalize(token: str) -> str:
    return re.sub(r"[^\w']+", "", token.lower(), flags=re.UNICODE)


def segment_sentences(
    words: list[Word], max_sentence_s: float = 30.0, max_gap_s: float = 1.5
) -> list[Sentence]:
    """Group words into sentences: terminal punctuation, speaker change, long pause, or max length."""
    sentences: list[Sentence] = []
    if not words:
        return sentences
    start = 0

    def close(end_idx: int) -> None:
        nonlocal start
        ws = words[start : end_idx + 1]
        sentences.append(
            Sentence(
                id=len(sentences),
                start_word=start,
                end_word=end_idx,
                start=ws[0].start,
                end=ws[-1].end,
                speaker=ws[0].speaker,
                text=join_words([w.word for w in ws]),
            )
        )
        start = end_idx + 1

    for i, w in enumerate(words):
        last = i == len(words) - 1
        if last:
            close(i)
            break
        nxt = words[i + 1]
        boundary = ends_sentence(w.word)
        if not boundary and nxt.speaker != w.speaker and (nxt.speaker or w.speaker):
            boundary = True
        if not boundary and nxt.start - w.end > max_gap_s:
            boundary = True
        if not boundary and w.end - words[start].start > max_sentence_s and ends_clause(w.word):
            boundary = True
        if not boundary and w.end - words[start].start > max_sentence_s * 1.5:
            boundary = True
        if boundary:
            close(i)
    return sentences


def fmt_ts(t: float, force_hours: bool = False) -> str:
    t = max(0, int(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    if h or force_hours:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def sentence_lines(tr: Transcript, sids: list[int]) -> str:
    """Prompt lines: [id] [mm:ss] SPEAKER: text"""
    hours = tr.duration >= 3600
    out = []
    for sid in sids:
        s = tr.sentences[sid]
        spk = f"{s.speaker}: " if s.speaker else ""
        out.append(f"[{s.id}] [{fmt_ts(s.start, hours)}] {spk}{s.text}")
    return "\n".join(out)


# ---------- quality check ----------


class QualityReport(BaseModel):
    ok: bool
    words: int
    low_confidence_share: float
    repetition_share: float
    fast_sentences: int
    repeated_sentence_runs: int
    language: str
    warnings: list[str]


# Languages Universal-3.5 Pro covers natively, per AssemblyAI's model page (checked 2026-11-04).
# Other languages fall back to universal-2. This list only drives warnings.
STRONG_ASR_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "ar", "da", "nl", "fi", "he", "hi", "ja", "zh", "no", "nb",
    "sv", "tr", "vi",
}


def quality_check(tr: Transcript, cfg) -> QualityReport:
    """Flag a transcript before any LLM tokens are spent. `cfg` is a QualityConfig."""
    warnings: list[str] = []
    n = len(tr.words)
    scored = [w.score for w in tr.words if w.score is not None]
    low = sum(1 for s in scored if s < cfg.low_confidence_threshold)
    low_share = low / len(scored) if scored else 0.0

    # Immediately repeated n-grams ("the the the" loops, ASR hallucination loops)
    toks = [normalize(w.word) for w in tr.words]
    k = cfg.repetition_ngram
    flagged = [False] * n
    i = 0
    while i + 2 * k <= n:
        a = toks[i : i + k]
        if all(a) and a == toks[i + k : i + 2 * k]:
            j = i + k
            while j + k <= n and toks[j : j + k] == a:
                for t in range(j, j + k):
                    flagged[t] = True
                j += k
            i = j
        else:
            i += 1
    rep_share = sum(flagged) / n if n else 0.0

    fast = 0
    for s in tr.sentences:
        dur = s.end - s.start
        nw = s.end_word - s.start_word + 1
        if dur > 0.5 and nw / dur > cfg.max_words_per_second:
            fast += 1

    runs = 0
    texts = [normalize(s.text) for s in tr.sentences]
    for a, b, c in zip(texts, texts[1:], texts[2:]):
        if a and a == b == c and len(a) > 8:
            runs += 1

    ok = True
    if n == 0:
        ok = False
        warnings.append("Transcript is empty.")
    if low_share > cfg.max_low_confidence_share:
        ok = False
        warnings.append(
            f"{low_share:.0%} of words have confidence below {cfg.low_confidence_threshold} "
            f"(limit {cfg.max_low_confidence_share:.0%})."
        )
    if rep_share > cfg.max_repetition_share:
        ok = False
        warnings.append(f"{rep_share:.1%} of words sit in repeated n-gram loops (possible ASR hallucination).")
    if runs:
        ok = False
        warnings.append(f"{runs} runs of the same sentence repeated 3+ times.")
    if fast > max(2, len(tr.sentences) * 0.05):
        warnings.append(f"{fast} sentences exceed {cfg.max_words_per_second} words/second.")
    lang = base_lang(tr.language)
    if lang and lang not in STRONG_ASR_LANGS:
        warnings.append(
            f"Language '{tr.language}' is outside the ASR model's strongest set; expect more errors."
        )
    if lang == "ar":
        warnings.append(
            "Arabic: dialects (for example Moroccan Darija) and code-switching are much harder than "
            "Modern Standard Arabic. Check captions by hand."
        )
    return QualityReport(
        ok=ok,
        words=n,
        low_confidence_share=round(low_share, 4),
        repetition_share=round(rep_share, 4),
        fast_sentences=fast,
        repeated_sentence_runs=runs,
        language=tr.language,
        warnings=warnings,
    )


# ---------- SRT ----------


def _srt_ts(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(words: list[Word], max_chars: int = 42, max_lines: int = 2, max_dur: float = 6.0) -> str:
    cues: list[tuple[float, float, str]] = []
    cur: list[Word] = []

    def flush() -> None:
        if not cur:
            return
        lines, line = [], ""
        for w in cur:
            cand = (line + " " + w.word).strip()
            if len(cand) > max_chars and line:
                lines.append(line)
                line = w.word
            else:
                line = cand
        if line:
            lines.append(line)
        cues.append((cur[0].start, cur[-1].end, "\n".join(lines)))
        cur.clear()

    for w in words:
        if cur:
            total = len(" ".join(x.word for x in cur + [w]))
            too_long = total > max_chars * max_lines
            too_slow = w.end - cur[0].start > max_dur
            gap = w.start - cur[-1].end > 1.0
            if too_long or too_slow or gap:
                flush()
        cur.append(w)
        if ends_sentence(w.word):
            flush()
    flush()
    out = []
    for i, (a, b, text) in enumerate(cues, 1):
        out.append(f"{i}\n{_srt_ts(a)} --> {_srt_ts(b)}\n{text}\n")
    return "\n".join(out)


def word_confidence(words: list[Word]) -> Optional[float]:
    s = [w.score for w in words if w.score is not None]
    return sum(s) / len(s) if s else None


def speaker_counts(tr: Transcript) -> dict[str, int]:
    return dict(Counter(w.speaker for w in tr.words if w.speaker))
