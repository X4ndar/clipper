"""Pass 1 (candidate generation), pass 2 (judging) and final selection."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from pydantic import BaseModel, Field

from .boundaries import Bounds, boundary_ok, compute_bounds
from .config import EVENT_TYPES
from .llm.client import BudgetExceeded, LLMCall, LLMError, LLMService, load_prompt
from .llm.schemas import JudgeOutput, Pass1Output
from .media import Wav
from .transcript import Transcript, fmt_ts, normalize, sentence_lines, word_confidence

SCORE_KEYS = ("hook", "standalone_sense", "payoff", "peak", "quotability", "clean_ending")

STATUS_SELECTED = "selected"


class ClipRecord(BaseModel):
    id: str
    status: str = "pending"  # pending|accepted|selected|rejected|duplicate|out_of_range|below_threshold|over_quota|invalid|error|skipped
    status_reason: Optional[str] = None
    pass1: dict
    judge: Optional[dict] = None
    start_sentence_id: int
    end_sentence_id: int
    start_word_offset: int = 0
    end_word_offset: int = 0
    w0: Optional[int] = None
    w1: Optional[int] = None
    start: Optional[float] = None
    end: Optional[float] = None
    duration: Optional[float] = None
    text: str = ""
    scores: dict[str, float] = Field(default_factory=dict)
    editorial: Optional[float] = None
    confidence: dict[str, Any] = Field(default_factory=dict)
    technical: dict[str, Any] = Field(default_factory=lambda: {"crop_ok": None, "captions_ok": None, "audio_ok": None})
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    hook_text: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    rationale: Optional[str] = None
    content_type: Optional[str] = None
    long_clip: bool = False
    rank: Optional[int] = None
    files: dict[str, str] = Field(default_factory=dict)
    # Hand-edited clips (picks.py) may join several spans with jump cuts, played in this order.
    # Each: {start_sentence_id, end_sentence_id} and, after finalize(), w0, w1, start, end.
    segments: list[dict[str, Any]] = Field(default_factory=list)


def clip_spans(r: ClipRecord) -> list[tuple[float, float, int, int]]:
    """(start, end, w0, w1) per piece of the clip, in play order. One piece unless jump-cut."""
    if r.segments:
        return [(g["start"], g["end"], g["w0"], g["w1"]) for g in r.segments]
    return [(r.start, r.end, r.w0, r.w1)]


def resolve_prompt(name: str, kind: str, content_type: str) -> str:
    """`auto` picks the prompt version by content type; anything else is used as given."""
    if name != "auto":
        return name
    return f"{kind}_v2" if content_type in EVENT_TYPES else f"{kind}_v1"


def selection_mode(cfg, content_type: str, duration_s: float, has_signals: bool) -> str:
    mode = cfg.selection.mode
    if mode == "auto":
        if content_type in EVENT_TYPES and has_signals:
            mode = "peaks" if duration_s > cfg.selection.peaks_only_over_s else "both"
        else:
            mode = "transcript"
    if mode != "transcript" and not has_signals:
        return "transcript"
    return mode


def _direction(spec: dict) -> str:
    d = (spec.get("direction") or "").strip()
    return f"Editor's direction (follow it when choosing and ranking): {d}\n" if d else ""


def peak_candidates(tr: Transcript, sig: dict, spec: dict, cfg, start_index: int = 0) -> list[ClipRecord]:
    """One candidate per audio/chat peak: the sentences from the build-up before it to the reaction
    after it. The judge then fixes the exact boundaries."""
    sc = cfg.signals
    span = min(spec["max_duration"], sc.pre_s + sc.post_s)
    pre = span * sc.pre_s / (sc.pre_s + sc.post_s)
    out: list[ClipRecord] = []
    for p in sig.get("peaks", []):
        t0, t1 = p["t"] - pre, p["t"] + (span - pre)
        ids = [s.id for s in tr.sentences if s.end > t0 and s.start < t1]
        if len(ids) < 2:
            continue  # nobody is talking around this peak; nothing to caption or judge
        why = f"audio {p['audio_z']:.1f} sigma above the usual level"
        if p.get("chat_z") is not None:
            why += f", chat {p['chat_z']:.1f} sigma above its usual rate"
        out.append(ClipRecord(
            id=f"p{start_index + len(out) + 1:02d}", start_sentence_id=ids[0], end_sentence_id=ids[-1],
            pass1={"source": "peaks", "peak_t": p["t"], "peak_score": p["score"],
                   "reason": f"Signal peak at {fmt_ts(p['t'])} ({why}). The outcome or reaction is probably "
                             "near that time; find the setup before it and end after the reaction."},
        ))
    return out


def estimate_tokens(text: str) -> int:
    # Conservative: ~3.5 characters per token for English; other scripts are denser.
    return int(len(text) / 3.5) + 1


def sentence_payload(tr: Transcript, sids: list[int]) -> list[dict]:
    return [
        {"id": s.id, "start": s.start, "end": s.end, "speaker": s.speaker, "text": s.text}
        for s in (tr.sentences[i] for i in sids)
    ]


def chunk_sentences(tr: Transcript, limit: int, overlap: int) -> list[list[int]]:
    lines = [sentence_lines(tr, [s.id]) for s in tr.sentences]
    toks = [estimate_tokens(line) for line in lines]
    if sum(toks) <= limit:
        return [list(range(len(lines)))]
    chunks: list[list[int]] = []
    i = 0
    while i < len(lines):
        cur, total, j = [], 0, i
        while j < len(lines) and (total + toks[j] <= limit or not cur):
            cur.append(j)
            total += toks[j]
            j += 1
        chunks.append(cur)
        if j >= len(lines):
            break
        # step back for overlap
        back, k = 0, j
        while k > i + 1 and back < overlap:
            k -= 1
            back += toks[k]
        i = k
    return chunks


# ---------------- pass 1 ----------------


def generate_candidates(tr: Transcript, spec: dict, cfg, llm: LLMService) -> tuple[list[ClipRecord], list[dict]]:
    """Returns (valid candidate records, invalid raw candidates with reasons)."""
    sel, lc = cfg.selection, cfg.llm
    want = max(1, math.ceil(spec["max_clips"] * lc.candidate_multiplier))
    prompt_name = resolve_prompt(lc.candidate_prompt, "candidates", spec["content_type"])
    system = load_prompt(prompt_name)
    chunks = chunk_sentences(tr, lc.chunk_token_limit, lc.chunk_overlap_tokens)
    total_sents = len(tr.sentences)
    raw: list[tuple[int, Any]] = []
    for ci, sids in enumerate(chunks):
        count = want if len(chunks) == 1 else max(2, math.ceil(want * len(sids) / total_sents) + 1)
        user = (
            f"Content type hint: {spec['content_type']}\n"
            + _direction(spec)
            + f"Language: {tr.language or 'unknown'}\n"
            f"Video duration: {fmt_ts(tr.duration)}\n"
            + (f"This is part {ci + 1} of {len(chunks)} of the transcript.\n" if len(chunks) > 1 else "")
            + f"Return up to {count} candidates. Target clip length: {spec['min_duration']:.0f}-"
            f"{spec['max_duration']:.0f} seconds (up to {sel.extended_max_duration:.0f} only if the idea "
            f"needs two beats).\n\nTranscript:\n{sentence_lines(tr, sids)}"
        )
        call = LLMCall(
            task="candidates", model=lc.candidate_model, prompt_version=prompt_name, system=system,
            user=user, schema=Pass1Output, effort=lc.candidate_effort, max_tokens=lc.max_tokens,
            payload={"sentences": sentence_payload(tr, sids), "count": count,
                     "min_duration": spec["min_duration"], "max_duration": spec["max_duration"]},
        )
        resp = llm.call(call, stage="candidates")
        raw.extend((ci, c) for c in resp.parsed.candidates)  # type: ignore[attr-defined]

    records: list[ClipRecord] = []
    invalid: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for ci, c in raw:
        d = c.model_dump()
        d["chunk"] = ci
        s0, s1 = c.start_sentence_id, c.end_sentence_id
        if not (0 <= s0 < total_sents and 0 <= s1 < total_sents):
            invalid.append({**d, "invalid_reason": "sentence id out of range"})
            continue
        if s1 < s0:
            invalid.append({**d, "invalid_reason": "end before start"})
            continue
        if (s0, s1) in seen:
            continue  # same span from overlapping chunks
        seen.add((s0, s1))
        records.append(ClipRecord(id=f"c{len(records) + 1:02d}", pass1=d, start_sentence_id=s0, end_sentence_id=s1,
                                  content_type=c.content_type))
    # Skip judging near-identical spans (saves judge tokens).
    kept: list[ClipRecord] = []
    for r in records:
        a = (tr.sentences[r.start_sentence_id].start, tr.sentences[r.end_sentence_id].end)
        dup = next((k for k in kept if iou(a, (tr.sentences[k.start_sentence_id].start,
                                                tr.sentences[k.end_sentence_id].end)) > 0.85), None)
        if dup:
            r.status, r.status_reason = "duplicate", f"near-identical to {dup.id} from pass 1"
        else:
            kept.append(r)
    return records, invalid


# ---------------- pass 2 ----------------


def judge_one(tr: Transcript, rec: ClipRecord, spec: dict, cfg, llm: LLMService, system: str) -> ClipRecord:
    lc, sel = cfg.llm, cfg.selection
    s0, s1 = rec.start_sentence_id, rec.end_sentence_id
    t0, t1 = tr.sentences[s0].start, tr.sentences[s1].end
    before = [s.id for s in tr.sentences[:s0] if s.end >= t0 - lc.context_seconds]
    after = [s.id for s in tr.sentences[s1 + 1:] if s.start <= t1 + lc.context_seconds]
    clip = list(range(s0, s1 + 1))
    user = (
        f"Content type hint: {spec['content_type']}. Target length {spec['min_duration']:.0f}-"
        f"{spec['max_duration']:.0f} s (hard max {sel.extended_max_duration:.0f} s).\n"
        + _direction(spec)
        +        f"First-pass proposal: sentences {s0}-{s1} (about {t1 - t0:.0f} s). "
        f"Reason given: {rec.pass1.get('reason', '')}\n\n"
        f"CONTEXT BEFORE:\n{sentence_lines(tr, before) or '(start of video)'}\n\n"
        f"CLIP:\n{sentence_lines(tr, clip)}\n\n"
        f"CONTEXT AFTER:\n{sentence_lines(tr, after) or '(end of video)'}"
    )
    call = LLMCall(
        task="judge", model=lc.judge_model,
        prompt_version=resolve_prompt(lc.judge_prompt, "judge", spec["content_type"]), system=system, user=user,
        schema=JudgeOutput, effort=lc.judge_effort, max_tokens=lc.max_tokens, fallbacks=lc.judge_fallbacks,
        payload={"clip_sentences": sentence_payload(tr, clip)},
    )
    resp = llm.call(call, stage="judge")
    j: JudgeOutput = resp.parsed  # type: ignore[assignment]
    rec.judge = j.model_dump()

    # Clamp boundary moves to sentences the judge actually saw, and to a few sentences.
    shown_lo = before[0] if before else s0
    shown_hi = after[-1] if after else s1
    m = lc.max_boundary_shift_sentences
    if rec.pass1.get("source") == "peaks":
        m = max(m, s1 - s0)  # a peak window is a rough guess: let the judge trim it freely
    ns0 = min(max(j.start_sentence_id, shown_lo, s0 - m), s0 + m)
    ns1 = min(max(j.end_sentence_id, s1 - m), shown_hi, s1 + m)
    if ns1 < ns0:
        ns0, ns1 = s0, s1
        rec.notes.append("judge boundary move ignored: end before start")
    elif (ns0, ns1) != (j.start_sentence_id, j.end_sentence_id):
        rec.notes.append(f"judge boundaries clamped to {ns0}-{ns1}")
    if (ns0, ns1) != (s0, s1):
        rec.notes.append(f"judge moved span {s0}-{s1} -> {ns0}-{ns1}")
    rec.start_sentence_id, rec.end_sentence_id = ns0, ns1
    rec.start_word_offset = max(0, j.start_word_offset)
    rec.end_word_offset = max(0, j.end_word_offset)
    rec.scores = {k: float(max(0, min(10, getattr(j.scores, k)))) for k in SCORE_KEYS}
    rec.editorial = editorial_score(rec.scores, sel.weights)
    rec.confidence["llm"] = round(max(0.0, min(1.0, j.confidence)), 3)
    rec.rationale = j.rationale
    rec.hook_text = j.hook_text
    rec.title, rec.description, rec.tags = j.title, j.description, list(j.tags)
    rec.content_type = j.content_type

    # Hard gates: whatever the total, no standalone sense or no payoff means no clip.
    gate_fail = [k for k in ("standalone_sense", "payoff") if rec.scores[k] < sel.gate_min]
    if gate_fail:
        rec.status = "rejected"
        rec.status_reason = (
            "hard gate: " + ", ".join(f"{k}={rec.scores[k]:.0f}<{sel.gate_min}" for k in gate_fail)
            + (f". {j.reject_reason}" if j.reject_reason else "")
        )
    elif j.reject:
        rec.status, rec.status_reason = "rejected", f"judge rejected: {j.reject_reason or 'no reason given'}"
    else:
        rec.status = "accepted"
    return rec


def judge_candidates(tr: Transcript, records: list[ClipRecord], spec: dict, cfg, llm: LLMService) -> list[ClipRecord]:
    system = load_prompt(resolve_prompt(cfg.llm.judge_prompt, "judge", spec["content_type"]))
    todo = [r for r in records if r.status == "pending"]
    budget_hit = False

    def work(r: ClipRecord) -> ClipRecord:
        nonlocal budget_hit
        if budget_hit:
            r.status, r.status_reason = "skipped", "LLM budget reached before judging"
            return r
        try:
            return judge_one(tr, r, spec, cfg, llm, system)
        except BudgetExceeded as e:
            budget_hit = True
            r.status, r.status_reason = "skipped", str(e)
        except LLMError as e:
            r.status, r.status_reason = "error", f"judge failed: {e}"
        return r

    with ThreadPoolExecutor(max_workers=max(1, cfg.llm.judge_concurrency)) as ex:
        list(ex.map(work, todo))
    return records


def editorial_score(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted mean on 0-10. Signals not available yet (e.g. speaker_energy in M1) are skipped."""
    num = den = 0.0
    for k, w in weights.items():
        if k in scores and w > 0:
            num += w * scores[k]
            den += w
    return round(num / den, 3) if den else 0.0


# ---------------- final selection ----------------


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def shorter_overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    short = min(a[1] - a[0], b[1] - b[0])
    return inter / short if short > 0 else 0.0


def jaccard(a: str, b: str) -> float:
    sa = {normalize(t) for t in a.split()} - {""}
    sb = {normalize(t) for t in b.split()} - {""}
    return len(sa & sb) / len(sa | sb) if sa and sb else 0.0


def manual(r: ClipRecord) -> bool:
    """Picked by hand (see picks.py), not by the LLM passes."""
    return r.pass1.get("source") == "manual"


def _finalize_segments(tr: Transcript, r: ClipRecord, wav: Optional[Wav], cfg) -> None:
    """Snap every piece of a jump-cut clip on its own; the clip's span and text cover all pieces."""
    oks, words, texts = [], [], []
    for g in r.segments:
        b = compute_bounds(tr, g["start_sentence_id"], g["end_sentence_id"], 0, 0, wav, cfg.snap)
        g.update({"w0": b.w0, "w1": b.w1, "start": b.start, "end": b.end})
        r.notes.extend(b.notes)
        oks.append(boundary_ok(tr, b))
        words += tr.words[b.w0 : b.w1 + 1]
        texts.append(tr.text_range(b.w0, b.w1))
    first, last = r.segments[0], r.segments[-1]
    r.w0, r.w1 = first["w0"], last["w1"]
    r.start = min(g["start"] for g in r.segments)
    r.end = max(g["end"] for g in r.segments)
    r.duration = round(sum(g["end"] - g["start"] for g in r.segments), 3)
    r.text = " … ".join(texts)
    asr = word_confidence(words)
    r.confidence.update({"asr": round(asr, 3) if asr is not None else None, "boundary_ok": all(oks),
                         "combined": round(asr, 3) if asr is not None else None})


def finalize(tr: Transcript, records: list[ClipRecord], spec: dict, cfg, wav: Optional[Wav]) -> list[ClipRecord]:
    sel = cfg.selection
    lo, hi = spec["min_duration"], spec["max_duration"]
    ext = max(hi, min(sel.extended_max_duration, sel.hard_cap_duration))
    pool: list[ClipRecord] = []
    for r in records:
        if r.status != "accepted":
            continue
        if r.segments:
            _finalize_segments(tr, r, wav, cfg)
            if r.duration > sel.hard_cap_duration:
                r.status = "out_of_range"
                r.status_reason = f"too long ({r.duration:.1f}s > {sel.hard_cap_duration:.0f}s hard cap)"
                continue
            r.long_clip = r.duration > hi
            pool.append(r)
            continue
        b: Bounds = compute_bounds(tr, r.start_sentence_id, r.end_sentence_id, r.start_word_offset,
                                   r.end_word_offset, wav, cfg.snap)
        r.w0, r.w1, r.start, r.end = b.w0, b.w1, b.start, b.end
        r.duration = round(b.duration, 3)
        r.start_word_offset, r.end_word_offset = b.start_offset, b.end_offset
        r.notes.extend(b.notes)
        r.text = tr.text_range(b.w0, b.w1)
        asr = word_confidence(tr.words[b.w0 : b.w1 + 1])
        bok = boundary_ok(tr, b)
        llm_c = r.confidence.get("llm")
        r.confidence.update({
            "asr": round(asr, 3) if asr is not None else None,
            "boundary_ok": bok,
            "combined": round(min(x for x in (llm_c, asr) if x is not None), 3)
            if (llm_c is not None or asr is not None) else None,
        })
        if manual(r):  # a person chose this span: only the hard cap applies
            if r.duration > sel.hard_cap_duration:
                r.status = "out_of_range"
                r.status_reason = f"too long ({r.duration:.1f}s > {sel.hard_cap_duration:.0f}s hard cap)"
                continue
            r.long_clip = r.duration > hi
            pool.append(r)
            continue
        if r.duration < lo:
            r.status, r.status_reason = "out_of_range", f"too short ({r.duration:.1f}s < {lo:.0f}s)"
            continue
        if r.duration > ext:
            r.status, r.status_reason = "out_of_range", f"too long ({r.duration:.1f}s > {ext:.0f}s)"
            continue
        bar = sel.min_editorial
        if r.duration > hi:
            r.long_clip = True
            bar += sel.extended_bonus_required
        if (r.editorial or 0) < bar:
            r.status = "below_threshold"
            r.status_reason = f"editorial {r.editorial:.2f} < {bar:.2f}" + (" (long clip bar)" if r.long_clip else "")
            continue
        pool.append(r)

    pool.sort(key=lambda r: (-(r.editorial or 0), -r.scores.get("hook", 0), r.start or 0))
    kept: list[ClipRecord] = []
    for r in pool:
        span = (r.start, r.end)
        dup = None
        for k in kept:
            kspan = (k.start, k.end)
            if iou(span, kspan) > sel.dedupe_iou or shorter_overlap(span, kspan) > sel.dedupe_shorter_overlap:
                dup = k
            elif shorter_overlap(span, kspan) > 0 and jaccard(r.text, k.text) > sel.dedupe_text_jaccard:
                dup = k
            if dup:
                break
        if dup:
            r.status, r.status_reason = "duplicate", f"overlaps higher-ranked {dup.id}"
            continue
        if len(kept) >= spec["max_clips"] and not manual(r):
            r.status, r.status_reason = "over_quota", f"max_clips={spec['max_clips']} already filled"
            continue
        kept.append(r)

    for rank, r in enumerate(kept, 1):
        r.status, r.rank = STATUS_SELECTED, rank
        r.status_reason = None
        reasons = []
        comb = r.confidence.get("combined")
        if comb is not None and comb < sel.min_confidence:
            reasons.append(f"low confidence {comb:.2f}")
        if not r.confidence.get("boundary_ok"):
            reasons.append("start or end may not be a complete thought")
        if r.scores.get("clean_ending", 10) < 5:
            reasons.append("weak ending")
        r.review_reasons = reasons
        r.needs_review = bool(reasons)
    return records
