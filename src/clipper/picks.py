"""Hand-picked clips: turn a picks file into accepted candidates so the normal cut/crop/caption path runs.

A picks file is JSON: a list of clips (or {"clips": [...]}) in the order you want them ranked.
Each clip gives its span by time or by sentence ID (see transcript.json), plus optional metadata:

  [{"start": "9:50.5", "end": "10:44", "title": "...", "hook_text": "...", "rationale": "..."},
   {"start_sentence": 812, "end_sentence": 840}]

Times snap to whole sentences: the clip starts at the sentence that contains `start` and ends at the
sentence that contains `end`. The usual word-boundary and quiet-point snapping then applies.

To tighten a clip, give `segments` instead of one span: the pieces are joined with jump cuts and
play in the order listed (so a clip can open on its best line, then go back to the build-up):

  {"segments": [{"start_sentence": 645, "end_sentence": 647}, {"start": "20:05", "end": "20:30"}],
   "title": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, model_validator

from .select import ClipRecord
from .transcript import Transcript

MANUAL = "manual"


class Span(BaseModel):
    start: Optional[Union[str, float]] = None
    end: Optional[Union[str, float]] = None
    start_sentence: Optional[int] = None
    end_sentence: Optional[int] = None

    @model_validator(mode="after")
    def _span(self) -> "Span":
        if (self.start is None) == (self.start_sentence is None):
            raise ValueError("give exactly one of start or start_sentence")
        if (self.end is None) == (self.end_sentence is None):
            raise ValueError("give exactly one of end or end_sentence")
        return self


class Pick(BaseModel):
    start: Optional[Union[str, float]] = None
    end: Optional[Union[str, float]] = None
    start_sentence: Optional[int] = None
    end_sentence: Optional[int] = None
    segments: list[Span] = []
    title: Optional[str] = None
    hook_text: Optional[str] = None
    description: Optional[str] = None
    tags: list[str] = []
    rationale: Optional[str] = None
    content_type: Optional[str] = None

    @model_validator(mode="after")
    def _span(self) -> "Pick":
        whole = any(v is not None for v in (self.start, self.end, self.start_sentence, self.end_sentence))
        if self.segments:
            if whole:
                raise ValueError("give either segments or one start/end span, not both")
            return self
        Span(start=self.start, end=self.end, start_sentence=self.start_sentence, end_sentence=self.end_sentence)
        return self

    def spans(self) -> list[Span]:
        if self.segments:
            return self.segments
        return [Span(start=self.start, end=self.end, start_sentence=self.start_sentence,
                     end_sentence=self.end_sentence)]


def parse_time(v: Union[str, float]) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    t = 0.0
    for part in v.strip().split(":"):
        t = t * 60 + float(part)
    return t


def load_picks(path: Path) -> list[Pick]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    items = data["clips"] if isinstance(data, dict) else data
    if not isinstance(items, list) or not items:
        raise ValueError(f"{path}: expected a non-empty list of clips")
    out = []
    for i, it in enumerate(items, 1):
        try:
            out.append(Pick.model_validate(it))
        except ValueError as e:
            raise ValueError(f"{path}: clip {i}: {e}") from e
    return out


def sentence_at(tr: Transcript, t: float) -> int:
    """The sentence that contains time t (or the last one starting before it)."""
    best = 0
    for s in tr.sentences:
        if s.start <= t + 0.05:
            best = s.id
        else:
            break
    return best


def picks_to_records(tr: Transcript, picks: list[Pick], source: str) -> list[ClipRecord]:
    n = len(tr.sentences)
    recs = []
    for i, p in enumerate(picks, 1):
        ranges = []
        for sp in p.spans():
            s0 = sp.start_sentence if sp.start_sentence is not None else sentence_at(tr, parse_time(sp.start))
            s1 = sp.end_sentence if sp.end_sentence is not None else sentence_at(tr, parse_time(sp.end))
            if not (0 <= s0 < n and 0 <= s1 < n) or s1 < s0:
                raise ValueError(f"clip {i}: sentence range {s0}-{s1} is outside the transcript (0-{n - 1}) "
                                 "or reversed")
            ranges.append((s0, s1))
        segs = [{"start_sentence_id": a, "end_sentence_id": b} for a, b in ranges] if len(ranges) > 1 else []
        recs.append(ClipRecord(
            id=f"m{i:02d}", status="accepted",
            start_sentence_id=min(a for a, _ in ranges), end_sentence_id=max(b for _, b in ranges),
            segments=segs,
            pass1={"source": MANUAL, "file": source, "order": i, "ranges": ranges},
            # Ranked in file order; finalize() sorts by editorial score.
            editorial=round(10.0 - i * 0.01, 2),
            confidence={"llm": None},
            title=p.title, hook_text=p.hook_text, description=p.description, tags=p.tags,
            rationale=p.rationale or "Picked by hand.", content_type=p.content_type,
            notes=[f"picked by hand ({source})"],
        ))
    return recs

