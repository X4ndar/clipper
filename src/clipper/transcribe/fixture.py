"""Read a word-level transcript from disk instead of calling an ASR service.

Accepted formats:
  - our own: {"language": "en", "words": [{"word", "start", "end", "speaker", "score"}]}
  - an AssemblyAI response: {"words": [{"text", "start"(ms), "end"(ms), "confidence", "speaker"}]}
Used by tests, and to bring your own transcript (for example from a local WhisperX run).
"""

from __future__ import annotations

import json
from pathlib import Path

from ..transcript import Transcript, Word, segment_sentences
from .base import TranscribeRequest, TranscriptionError


class FixtureTranscriber:
    name = "fixture"

    def __init__(self, cfg):
        self.cfg = cfg

    def _path(self, req: TranscribeRequest) -> Path:
        if self.cfg.fixture_path:
            return Path(self.cfg.fixture_path)
        return req.video_path.with_name(req.video_path.stem + ".words.json")

    def cache_key(self, req: TranscribeRequest) -> str:
        return f"fixture-{self._path(req).name}"

    def estimate_cost(self, audio_seconds: float, req: TranscribeRequest) -> float:
        return 0.0

    def transcribe(self, req: TranscribeRequest) -> tuple[dict, Transcript]:
        p = self._path(req)
        if not p.is_file():
            raise TranscriptionError(f"fixture transcript not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        raw_words = data.get("words") or []
        words = []
        for w in raw_words:
            if "text" in w and "word" not in w:  # AssemblyAI shape
                words.append(Word(word=w["text"], start=w["start"] / 1000, end=w["end"] / 1000,
                                  speaker=w.get("speaker"), score=w.get("confidence")))
            else:
                words.append(Word(**w))
        duration = float(data.get("duration") or data.get("audio_duration") or (words[-1].end if words else 0))
        tr = Transcript(
            language=data.get("language") or data.get("language_code") or (req.language or "en"),
            duration=duration,
            backend=self.name,
            model=p.name,
            words=words,
            sentences=segment_sentences(words),
        )
        return data, tr
