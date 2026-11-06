"""Transcriber interface. Backends return the provider's raw response plus our Transcript."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from ..transcript import Transcript


class TranscriptionError(RuntimeError):
    pass


@dataclass
class TranscribeRequest:
    audio_path: Path
    video_path: Path
    language: Optional[str]  # None = auto-detect
    speaker_labels: bool
    speakers_expected: Optional[int] = None


class Transcriber(Protocol):
    name: str

    def cache_key(self, req: TranscribeRequest) -> str:
        """Everything except the audio hash that changes the output."""

    def estimate_cost(self, audio_seconds: float, req: TranscribeRequest) -> float: ...

    def transcribe(self, req: TranscribeRequest) -> tuple[dict, Transcript]: ...


def get_transcriber(cfg) -> Transcriber:
    """`cfg` is a TranscribeConfig."""
    if cfg.backend == "assemblyai":
        from .assemblyai import AssemblyAITranscriber

        return AssemblyAITranscriber(cfg)
    if cfg.backend == "fixture":
        from .fixture import FixtureTranscriber

        return FixtureTranscriber(cfg)
    raise ValueError(f"unknown transcriber backend {cfg.backend!r}")
