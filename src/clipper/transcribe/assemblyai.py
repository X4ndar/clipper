"""AssemblyAI pre-recorded transcription over the REST API.

- Always sends `speech_models` (plural) explicitly.
- Uploads the mono 16 kHz WAV we extracted (multichannel files are billed per channel).
- Word timestamps arrive in milliseconds; we store seconds.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import httpx

from ..transcript import Transcript, Word, segment_sentences
from .base import TranscribeRequest, TranscriptionError

HOSTS = {"us": "https://api.assemblyai.com", "eu": "https://api.eu.assemblyai.com"}


class AssemblyAITranscriber:
    name = "assemblyai"

    def __init__(self, cfg, client: Optional[httpx.Client] = None):
        self.cfg = cfg
        self.base = HOSTS[cfg.region]
        self._client = client

    def _http(self) -> httpx.Client:
        if self._client is None:
            key = os.environ.get("ASSEMBLYAI_API_KEY")
            if not key:
                raise TranscriptionError(
                    "ASSEMBLYAI_API_KEY is not set. Put it in .env (see .env.example) or the environment."
                )
            self._client = httpx.Client(
                base_url=self.base, headers={"authorization": key}, timeout=httpx.Timeout(120.0)
            )
        return self._client

    def cache_key(self, req: TranscribeRequest) -> str:
        models = "+".join(self.cfg.speech_models)
        return f"assemblyai-{models}-spk{int(req.speaker_labels)}-{req.language or 'auto'}"

    def estimate_cost(self, audio_seconds: float, req: TranscribeRequest) -> float:
        rate = self.cfg.price_per_hour_usd + (self.cfg.diarization_price_per_hour_usd if req.speaker_labels else 0)
        return audio_seconds / 3600.0 * rate

    def build_request(self, audio_url: str, req: TranscribeRequest) -> dict:
        body: dict = {
            "audio_url": audio_url,
            "speech_models": list(self.cfg.speech_models),
            "punctuate": True,
            "format_text": True,
            "speaker_labels": bool(req.speaker_labels),
        }
        if req.language:
            body["language_code"] = req.language
        else:
            body["language_detection"] = True
        if req.speaker_labels and req.speakers_expected and req.speakers_expected > 1:
            body["speakers_expected"] = int(req.speakers_expected)
        return body

    def upload(self, req: TranscribeRequest) -> str:
        # Prefer the FLAC copy made at ingest (same audio, smaller upload).
        path = req.audio_path.with_suffix(".flac")
        if not path.is_file():
            path = req.audio_path

        def chunks():
            with open(path, "rb") as f:
                while True:
                    b = f.read(5 * 1024 * 1024)
                    if not b:
                        break
                    yield b

        r = self._http().post("/v2/upload", content=chunks(), headers={"content-type": "application/octet-stream"})
        self._check(r, "upload")
        return r.json()["upload_url"]

    def transcribe(self, req: TranscribeRequest) -> tuple[dict, Transcript]:
        audio_url = self.upload(req)
        r = self._http().post("/v2/transcript", json=self.build_request(audio_url, req))
        self._check(r, "submit")
        tid = r.json()["id"]
        deadline = time.monotonic() + self.cfg.timeout_s
        while True:
            r = self._http().get(f"/v2/transcript/{tid}")
            self._check(r, "poll")
            data = r.json()
            status = data.get("status")
            if status == "completed":
                return data, self.to_transcript(data)
            if status == "error":
                raise TranscriptionError(f"AssemblyAI error: {data.get('error')}")
            if time.monotonic() > deadline:
                raise TranscriptionError(f"AssemblyAI transcript {tid} timed out after {self.cfg.timeout_s}s")
            time.sleep(self.cfg.poll_interval_s)

    @staticmethod
    def _check(r: httpx.Response, what: str) -> None:
        if r.status_code >= 400:
            msg = r.text[:500]
            if r.status_code in (401, 403):
                msg += " (check ASSEMBLYAI_API_KEY)"
            raise TranscriptionError(f"AssemblyAI {what} failed: HTTP {r.status_code}: {msg}")

    def to_transcript(self, data: dict) -> Transcript:
        words = [
            Word(
                word=w["text"],
                start=w["start"] / 1000.0,
                end=w["end"] / 1000.0,
                speaker=w.get("speaker"),
                score=w.get("confidence"),
            )
            for w in data.get("words") or []
            if (w.get("text") or "").strip()
        ]
        duration = float(data.get("audio_duration") or (words[-1].end if words else 0.0))
        return Transcript(
            language=data.get("language_code") or "",
            language_confidence=data.get("language_confidence"),
            duration=duration,
            backend=self.name,
            model=",".join(self.cfg.speech_models),
            words=words,
            sentences=segment_sentences(words),
        )
