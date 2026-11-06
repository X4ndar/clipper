import json

import httpx
import pytest

from clipper.config import TranscribeConfig
from clipper.transcribe.assemblyai import AssemblyAITranscriber
from clipper.transcribe.base import TranscribeRequest, TranscriptionError


def req(tmp_path, lang=None, spk=True, n=None):
    a = tmp_path / "a.wav"
    a.write_bytes(b"RIFF0000")
    return TranscribeRequest(audio_path=a, video_path=tmp_path / "v.mp4", language=lang, speaker_labels=spk,
                             speakers_expected=n)


def test_request_uses_plural_speech_models_and_detection(tmp_path):
    t = AssemblyAITranscriber(TranscribeConfig())
    body = t.build_request("https://x/u", req(tmp_path))
    assert body["speech_models"] == ["universal-3-5-pro"] and "speech_model" not in body
    assert body["language_detection"] is True and "language_code" not in body
    assert body["speaker_labels"] is True
    body = t.build_request("https://x/u", req(tmp_path, lang="fr", spk=False, n=3))
    assert body["language_code"] == "fr" and "language_detection" not in body and "speakers_expected" not in body


def test_cost_estimate():
    t = AssemblyAITranscriber(TranscribeConfig())
    r = TranscribeRequest(audio_path=None, video_path=None, language=None, speaker_labels=True)  # type: ignore
    assert t.estimate_cost(3600, r) == pytest.approx(0.23)
    r.speaker_labels = False
    assert t.estimate_cost(3600, r) == pytest.approx(0.21)


def test_full_flow_with_mock_transport(tmp_path, monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "k")
    seen = {}
    polls = {"n": 0}

    def handler(request: httpx.Request):
        assert request.headers["authorization"] == "k"
        if request.url.path == "/v2/upload":
            return httpx.Response(200, json={"upload_url": "https://cdn/x"})
        if request.url.path == "/v2/transcript" and request.method == "POST":
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json={"id": "t1", "status": "queued"})
        if request.url.path == "/v2/transcript/t1":
            polls["n"] += 1
            if polls["n"] < 2:
                return httpx.Response(200, json={"id": "t1", "status": "processing"})
            return httpx.Response(200, json={
                "id": "t1", "status": "completed", "language_code": "en_us", "audio_duration": 3,
                "words": [{"text": "Hello", "start": 100, "end": 400, "confidence": 0.9, "speaker": "A"},
                          {"text": "world.", "start": 450, "end": 900, "confidence": 0.8, "speaker": "A"}],
            })
        return httpx.Response(404)

    cfg = TranscribeConfig(poll_interval_s=0)
    client = httpx.Client(base_url="https://api.assemblyai.com", headers={"authorization": "k"},
                          transport=httpx.MockTransport(handler))
    raw, tr = AssemblyAITranscriber(cfg, client).transcribe(req(tmp_path))
    assert seen["body"]["audio_url"] == "https://cdn/x"
    assert tr.words[0].start == 0.1 and tr.words[1].end == 0.9 and tr.words[0].score == 0.9
    assert tr.language == "en_us" and len(tr.sentences) == 1 and tr.duration == 3


def test_error_status_raises(tmp_path):
    def handler(request):
        if request.url.path == "/v2/upload":
            return httpx.Response(401, text="bad key")
        return httpx.Response(404)

    client = httpx.Client(base_url="https://api.assemblyai.com", transport=httpx.MockTransport(handler))
    with pytest.raises(TranscriptionError, match="ASSEMBLYAI_API_KEY"):
        AssemblyAITranscriber(TranscribeConfig(), client).transcribe(req(tmp_path))


def test_eu_region_host():
    assert AssemblyAITranscriber(TranscribeConfig(region="eu")).base == "https://api.eu.assemblyai.com"
