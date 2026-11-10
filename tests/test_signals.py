"""Audio and chat signals on synthetic data."""

import json
import wave

import numpy as np
import pytest

from clipper.config import Config, SignalsConfig
from clipper.select import peak_candidates, resolve_prompt, selection_mode
from clipper.signals import SignalError, compute_signals, energy_score, find_peaks, load_chat, rolling_z

from conftest import make_transcript

SR = 16000


def write_wav(path, seconds, loud_at=(), level=0.02, loud=0.4):
    rng = np.random.default_rng(1)
    x = rng.normal(0, level, int(seconds * SR))
    for t in loud_at:
        x[int(t * SR): int((t + 3) * SR)] = rng.normal(0, loud, 3 * SR)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return path


def test_rolling_z_flags_a_spike_but_not_a_flat_signal():
    x = np.full(600, -30.0)
    assert np.abs(rolling_z(x, 120, 1.5)).max() == 0
    x[300:303] = -10
    z = rolling_z(x, 120, 1.5)
    assert z[301] > 5 and abs(z[100]) < 0.1


def test_find_peaks_keeps_the_strongest_and_spaces_them():
    s = np.zeros(400)
    s[50], s[60], s[200], s[300] = 3, 5, 4, 0.5
    assert find_peaks(s, 1.0, 45, 10) == [60, 200]
    assert find_peaks(s, 1.0, 45, 1) == [60]


def test_audio_peaks_are_found(tmp_path):
    wav = write_wav(tmp_path / "a.wav", 600, loud_at=(120, 400))
    sig = compute_signals(wav, None, SignalsConfig(), 10)
    assert [round(p["t"] / 10) * 10 for p in sig["peaks"]] == [120, 400]
    assert not sig["has_chat"] and sig["chat_z"] is None
    assert energy_score(sig, 110, 130) > 8 > energy_score(sig, 200, 260)


def test_chat_formats(tmp_path):
    td = tmp_path / "td.json"
    td.write_text(json.dumps({"comments": [
        {"content_offset_seconds": 12.5, "message": {"body": "LUL", "fragments": [{"text": "LUL", "emoticon": {"emoticon_id": "1"}}]}},
        {"content_offset_seconds": 3, "message": {"body": "hi", "fragments": [{"text": "hi"}]}}]}), encoding="utf-8")
    m = load_chat(td)
    assert [x.t for x in m] == [3, 12.5] and m[1].emotes == 1
    yt = tmp_path / "v.live_chat.json"
    line = {"replayChatItemAction": {"videoOffsetTimeMsec": "61000", "actions": [{"addChatItemAction": {"item": {
        "liveChatTextMessageRenderer": {"message": {"runs": [{"text": "no way "}, {"emoji": {"emojiId": "x"}}]}}}}}]}}
    yt.write_text(json.dumps(line) + "\n" + json.dumps({"other": 1}) + "\n", encoding="utf-8")
    m = load_chat(yt)
    assert m[0].t == 61.0 and m[0].text == "no way " and m[0].emotes == 1
    txt = tmp_path / "c.txt"
    txt.write_text("[00:01:05] bob: KEKW\n[01:00:00] amy: what: happened\nnot a chat line\n", encoding="utf-8")
    m = load_chat(txt)
    assert [x.t for x in m] == [65, 3600] and m[1].text == "what: happened"
    txt.write_text("nothing here\n", encoding="utf-8")
    with pytest.raises(SignalError):
        load_chat(txt)


def test_chat_burst_makes_a_peak_and_agreement_scores_higher(tmp_path):
    wav = write_wav(tmp_path / "a.wav", 900, loud_at=(600,))
    rng = np.random.default_rng(2)
    lines = [(float(t), "hello") for t in rng.uniform(0, 900, 300)]
    lines += [(305 + float(t), "LUL") for t in rng.uniform(0, 8, 80)]  # chat-only moment at 300 (5 s lag)
    lines += [(605 + float(t), "OMG") for t in rng.uniform(0, 8, 80)]  # chat + audio at 600
    chat = tmp_path / "c.txt"
    chat.write_text("\n".join(f"[{int(t)//3600:02d}:{int(t)%3600//60:02d}:{int(t)%60:02d}] u: {m}" for t, m in sorted(lines)),
                    encoding="utf-8")
    sig = compute_signals(wav, chat, SignalsConfig(), 5)
    by_t = {round(p["t"] / 20) * 20: p["score"] for p in sig["peaks"]}
    assert 300 in by_t and 600 in by_t and by_t[600] > by_t[300]
    assert sig["has_chat"] and sig["chat_messages"] == 460


def test_peak_candidates_cover_build_up_and_reaction(tmp_path):
    tr = make_transcript([f"Sentence number {i} goes here." for i in range(60)])
    peak_t = tr.sentences[40].start
    sig = {"peaks": [{"t": peak_t, "score": 3.0, "audio_z": 3.0, "chat_z": None}, {"t": 99999, "score": 2, "audio_z": 2, "chat_z": None}]}
    cfg = Config(home=str(tmp_path))
    recs = peak_candidates(tr, sig, {"max_duration": 60}, cfg)
    assert len(recs) == 1  # the second peak has no speech around it
    r = recs[0]
    s0, s1 = tr.sentences[r.start_sentence_id], tr.sentences[r.end_sentence_id]
    assert s0.start < peak_t - 30 and peak_t + 8 < s1.end < peak_t + 20
    assert r.pass1["source"] == "peaks" and "Signal peak" in r.pass1["reason"]


def test_mode_and_prompt_resolution(tmp_path):
    cfg = Config(home=str(tmp_path))
    assert selection_mode(cfg, "podcast", 3600, True) == "transcript"
    assert selection_mode(cfg, "stream", 1200, True) == "both"
    assert selection_mode(cfg, "stream", 4 * 3600, True) == "peaks"
    assert selection_mode(cfg, "stream", 4 * 3600, False) == "transcript"
    assert resolve_prompt("auto", "judge", "challenge") == "judge_v2"
    assert resolve_prompt("auto", "candidates", "podcast") == "candidates_v1"
    assert resolve_prompt("judge_v1", "judge", "stream") == "judge_v1"
