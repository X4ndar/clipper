import wave

import numpy as np

from clipper.boundaries import apply_offsets, boundary_ok, compute_bounds, drop_leading_fillers, snap
from clipper.config import SnapConfig
from clipper.media import Wav
from clipper.transcript import Transcript, Word, segment_sentences

from conftest import make_transcript


def test_offsets_respect_clause_boundaries():
    tr = make_transcript(["I mean, honestly the real secret is focus.", "It changes everything."])
    # "I mean," -> drop two words: previous word "mean," ends a clause -> valid
    w0, w1, so, eo, notes = apply_offsets(tr, 0, 1, 2, 0)
    assert so == 2 and tr.words[w0].word == "honestly"
    # dropping one word would cut after "I" -> not a clause boundary -> ignored
    w0, w1, so, eo, notes = apply_offsets(tr, 0, 1, 1, 0)
    assert so == 0 and w0 == 0 and notes


def test_end_offset_never_leaves_fragment():
    tr = make_transcript(["Focus wins, but nobody believes it at first."])
    # keep "Focus wins," (cut before "but") -> valid
    n = len(tr.words)
    w0, w1, so, eo, _ = apply_offsets(tr, 0, 0, 0, n - 2)
    assert eo == 0  # would leave only 2 words -> fragment -> refused
    tr2 = make_transcript(["Focus always wins in the end, but nobody believes it."])
    n2 = len(tr2.words)
    w0, w1, so, eo, _ = apply_offsets(tr2, 0, 0, 0, 4)
    assert eo == 4 and tr2.words[w1].word == "end,"


def test_leading_fillers_dropped():
    tr = make_transcript(["um uh the point is simple."])
    w0, dropped = drop_leading_fillers(tr.words, 0, len(tr.words) - 1, ["um", "uh"])
    assert dropped == 2 and tr.words[w0].word == "the"


def test_snap_without_audio_pads_and_never_cuts_words():
    tr = make_transcript(["First sentence here.", "Second sentence is the clip.", "Third one after."])
    s = tr.sentences[1]
    start, end, _ = snap(tr, s.start_word, s.end_word, None, SnapConfig())
    first, last = tr.words[s.start_word], tr.words[s.end_word]
    prev, nxt = tr.words[s.start_word - 1], tr.words[s.end_word + 1]
    assert prev.end <= start <= first.start
    assert abs((first.start - start) - 0.08) < 1e-6
    assert last.end <= end <= nxt.start
    assert abs((end - last.end) - 0.15) < 1e-6


def _write_wav(path, seconds, bursts, sr=16000):
    x = np.zeros(int(seconds * sr), dtype=np.float32)
    rng = np.random.default_rng(0)
    for a, b, amp in bursts:
        x[int(a * sr): int(b * sr)] = rng.normal(0, amp, int(b * sr) - int(a * sr))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def test_snap_moves_to_quietest_frame(tmp_path):
    words = [Word(word="Hello.", start=0.5, end=1.0), Word(word="Breath", start=2.0, end=2.5),
             Word(word="world.", start=2.6, end=3.0), Word(word="Next.", start=4.0, end=4.5)]
    tr = Transcript(language="en", duration=5, words=words, sentences=segment_sentences(words))
    # Loud noise (a breath) from 1.86 to 1.95 s, just before "Breath"; quiet elsewhere.
    p = tmp_path / "a.wav"
    _write_wav(p, 5.0, [(0.5, 1.0, 0.3), (1.86, 1.95, 0.3), (2.0, 3.0, 0.3), (4.0, 4.5, 0.3)])
    wav = Wav(p)
    start, end, _ = snap(tr, 1, 2, wav, SnapConfig())
    assert 1.95 <= start <= 2.0  # quiet gap between breath and word, not inside the breath
    assert 3.0 <= end <= 3.3


def test_compute_bounds_and_boundary_ok():
    tr = make_transcript(["Intro words here.", "uh So this is the idea.", "And this is the payoff."])
    b = compute_bounds(tr, 1, 2, 0, 0, None, SnapConfig())
    assert tr.words[b.w0].word == "So" and b.fillers_dropped == 1
    assert boundary_ok(tr, b)
