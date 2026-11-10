import json

import pytest

from clipper.config import Config
from clipper.picks import load_picks, parse_time, picks_to_records, sentence_at
from clipper.select import finalize

from conftest import make_transcript

SENTS = [
    "Welcome back to the show.",
    "So why do most plans fail?",
    "Because nobody decides what to drop.",
    "That is the whole trick.",
    "Thanks for listening.",
]


def test_parse_time_formats():
    assert parse_time("9:50.5") == 590.5
    assert parse_time("1:02:03") == 3723
    assert parse_time(12.5) == 12.5


def test_times_snap_to_the_sentences_that_contain_them():
    tr = make_transcript(SENTS)
    s1, s3 = tr.sentences[1], tr.sentences[3]
    assert sentence_at(tr, s1.start + 0.3) == 1
    assert sentence_at(tr, s3.end - 0.1) == 3
    assert sentence_at(tr, 0.0) == 0


def test_load_picks_validates(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"start": "0:01", "end_sentence": 3, "title": "T"}]), encoding="utf-8")
    assert load_picks(f)[0].end_sentence == 3
    f.write_text(json.dumps({"clips": [{"start": "0:01"}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="clip 1"):
        load_picks(f)
    f.write_text(json.dumps([{"start": 1, "start_sentence": 0, "end": 3}]), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly one"):
        load_picks(f)


def test_manual_picks_skip_length_score_and_quota_limits(tmp_path):
    tr = make_transcript(SENTS)
    f = tmp_path / "p.json"
    # Two short picks (well under the 20 s minimum), more picks than max_clips.
    f.write_text(json.dumps([
        {"start_sentence": 1, "end_sentence": 3, "title": "Why plans fail", "hook_text": "Why plans fail"},
        {"start_sentence": 4, "end_sentence": 4},
    ]), encoding="utf-8")
    recs = picks_to_records(tr, load_picks(f), "p.json")
    spec = {"min_duration": 20, "max_duration": 60, "max_clips": 1}
    out = finalize(tr, recs, spec, Config(home=str(tmp_path / "h")), None)
    sel = sorted((r for r in out if r.status == "selected"), key=lambda r: r.rank)
    assert [r.id for r in sel] == ["m01", "m02"]  # file order is the rank
    assert sel[0].text.startswith("So why") and sel[0].text.endswith("trick.")
    assert sel[0].title == "Why plans fail" and "picked by hand" in sel[0].notes[0]


def test_out_of_range_sentences_are_rejected(tmp_path):
    tr = make_transcript(SENTS)
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"start_sentence": 3, "end_sentence": 99}]), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the transcript"):
        picks_to_records(tr, load_picks(f), "p.json")


def test_segments_join_pieces_in_the_given_order(tmp_path):
    from clipper.pipeline import clip_caption_doc, source_offset
    from clipper.select import clip_spans

    tr = make_transcript(SENTS)
    f = tmp_path / "p.json"
    # Cold open on sentence 3, then back to 1-2.
    f.write_text(json.dumps([{"segments": [{"start_sentence": 3, "end_sentence": 3},
                                           {"start_sentence": 1, "end_sentence": 2}]}]), encoding="utf-8")
    recs = picks_to_records(tr, load_picks(f), "p.json")
    out = finalize(tr, recs, {"min_duration": 20, "max_duration": 60, "max_clips": 8},
                   Config(home=str(tmp_path / "h")), None)
    r = next(x for x in out if x.status == "selected")
    spans = clip_spans(r)
    assert len(spans) == 2 and spans[0][0] > spans[1][0]  # played out of source order
    assert r.duration == pytest.approx(sum(b - a for a, b, _, _ in spans), abs=1e-3)
    assert r.text.startswith("That is the whole trick.") and " … So why" in r.text
    doc = clip_caption_doc(tr, r, None)
    starts = [w.start for w in doc.words]
    assert starts == sorted(starts) and doc.words[0].word == "That"
    assert doc.duration == pytest.approx(r.duration, abs=1e-3) and starts[-1] < doc.duration
    first_len = spans[0][1] - spans[0][0]
    assert source_offset(r, 0.1) == pytest.approx(spans[0][0])
    assert source_offset(r, first_len + 0.1) == pytest.approx(spans[1][0] - first_len)


def test_segments_and_a_single_span_cannot_be_mixed(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps([{"start": 1, "end": 5, "segments": [{"start": 1, "end": 2}]}]), encoding="utf-8")
    with pytest.raises(ValueError, match="not both"):
        load_picks(f)
