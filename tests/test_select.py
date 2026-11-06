from clipper.config import Config
from clipper.select import ClipRecord, editorial_score, finalize, iou, jaccard

from conftest import make_transcript


def test_editorial_skips_missing_signals():
    w = {"hook": 30, "peak": 25, "standalone_sense": 20, "speaker_energy": 15, "quotability": 10}
    s = {"hook": 10, "peak": 10, "standalone_sense": 10, "quotability": 10, "payoff": 0}
    assert editorial_score(s, w) == 10.0  # speaker_energy absent -> renormalized


def test_overlap_helpers():
    assert iou((0, 10), (5, 15)) == 5 / 15
    assert jaccard("the cat sat", "the cat ran") == 0.5


def long_transcript():
    sents = [f"Sentence number {i} says something useful about topic {i % 5}." for i in range(80)]
    return make_transcript(sents)


def rec(id, s0, s1, editorial, **kw):
    scores = {"hook": editorial, "standalone_sense": 8, "payoff": 8, "peak": editorial, "quotability": 7,
              "clean_ending": 8}
    return ClipRecord(id=id, status="accepted", pass1={}, start_sentence_id=s0, end_sentence_id=s1,
                      scores=scores, editorial=editorial, confidence={"llm": 0.9}, **kw)


SPEC = {"content_type": "podcast", "max_clips": 2, "min_duration": 10, "max_duration": 30}


def test_finalize_dedupes_ranks_and_caps():
    tr = long_transcript()
    cfg = Config(home="/tmp/x")
    records = [rec("a", 0, 5, 8.0), rec("b", 1, 6, 7.5), rec("c", 20, 25, 7.0), rec("d", 40, 45, 6.5)]
    out = {r.id: r for r in finalize(tr, records, SPEC, cfg, None)}
    assert out["a"].status == "selected" and out["a"].rank == 1
    assert out["b"].status == "duplicate"
    assert out["c"].status == "selected" and out["c"].rank == 2
    assert out["d"].status == "over_quota"


def test_finalize_duration_and_threshold():
    tr = long_transcript()
    cfg = Config(home="/tmp/x")
    records = [rec("short", 0, 0, 9.0), rec("low", 10, 15, 4.0), rec("huge", 30, 60, 9.0)]
    out = {r.id: r for r in finalize(tr, records, SPEC, cfg, None)}
    assert out["short"].status == "out_of_range" and "too short" in out["short"].status_reason
    assert out["low"].status == "below_threshold"
    assert out["huge"].status == "out_of_range"


def test_zero_clips_is_valid():
    tr = long_transcript()
    out = finalize(tr, [rec("low", 10, 15, 2.0)], SPEC, Config(home="/tmp/x"), None)
    assert not [r for r in out if r.status == "selected"]


def test_rejected_records_untouched():
    tr = long_transcript()
    r = rec("x", 0, 5, 9.0)
    r.status = "rejected"
    out = finalize(tr, [r], SPEC, Config(home="/tmp/x"), None)
    assert out[0].status == "rejected" and out[0].start is None


class JudgeBackend:
    name = "j"

    def __init__(self, **over):
        self.over = over

    def complete(self, c):
        clip = c.payload["clip_sentences"]
        out = {"start_sentence_id": clip[0]["id"], "end_sentence_id": clip[-1]["id"], "start_word_offset": 0,
               "end_word_offset": 0, "scores": {"hook": 10, "standalone_sense": 10, "payoff": 10, "peak": 10,
                                                "quotability": 10, "clean_ending": 10},
               "reject": False, "reject_reason": None, "rationale": "r", "confidence": 0.9, "hook_text": "h",
               "title": "t", "description": "d", "tags": ["a", "b", "c"], "content_type": "claim"}
        for k, v in self.over.items():
            if k in out["scores"]:
                out["scores"][k] = v
            else:
                out[k] = v
        return c.schema.model_validate(out), {}, "end_turn", c.model


def _judge(tmp_path, **over):
    from clipper.db import DB
    from clipper.llm.client import LLMService
    from clipper.select import judge_candidates

    tr = long_transcript()
    cfg = Config(home=str(tmp_path))
    llm = LLMService(JudgeBackend(**over), DB(tmp_path / "d.db"), cfg.llm, "j")
    r = ClipRecord(id="c01", pass1={"reason": "x"}, start_sentence_id=10, end_sentence_id=14)
    return judge_candidates(tr, [r], SPEC, cfg, llm)[0]


def test_hard_gate_beats_high_total(tmp_path):
    r = _judge(tmp_path, standalone_sense=5)
    assert r.status == "rejected" and "hard gate" in r.status_reason and r.editorial > 8


def test_judge_reject_flag_respected(tmp_path):
    r = _judge(tmp_path, reject=True, reject_reason="needs context")
    assert r.status == "rejected" and "needs context" in r.status_reason


def test_judge_boundary_moves_are_clamped(tmp_path):
    r = _judge(tmp_path, start_sentence_id=0, end_sentence_id=79)
    assert r.status == "accepted"
    assert r.start_sentence_id >= 5 and r.end_sentence_id <= 19
