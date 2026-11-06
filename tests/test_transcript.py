from clipper.config import QualityConfig
from clipper.transcript import Word, ends_sentence, quality_check, segment_sentences, to_srt

from conftest import make_transcript


def test_segmentation_on_punctuation_speaker_and_pause():
    words = [Word(word="Hi", start=0, end=0.2, speaker="A"), Word(word="there.", start=0.3, end=0.5, speaker="A"),
             Word(word="Dr.", start=0.6, end=0.8, speaker="A"), Word(word="Smith", start=0.9, end=1.1, speaker="A"),
             Word(word="agrees", start=1.2, end=1.4, speaker="B"), Word(word="later", start=4.0, end=4.2, speaker="B")]
    s = segment_sentences(words)
    assert [x.text for x in s] == ["Hi there.", "Dr. Smith", "agrees", "later"]


def test_ends_sentence_handles_quotes_and_arabic():
    assert ends_sentence('done."') and ends_sentence("لماذا؟") and not ends_sentence("e.g.")


def test_quality_flags_repetition_and_low_confidence():
    tr = make_transcript(["thank you so much " * 6 + "end."])
    q = quality_check(tr, QualityConfig())
    assert not q.ok and q.repetition_share > 0.5
    tr2 = make_transcript(["A normal sentence about focus.", "Another one here."], score=0.3)
    q2 = quality_check(tr2, QualityConfig())
    assert not q2.ok and q2.low_confidence_share == 1.0
    tr3 = make_transcript(["A normal sentence about focus.", "Another one here."])
    assert quality_check(tr3, QualityConfig()).ok


def test_quality_warns_for_darija_risk():
    tr = make_transcript(["kayn chi haja."])
    tr.language = "ar"
    q = quality_check(tr, QualityConfig())
    assert any("Darija" in w for w in q.warnings)


def test_srt_format():
    tr = make_transcript(["Hello world.", "Second line here."])
    srt = to_srt(tr.words)
    assert srt.startswith("1\n00:00:01,000 --> ")
    assert "\n2\n" in srt
