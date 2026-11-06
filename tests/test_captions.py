import re

from clipper.captions import (
    CaptionDoc, CaptionWord, ass_color, ass_inline_color, ass_time, build_ass, escape_text, layout_pages,
)
from clipper.config import CaptionsConfig, HookConfig


def doc(words, lang="en", hook=None, dur=None):
    ws, t = [], 0.0
    for w in words:
        ws.append(CaptionWord(word=w, start=round(t, 3), end=round(t + 0.3, 3)))
        t += 0.35
    return CaptionDoc(language=lang, duration=dur or t + 0.5, hook_text=hook, words=ws)


def test_colors_and_times():
    assert ass_color("#FFD400") == "&H0000D4FF"
    assert ass_color("#000000A0") == "&H5F000000"  # 0xA0 opacity -> 0x5F transparency
    assert ass_inline_color("#FFD400") == "&H00D4FF&"
    assert ass_time(3723.456) == "1:02:03.46"  # centiseconds


def test_escape():
    assert escape_text("a {b} c\\d") == "a (b) c/d"


def test_layout_limits():
    words = doc("the quick brown fox jumps over the lazy dog again and again until night falls".split()).words
    pages = layout_pages(words, 22, 2, 0.6)
    for page in pages:
        assert len(page) <= 2
        for line in page:
            assert len(" ".join(words[i].word for i in line)) <= 22
    assert [i for p in pages for ln in p for i in ln] == list(range(len(words)))


def test_word_mode_one_event_per_word_with_highlight():
    d = doc("why most systems fail is simple".split())
    ass = build_ass(d, CaptionsConfig(), HookConfig(enabled=False))
    events = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(events) == len(d.words)
    assert all("\\1c&H00D4FF&" in e for e in events)
    assert "PlayResX: 1080" in ass and "PlayResY: 1920" in ass


def test_karaoke_uses_centiseconds():
    d = doc("one two three".split())
    ass = build_ass(d, CaptionsConfig(mode="karaoke"), HookConfig(enabled=False))
    ev = [l for l in ass.splitlines() if l.startswith("Dialogue:")]
    assert len(ev) == 1
    ks = [int(x) for x in re.findall(r"\\k(\d+)", ev[0])]
    start, end = ev[0].split(",")[1:3]
    def secs(t):
        h, m, s = t.split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    assert abs(sum(ks) / 100 - (secs(end) - secs(start))) < 0.05


def test_pop_animation_in_milliseconds():
    ass = build_ass(doc(["hello", "world."]), CaptionsConfig(pop=True), HookConfig(enabled=False))
    assert "\\t(0,80," in ass


def test_hook_overlay_max_words_and_top_alignment():
    d = doc("a b c".split(), hook="one two three four five six seven eight nine ten")
    ass = build_ass(d, CaptionsConfig(), HookConfig(duration_s=3.0))
    hook = [l for l in ass.splitlines() if ",Hook," in l and l.startswith("Dialogue")][0]
    text = hook.split(",,", 1)[1].split("}", 1)[1].replace("\\N", " ")
    assert len(text.split()) == 8
    assert ",0:00:00.00,0:00:03.00," in hook or ",0:00:00.00,0:00:01.55," in hook
    style = [l for l in ass.splitlines() if l.startswith("Style: Hook")][0]
    assert style.split(",")[18] == "8"  # \an8: top center


def test_rtl_uses_rtl_font():
    ass = build_ass(doc(["مرحبا", "بالعالم."], lang="ar"), CaptionsConfig(), HookConfig(enabled=False))
    assert "Style: Caption,Noto Sans Arabic," in ass


def test_disabled_captions_keep_hook_only():
    d = doc("a b".split(), hook="big claim")
    ass = build_ass(d, CaptionsConfig(enabled=False), HookConfig())
    ev = [l for l in ass.splitlines() if l.startswith("Dialogue")]
    assert len(ev) == 1 and ",Hook," in ev[0]
