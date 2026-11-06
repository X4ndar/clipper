"""Caption layout and ASS generation.

Timing units: ASS event times are h:mm:ss.cc (centiseconds); karaoke tags (\\k) take centiseconds;
animation tags (\\t, \\fad) take milliseconds.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .transcript import Word, ends_clause, ends_sentence, is_rtl


class CaptionWord(BaseModel):
    word: str
    start: float  # seconds, relative to clip start
    end: float
    speaker: Optional[str] = None


class CaptionDoc(BaseModel):
    """Editable per-clip captions. Fix ASR errors here and re-render without re-transcribing."""

    language: str = "en"
    duration: float
    hook_text: Optional[str] = None
    words: list[CaptionWord]


def ass_color(hex_color: str) -> str:
    """#RRGGBB or #RRGGBBAA (AA = opacity, FF opaque) -> &HAABBGGRR (AA = transparency)."""
    h = hex_color.lstrip("#")
    if len(h) not in (6, 8):
        raise ValueError(f"bad color {hex_color!r}")
    r, g, b = h[0:2], h[2:4], h[4:6]
    alpha = 255 - int(h[6:8], 16) if len(h) == 8 else 0
    return f"&H{alpha:02X}{b}{g}{r}".upper()


def ass_inline_color(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return f"&H{h[4:6]}{h[2:4]}{h[0:2]}&".upper()


def ass_time(t: float) -> str:
    cs = int(round(max(0.0, t) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def escape_text(s: str) -> str:
    s = s.replace("\\", "/").replace("{", "(").replace("}", ")")
    return re.sub(r"\s+", " ", s).strip()


def doc_from_words(words: list[Word], clip_start: float, duration: float, language: str,
                   hook_text: Optional[str]) -> CaptionDoc:
    out = []
    for w in words:
        a = max(0.0, w.start - clip_start)
        b = min(duration, max(a + 0.02, w.end - clip_start))
        out.append(CaptionWord(word=w.word, start=round(a, 3), end=round(b, 3), speaker=w.speaker))
    return CaptionDoc(language=language, duration=round(duration, 3), hook_text=hook_text, words=out)


def layout_pages(words: list[CaptionWord], max_chars: int, max_lines: int, max_gap: float) -> list[list[list[int]]]:
    """Group word indices into pages of up to `max_lines` lines of up to `max_chars` characters."""
    pages: list[list[list[int]]] = []
    page: list[list[int]] = []
    line: list[int] = []
    line_len = 0

    def close_page() -> None:
        nonlocal page, line, line_len
        if line:
            page.append(line)
        if page:
            pages.append(page)
        page, line, line_len = [], [], 0

    for i, w in enumerate(words):
        if line or page:
            prev = words[i - 1]
            if w.start - prev.end > max_gap or (w.speaker != prev.speaker and (w.speaker or prev.speaker)):
                close_page()
        wl = len(w.word)
        if line and line_len + 1 + wl > max_chars:
            page.append(line)
            line, line_len = [], 0
            if len(page) >= max_lines:
                close_page()
        line.append(i)
        line_len = wl if line_len == 0 else line_len + 1 + wl
        if ends_sentence(w.word):
            close_page()
        elif ends_clause(w.word) and len(page) == max_lines - 1 and line_len > max_chars * 0.6:
            close_page()
    close_page()
    return pages


def wrap_text(text: str, max_chars: int) -> str:
    lines, line = [], ""
    for tok in text.split():
        cand = (line + " " + tok).strip()
        if line and len(cand) > max_chars:
            lines.append(line)
            line = tok
        else:
            line = cand
    if line:
        lines.append(line)
    return r"\N".join(lines)


def build_ass(doc: CaptionDoc, ccfg, hcfg, width: int = 1080, height: int = 1920) -> str:
    """`ccfg` is CaptionsConfig, `hcfg` HookConfig."""
    rtl = is_rtl(doc.language)
    font = ccfg.rtl_font if rtl else ccfg.font
    bold = -1 if ccfg.bold else 0
    border_style = 3 if ccfg.box else 1
    outline_col = ass_color(ccfg.box_color if ccfg.box else ccfg.outline_color)
    back_col = ass_color(ccfg.box_color if ccfg.box else "#00000080")
    primary = ass_color(ccfg.primary_color)
    highlight = ass_color(ccfg.highlight_color)
    # Karaoke: \k fills from SecondaryColour to PrimaryColour, so primary = highlight there.
    if ccfg.mode == "karaoke":
        style_primary, style_secondary = highlight, primary
    else:
        style_primary, style_secondary = primary, highlight
    side = 60
    hook_font = font
    header = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,{font},{ccfg.size},{style_primary},{style_secondary},{outline_col},{back_col},{bold},0,0,0,"
        f"100,100,0,0,{border_style},{ccfg.outline},{ccfg.shadow},2,{side},{side},{ccfg.margin_v},1",
        f"Style: Hook,{hook_font},{hcfg.size},{ass_color(hcfg.primary_color)},{ass_color(hcfg.primary_color)},"
        f"{ass_color(hcfg.box_color)},{ass_color(hcfg.box_color)},-1,0,0,0,100,100,0,0,3,{hcfg.box_padding},0,8,{side},{side},"
        f"{hcfg.margin_v},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    events: list[str] = []

    def ev(layer: int, a: float, b: float, style: str, text: str) -> None:
        if b - a < 0.01:
            return
        events.append(f"Dialogue: {layer},{ass_time(a)},{ass_time(b)},{style},,0,0,0,,{text}")

    words = doc.words
    tx = (lambda s: escape_text(s).upper()) if ccfg.uppercase else escape_text
    if ccfg.enabled and words:
        pages = layout_pages(words, ccfg.max_chars_per_line, ccfg.max_lines, ccfg.max_page_gap_s)
        for pi, page in enumerate(pages):
            idxs = [i for line in page for i in line]
            p_start = words[idxs[0]].start
            p_end = words[idxs[-1]].end
            # Hold the page until the next one starts if the gap is short, so captions do not flicker.
            if pi + 1 < len(pages):
                nxt = words[pages[pi + 1][0][0]].start
                if nxt - p_end < 0.35:
                    p_end = nxt
                p_end = min(max(p_end, p_start + 0.3), nxt)
            else:
                p_end = min(doc.duration, max(p_end + 0.2, p_start + 0.3))
            if ccfg.mode == "plain":
                ev(0, p_start, p_end, "Caption", r"\N".join(" ".join(tx(words[i].word) for i in ln) for ln in page))
            elif ccfg.mode == "karaoke":
                parts = []
                t = p_start
                for li, ln in enumerate(page):
                    seg = []
                    for i in ln:
                        w = words[i]
                        nxt_t = words[i + 1].start if i + 1 < len(words) and i != idxs[-1] else p_end
                        lead = max(0.0, w.start - t)
                        if lead >= 0.01:
                            seg.append(f"{{\\k{int(round(lead * 100))}}}")
                        dur_cs = max(1, int(round((max(nxt_t, w.end) - max(t, w.start)) * 100)))
                        seg.append(f"{{\\k{dur_cs}}}{tx(w.word)} ")
                        t = max(nxt_t, w.end)
                    parts.append("".join(seg).rstrip())
                ev(0, p_start, p_end, "Caption", r"\N".join(parts))
            else:  # word highlight: one event per word, current word recolored
                hl = ass_inline_color(ccfg.highlight_color)
                for k, cur in enumerate(idxs):
                    a = words[cur].start if k else p_start
                    b = words[idxs[k + 1]].start if k + 1 < len(idxs) else p_end
                    lines = []
                    for ln in page:
                        toks = []
                        for i in ln:
                            t = tx(words[i].word)
                            if i == cur:
                                pop = r"\fscx100\fscy100\t(0,80,\fscx110\fscy110)" if ccfg.pop else ""
                                toks.append(f"{{\\1c{hl}{pop}}}{t}{{\\r}}")
                            else:
                                toks.append(t)
                        lines.append(" ".join(toks))
                    ev(0, a, b, "Caption", r"\N".join(lines))

    if hcfg.enabled and doc.hook_text:
        hook = " ".join(doc.hook_text.split()[: hcfg.max_words])
        dur = min(hcfg.duration_s, doc.duration)
        ev(1, 0.0, dur, "Hook", r"{\fad(150,250)}" + wrap_text(tx(hook), hcfg.max_chars_per_line))
    return "\n".join(header + events) + "\n"


def write_caption_files(doc: CaptionDoc, ass_path: Path, json_path: Path, ccfg, hcfg, width: int, height: int) -> None:
    json_path.write_text(doc.model_dump_json(indent=2), encoding="utf-8")
    ass_path.write_text(build_ass(doc, ccfg, hcfg, width, height), encoding="utf-8")


def load_doc(path: Path) -> CaptionDoc:
    return CaptionDoc.model_validate(json.loads(path.read_text(encoding="utf-8")))


def full_transcript_ass(words: list[Word], language: str, duration: float, ccfg, hcfg) -> str:
    """Whole-video ASS (plain pages) for reuse in other editors. 16:9 play resolution."""
    doc = doc_from_words(words, 0.0, duration, language, None)
    plain = ccfg.model_copy(update={"mode": "plain", "margin_v": 80, "size": 54, "max_chars_per_line": 42})
    no_hook = hcfg.model_copy(update={"enabled": False})
    return build_ass(doc, plain, no_hook, 1920, 1080)
