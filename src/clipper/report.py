"""Self-contained HTML review report: selected clips plus every rejected candidate and why."""

from __future__ import annotations

import html
import json
from pathlib import Path

from .jobs import Job
from .transcript import fmt_ts

STATUS_ORDER = ["selected", "over_quota", "duplicate", "below_threshold", "out_of_range", "rejected", "error",
                "skipped", "accepted", "pending"]
STATUS_LABEL = {
    "selected": "Selected", "over_quota": "Good, over quota", "duplicate": "Duplicate",
    "below_threshold": "Below score bar", "out_of_range": "Wrong length", "rejected": "Rejected by judge",
    "error": "Error", "skipped": "Skipped", "accepted": "Accepted (not finalized)", "pending": "Not judged",
}

CSS = """
:root{--bg:#fafaf9;--fg:#1c1917;--muted:#78716c;--card:#fff;--line:#e7e5e4;--ok:#15803d;--bad:#b91c1c;
--warn:#b45309;--accent:#4338ca;--bar:#e0e7ff}
@media (prefers-color-scheme:dark){:root{--bg:#0c0a09;--fg:#e7e5e4;--muted:#a8a29e;--card:#1c1917;--line:#292524;
--ok:#4ade80;--bad:#f87171;--warn:#fbbf24;--accent:#a5b4fc;--bar:#312e81}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:0 auto;padding:24px 16px 64px}h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:32px 0 12px}.muted{color:var(--muted)}.grid{display:grid;gap:16px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px;margin:16px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.stat b{display:block;font-size:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:16px;display:grid;grid-template-columns:220px 1fr;gap:16px}
@media (max-width:640px){.card{grid-template-columns:1fr}}video{width:100%;border-radius:8px;background:#000}
.title{font-weight:600;font-size:16px}.hook{color:var(--accent);font-weight:600}
.flag{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;border:1px solid currentColor;margin:2px 4px 2px 0}
.flag.warn{color:var(--warn)}.flag.ok{color:var(--ok)}.flag.bad{color:var(--bad)}
.scores{display:grid;grid-template-columns:110px 1fr 28px;gap:2px 8px;font-size:13px;margin:8px 0}
.bar{background:var(--line);border-radius:4px;height:8px;margin-top:6px}.bar i{display:block;height:8px;border-radius:4px;background:var(--accent)}
.quote{border-left:3px solid var(--line);padding-left:10px;color:var(--muted);font-size:14px;max-height:9em;overflow:auto}
table{width:100%;border-collapse:collapse;font-size:13px;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
th,td{text-align:left;padding:8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:600}
.tablewrap{overflow-x:auto}.warnbox{border:1px solid var(--warn);color:var(--warn);border-radius:10px;padding:10px 12px;margin:12px 0}
details summary{cursor:pointer;color:var(--muted)}code{font-size:12px}
"""


def e(x) -> str:
    return html.escape("" if x is None else str(x))


def _scores(scores: dict) -> str:
    rows = []
    for k in ("hook", "standalone_sense", "payoff", "peak", "quotability", "clean_ending"):
        if k in scores:
            v = scores[k]
            rows.append(f"<span>{e(k.replace('_', ' '))}</span><div class='bar'><i style='width:{v * 10:.0f}%'></i>"
                        f"</div><span>{v:.0f}</span>")
    return f"<div class='scores'>{''.join(rows)}</div>"


def write_report(job: Job, cfg) -> Path:
    sel_path = job.path("selection.json")
    records = json.loads(sel_path.read_text())["records"] if sel_path.is_file() else []
    if not records and job.path("judged.json").is_file():
        records = json.loads(job.path("judged.json").read_text())["records"]
    if not records and job.path("candidates.json").is_file():
        records = json.loads(job.path("candidates.json").read_text())["records"]
    invalid = json.loads(job.path("candidates.json").read_text()).get("invalid", []) \
        if job.path("candidates.json").is_file() else []
    clips = json.loads(job.path("clips.json").read_text())["clips"] if job.path("clips.json").is_file() else []
    st = job.state.get("stats", {})
    spec = job.state.get("spec", {})
    warnings = job.state.get("warnings", [])
    sentences = []
    if job.path("transcript.json").is_file():
        sentences = json.loads(job.path("transcript.json").read_text()).get("sentences", [])

    def cost(k):
        v = st.get(k)
        return f"${v:.3f}" if isinstance(v, (int, float)) else "–"

    stats = [
        ("Source", fmt_ts(st.get("source_duration_s", 0))),
        ("Clips", str(len(clips))),
        ("Candidates", str(len(records))),
        ("ASR cost", cost("asr_cost_usd") + (" (cached)" if st.get("asr_cached") else "")),
        ("LLM cost", cost("llm_cost_usd")),
        ("Total", cost("total_cost_usd")),
        ("Per source hour", cost("cost_per_source_hour_usd")),
        ("Processing time", f"{sum((st.get('stage_seconds') or {}).values()):.0f}s"),
    ]
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>Clip review</title><style>{CSS}</style></head><body><main>",
        f"<h1>Clip review</h1><div class='muted'>{e(Path(spec.get('video', '')).name)} · job <code>{e(job.id)}</code>"
        f" · {e(spec.get('content_type'))} · prompts {e(cfg.llm.candidate_prompt)}/{e(cfg.llm.judge_prompt)}"
        f" · models {e(cfg.llm.candidate_model)}/{e(cfg.llm.judge_model)}</div>",
        "<div class='stats'>" + "".join(f"<div class='stat'><span class='muted'>{e(a)}</span><b>{e(b)}</b></div>"
                                        for a, b in stats) + "</div>",
    ]
    if warnings:
        parts.append("<div class='warnbox'><b>Warnings</b><ul>" + "".join(f"<li>{e(w)}</li>" for w in warnings)
                     + "</ul></div>")
    parts.append("<p class='muted'>Scores are editorial signals from the judge model, not predicted views. "
                 "Review every clip before publishing; uploads default to private.</p>")

    parts.append(f"<h2>Selected clips ({len(clips)})</h2>")
    if not clips:
        parts.append("<p>No clip passed selection. That is a valid outcome; the table below says why.</p>")
    for c in clips:
        flags = []
        if c.get("needs_review"):
            flags += [f"<span class='flag warn'>{e(x)}</span>" for x in c.get("review_reasons", [])]
        else:
            flags.append("<span class='flag ok'>no review flags</span>")
        conf = c.get("confidence", {})
        video = f"<video controls preload='metadata' src='{e(c['file'])}'></video>" if c.get("file") else ""
        parts.append(
            "<div class='card'><div>" + video + "</div><div>"
            f"<div class='muted'>#{c['rank']} · {fmt_ts(c['start'])}–{fmt_ts(c['end'])} · {c['duration']:.1f}s · "
            f"{e(c.get('content_type'))} · editorial <b>{c['editorial']:.2f}</b></div>"
            f"<div class='title'>{e(c.get('title'))}</div>"
            + (f"<div class='hook'>Hook overlay: “{e(c['hook_text'])}”</div>" if c.get("hook_text") else "")
            + f"<div>{''.join(flags)}</div>{_scores(c.get('scores', {}))}"
            f"<div class='muted'>confidence: judge {e(conf.get('llm'))}, ASR {e(conf.get('asr'))}, "
            f"boundaries {'ok' if conf.get('boundary_ok') else 'check'}</div>"
            f"<p>{e(c.get('rationale'))}</p>"
            f"<div class='muted'>{e(c.get('description'))}<br>tags: {e(', '.join(c.get('tags', [])))}</div>"
            f"<details><summary>Transcript</summary><div class='quote'>{e(c.get('text'))}</div></details>"
            + (f"<details><summary>Notes</summary><ul>{''.join(f'<li>{e(n)}</li>' for n in c.get('notes', []))}"
               "</ul></details>" if c.get("notes") else "")
            + "</div></div>"
        )

    parts.append(f"<h2>All candidates ({len(records)})</h2><div class='tablewrap'><table><tr><th>ID</th><th>Status</th>"
                 "<th>Span</th><th>Scores</th><th>Why</th><th>Opening</th></tr>")
    records.sort(key=lambda r: (STATUS_ORDER.index(r["status"]) if r["status"] in STATUS_ORDER else 99,
                                -(r.get("editorial") or 0)))
    for r in records:
        if r.get("start") is not None:
            span = f"{fmt_ts(r['start'])}–{fmt_ts(r['end'])} ({r['duration']:.0f}s)"
        elif sentences and r["end_sentence_id"] < len(sentences):
            a, b = sentences[r["start_sentence_id"]], sentences[r["end_sentence_id"]]
            span = f"{fmt_ts(a['start'])}–{fmt_ts(b['end'])} (~{b['end'] - a['start']:.0f}s)"
        else:
            span = f"s{r['start_sentence_id']}–s{r['end_sentence_id']}"
        sc = r.get("scores") or {}
        sc_txt = (f"ed {r['editorial']:.1f} · hook {sc.get('hook', 0):.0f} · stand {sc.get('standalone_sense', 0):.0f}"
                  f" · pay {sc.get('payoff', 0):.0f}") if r.get("editorial") is not None else "–"
        why = r.get("status_reason") or (r.get("rationale") if r["status"] == "selected" else "") or ""
        if r.get("judge") and r["status"] != "selected" and r["judge"].get("rationale"):
            why += f" — {r['judge']['rationale']}"
        cls = "ok" if r["status"] == "selected" else ("warn" if r["status"] in ("over_quota", "duplicate") else "bad")
        opening = r.get("text") or (sentences[r["start_sentence_id"]]["text"] if sentences else "")
        parts.append(
            f"<tr><td>{e(r['id'])}</td><td><span class='flag {cls}'>{e(STATUS_LABEL.get(r['status'], r['status']))}"
            f"</span></td><td>{e(span)}</td><td>{e(sc_txt)}</td><td>{e(why)}<div class='muted'>pass 1: "
            f"{e(r.get('pass1', {}).get('reason'))}</div></td><td>{e(opening[:160])}</td></tr>"
        )
    parts.append("</table></div>")
    if invalid:
        parts.append(f"<h2>Invalid pass-1 output ({len(invalid)})</h2><ul>"
                     + "".join(f"<li>{e(i.get('start_sentence_id'))}–{e(i.get('end_sentence_id'))}: "
                               f"{e(i.get('invalid_reason'))}</li>" for i in invalid) + "</ul>")
    parts.append("</main></body></html>")
    out = job.path("report.html")
    out.write_text("".join(parts), encoding="utf-8")
    return out
