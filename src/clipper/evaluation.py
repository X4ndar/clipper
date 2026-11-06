"""Evaluation harness: gold set format, labeling sheet, metrics, run comparison.

Gold file (one per video), JSON:
{
  "name": "ep12",
  "video": "/path/to/ep12.mp4",           # optional, used by `clipper eval run`
  "source_sha256": "...",                  # optional, guards against labeling the wrong file
  "content_type": "podcast",               # optional
  "labels": [
    {"start": 312.4, "end": 351.0, "label": "good", "reason": "clear story with a payoff",
     "needs_trim": false, "candidate_id": "c03"}
  ]
}
label is one of good | ok | bad. Human-picked clips the system missed are labels too.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel

from .jobs import Job
from .select import iou
from .transcript import fmt_ts


class GoldLabel(BaseModel):
    start: float
    end: float
    label: Literal["good", "ok", "bad"]
    reason: str = ""
    needs_trim: bool = False
    candidate_id: Optional[str] = None


class GoldFile(BaseModel):
    name: str
    video: Optional[str] = None
    source_sha256: Optional[str] = None
    content_type: Optional[str] = None
    labels: list[GoldLabel]


SHEET_COLUMNS = ["candidate_id", "status", "start", "end", "duration", "editorial", "title", "opening",
                 "label", "needs_trim", "reason"]


def parse_time(s: str) -> float:
    s = s.strip()
    if ":" in s:
        parts = [float(p) for p in s.split(":")]
        t = 0.0
        for p in parts:
            t = t * 60 + p
        return t
    return float(s)


def _records(job: Job) -> list[dict]:
    for name in ("selection.json", "judged.json", "candidates.json"):
        if job.path(name).is_file():
            return json.loads(job.path(name).read_text())["records"]
    raise FileNotFoundError(f"{job.id}: no candidates yet; run at least to --stop-at candidates")


def write_label_sheet(job: Job, out: Path, blank_rows: int = 10) -> Path:
    """CSV with every candidate the system considered, plus blank rows for clips a human would pick."""
    tr = json.loads(job.path("transcript.json").read_text())
    sents = tr["sentences"]
    rows = []
    for r in sorted(_records(job), key=lambda r: sents[r["start_sentence_id"]]["start"]):
        start = r.get("start") if r.get("start") is not None else sents[r["start_sentence_id"]]["start"]
        end = r.get("end") if r.get("end") is not None else sents[r["end_sentence_id"]]["end"]
        opening = r.get("text") or sents[r["start_sentence_id"]]["text"]
        rows.append({
            "candidate_id": r["id"], "status": r["status"], "start": fmt_ts(start) + f".{int(start % 1 * 10)}",
            "end": fmt_ts(end) + f".{int(end % 1 * 10)}", "duration": f"{end - start:.1f}",
            "editorial": "" if r.get("editorial") is None else f"{r['editorial']:.2f}",
            "title": r.get("title") or "", "opening": opening[:200], "label": "", "needs_trim": "", "reason": "",
        })
    rows += [{c: "" for c in SHEET_COLUMNS} | {"candidate_id": "human"} for _ in range(blank_rows)]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SHEET_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out


def import_label_sheet(sheet: Path, name: str, video: Optional[str] = None, sha: Optional[str] = None,
                       content_type: Optional[str] = None) -> GoldFile:
    labels = []
    with open(sheet, newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            lab = (row.get("label") or "").strip().lower()
            if not lab:
                continue
            if lab not in ("good", "ok", "bad"):
                raise ValueError(f"{sheet}:{i}: label must be good, ok or bad, got {lab!r}")
            if not (row.get("start") or "").strip() or not (row.get("end") or "").strip():
                raise ValueError(f"{sheet}:{i}: start and end are required for labeled rows")
            cid = (row.get("candidate_id") or "").strip()
            labels.append(GoldLabel(
                start=parse_time(row["start"]), end=parse_time(row["end"]), label=lab,  # type: ignore[arg-type]
                reason=(row.get("reason") or "").strip(),
                needs_trim=(row.get("needs_trim") or "").strip().lower() in ("1", "y", "yes", "true", "x"),
                candidate_id=None if cid in ("", "human") else cid,
            ))
    return GoldFile(name=name, video=video, source_sha256=sha, content_type=content_type, labels=labels)


def score_job(job: Job, gold: GoldFile, k: Optional[int] = None, iou_threshold: float = 0.5) -> dict:
    if gold.source_sha256 and job.state.get("source_sha256") and gold.source_sha256 != job.state["source_sha256"]:
        raise ValueError(f"gold {gold.name} was labeled on a different file than job {job.id}")
    recs = [r for r in _records(job) if r["status"] == "selected"]
    recs.sort(key=lambda r: r.get("rank") or 0)
    k = k or len(recs)
    top = recs[:k]

    def best(span: tuple[float, float]) -> tuple[Optional[GoldLabel], float]:
        scored = [(g, iou(span, (g.start, g.end))) for g in gold.labels]
        if not scored:
            return None, 0.0
        g, v = max(scored, key=lambda x: x[1])
        return (g, v) if v >= iou_threshold else (None, v)

    matches = []
    for r in top:
        g, v = best((r["start"], r["end"]))
        matches.append({"id": r["id"], "rank": r.get("rank"), "label": g.label if g else None, "iou": round(v, 3),
                        "needs_trim": g.needs_trim if g else None,
                        "boundary_ok": bool((r.get("confidence") or {}).get("boundary_ok"))})
    n = len(top)
    good = sum(m["label"] == "good" for m in matches)
    ok = sum(m["label"] == "ok" for m in matches)
    labeled = [m for m in matches if m["label"]]
    gold_good = [g for g in gold.labels if g.label == "good"]
    hit = sum(any(iou((g.start, g.end), (r["start"], r["end"])) >= iou_threshold for r in recs) for g in gold_good)
    st = job.state.get("stats", {})
    hours = (st.get("source_duration_s") or 0) / 3600
    metrics = {
        "k": k,
        "selected": len(recs),
        "precision_at_k": round(good / n, 4) if n else None,
        "precision_at_k_lenient": round((good + 0.5 * ok) / n, 4) if n else None,
        "bad_at_k": sum(m["label"] == "bad" for m in matches),
        "unlabeled_at_k": n - len(labeled),
        "complete_thought_share": round(sum(m["boundary_ok"] for m in matches) / n, 4) if n else None,
        "needs_trim_share": round(sum(bool(m["needs_trim"]) for m in labeled) / len(labeled), 4) if labeled else None,
        "human_good_recall": round(hit / len(gold_good), 4) if gold_good else None,
        "cost_per_hour_usd": round(st.get("total_cost_usd", 0) / hours, 4)
        if hours and st.get("total_cost_usd") is not None else None,
        "llm_cost_usd": st.get("llm_cost_usd"),
        "wall_seconds_per_hour": round(st.get("last_run_wall_seconds", 0) / hours, 1) if hours else None,
        "matches": matches,
    }
    return metrics


def summarize(rows) -> list[dict]:
    """Group eval runs by prompt and model versions; mean of each numeric metric."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["candidate_prompt"], r["judge_prompt"], r["candidate_model"], r["judge_model"])
        groups[key].append(json.loads(r["metrics_json"]))
    out = []
    for key, ms in groups.items():
        agg: dict = {"candidate_prompt": key[0], "judge_prompt": key[1], "candidate_model": key[2],
                     "judge_model": key[3], "runs": len(ms)}
        for m in ("precision_at_k", "precision_at_k_lenient", "complete_thought_share", "needs_trim_share",
                  "human_good_recall", "cost_per_hour_usd"):
            vals = [x[m] for x in ms if isinstance(x.get(m), (int, float))]
            agg[m] = round(sum(vals) / len(vals), 4) if vals else None
        out.append(agg)
    return out
