"""Stage registry and job runner. Every front end (CLI now; API, UI, MCP later) calls run_job()."""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from . import media
from .captions import doc_from_words, full_transcript_ass, load_doc, write_caption_files, build_ass
from .config import Config
from .db import DB
from .jobs import STAGES, Job, JobSpec, jobs_root, resolve_stage, slugify
from .llm.client import LLMService, get_backend
from .reframe import CropPlan, ReframeError, concat_plans, get_detector, reframe_clip, write_plan
from .render import DynamicCrop, prepare_fonts_dir, render_clip
from .select import ClipRecord, clip_spans, finalize, generate_candidates, judge_candidates
from .transcribe import TranscribeRequest, TranscriptionError, get_transcriber
from .transcript import Transcript, is_rtl, quality_check, to_srt

Progress = Callable[[str, str, str], None]


class PipelineError(RuntimeError):
    pass


class QualityGateError(PipelineError):
    pass


class RightsError(PipelineError):
    pass


@dataclass
class Ctx:
    job: Job
    cfg: Config
    db: DB
    progress: Progress
    llm_backend: object = None  # injectable for tests
    transcriber: object = None
    face_detector: object = None
    _cache: dict = field(default_factory=dict)

    @property
    def spec(self) -> JobSpec:
        return self.job.spec

    def transcript(self) -> Transcript:
        if "tr" not in self._cache:
            self._cache["tr"] = Transcript.model_validate(self.job.read_json("transcript.json"))
        return self._cache["tr"]

    def wav(self) -> Optional[media.Wav]:
        p = self.job.path("audio.wav")
        return media.Wav(p) if p.is_file() else None

    def llm(self) -> LLMService:
        backend = self.llm_backend or get_backend(self.cfg.llm)
        hours = self.job.stats().get("source_duration_s", 0) / 3600
        budget = max(self.cfg.llm.min_budget_usd, self.cfg.llm.max_cost_usd_per_hour * hours)
        return LLMService(backend, self.db, self.cfg.llm, job_id=self.job.id, budget_usd=budget)

    def records(self, name: str) -> list[ClipRecord]:
        return [ClipRecord.model_validate(r) for r in self.job.read_json(name)["records"]]

    def save_records(self, name: str, records: list[ClipRecord], **extra) -> None:
        self.job.write_json(name, {"records": [r.model_dump() for r in records], **extra})


# ---------------- job creation ----------------


def create_job(video: Path, spec: JobSpec, cfg: Config, db: DB, job_id: Optional[str] = None) -> Job:
    if not spec.rights_confirmed:
        raise RightsError(
            "Refusing to process: pass --i-own-rights to confirm you own or have rights to this video."
        )
    video = video.expanduser().resolve()
    if not video.is_file():
        raise PipelineError(f"video not found: {video}")
    sha = media.sha256_file(video)
    jid = job_id or f"{slugify(video.stem)}-{sha[:8]}"
    job = Job(jobs_root(cfg.home_path) / jid)
    prev = job.state.get("spec")
    job.state["id"] = jid
    job.state["spec"] = spec.model_copy(update={"video": str(video)}).model_dump()
    job.state["source_sha256"] = sha
    job.state.setdefault("created_at", time.time())
    if prev and {k: v for k, v in prev.items() if k != "rights_confirmed"} != {
        k: v for k, v in job.state["spec"].items() if k != "rights_confirmed"
    }:
        # Inputs changed: keep transcript work, redo selection onward.
        for st in STAGES[STAGES.index("candidates"):]:
            job.state.get("stages", {}).pop(st, None)
    job.save()
    db.upsert_job(jid, str(video), sha, "created", job.state["spec"])
    return job


# ---------------- stages ----------------


def st_ingest(ctx: Ctx) -> None:
    media.require_tools()
    src = Path(ctx.spec.video)
    pr = media.probe(src)
    if not pr.has_audio:
        raise PipelineError(f"{src}: no audio stream; nothing to transcribe")
    if pr.duration <= 0:
        raise PipelineError(f"{src}: could not read duration")
    ctx.job.write_json("source.json", {
        "path": str(src), "duration": pr.duration, "width": pr.width, "height": pr.height, "fps": pr.fps,
        "audio_channels": pr.audio_channels,
    })
    audio = ctx.job.path("audio.wav")
    if not audio.is_file():
        media.extract_audio(src, audio)
    if ctx.cfg.transcribe.backend == "assemblyai" and not audio.with_suffix(".flac").is_file():
        media.encode_flac(audio, audio.with_suffix(".flac"))
    st = ctx.job.stats()
    st["source_duration_s"] = pr.duration
    st["audio_sha256"] = media.sha256_file(audio)
    if pr.audio_channels > 1:
        ctx.progress("ingest", "info", f"downmixed {pr.audio_channels} audio channels to mono before upload")


def speaker_labels_for(spec: JobSpec, mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    if spec.speakers is not None:
        return spec.speakers > 1
    return spec.content_type in ("podcast", "interview")


def st_transcribe(ctx: Ctx) -> None:
    cfg, job = ctx.cfg, ctx.job
    tcfg = cfg.transcribe
    tr_backend = ctx.transcriber or get_transcriber(tcfg)
    spk = speaker_labels_for(ctx.spec, tcfg.speaker_labels)
    req = TranscribeRequest(
        audio_path=job.path("audio.wav"), video_path=Path(ctx.spec.video), language=ctx.spec.language,
        speaker_labels=spk, speakers_expected=ctx.spec.speakers,
    )
    audio_sha = job.stats()["audio_sha256"]
    cache = cfg.home_path / "cache" / "transcripts" / f"{audio_sha}-{tr_backend.cache_key(req)}.json"
    seconds = media.Wav(req.audio_path).duration
    if cache.is_file():
        blob = json.loads(cache.read_text(encoding="utf-8"))
        raw, tr = blob["raw"], Transcript.model_validate(blob["transcript"])
        ctx.progress("transcribe", "info", f"cache hit ({cache.name}); no ASR cost")
        job.stats()["asr_cost_usd"] = 0.0
        job.stats()["asr_cached"] = True
    else:
        est = tr_backend.estimate_cost(seconds, req)
        if est > 0:
            used_s, used_usd = ctx.db.asr_totals(tr_backend.name)
            remaining = tcfg.credit_usd - used_usd
            if est > remaining - tcfg.credit_reserve_usd:
                raise PipelineError(
                    f"ASR credit low: this file needs about ${est:.2f}, about ${remaining:.2f} of "
                    f"${tcfg.credit_usd:.2f} remains (reserve ${tcfg.credit_reserve_usd:.2f}). Top up and raise "
                    "transcribe.credit_usd, or switch backend."
                )
        ctx.progress("transcribe", "info",
                     f"{tr_backend.name}: {seconds / 60:.1f} min of audio, speaker labels {'on' if spk else 'off'}, "
                     f"est. ${est:.3f}")
        try:
            raw, tr = tr_backend.transcribe(req)
        except TranscriptionError as e:
            raise PipelineError(str(e)) from e
        billed = float(raw.get("audio_duration") or seconds) if isinstance(raw, dict) else seconds
        cost = tr_backend.estimate_cost(billed, req)
        ctx.db.log_asr(job.id, tr_backend.name, tr.model, audio_sha, billed, cost)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"raw": raw, "transcript": tr.model_dump()}), encoding="utf-8")
        job.stats()["asr_cost_usd"] = round(cost, 4)
        job.stats()["asr_cached"] = False
        job.stats()["audio_hours"] = round(billed / 3600, 4)
    if not tr.duration:
        tr.duration = seconds
    job.write_json("transcript.raw.json", raw)
    job.write_json("transcript.json", tr.model_dump())
    job.path("transcript.srt").write_text(to_srt(tr.words), encoding="utf-8")
    job.path("transcript.ass").write_text(
        full_transcript_ass(tr.words, tr.language, tr.duration, cfg.captions, cfg.hook), encoding="utf-8"
    )
    ctx._cache["tr"] = tr
    used_s, used_usd = ctx.db.asr_totals(tr_backend.name)
    if tr_backend.name == "assemblyai":
        job.stats()["asr_credit_remaining_usd"] = round(tcfg.credit_usd - used_usd, 2)
    job.stats()["language"] = tr.language
    job.stats()["sentences"] = len(tr.sentences)


def st_quality(ctx: Ctx) -> None:
    tr = ctx.transcript()
    q = quality_check(tr, ctx.cfg.quality)
    ctx.job.write_json("quality.json", q.model_dump())
    for w in q.warnings:
        ctx.job.warn(w)
        ctx.progress("quality", "warn", w)
    if not q.ok and ctx.cfg.quality.block_on_fail:
        raise QualityGateError(
            "Transcript quality check failed, so no LLM tokens were spent. See quality.json. "
            "To continue anyway: --set quality.block_on_fail=false"
        )


def _spec_dict(ctx: Ctx) -> dict:
    s = ctx.spec
    return {"content_type": s.content_type, "max_clips": s.max_clips, "min_duration": s.min_duration,
            "max_duration": s.max_duration}


def st_candidates(ctx: Ctx) -> None:
    ctx.job.state.pop("selection_source", None)  # the LLM selects again, replacing any hand picks
    records, invalid = generate_candidates(ctx.transcript(), _spec_dict(ctx), ctx.cfg, ctx.llm())
    ctx.save_records("candidates.json", records, invalid=invalid)
    ctx.progress("candidates", "info", f"{len(records)} candidates ({len(invalid)} invalid dropped)")


def st_judge(ctx: Ctx) -> None:
    records = ctx.records("candidates.json")
    records = judge_candidates(ctx.transcript(), records, _spec_dict(ctx), ctx.cfg, ctx.llm())
    ctx.save_records("judged.json", records)
    acc = sum(r.status == "accepted" for r in records)
    ctx.progress("judge", "info", f"{acc} of {len(records)} passed the judge")


def st_snap(ctx: Ctx) -> None:
    records = ctx.records("judged.json")
    records = finalize(ctx.transcript(), records, _spec_dict(ctx), ctx.cfg, ctx.wav())
    ctx.save_records("selection.json", records)
    n = sum(r.status == "selected" for r in records)
    ctx.progress("snap", "info", f"{n} clips selected")
    if n == 0:
        ctx.job.warn("No clip passed selection. Zero is a valid result; see report.html for why.")


def selected(records: list[ClipRecord]) -> list[ClipRecord]:
    return sorted((r for r in records if r.status == "selected"), key=lambda r: r.rank or 0)


def clip_name(r: ClipRecord) -> str:
    return f"clip_{r.rank:02d}"


def _is_vertical(src: dict) -> bool:
    return src["height"] >= src["width"] * 1.7


def st_reframe(ctx: Ctx) -> None:
    """Plan a face-following crop path per selected clip (writes clips/clip_NN.crop.json + .crop.cmd)."""
    records = ctx.records("selection.json")
    sel = selected(records)
    src = ctx.job.read_json("source.json")
    rc = ctx.cfg.render
    for r in records:
        r.files.pop("crop", None)
        r.files.pop("crop_cmd", None)
    detector = None
    if sel and rc.reframe == "face" and not _is_vertical(src):
        try:
            detector = ctx.face_detector or get_detector(ctx.cfg.reframe, ctx.cfg.home_path)
        except ReframeError as e:
            ctx.job.warn(f"Face reframing is off, using a center crop: {e}")
            ctx.progress("reframe", "warn", str(e))
    if detector is None:
        ctx.save_records("selection.json", records)
        return
    ctx.job.clips_dir.mkdir(parents=True, exist_ok=True)
    fps = src.get("fps") or 30.0
    for i, r in enumerate(sel, 1):
        name = clip_name(r)
        spans = clip_spans(r)
        try:
            plans = [
                reframe_clip(Path(ctx.spec.video), a, b, src["width"], src["height"], fps, rc.width, rc.height,
                             ctx.cfg.reframe, detector,
                             faces_cache=ctx.job.clips_dir / (f"{name}.faces.json" if len(spans) == 1
                                                              else f"{name}.faces.{k}.json"))
                for k, (a, b, _, _) in enumerate(spans)
            ]
            plan = concat_plans(plans, [b - a for a, b, _, _ in spans])
        except ReframeError as e:
            ctx.job.warn(f"{name}: face reframing failed, using a center crop: {e}")
            ctx.progress("reframe", "warn", f"{name}: {e}")
            continue
        write_plan(plan, ctx.job.clips_dir / f"{name}.crop.json", ctx.job.clips_dir / f"{name}.crop.cmd",
                   r.duration, fps)
        r.files.update({"crop": f"clips/{name}.crop.json", "crop_cmd": f"clips/{name}.crop.cmd"})
        m = plan.metrics
        inside = "n/a" if m["face_in_frame"] is None else f"{m['face_in_frame']:.0%}"
        center = "n/a" if m["face_in_frame_center_crop"] is None else f"{m['face_in_frame_center_crop']:.0%}"
        cut = "n/a" if m["face_cut"] is None else f"{m['face_cut']:.0%}"
        cut_c = "n/a" if m["face_cut_center_crop"] is None else f"{m['face_cut_center_crop']:.0%}"
        ctx.progress("reframe", "info", f"clip {i}/{len(sel)}: {m['shots']} shots, faces in {m['face_share']:.0%} "
                     f"of frames; a face fully in frame {inside} (center crop {center}), "
                     f"a face sliced by the edge {cut} (center crop {cut_c})")
    ctx.save_records("selection.json", records)


def load_crop(job: Job, r: ClipRecord) -> Optional[tuple[CropPlan, DynamicCrop]]:
    pj, pc = r.files.get("crop"), r.files.get("crop_cmd")
    if not pj or not pc or not (job.dir / pj).is_file() or not (job.dir / pc).is_file():
        return None
    plan = CropPlan.from_json(json.loads((job.dir / pj).read_text(encoding="utf-8")))
    x0 = int(plan.xs[0]) if plan.xs else (plan.src_w - plan.crop_w) // 2
    return plan, DynamicCrop(plan.crop_w, plan.crop_h, x0 - x0 % 2, plan.y, job.dir / pc)


def st_captions(ctx: Ctx) -> None:
    tr = ctx.transcript()
    records = ctx.records("selection.json")
    ctx.job.clips_dir.mkdir(parents=True, exist_ok=True)
    rc = ctx.cfg.render
    for r in selected(records):
        name = clip_name(r)
        hook = r.hook_text if ctx.cfg.hook.enabled else None
        doc = clip_caption_doc(tr, r, hook)
        write_caption_files(doc, ctx.job.clips_dir / f"{name}.ass", ctx.job.clips_dir / f"{name}.captions.json",
                            ctx.cfg.captions, ctx.cfg.hook, rc.width, rc.height)
        r.files.update({"ass": f"clips/{name}.ass", "captions_json": f"clips/{name}.captions.json"})
    ctx.save_records("selection.json", records)


def clip_caption_doc(tr: Transcript, r: ClipRecord, hook: Optional[str]):
    """Caption words on the clip's own timeline; jump-cut pieces are laid end to end."""
    doc, off = None, 0.0
    for a, b, w0, w1 in clip_spans(r):
        part = doc_from_words(tr.words[w0 : w1 + 1], a - off, off + (b - a), tr.language, hook)
        if doc is None:
            doc = part
        else:
            doc.words += [w for w in part.words if w.start >= off - 1e-6]
        off += b - a
    doc.duration = round(off, 3)
    return doc


def source_offset(r: ClipRecord, at: float) -> float:
    """For a time `at` on the clip's timeline, the source time where the clip would start if it were
    one continuous piece there (what render_frame needs to find the right frame)."""
    off = 0.0
    spans = clip_spans(r)
    for a, b, _, _ in spans:
        if at < off + (b - a) or (a, b) == spans[-1][:2]:
            return a - off
        off += b - a
    return r.start


def _fonts(ctx: Ctx) -> Path:
    return prepare_fonts_dir(ctx.cfg.captions, ctx.cfg.home_path, ctx.job.path("fonts"))


def render_one(ctx: Ctx, r: ClipRecord, fonts: Path, wav: Optional[media.Wav]) -> None:
    name = clip_name(r)
    out = ctx.job.clips_dir / f"{name}.mp4"
    ass = ctx.job.clips_dir / f"{name}.ass"
    use_ass = ass if (ctx.cfg.captions.enabled or ctx.cfg.hook.enabled) and ass.is_file() else None
    crop = load_crop(ctx.job, r)
    t0 = time.monotonic()
    try:
        spans = clip_spans(r)
        render_clip(Path(ctx.spec.video), out, r.start, r.end, ctx.cfg.render, ctx.cfg.snap.fade_ms, use_ass,
                    fonts, crop[1] if crop else None,
                    segments=[(a, b) for a, b, _, _ in spans] if len(spans) > 1 else None)
    except PermissionError as e:  # Windows: the old file is open in a player and cannot be replaced
        raise PipelineError(
            f"{out} is open in another program (a video player?). Close it, then run: "
            f"clipper resume {ctx.job.id} --from render"
        ) from e
    r.files["video"] = f"clips/{name}.mp4"
    src = ctx.job.read_json("source.json")
    audio_rms = wav.rms(r.start, r.end) if wav else None
    render_reasons: set[str] = set()
    framing = None
    if _is_vertical(src):
        crop_ok = True
    elif crop is None:
        crop_ok = None
        render_reasons.add("center crop on wide source: check framing")
    else:
        framing = crop[0].metrics
        inside = framing.get("face_in_frame")
        if inside is None:
            crop_ok = None
            render_reasons.add("no faces found: check framing")
        else:
            crop_ok = inside >= ctx.cfg.reframe.min_face_in_frame
            if not crop_ok:
                render_reasons.add(f"no face fully in frame in {1 - inside:.0%} of frames with faces")
    tech = {
        "crop_ok": crop_ok,
        "framing": framing,
        "captions_ok": None,
        "audio_ok": None if audio_rms is None else audio_rms > 0.003,
        "render_seconds": round(time.monotonic() - t0, 2),
    }
    if use_ass is not None:
        lang = ctx.transcript().language
        rtl_font = (fonts / "NotoSansArabic-Bold.ttf").exists() or bool(ctx.cfg.captions.fonts_dir)
        tech["captions_ok"] = (not is_rtl(lang)) or rtl_font
    if tech["audio_ok"] is False:
        render_reasons.add("audio nearly silent")
    if tech["captions_ok"] is False:
        render_reasons.add("no font for this script")
    # Replace (not pile up) the reasons an earlier render added; keep selection's own reasons.
    previous = set(r.technical.get("review_reasons") or []) | {"center crop on wide source: check framing"}
    tech["review_reasons"] = sorted(render_reasons)
    r.technical = tech
    r.review_reasons = sorted((set(r.review_reasons) - previous) | render_reasons)
    r.needs_review = bool(r.review_reasons)


def st_render(ctx: Ctx) -> None:
    records = ctx.records("selection.json")
    fonts = _fonts(ctx)
    wav = ctx.wav()
    sel = selected(records)
    t0 = time.monotonic()
    for i, r in enumerate(sel, 1):
        ctx.progress("render", "info", f"clip {i}/{len(sel)} ({r.duration:.0f}s)")
        render_one(ctx, r, fonts, wav)
    ctx.job.stats()["render_seconds"] = round(time.monotonic() - t0, 2)
    ctx.save_records("selection.json", records)


def _truncate(s: Optional[str], n: int) -> Optional[str]:
    if s is None:
        return None
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def st_metadata(ctx: Ctx) -> None:
    records = ctx.records("selection.json")
    hk = ctx.cfg.hook
    clips = []
    for r in selected(records):
        tags = []
        for t in r.tags:
            t = t.strip().lstrip("#")
            if t and t.lower() not in {x.lower() for x in tags}:
                tags.append(t)
        hook = " ".join((r.hook_text or "").split()[: hk.max_words]) or None
        if len(tags) < 3:
            r.notes.append("fewer than 3 tags from the judge")
        clips.append({
            "rank": r.rank, "id": r.id, "source": r.pass1.get("source", "llm"), "file": r.files.get("video"), "start": r.start, "end": r.end,
            "segments": [{"start": g["start"], "end": g["end"]} for g in r.segments],
            "duration": r.duration, "title": _truncate(r.title, 100), "description": _truncate(r.description, 5000),
            "tags": tags[:5], "hook_text": hook, "hook_line": r.pass1.get("hook_line"), "rationale": r.rationale,
            "content_type": r.content_type, "scores": r.scores, "editorial": r.editorial,
            "confidence": r.confidence, "technical": r.technical, "needs_review": r.needs_review,
            "review_reasons": r.review_reasons, "start_sentence_id": r.start_sentence_id,
            "end_sentence_id": r.end_sentence_id, "start_word_offset": r.start_word_offset,
            "end_word_offset": r.end_word_offset, "text": r.text, "files": r.files, "notes": r.notes,
        })
    st = ctx.job.stats()
    st["llm_cost_usd"] = round(ctx.db.llm_cost_for_job(ctx.job.id), 4)
    st["total_cost_usd"] = round(st["llm_cost_usd"] + st.get("asr_cost_usd", 0.0), 4)
    hours = st.get("source_duration_s", 0) / 3600
    st["cost_per_source_hour_usd"] = round(st["total_cost_usd"] / hours, 4) if hours else None
    ctx.job.write_json("clips.json", {
        "job_id": ctx.job.id, "source": ctx.spec.video, "privacy_default": "private",
        "prompts": {"candidates": ctx.cfg.llm.candidate_prompt, "judge": ctx.cfg.llm.judge_prompt},
        "models": {"candidates": ctx.cfg.llm.candidate_model, "judge": ctx.cfg.llm.judge_model},
        "clips": clips,
    })


def st_report(ctx: Ctx) -> None:
    from .report import write_report

    write_report(ctx.job, ctx.cfg)


STAGE_FUNCS: dict[str, Callable[[Ctx], None]] = {
    "ingest": st_ingest, "transcribe": st_transcribe, "quality": st_quality, "candidates": st_candidates,
    "judge": st_judge, "snap": st_snap, "reframe": st_reframe, "captions": st_captions, "render": st_render, "metadata": st_metadata,
    "report": st_report,
}


def _print_progress(stage: str, status: str, msg: str) -> None:
    print(f"[{stage}] {status}: {msg}", flush=True)


def run_job(
    job: Job,
    cfg: Config,
    db: DB,
    stop_at: Optional[str] = None,
    from_stage: Optional[str] = None,
    only: Optional[str] = None,
    progress: Progress = _print_progress,
    llm_backend=None,
    transcriber=None,
    face_detector=None,
) -> dict:
    """Run stages in order, skipping ones already done. A stage that runs forces all later ones."""
    stop_at, from_stage, only = resolve_stage(stop_at), resolve_stage(from_stage), resolve_stage(only)
    ctx = Ctx(job, cfg, db, progress, llm_backend=llm_backend, transcriber=transcriber,
              face_detector=face_detector)
    job.state["config"] = cfg.model_dump(exclude={"home"})
    job.save()
    rerun = False
    t_job = time.monotonic()
    db.set_job_status(job.id, "running")
    for stage in STAGES:
        if only and stage != only:
            continue
        if from_stage and stage == from_stage:
            rerun = True
        if not only and not rerun and job.stage_done(stage):
            progress(stage, "skip", "already done")
        else:
            rerun = True
            progress(stage, "start", "")
            run_id = db.stage_start(job.id, stage)
            t0 = time.monotonic()
            try:
                STAGE_FUNCS[stage](ctx)
            except Exception as e:  # record and re-raise with the stage name
                secs = time.monotonic() - t0
                job.mark(stage, "failed", secs, str(e))
                db.stage_end(run_id, "failed", traceback.format_exc()[-4000:])
                db.set_job_status(job.id, f"failed:{stage}")
                raise
            secs = time.monotonic() - t0
            job.stats().setdefault("stage_seconds", {})[stage] = round(secs, 2)
            job.mark(stage, "done", secs)
            db.stage_end(run_id, "done")
            progress(stage, "done", f"{secs:.1f}s")
        if stop_at and stage == stop_at:
            break
    job.stats()["last_run_wall_seconds"] = round(time.monotonic() - t_job, 2)
    job.save()
    finished = all(job.stage_done(s) for s in STAGES)
    db.set_job_status(job.id, "done" if finished else f"stopped:{stop_at or only}")
    return job.state


def apply_picks(job: Job, cfg: Config, db: DB, picks_path: Path, progress: Progress = _print_progress,
                face_detector=None) -> dict:
    """Use hand-picked spans (picks.py) as the selection, then cut, crop, caption and report as usual."""
    import shutil

    from .picks import load_picks, picks_to_records

    if not job.path("transcript.json").is_file():
        raise PipelineError(f"{job.id} has no transcript yet; run: clipper run VIDEO --i-own-rights --stop-at transcript")
    tr = Transcript.model_validate(job.read_json("transcript.json"))
    try:
        records = picks_to_records(tr, load_picks(picks_path), picks_path.name)
    except ValueError as e:
        raise PipelineError(str(e)) from e
    shutil.copyfile(picks_path, job.path("picks.json"))
    blob = {"records": [r.model_dump() for r in records], "invalid": [], "source": "manual"}
    job.write_json("candidates.json", blob)
    job.write_json("judged.json", blob)
    job.state["selection_source"] = f"hand-picked ({picks_path.name})"
    for st in ("candidates", "judge"):
        job.mark(st, "done", 0.0)
    progress("picks", "info", f"{len(records)} hand-picked clips from {picks_path.name}")
    return run_job(job, cfg, db, from_stage="snap", progress=progress, face_detector=face_detector)


def rerender_clip(job: Job, cfg: Config, db: DB, rank: int, from_json: bool = True) -> Path:
    """Re-render one clip after editing clips/clip_NN.captions.json (or changing caption config)."""
    ctx = Ctx(job, cfg, db, _print_progress)
    records = ctx.records("selection.json")
    r = next((x for x in selected(records) if x.rank == rank), None)
    if r is None:
        raise PipelineError(f"no selected clip with rank {rank}")
    name = clip_name(r)
    cj = job.clips_dir / f"{name}.captions.json"
    if from_json and cj.is_file():
        doc = load_doc(cj)
        (job.clips_dir / f"{name}.ass").write_text(
            build_ass(doc, cfg.captions, cfg.hook, cfg.render.width, cfg.render.height), encoding="utf-8")
    render_one(ctx, r, _fonts(ctx), ctx.wav())
    ctx.save_records("selection.json", records)
    st_metadata(ctx)
    st_report(ctx)
    return job.clips_dir / f"{name}.mp4"

