"""Command line interface. Run `clipper --help`."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml

from .config import (
    Config, export_settings, load_config, load_layers, parse_set_overrides, settings_path,
)
from .db import DB
from .jobs import STAGE_ALIASES, STAGES, JobSpec, find_job
from .pipeline import PipelineError, apply_picks, create_job, rerender_clip, run_job

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Long video to vertical Shorts clipper.")
eval_app = typer.Typer(no_args_is_help=True, help="Gold set, labeling and metrics.")
config_app = typer.Typer(no_args_is_help=True, help="Inspect, export and import settings.")
app.add_typer(eval_app, name="eval")
app.add_typer(config_app, name="config")

STAGE_HELP = f"one of: {', '.join(STAGES)} (aliases: {', '.join(STAGE_ALIASES)})"


def _cfg(config: Optional[str], sets: Optional[list[str]], flags: Optional[dict] = None) -> Config:
    over = parse_set_overrides(sets or [])
    if flags:
        from .config import deep_merge

        over = deep_merge(over, flags)
    return load_config(config, over)


def _db(cfg: Config) -> DB:
    return DB(cfg.home_path / "clipper.db")


def _fail(msg: str, code: int = 1) -> None:
    typer.secho(f"error: {msg}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code)


ConfigOpt = typer.Option(None, "--config", "-c", help="config.yaml path (default: $CLIPPER_HOME/config.yaml)")
SetOpt = typer.Option(None, "--set", "-s", help="override any config key, e.g. --set llm.judge_effort=high")


@app.command()
def run(
    video: Path = typer.Argument(..., help="Local video file you own or have rights to."),
    i_own_rights: bool = typer.Option(False, "--i-own-rights", help="Required. Confirms you own or have rights to the video."),
    language: Optional[str] = typer.Option(None, help="Language code, e.g. en, fr, ar. Default: auto-detect."),
    content_type: str = typer.Option("podcast", help="podcast | interview | talking_head | tutorial | other"),
    max_clips: Optional[int] = typer.Option(None, help="Maximum clips to output (default from config: 8)."),
    min_duration: Optional[float] = typer.Option(None, help="Minimum clip length in seconds (default 20)."),
    max_duration: Optional[float] = typer.Option(None, help="Target maximum clip length in seconds (default 60)."),
    speakers: Optional[int] = typer.Option(None, help="Number of speakers, if known. >1 turns on speaker labels."),
    stop_at: Optional[str] = typer.Option(None, help=f"Stop after this stage, {STAGE_HELP}."),
    from_stage: Optional[str] = typer.Option(None, "--from", help="Re-run from this stage onward."),
    job_id: Optional[str] = typer.Option(None, help="Use this job id instead of <name>-<hash>."),
    config: Optional[str] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
):
    """Process one video: transcribe, select, cut, caption, report."""
    cfg = _cfg(config, set_)
    sel = cfg.selection
    spec = JobSpec(
        video=str(video), language=language, content_type=content_type,  # type: ignore[arg-type]
        max_clips=max_clips if max_clips is not None else sel.max_clips,
        min_duration=min_duration if min_duration is not None else sel.min_duration,
        max_duration=max_duration if max_duration is not None else sel.max_duration,
        speakers=speakers, rights_confirmed=i_own_rights,
    )
    if spec.max_duration > sel.hard_cap_duration:
        _fail(f"--max-duration cannot exceed {sel.hard_cap_duration:.0f}s (Shorts maximum)")
    if spec.min_duration >= spec.max_duration:
        _fail("--min-duration must be below --max-duration")
    db = _db(cfg)
    try:
        job = create_job(video, spec, cfg, db, job_id)
        typer.echo(f"job {job.id} -> {job.dir}")
        state = run_job(job, cfg, db, stop_at=stop_at, from_stage=from_stage)
    except (PipelineError, ValueError, FileNotFoundError) as e:
        _fail(str(e))
    _summary(job, state)


def _summary(job, state: dict) -> None:
    st = state.get("stats", {})
    out = {k: st.get(k) for k in ("asr_cost_usd", "llm_cost_usd", "total_cost_usd", "cost_per_source_hour_usd",
                                  "last_run_wall_seconds", "asr_credit_remaining_usd") if k in st}
    if job.path("clips.json").is_file():
        clips = job.read_json("clips.json")["clips"]
        out["clips"] = [{"rank": c["rank"], "file": c["file"], "duration": c["duration"], "title": c["title"],
                         "needs_review": c["needs_review"]} for c in clips]
    if job.path("report.html").is_file():
        out["report"] = str(job.path("report.html"))
    typer.echo(json.dumps(out, indent=2, ensure_ascii=False))


@app.command()
def stage(
    job: str = typer.Argument(..., help="Job id, prefix, or folder."),
    name: str = typer.Argument(..., help=f"Stage to re-run alone, {STAGE_HELP}."),
    config: Optional[str] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
):
    """Re-run a single stage of an existing job (later stages are not touched)."""
    cfg = _cfg(config, set_)
    db = _db(cfg)
    try:
        j = find_job(cfg.home_path, job)
        run_job(j, cfg, db, only=name)
    except (PipelineError, ValueError, FileNotFoundError) as e:
        _fail(str(e))


@app.command()
def resume(
    job: str = typer.Argument(..., help="Job id, prefix, or folder."),
    stop_at: Optional[str] = typer.Option(None, help=f"Stop after this stage, {STAGE_HELP}."),
    from_stage: Optional[str] = typer.Option(None, "--from", help="Re-run from this stage onward."),
    config: Optional[str] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
):
    """Continue a job (for example after --stop-at), optionally re-running from a stage."""
    cfg = _cfg(config, set_)
    db = _db(cfg)
    try:
        j = find_job(cfg.home_path, job)
        state = run_job(j, cfg, db, stop_at=stop_at, from_stage=from_stage)
    except (PipelineError, ValueError, FileNotFoundError) as e:
        _fail(str(e))
    _summary(j, state)


@app.command()
def picks(
    job: str = typer.Argument(..., help="Job id (or a unique prefix)."),
    file: Path = typer.Argument(..., help="JSON list of clips: start/end times (\"9:50.5\") or sentence IDs."),
    config: Optional[str] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
):
    """Render hand-picked clips instead of the LLM's selection (see src/clipper/picks.py for the format)."""
    cfg = _cfg(config, set_)
    db = _db(cfg)
    try:
        j = find_job(cfg.home_path, job)
        state = apply_picks(j, cfg, db, file)
    except (PipelineError, ValueError, FileNotFoundError) as e:
        _fail(str(e))
    _summary(j, state)


@app.command()
def jobs(limit: int = 20, config: Optional[str] = ConfigOpt):
    """List job history."""
    cfg = _cfg(config, None)
    for r in _db(cfg).list_jobs(limit):
        typer.echo(f"{r['id']:<50} {r['status']:<20} {Path(r['source']).name}")


@app.command()
def report(job: str, config: Optional[str] = ConfigOpt, set_: Optional[list[str]] = SetOpt):
    """Rebuild report.html for a job."""
    cfg = _cfg(config, set_)
    from .report import write_report

    typer.echo(write_report(find_job(cfg.home_path, job), cfg))


@app.command()
def rerender(
    job: str,
    clip: int = typer.Option(..., help="Clip rank (clip_01 = 1)."),
    config: Optional[str] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
):
    """Re-render one clip from its (edited) captions JSON and current caption style. No re-transcription."""
    cfg = _cfg(config, set_)
    try:
        out = rerender_clip(find_job(cfg.home_path, job), cfg, _db(cfg), clip)
    except (PipelineError, FileNotFoundError) as e:
        _fail(str(e))
    typer.echo(out)


@app.command()
def preview(
    job: str,
    clip: int = typer.Option(1, help="Clip rank."),
    at: float = typer.Option(1.5, help="Seconds into the clip."),
    out: Optional[Path] = typer.Option(None, help="PNG path (default: clips/clip_NN.preview.png)."),
    config: Optional[str] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
):
    """Render a single frame with the current caption style, to tune it in seconds."""
    from .captions import build_ass, load_doc
    from .pipeline import Ctx, clip_name, load_crop, selected, source_offset, _print_progress
    from .render import prepare_fonts_dir, render_frame

    cfg = _cfg(config, set_)
    j = find_job(cfg.home_path, job)
    ctx = Ctx(j, cfg, _db(cfg), _print_progress)
    r = next((x for x in selected(ctx.records("selection.json")) if x.rank == clip), None)
    if r is None:
        _fail(f"no selected clip with rank {clip}")
    name = clip_name(r)
    cj = j.clips_dir / f"{name}.captions.json"
    if not cj.is_file():
        _fail(f"{cj} missing; run the captions stage first")
    ass = j.clips_dir / f"{name}.preview.ass"
    ass.write_text(build_ass(load_doc(cj), cfg.captions, cfg.hook, cfg.render.width, cfg.render.height),
                   encoding="utf-8")
    png = out or (j.clips_dir / f"{name}.preview.png")
    fonts = prepare_fonts_dir(cfg.captions, cfg.home_path, j.path("fonts"))
    crop = load_crop(j, r)
    render_frame(Path(j.spec.video), png, source_offset(r, at), at, cfg.render, ass, fonts,
                 crop[1] if crop else None)
    typer.echo(png)


@app.command()
def usage(config: Optional[str] = ConfigOpt):
    """Show ASR credit used and remaining, and LLM spend."""
    cfg = _cfg(config, None)
    db = _db(cfg)
    secs, usd = db.asr_totals("assemblyai")
    llm = db.conn.execute("SELECT COALESCE(SUM(cost_usd),0), COUNT(*) FROM llm_runs WHERE cached=0").fetchone()
    typer.echo(json.dumps({
        "assemblyai_audio_hours": round(secs / 3600, 3), "assemblyai_cost_usd": round(usd, 3),
        "assemblyai_credit_remaining_usd": round(cfg.transcribe.credit_usd - usd, 2),
        "llm_cost_usd": round(llm[0], 4), "llm_calls": llm[1],
    }, indent=2))


# ---------------- config ----------------


@config_app.command("show")
def config_show(config: Optional[str] = ConfigOpt, set_: Optional[list[str]] = SetOpt):
    """Print the effective config and which files contributed."""
    cfg = _cfg(config, set_)
    layers = [p for p, _ in load_layers(config, cfg.home_path)]
    typer.echo("# layers (low to high): defaults, " + ", ".join(layers + (["--set"] if set_ else [])))
    typer.echo(yaml.safe_dump(cfg.model_dump(), sort_keys=False, allow_unicode=True))


@config_app.command("export")
def config_export(
    out: Path,
    include_secrets: bool = typer.Option(False, "--include-secrets", help="Also write API keys in plain text."),
    config: Optional[str] = ConfigOpt,
):
    """Export effective settings to a JSON file. API keys are left out unless --include-secrets."""
    cfg = _cfg(config, None)
    out.write_text(json.dumps(export_settings(cfg, include_secrets), indent=2), encoding="utf-8")
    if include_secrets:
        typer.secho("warning: file contains API keys in plain text", fg=typer.colors.YELLOW, err=True)
    typer.echo(out)


@config_app.command("import")
def config_import(src: Path, config: Optional[str] = ConfigOpt):
    """Import settings JSON into $CLIPPER_HOME/settings.json (the layer the web UI writes). Keys are ignored."""
    cfg = _cfg(config, None)
    data = json.loads(src.read_text(encoding="utf-8"))
    data.pop("_secrets", None)
    data.pop("home", None)
    Config.model_validate({**data, "home": cfg.home})  # validate before writing
    p = settings_path(cfg.home_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    typer.echo(p)


# ---------------- eval ----------------


@eval_app.command("label-sheet")
def eval_label_sheet(job: str, out: Optional[Path] = None, config: Optional[str] = ConfigOpt):
    """Write a CSV labeling sheet with every candidate the system considered, plus rows for your own picks."""
    from .evaluation import write_label_sheet

    cfg = _cfg(config, None)
    j = find_job(cfg.home_path, job)
    typer.echo(write_label_sheet(j, out or j.path("labels.csv")))


@eval_app.command("import")
def eval_import(
    sheet: Path,
    out: Path = typer.Option(..., help="Gold JSON to write, e.g. gold/ep12.json"),
    job: Optional[str] = typer.Option(None, help="Job the sheet came from (records the video and its hash)."),
    name: Optional[str] = None,
    config: Optional[str] = ConfigOpt,
):
    """Turn a filled-in labeling sheet into a gold file."""
    from .evaluation import import_label_sheet

    cfg = _cfg(config, None)
    video = sha = ctype = None
    if job:
        j = find_job(cfg.home_path, job)
        video, sha, ctype = j.spec.video, j.state.get("source_sha256"), j.spec.content_type
    gold = import_label_sheet(sheet, name or out.stem, video, sha, ctype)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(gold.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(f"{out}: {len(gold.labels)} labels")


@eval_app.command("score")
def eval_score(
    job: str,
    gold: Path = typer.Option(..., help="Gold JSON file."),
    k: Optional[int] = typer.Option(None, help="k for precision@k (default: number selected)."),
    iou_threshold: float = 0.5,
    config: Optional[str] = ConfigOpt,
):
    """Score a job against a gold file and log the result."""
    from .evaluation import GoldFile, score_job

    cfg = _cfg(config, None)
    j = find_job(cfg.home_path, job)
    g = GoldFile.model_validate_json(gold.read_text(encoding="utf-8"))
    metrics = score_job(j, g, k, iou_threshold)
    jc = j.state.get("config", {}).get("llm", {})
    _db(cfg).log_eval(j.id, str(gold), jc, metrics)
    typer.echo(json.dumps(metrics, indent=2))


@eval_app.command("run")
def eval_run(
    gold_dir: Path = typer.Argument(..., help="Folder of gold JSON files with a `video` field."),
    i_own_rights: bool = typer.Option(False, "--i-own-rights"),
    k: Optional[int] = None,
    config: Optional[str] = ConfigOpt,
    set_: Optional[list[str]] = SetOpt,
):
    """Run selection (no rendering) on every gold video and score it. Transcripts come from cache."""
    from .evaluation import GoldFile, score_job, summarize

    cfg = _cfg(config, set_)
    db = _db(cfg)
    rows = []
    for gp in sorted(gold_dir.glob("*.json")):
        g = GoldFile.model_validate_json(gp.read_text(encoding="utf-8"))
        if not g.video:
            typer.echo(f"skip {gp.name}: no video path")
            continue
        spec = JobSpec(video=g.video, content_type=(g.content_type or "podcast"),  # type: ignore[arg-type]
                       max_clips=cfg.selection.max_clips, min_duration=cfg.selection.min_duration,
                       max_duration=cfg.selection.max_duration, rights_confirmed=i_own_rights)
        try:
            j = create_job(Path(g.video), spec, cfg, db)
            run_job(j, cfg, db, stop_at="snap", from_stage="candidates")
        except PipelineError as e:
            typer.echo(f"{gp.name}: {e}", err=True)
            continue
        st = j.state.setdefault("stats", {})
        st["llm_cost_usd"] = round(db.llm_cost_for_job(j.id), 4)
        st["total_cost_usd"] = round(st["llm_cost_usd"] + st.get("asr_cost_usd", 0.0), 4)
        j.save()
        m = score_job(j, g, k)
        db.log_eval(j.id, str(gp), cfg.llm.model_dump(), m)
        rows.append({"gold": gp.name, **{x: m[x] for x in ("precision_at_k", "complete_thought_share",
                                                            "human_good_recall", "selected")}})
    typer.echo(json.dumps({"runs": rows, "summary": summarize(db.list_evals())}, indent=2))


@eval_app.command("summary")
def eval_summary(config: Optional[str] = ConfigOpt):
    """Compare logged eval runs, grouped by prompt and model versions."""
    from .evaluation import summarize

    cfg = _cfg(config, None)
    typer.echo(json.dumps(summarize(_db(cfg).list_evals()), indent=2))


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
