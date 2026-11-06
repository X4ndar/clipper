"""End to end on a generated 70-second two-speaker sample: real ffmpeg, fixture transcript, fake LLM."""

import json
import subprocess

import pytest

from clipper.config import load_config
from clipper.db import DB
from clipper.evaluation import GoldFile, GoldLabel, import_label_sheet, score_job, write_label_sheet
from clipper.jobs import JobSpec
from clipper.pipeline import RightsError, create_job, rerender_clip, run_job

pytestmark = pytest.mark.integration

OVR = {"transcribe": {"backend": "fixture"}, "llm": {"backend": "fake"}}


def setup(tmp_path, sample_video, max_clips=3, **spec_kw):
    cfg = load_config(cli_overrides=OVR, home=tmp_path / "home")
    db = DB(cfg.home_path / "clipper.db")
    spec = JobSpec(video=str(sample_video), max_clips=max_clips, min_duration=10, max_duration=40,
                   rights_confirmed=True, **spec_kw)
    return cfg, db, create_job(sample_video, spec, cfg, db)


def quiet(*a):
    pass


def test_rights_flag_required(tmp_path, sample_video):
    cfg = load_config(cli_overrides=OVR, home=tmp_path / "home")
    with pytest.raises(RightsError):
        create_job(sample_video, JobSpec(video=str(sample_video)), cfg, DB(cfg.home_path / "x.db"))


def test_full_pipeline(tmp_path, sample_video):
    cfg, db, job = setup(tmp_path, sample_video, max_clips=1)
    run_job(job, cfg, db, progress=quiet)
    clips = job.read_json("clips.json")["clips"]
    assert len(clips) == 1
    for c in clips:
        f = job.dir / c["file"]
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(f)],
            capture_output=True, text=True, check=True).stdout)
        v = next(s for s in probe["streams"] if s["codec_type"] == "video")
        assert (v["width"], v["height"]) == (1080, 1920)
        assert any(s["codec_type"] == "audio" for s in probe["streams"])
        assert abs(float(probe["format"]["duration"]) - c["duration"]) < 0.25
        assert 10 <= c["duration"] <= 40 * 1.5
        assert c["title"] and len(c["title"]) <= 100 and len(c["tags"]) >= 3
        assert (job.dir / c["files"]["ass"]).is_file() and (job.dir / c["files"]["captions_json"]).is_file()
        # starts on a sentence start, ends on a sentence end
        assert c["confidence"]["boundary_ok"] is True
    for name in ("transcript.json", "transcript.srt", "transcript.ass", "quality.json", "report.html"):
        assert job.path(name).is_file()
    html = job.path("report.html").read_text()
    sel = job.read_json("selection.json")["records"]
    assert any(r["status"] == "over_quota" for r in sel)  # non-selected ones are listed with reasons
    assert "Good, over quota" in html and "max_clips=1 already filled" in html
    # stage timings and costs recorded
    st = job.state["stats"]
    assert st["total_cost_usd"] == 0 and "render" in st["stage_seconds"]


def test_stop_at_and_resume_and_cache(tmp_path, sample_video):
    cfg, db, job = setup(tmp_path, sample_video)
    run_job(job, cfg, db, stop_at="transcript", progress=quiet)
    assert job.stage_done("quality") and not job.stage_done("candidates")
    assert not job.path("candidates.json").exists()
    run_job(job, cfg, db, stop_at="candidates", progress=quiet)
    assert job.path("candidates.json").exists() and not job.path("judged.json").exists()
    # same audio under a new job id: transcript comes from cache
    spec = JobSpec(video=str(sample_video), max_clips=2, min_duration=10, max_duration=40, rights_confirmed=True)
    job2 = create_job(sample_video, spec, cfg, db, job_id="second")
    msgs = []
    run_job(job2, cfg, db, stop_at="transcribe", progress=lambda s, st, m: msgs.append(m))
    assert any("cache hit" in m for m in msgs)
    assert job2.state["stats"]["asr_cached"] is True


def test_single_stage_rerun_and_rerender(tmp_path, sample_video):
    cfg, db, job = setup(tmp_path, sample_video)
    run_job(job, cfg, db, progress=quiet)
    before = job.read_json("clips.json")
    run_job(job, cfg, db, only="report", progress=quiet)
    assert job.read_json("clips.json") == before
    # edit captions JSON (fix an "ASR error") and re-render without re-transcribing
    cj = job.clips_dir / "clip_01.captions.json"
    d = json.loads(cj.read_text())
    d["words"][0]["word"] = "EDITED"
    cj.write_text(json.dumps(d))
    out = rerender_clip(job, cfg, db, 1)
    assert out.is_file() and "EDITED" in (job.clips_dir / "clip_01.ass").read_text()


def test_eval_roundtrip(tmp_path, sample_video):
    cfg, db, job = setup(tmp_path, sample_video)
    run_job(job, cfg, db, stop_at="snap", progress=quiet)
    sheet = write_label_sheet(job, tmp_path / "labels.csv")
    rows = sheet.read_text().splitlines()
    assert rows[0].startswith("candidate_id,status,start,end")
    # label: first selected clip good, everything else bad
    import csv
    data = list(csv.DictReader(open(sheet)))
    sel = [r for r in job.read_json("selection.json")["records"] if r["status"] == "selected"]
    for r in data:
        if r["candidate_id"] == sel[0]["id"]:
            r["label"], r["reason"] = "good", "clear question and answer"
        elif r["candidate_id"] not in ("human", ""):
            r["label"], r["needs_trim"] = "bad", "y"
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader()
        w.writerows(data)
    gold = import_label_sheet(sheet, "sample", sha=job.state["source_sha256"])
    assert any(l.label == "good" for l in gold.labels)
    m = score_job(job, gold)
    assert m["precision_at_k"] == pytest.approx(1 / len(sel))
    assert m["human_good_recall"] == 1.0
    with pytest.raises(ValueError):
        score_job(job, GoldFile(name="x", source_sha256="other", labels=[]))
