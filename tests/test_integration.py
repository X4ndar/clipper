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

OVR = {"transcribe": {"backend": "fixture"}, "llm": {"backend": "fake"}, "render": {"reframe": "center"}}


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
    html = job.path("report.html").read_text(encoding="utf-8")
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
    d = json.loads(cj.read_text(encoding="utf-8"))
    d["words"][0]["word"] = "EDITED"
    cj.write_text(json.dumps(d), encoding="utf-8")
    out = rerender_clip(job, cfg, db, 1)
    assert out.is_file() and "EDITED" in (job.clips_dir / "clip_01.ass").read_text(encoding="utf-8")


def test_eval_roundtrip(tmp_path, sample_video):
    cfg, db, job = setup(tmp_path, sample_video)
    run_job(job, cfg, db, stop_at="snap", progress=quiet)
    sheet = write_label_sheet(job, tmp_path / "labels.csv")
    rows = sheet.read_text(encoding="utf-8").splitlines()
    assert rows[0].startswith("candidate_id,status,start,end")
    # label: first selected clip good, everything else bad
    import csv
    data = list(csv.DictReader(open(sheet, encoding="utf-8")))
    sel = [r for r in job.read_json("selection.json")["records"] if r["status"] == "selected"]
    for r in data:
        if r["candidate_id"] == sel[0]["id"]:
            r["label"], r["reason"] = "good", "clear question and answer"
        elif r["candidate_id"] not in ("human", ""):
            r["label"], r["needs_trim"] = "bad", "y"
    with open(sheet, "w", newline="", encoding="utf-8") as f:
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


class LeftFace:
    """Stand-in detector: one face on the left third (the test pattern has no faces)."""

    name = "test-left-face"

    def detect(self, rgb):
        from clipper.reframe import Face

        return [Face(cx=0.2, cy=0.4, w=0.08, h=0.14, score=0.9)]


def test_face_reframe_moves_the_crop(tmp_path, sample_video):
    cfg = load_config(cli_overrides={**OVR, "render": {"reframe": "face"}}, home=tmp_path / "home")
    db = DB(cfg.home_path / "clipper.db")
    spec = JobSpec(video=str(sample_video), max_clips=1, min_duration=10, max_duration=40, rights_confirmed=True)
    job = create_job(sample_video, spec, cfg, db)
    run_job(job, cfg, db, progress=quiet, face_detector=LeftFace())
    c = job.read_json("clips.json")["clips"][0]
    plan = json.loads((job.dir / c["files"]["crop"]).read_text(encoding="utf-8"))
    assert plan["mode"] == "face" and plan["crop_w"] == 404
    assert max(plan["xs"]) < (1280 - 404) / 2  # crop sits left of center, on the face
    tech = c["technical"]
    assert tech["crop_ok"] is True and tech["framing"]["face_in_frame"] == 1.0
    assert tech["framing"]["face_in_frame_center_crop"] == 0.0
    assert not any("crop" in r or "face" in r for r in c["review_reasons"])
    cmds = (job.dir / c["files"]["crop_cmd"]).read_text(encoding="utf-8")
    assert cmds.startswith("0.0000 crop@rf x ")
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(job.dir / c["file"])],
        capture_output=True, text=True, check=True).stdout)
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)


def test_hand_picks_replace_the_selection(tmp_path, sample_video):
    from clipper.pipeline import apply_picks

    cfg, db, job = setup(tmp_path, sample_video)
    run_job(job, cfg, db, stop_at="transcript", progress=quiet)
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps([
        {"start_sentence": 12, "end_sentence": 16, "title": "Hiring too fast", "hook_text": "My biggest mistake"},
        {"start_sentence": 3, "end_sentence": 9},
    ]), encoding="utf-8")
    apply_picks(job, cfg, db, picks, progress=quiet)
    clips = job.read_json("clips.json")["clips"]
    assert [c["source"] for c in clips] == ["manual", "manual"]
    assert clips[0]["title"] == "Hiring too fast" and clips[0]["text"].startswith("The biggest mistake")
    assert all((job.dir / c["file"]).is_file() for c in clips)
    html = job.path("report.html").read_text(encoding="utf-8")
    assert "hand-picked (picks.json)" in html and "<b>hand-picked</b>" in html
    assert job.path("picks.json").is_file()


def test_jump_cut_clip_renders_the_pieces_back_to_back(tmp_path, sample_video):
    from clipper.pipeline import apply_picks

    cfg, db, job = setup(tmp_path, sample_video)
    run_job(job, cfg, db, stop_at="transcript", progress=quiet)
    picks = tmp_path / "picks.json"
    picks.write_text(json.dumps([{"segments": [{"start_sentence": 15, "end_sentence": 16},
                                               {"start_sentence": 12, "end_sentence": 13}],
                                  "title": "Cold open"}]), encoding="utf-8")
    apply_picks(job, cfg, db, picks, progress=quiet)
    c = job.read_json("clips.json")["clips"][0]
    assert len(c["segments"]) == 2 and c["segments"][0]["start"] > c["segments"][1]["start"]
    want = sum(g["end"] - g["start"] for g in c["segments"])
    assert c["duration"] == pytest.approx(want, abs=0.01)
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(job.dir / c["file"])],
        capture_output=True, text=True, check=True).stdout)
    assert abs(float(probe["format"]["duration"]) - want) < 0.25
    assert {s["codec_type"] for s in probe["streams"]} == {"video", "audio"}
    doc = json.loads((job.dir / c["files"]["captions_json"]).read_text(encoding="utf-8"))
    assert doc["words"][0]["word"].startswith("So") and doc["words"][0]["start"] < 0.5


def test_stack_layout_renders_camera_over_gameplay(tmp_path, sample_video):
    cfg = load_config(cli_overrides={**OVR, "render": {"reframe": "center", "layout": "stack"},
                                     "reframe": {"facecam_region": "0.70,0.60,0.30,0.40"}}, home=tmp_path / "home")
    db = DB(cfg.home_path / "clipper.db")
    spec = JobSpec(video=str(sample_video), max_clips=1, min_duration=10, max_duration=40, rights_confirmed=True)
    job = create_job(sample_video, spec, cfg, db)
    run_job(job, cfg, db, progress=quiet)
    c = job.read_json("clips.json")["clips"][0]
    assert c["technical"]["framing"] == {"layout": "stack"} and c["technical"]["crop_ok"] is True
    g = job.read_json("facecam.json")
    assert g["top_h"] + g["bot_h"] == 1920 and g["cam"][0] + g["cam"][2] <= 1280
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", str(job.dir / c["file"])],
        capture_output=True, text=True, check=True).stdout)
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    assert (v["width"], v["height"]) == (1080, 1920)
    assert "camera on top" in job.path("report.html").read_text(encoding="utf-8")


def test_signals_stage_and_direction(tmp_path, sample_video):
    cfg, db, job = setup(tmp_path, sample_video, direction="only the hiring story")
    run_job(job, cfg, db, stop_at="judge", progress=quiet)
    sig = job.read_json("signals.json")
    assert sig["seconds"] > 60 and len(sig["combined"]) == sig["seconds"] and not sig["has_chat"]
    assert job.state["stats"]["selection_mode"] == "transcript"  # podcast content: peaks are not used
    accepted = [r for r in job.read_json("judged.json")["records"] if r["status"] == "accepted"]
    assert accepted and all("speaker_energy" in r["scores"] for r in accepted)
