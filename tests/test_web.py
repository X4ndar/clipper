"""Web API: job queue, progress events, job pages, settings, file access rules. No ffmpeg, no network."""

import json
import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from clipper.jobs import STAGES, Job, jobs_root  # noqa: E402
from clipper.web.manager import Cancelled, Manager  # noqa: E402
from clipper.web.server import create_app, downsample  # noqa: E402


def make_job(home, jid="demo-1234", done=True):
    job = Job(jobs_root(home) / jid)
    job.state = {"id": jid, "created_at": time.time(), "warnings": ["x"],
                 "spec": {"video": str(home / "demo.mp4"), "content_type": "stream", "rights_confirmed": True},
                 "stats": {"source_duration_s": 600.0, "total_cost_usd": 0.0},
                 "stages": {s: {"status": "done", "seconds": 1.0, "error": None} for s in (STAGES if done else STAGES[:2])}}
    job.save()
    job.clips_dir.mkdir(parents=True, exist_ok=True)
    (job.clips_dir / "clip_01.mp4").write_bytes(b"0123456789")
    job.write_json("clips.json", {"clips": [{"rank": 1, "file": "clips/clip_01.mp4", "start": 100.0, "end": 140.0,
                                             "duration": 40.0, "title": "A <b>title</b>", "scores": {"hook": 8},
                                             "source": "peaks", "files": {"video": "clips/clip_01.mp4"}}]})
    job.write_json("selection.json", {"records": [
        {"id": "c01", "status": "selected", "start": 100.0, "end": 140.0, "duration": 40.0, "editorial": 7.5, "scores": {},
         "rank": 1, "pass1": {"source": "peaks", "reason": "loud"}, "text": "hello", "status_reason": None, "judge": None},
        {"id": "c02", "status": "rejected", "start": 300.0, "end": 330.0, "duration": 30.0, "editorial": 4.0, "scores": {},
         "rank": None, "pass1": {"reason": "meh"}, "text": "nope", "status_reason": "gate", "judge": {"rationale": "no payoff"}}]})
    job.write_json("signals.json", {"seconds": 600, "has_chat": True, "chat_messages": 50, "audio_z": [0.5] * 599 + [9.0],
                                    "chat_z": [-1.0] * 600, "peaks": [{"t": 120.0, "score": 2.0}]})
    return job


@pytest.fixture
def web(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPER_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    ran = []

    def runner(mgr, t):
        ran.append(t)
        progress = mgr.progress_for(t)
        t.status = "running"
        progress("ingest", "start", "")
        if t.params.get("source", "").endswith("boom.mp4"):
            raise RuntimeError("ffmpeg exploded")
        while t.params.get("wait") and not t.cancel.is_set():
            time.sleep(0.01)
        progress("ingest", "done", "0.1s")
        t.job_id = "demo-1234"

    mgr = Manager(runner=runner)
    client = TestClient(create_app(manager=mgr))
    client.ran, client.mgr, client.home = ran, mgr, tmp_path
    return client


def wait(client, key, statuses=("done", "failed", "cancelled")):
    for _ in range(300):
        t = client.get(f"/api/tasks/{key}").json()
        if t["status"] in statuses:
            return t
        time.sleep(0.01)
    raise AssertionError("task did not finish")


def test_downsample_keeps_spikes_and_clips_the_range():
    out = downsample([0.5] * 599 + [9.0], 60)
    assert len(out) == 60 and out[-1] == 6.0 and out[0] == 0.5
    assert downsample([-2.0, -1.0]) == [0.0, 0.0] and downsample(None) is None


def test_page_and_meta_never_return_keys(web, monkeypatch):
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", "secret-value-123")
    assert "clipper" in web.get("/").text and web.get("/static/app.js").status_code == 200
    assert web.get("/static/../server.py").status_code == 404
    r = web.get("/api/meta")
    assert r.json()["keys"]["assemblyai"] is True and "secret-value-123" not in r.text
    assert "stream" in r.json()["content_types"] and r.headers["cache-control"] == "no-store"


def test_requests_from_other_sites_are_refused(web):
    assert web.get("/api/meta", headers={"host": "evil.example"}).status_code == 403
    assert web.get("/api/meta", headers={"origin": "https://evil.example"}).status_code == 403
    assert web.get("/api/meta", headers={"origin": "http://localhost:8765"}).status_code == 200


def test_new_job_needs_a_source_and_the_rights_box(web):
    video = web.home / "demo.mp4"
    video.write_bytes(b"x")
    assert web.post("/api/jobs", json={"source": "", "rights": True}).status_code == 422
    assert "rights" in web.post("/api/jobs", json={"source": str(video), "rights": False}).json()["detail"]
    assert web.post("/api/jobs", json={"source": str(web.home / "nope.mp4"), "rights": True}).status_code == 422
    assert web.post("/api/jobs", json={"source": str(video), "rights": True, "chat": "missing.json"}).status_code == 422
    assert web.ran == []


def test_job_runs_and_its_events_stream(web):
    make_job(web.home)
    video = web.home / "demo.mp4"
    video.write_bytes(b"x")
    key = web.post("/api/jobs", json={"source": str(video), "rights": True, "content_type": "stream"}).json()["task"]
    t = wait(web, key)
    assert t["status"] == "done" and t["job_id"] == "demo-1234"
    body = web.get(f"/api/tasks/{key}/events").text
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: {\"")]
    assert [(e["stage"], e["status"]) for e in events] == [("ingest", "start"), ("ingest", "done"), ("job", "done")]
    assert body.rstrip().endswith("event: end\ndata: {}")
    assert web.get(f"/api/tasks/{key}/events?after=1").text.count("data: {\"") == 1


def test_a_link_is_queued_as_a_stream_download(web):
    key = web.post("/api/jobs", json={"source": "https://www.twitch.tv/videos/123", "rights": True}).json()["task"]
    assert web.ran[0].title == "Twitch stream" or wait(web, key)
    assert web.get(f"/api/tasks/{key}").json()["source"] == "https://www.twitch.tv/videos/123"


def test_failure_and_stop_are_reported(web):
    boom = web.home / "boom.mp4"
    boom.write_bytes(b"x")
    t = wait(web, web.post("/api/jobs", json={"source": str(boom), "rights": True}).json()["task"])
    assert t["status"] == "failed" and t["error"] == "ffmpeg exploded"
    assert web.get("/api/jobs").json()["pending"][0]["status"] == "failed"
    ok = web.home / "ok.mp4"
    ok.write_bytes(b"x")
    key = web.post("/api/jobs", json={"source": str(ok), "rights": True, "wait": True}).json()["task"]
    wait(web, key, ("running",))
    assert web.post(f"/api/tasks/{key}/cancel").json() == {"ok": True}
    web.mgr.get(key).params["wait"] = False
    # the runner's next progress call raises Cancelled
    assert wait(web, key)["status"] == "cancelled"
    with pytest.raises(Cancelled):
        web.mgr.progress_for(web.mgr.get(key))("x", "info", "")


def test_job_list_and_detail(web):
    make_job(web.home)
    make_job(web.home, "half-0000", done=False)
    jobs = {j["id"]: j for j in web.get("/api/jobs").json()["jobs"]}
    assert jobs["demo-1234"]["status"] == "done" and jobs["demo-1234"]["clips"] == 1 and jobs["demo-1234"]["posters"] == [1]
    assert jobs["half-0000"]["status"] == "stopped"
    d = web.get("/api/jobs/demo-1234").json()
    assert d["clips"][0]["has_video"] and "files" not in d["clips"][0] and "rights_confirmed" not in d["spec"]
    assert [c["label"] for c in d["candidates"]] == ["Selected", "Rejected by judge"]
    assert d["candidates"][1]["why"] == "gate — no payoff" and d["candidates"][0]["source"] == "peaks"
    assert d["signals"]["audio"][-1] == 6.0 and set(d["signals"]["chat"]) == {0.0} and len(d["stages"]) == len(STAGES)
    assert web.get("/api/jobs/nope").status_code == 404
    assert web.get("/api/jobs/..%2F..%2Fsecrets").status_code == 404


def test_clip_files_support_seeking_and_download_names(web):
    make_job(web.home)
    r = web.get("/media/demo-1234/clip/1", headers={"range": "bytes=2-5"})
    assert r.status_code == 206 and r.content == b"2345"
    r = web.get("/media/demo-1234/clip/1?download=1")
    assert "01-a-b-title-b.mp4" in r.headers["content-disposition"]
    assert web.get("/media/demo-1234/clip/9").status_code == 404
    assert web.get("/media/demo-1234/clips/clip_01.mp4").status_code == 200
    assert web.get("/media/demo-1234/clips/..%2Fjob.json").status_code == 404


def test_resume_and_delete(web):
    make_job(web.home)
    assert web.post("/api/jobs/demo-1234/resume", json={"from_stage": "nonsense"}).status_code == 422
    key = web.post("/api/jobs/demo-1234/resume", json={"from_stage": "reframe"}).json()["task"]
    wait(web, key)
    assert web.ran[-1].kind == "resume" and web.ran[-1].params["from_stage"] == "reframe"
    assert web.delete("/api/jobs/demo-1234").json() == {"ok": True}
    assert not (jobs_root(web.home) / "demo-1234").exists() and web.get("/api/jobs/demo-1234").status_code == 404


def test_settings_are_validated_and_only_listed_keys_are_written(web):
    s = web.get("/api/settings").json()
    assert s["selection"]["max_clips"] == 8 and "prices_per_mtok" not in s["llm"]
    r = web.put("/api/settings", json={"selection": {"max_clips": 5}, "llm": {"backend": "claude_code", "max_tokens": 1},
                                       "transcribe": {"credit_usd": 0}})
    assert r.json()["selection"]["max_clips"] == 5 and r.json()["llm"]["backend"] == "claude_code"
    saved = json.loads((web.home / "settings.json").read_text(encoding="utf-8"))
    assert saved == {"selection": {"max_clips": 5}, "llm": {"backend": "claude_code"}}
    assert web.put("/api/settings", json={"llm": {"backend": "gpt"}}).status_code == 422
    assert json.loads((web.home / "settings.json").read_text(encoding="utf-8")) == saved


def test_file_picker_and_upload(web):
    (web.home / "vods").mkdir()
    (web.home / "vods" / "a.mp4").write_bytes(b"12345")
    (web.home / "vods" / "notes.docx").write_bytes(b"1")
    (web.home / "vods" / "chat.json").write_text("{}", encoding="utf-8")
    d = web.get("/api/fs", params={"path": str(web.home / "vods")}).json()
    assert [f["name"] for f in d["files"]] == ["a.mp4"] and d["files"][0]["size"] == 5
    assert [f["name"] for f in web.get("/api/fs", params={"path": str(web.home / "vods"), "kind": "chat"}).json()["files"]] == ["chat.json"]
    assert web.get("/api/fs").json()["dirs"] and web.get("/api/fs", params={"path": str(web.home / "zz")}).status_code == 404
    r = web.put("/api/upload", params={"name": "..\\..\\My Stream.mp4"}, content=b"videobytes")
    saved = r.json()["path"]
    assert r.json()["size"] == 10 and str(web.home / "uploads") in saved and "my-stream" in saved
    assert web.put("/api/upload", params={"name": "run.exe"}, content=b"x").status_code == 415
    assert web.put("/api/upload", params={"name": "empty.mp4"}, content=b"").status_code == 400
