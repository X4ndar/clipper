"""HTTP API and static page for the web interface (`clipper serve`).

Local-first: it listens on 127.0.0.1 and has no login. Requests whose Host or Origin is not this
machine are refused, so a web page you visit cannot drive it from your browser.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from ..config import SECRET_KEYS, Config, settings_path
from ..fetch import VIDEO_EXTS, is_url, platform_of
from ..jobs import STAGES, Job, find_job, jobs_root, slugify
from .manager import Manager

STATIC = Path(__file__).resolve().parent / "static"
LOCAL_HOSTS = ("127.0.0.1", "localhost", "[::1]", "::1", "testserver")
CHAT_EXTS = (".json", ".txt", ".log")
STATUS_LABEL = {
    "selected": "Selected", "over_quota": "Good, over quota", "duplicate": "Duplicate",
    "below_threshold": "Below score bar", "out_of_range": "Wrong length", "rejected": "Rejected by judge",
    "error": "Error", "skipped": "Skipped", "accepted": "Accepted", "pending": "Not judged",
}
# Settings the page may change (written to settings.json, the layer between config.yaml and the CLI).
EDITABLE = {
    "llm": ("backend", "candidate_model", "judge_model"),
    "selection": ("max_clips", "min_duration", "max_duration", "mode"),
    "quality": ("block_on_fail",),
    "render": ("layout", "reframe"),
    "reframe": ("follow_speaker",),
    "captions": ("enabled", "mode", "size", "uppercase", "primary_color", "highlight_color", "outline_color"),
    "hook": ("enabled",),
    "fetch": ("max_height", "chat"),
}


def _host_only(value: str) -> str:
    v = value.split("://", 1)[-1].split("/", 1)[0]
    if v.startswith("["):
        return v.split("]")[0] + "]"
    return v.rsplit(":", 1)[0] if ":" in v else v


def downsample(xs: Optional[list], n: int = 600) -> Optional[list]:
    """Max per bucket, clipped to 0..6 sigma: enough to draw the strip, small enough to send."""
    if not xs:
        return None
    step = max(1, -(-len(xs) // n))
    return [round(max(0.0, min(6.0, max(xs[i:i + step]))), 2) for i in range(0, len(xs), step)]


def job_summary(job: Job, mgr: Manager) -> dict:
    st, spec = job.state.get("stats", {}), job.state.get("spec", {})
    stages = job.state.get("stages", {})
    task = mgr.for_job(job.id)
    failed = next((s for s in STAGES if stages.get(s, {}).get("status") == "failed"), None)
    # "report" is the last stage; older jobs may lack stages added since (they still count as finished)
    done = not failed and stages.get(STAGES[-1], {}).get("status") == "done"
    status = "running" if task and task.active else ("failed" if failed else ("done" if done else "stopped"))
    clips = []
    if job.path("clips.json").is_file():
        try:
            clips = job.read_json("clips.json")["clips"]
        except (ValueError, KeyError):
            clips = []
    src = Path(spec.get("video", ""))
    title = job.state.get("title") or src.stem
    if src.stem == "source" and src.parent.joinpath("info.json").is_file():  # downloaded: use the stream's title
        try:
            title = json.loads(src.parent.joinpath("info.json").read_text(encoding="utf-8")).get("title") or title
        except ValueError:
            pass
    return {
        "id": job.id, "title": title, "status": status, "failed_stage": failed,
        "content_type": spec.get("content_type"), "duration": st.get("source_duration_s"),
        "created_at": job.state.get("created_at"), "clips": len(clips) if done or clips else None,
        "posters": [c["rank"] for c in clips[:4] if c.get("file")] if done else [],
        "cost": st.get("total_cost_usd"), "source_url": job.state.get("source_url"),
        "has_chat": bool(spec.get("chat")), "task": task.key if task and task.active else None,
    }


def job_detail(job: Job, mgr: Manager) -> dict:
    out = job_summary(job, mgr)
    stages = job.state.get("stages", {})
    out["stages"] = [{"name": s, **{k: stages.get(s, {}).get(k) for k in ("status", "seconds", "error")}} for s in STAGES]
    out["spec"] = {k: v for k, v in job.state.get("spec", {}).items() if k != "rights_confirmed"}
    out["stats"] = job.state.get("stats", {})
    out["warnings"] = job.state.get("warnings", [])
    out["selection_source"] = job.state.get("selection_source")
    out["clips"], out["candidates"], out["signals"] = [], [], None
    if job.path("clips.json").is_file():
        blob = job.read_json("clips.json")
        out["prompts"], out["models"] = blob.get("prompts"), blob.get("models")
        for c in blob["clips"]:
            c = dict(c)
            c.pop("files", None)
            c["has_video"] = bool(c.get("file")) and (job.dir / str(c.get("file"))).is_file()
            out["clips"].append(c)
    for name in ("selection.json", "judged.json", "candidates.json"):
        if job.path(name).is_file():
            for r in job.read_json(name)["records"]:
                why = r.get("status_reason") or ""
                if r.get("judge") and r["status"] != "selected" and r["judge"].get("rationale"):
                    why = f"{why} — {r['judge']['rationale']}" if why else r["judge"]["rationale"]
                out["candidates"].append({
                    "id": r["id"], "status": r["status"], "label": STATUS_LABEL.get(r["status"], r["status"]),
                    "start": r.get("start"), "end": r.get("end"), "duration": r.get("duration"),
                    "editorial": r.get("editorial"), "scores": r.get("scores") or {}, "why": why,
                    "rank": r.get("rank"), "source": (r.get("pass1") or {}).get("source", "llm"),
                    "pass1": (r.get("pass1") or {}).get("reason"), "opening": (r.get("text") or "")[:200],
                })
            break
    if job.path("signals.json").is_file():
        sig = job.read_json("signals.json")
        out["signals"] = {"seconds": sig.get("seconds"), "has_chat": sig.get("has_chat"),
                          "chat_messages": sig.get("chat_messages"), "audio": downsample(sig.get("audio_z")),
                          "chat": downsample(sig.get("chat_z")), "peaks": sig.get("peaks", [])}
    task = mgr.for_job(job.id)
    out["events"] = task.events[-40:] if task else []
    return out


def create_app(config_path: Optional[str] = None, overrides: Optional[dict] = None, allow_remote: bool = False,
               manager: Optional[Manager] = None) -> FastAPI:
    mgr = manager or Manager(config_path, overrides)
    app = FastAPI(title="clipper", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.manager = mgr

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if not allow_remote:
            host = _host_only(request.headers.get("host", ""))
            origin = request.headers.get("origin")
            if host not in LOCAL_HOSTS or (origin and _host_only(origin) not in LOCAL_HOSTS):
                return JSONResponse({"detail": "This page only answers requests from this computer."}, status_code=403)
        resp = await call_next(request)
        if request.url.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    def cfg() -> Config:
        return mgr.cfg()

    def job_or_404(job_id: str) -> Job:
        root = jobs_root(cfg().home_path)
        p = root / job_id
        if job_id != Path(job_id).name or not (p / "job.json").is_file():
            raise HTTPException(404, "No such job.")
        return Job(p)

    # ---------------- page ----------------

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/static/{path:path}")
    def static(path: str):
        f = (STATIC / path).resolve()
        if STATIC not in f.parents or not f.is_file():
            raise HTTPException(404)
        return FileResponse(f, headers={"Cache-Control": "no-cache"})

    # ---------------- meta, settings, usage ----------------

    @app.get("/api/meta")
    def meta():
        c = cfg()
        try:
            import yt_dlp  # noqa: F401
            fetch_ok = True
        except ImportError:
            fetch_ok = False
        db = mgr.db(c)
        try:
            secs, usd = db.asr_totals("assemblyai")
            llm = db.conn.execute("SELECT COALESCE(SUM(cost_usd),0), COUNT(*) FROM llm_runs WHERE cached=0").fetchone()
        finally:
            db.close()
        return {
            "content_types": ["stream", "challenge", "podcast", "interview", "talking_head", "tutorial", "other"],
            "stages": STAGES,
            "defaults": {"max_clips": c.selection.max_clips, "min_duration": c.selection.min_duration,
                         "max_duration": c.selection.max_duration, "llm_backend": c.llm.backend,
                         "hard_cap_duration": c.selection.hard_cap_duration},
            "keys": {"assemblyai": bool(os.environ.get("ASSEMBLYAI_API_KEY")),
                     "anthropic": bool(os.environ.get("ANTHROPIC_API_KEY"))},
            "claude_cli": bool(shutil.which(c.llm.claude_code_path or "claude")),
            "ffmpeg": bool(shutil.which("ffmpeg")), "fetch": fetch_ok,
            "home": str(c.home_path),
            "usage": {"asr_hours": round(secs / 3600, 2), "asr_cost_usd": round(usd, 3),
                      "asr_credit_left_usd": round(c.transcribe.credit_usd - usd, 2),
                      "llm_cost_usd": round(float(llm[0]), 3), "llm_calls": int(llm[1])},
        }

    @app.get("/api/settings")
    def get_settings():
        data = cfg().model_dump()
        return {sec: {k: data[sec][k] for k in keys} for sec, keys in EDITABLE.items()}

    @app.put("/api/settings")
    async def put_settings(request: Request):
        body = await request.json()
        c = cfg()
        sp = settings_path(c.home_path)
        current: dict[str, Any] = json.loads(sp.read_text(encoding="utf-8")) if sp.is_file() else {}
        for sec, keys in EDITABLE.items():
            for k in keys:
                if k in (body.get(sec) or {}):
                    current.setdefault(sec, {})[k] = body[sec][k]
        try:
            Config.model_validate({**current, "home": c.home})
        except Exception as e:
            raise HTTPException(422, f"That setting is not valid: {str(e).splitlines()[1].strip() if chr(10) in str(e) else e}")
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(current, indent=2), encoding="utf-8")
        return get_settings()

    # ---------------- files on this computer ----------------

    @app.get("/api/fs")
    def browse(path: str = "", kind: str = "video"):
        exts = VIDEO_EXTS if kind == "video" else CHAT_EXTS
        if not path:
            home = Path.home()
            roots = [home / n for n in ("Videos", "Downloads", "Desktop", "Documents") if (home / n).is_dir()]
            roots.append(cfg().home_path / "downloads")
            if os.name == "nt":
                roots += [Path(f"{d}:\\") for d in "CDEFGHIJ" if Path(f"{d}:\\").exists()]
            else:
                roots.append(Path("/"))
            return {"path": "", "parent": None, "dirs": [{"name": str(r), "path": str(r)} for r in roots if r.exists()],
                    "files": []}
        p = Path(path).expanduser()
        if not p.is_dir():
            raise HTTPException(404, "That folder does not exist.")
        dirs, files = [], []
        try:
            for ch in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                try:
                    if ch.name.startswith((".", "$")):
                        continue
                    if ch.is_dir():
                        dirs.append({"name": ch.name, "path": str(ch)})
                    elif ch.suffix.lower() in exts:
                        files.append({"name": ch.name, "path": str(ch), "size": ch.stat().st_size})
                except OSError:
                    continue
        except OSError:
            raise HTTPException(403, "This folder cannot be read.")
        return {"path": str(p), "parent": str(p.parent) if p.parent != p else "", "dirs": dirs, "files": files}

    @app.put("/api/upload")
    async def upload(request: Request, name: str):
        """Save a dropped file under $CLIPPER_HOME/uploads (the body is the raw file)."""
        safe = Path(name).name
        ext = Path(safe).suffix.lower()
        if ext not in VIDEO_EXTS + CHAT_EXTS:
            raise HTTPException(415, "Drop a video file (mp4, mkv, mov, webm) or a chat file (json, txt).")
        folder = cfg().home_path / "uploads"
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / f"{slugify(Path(safe).stem, 60)}-{int(time.time())}{ext}"
        size = 0
        try:
            with dest.open("wb") as f:
                async for chunk in request.stream():
                    f.write(chunk)
                    size += len(chunk)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
        if size == 0:
            dest.unlink(missing_ok=True)
            raise HTTPException(400, "The file arrived empty.")
        return {"path": str(dest), "name": safe, "size": size}

    # ---------------- jobs ----------------

    @app.get("/api/jobs")
    def list_jobs():
        root = jobs_root(cfg().home_path)
        jobs = []
        if root.is_dir():
            for d in root.iterdir():
                if (d / "job.json").is_file():
                    try:
                        jobs.append(job_summary(Job(d), mgr))
                    except Exception:
                        continue
        jobs.sort(key=lambda j: -(j.get("created_at") or 0))
        pending = [t.public() for t in sorted(mgr.pending(), key=lambda t: -t.created_at)]
        return {"jobs": jobs, "pending": pending}

    @app.post("/api/jobs")
    async def new_job(request: Request):
        p = await request.json()
        source = str(p.get("source") or "").strip().strip('"')
        if not source:
            raise HTTPException(422, "Paste a link or choose a video file first.")
        if not p.get("rights"):
            raise HTTPException(422, "Confirm that this video is yours, or that you have the rights to clip it.")
        if is_url(source):
            title = f"{platform_of(source).title()} stream"
        else:
            f = Path(source).expanduser()
            if not f.is_file():
                raise HTTPException(422, f"No file at {source}")
            title = f.stem
        chat = str(p.get("chat") or "").strip().strip('"')
        if chat and not Path(chat).expanduser().is_file():
            raise HTTPException(422, f"No chat file at {chat}")
        p["source"], p["chat"] = source, (str(Path(chat).expanduser().resolve()) if chat else None)
        t = mgr.submit("new", p, title)
        return {"task": t.key}

    @app.get("/api/tasks/{key}")
    def task(key: str):
        t = mgr.get(key)
        if not t:
            raise HTTPException(404, "No such task.")
        return {**t.public(), "events": t.events[-40:]}

    @app.post("/api/tasks/{key}/cancel")
    def cancel(key: str):
        t = mgr.get(key)
        if not t:
            raise HTTPException(404, "No such task.")
        t.cancel.set()
        return {"ok": True}

    @app.get("/api/tasks/{key}/events")
    async def events(key: str, request: Request, after: int = -1):
        t = mgr.get(key)
        if not t:
            raise HTTPException(404, "No such task.")

        async def gen():
            sent = after
            idle = 0.0
            while True:
                fresh = [e for e in t.events if e["seq"] > sent]
                for ev in fresh:
                    sent = ev["seq"]
                    yield f"data: {json.dumps({**ev, 'task': t.public()})}\n\n"
                if not t.active and not fresh:
                    yield "event: end\ndata: {}\n\n"
                    return
                if await request.is_disconnected():
                    return
                await asyncio.sleep(0.4)
                idle = 0.0 if fresh else idle + 0.4
                if idle >= 15:
                    idle = 0.0
                    yield ": keep-alive\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        return job_detail(job_or_404(job_id), mgr)

    @app.post("/api/jobs/{job_id}/resume")
    async def resume(job_id: str, request: Request):
        job = job_or_404(job_id)
        body = await request.json() if await request.body() else {}
        active = mgr.for_job(job.id)
        if active and active.active:
            raise HTTPException(409, "This job is already running.")
        stage = body.get("from_stage")
        if stage and stage not in STAGES:
            raise HTTPException(422, "Unknown stage.")
        params = {k: body[k] for k in ("llm_backend", "ignore_quality", "layout") if k in body}
        params["from_stage"] = stage
        t = mgr.submit("resume", params, job_summary(job, mgr)["title"], job_id=job.id)
        return {"task": t.key}

    @app.delete("/api/jobs/{job_id}")
    def delete_job(job_id: str):
        job = job_or_404(job_id)
        active = mgr.for_job(job.id)
        if active and active.active:
            raise HTTPException(409, "Stop the job before deleting it.")
        try:
            shutil.rmtree(job.dir)
        except OSError as e:
            raise HTTPException(409, f"Could not delete the job folder (is a clip open in a player?): {e}")
        c = cfg()
        db = mgr.db(c)
        try:
            db.conn.execute("DELETE FROM jobs WHERE id=?", (job.id,))
            db.conn.commit()
        finally:
            db.close()
        return {"ok": True}

    @app.post("/api/jobs/{job_id}/open")
    def open_folder(job_id: str):
        """Show the job's clips folder in the file manager of this computer."""
        job = job_or_404(job_id)
        folder = job.clips_dir if job.clips_dir.is_dir() else job.dir
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # noqa: S606
            else:
                subprocess.Popen(["open" if shutil.which("open") else "xdg-open", str(folder)])
        except OSError as e:
            raise HTTPException(500, f"Could not open {folder}: {e}")
        return {"path": str(folder)}

    # ---------------- media ----------------

    def clip_file(job: Job, rank: int) -> Path:
        f = job.clips_dir / f"clip_{rank:02d}.mp4"
        if not f.is_file():
            raise HTTPException(404, "This clip has no video file.")
        return f

    @app.get("/media/{job_id}/clip/{rank}")
    def clip_video(job_id: str, rank: int, download: int = 0):
        job = job_or_404(job_id)
        f = clip_file(job, rank)
        name = None
        if download and job.path("clips.json").is_file():
            c = next((c for c in job.read_json("clips.json")["clips"] if c["rank"] == rank), None)
            name = f"{rank:02d}-{slugify(c.get('title') or 'clip', 60)}.mp4" if c else f.name
        return FileResponse(f, media_type="video/mp4", filename=name)

    @app.get("/media/{job_id}/poster/{rank}")
    def poster(job_id: str, rank: int):
        job = job_or_404(job_id)
        f = clip_file(job, rank)
        out = job.clips_dir / f"clip_{rank:02d}.poster.jpg"
        if not out.is_file() or out.stat().st_mtime < f.stat().st_mtime:
            try:
                subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "1.2", "-i", str(f), "-frames:v", "1",
                                "-vf", "scale=360:-2", "-q:v", "4", str(out)], check=True, timeout=30,
                               capture_output=True)
            except (OSError, subprocess.SubprocessError):
                raise HTTPException(404, "No preview image.")
        return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    @app.get("/media/{job_id}/report")
    def report(job_id: str):
        f = job_or_404(job_id).path("report.html")
        if not f.is_file():
            raise HTTPException(404, "No report yet.")
        return FileResponse(f, media_type="text/html")

    @app.get("/media/{job_id}/clips/{name}")
    def report_asset(job_id: str, name: str):  # the report links its videos as clips/<file>
        job = job_or_404(job_id)
        f = job.clips_dir / Path(name).name
        if not f.is_file() or f.suffix.lower() not in (".mp4", ".jpg", ".png", ".srt"):
            raise HTTPException(404)
        return FileResponse(f)

    _ = SECRET_KEYS  # keys are only ever reported as present or missing, never returned
    return app
