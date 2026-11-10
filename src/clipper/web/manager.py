"""Job queue for the web UI: one worker thread runs jobs in order and records progress events.

The pipeline itself is unchanged; this only calls fetch_source(), create_job() and run_job() with a
progress callback that stores events for the browser (sent over SSE by server.py).
"""

from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from ..config import Config, deep_merge, load_config
from ..db import DB
from ..jobs import JobSpec, find_job


class Cancelled(Exception):
    pass


@dataclass
class Task:
    key: str
    kind: str  # "new" | "resume"
    params: dict
    title: str
    status: str = "queued"  # queued | fetching | running | done | failed | cancelled
    job_id: Optional[str] = None
    error: Optional[str] = None
    stage: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    events: list[dict] = field(default_factory=list)
    cancel: threading.Event = field(default_factory=threading.Event)

    @property
    def active(self) -> bool:
        return self.status in ("queued", "fetching", "running")

    def public(self) -> dict:
        return {"key": self.key, "kind": self.kind, "title": self.title, "status": self.status,
                "job_id": self.job_id, "error": self.error, "stage": self.stage,
                "created_at": self.created_at, "finished_at": self.finished_at,
                "source": self.params.get("source"), "content_type": self.params.get("content_type")}


class Manager:
    """Runs tasks one at a time. `runner` is injectable so tests need no ffmpeg or network."""

    def __init__(self, config_path: Optional[str] = None, base_overrides: Optional[dict] = None,
                 runner: Optional[Callable[["Manager", Task], None]] = None):
        self.config_path = config_path
        self.base_overrides = base_overrides or {}
        self.tasks: dict[str, Task] = {}
        self.lock = threading.Lock()
        self._q: "queue.Queue[Task]" = queue.Queue()
        self._runner = runner or _run_task
        self._thread: Optional[threading.Thread] = None

    # ---- config
    def cfg(self, overrides: Optional[dict] = None) -> Config:
        over = deep_merge(self.base_overrides, overrides or {})
        return load_config(self.config_path, over)

    def db(self, cfg: Config) -> DB:
        return DB(cfg.home_path / "clipper.db")

    # ---- queue
    def submit(self, kind: str, params: dict, title: str, job_id: Optional[str] = None) -> Task:
        t = Task(key=uuid.uuid4().hex[:12], kind=kind, params=params, title=title, job_id=job_id)
        with self.lock:
            self.tasks[t.key] = t
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="clipper-worker", daemon=True)
                self._thread.start()
        self._q.put(t)
        return t

    def _loop(self) -> None:
        while True:
            t = self._q.get()
            if t.cancel.is_set():
                self._finish(t, "cancelled")
                continue
            try:
                self._runner(self, t)
                self._finish(t, "done")
            except Cancelled:
                self._finish(t, "cancelled")
            except Exception as e:  # shown in the UI; the pipeline already recorded the stage error
                if t.cancel.is_set():  # a library wrapped our Cancelled in its own error
                    self._finish(t, "cancelled")
                    continue
                t.error = str(e) or e.__class__.__name__
                self.emit(t, t.stage or "job", "error", t.error)
                self._finish(t, "failed")

    def _finish(self, t: Task, status: str) -> None:
        t.status, t.finished_at = status, time.time()
        self.emit(t, "job", status, t.error or "")

    def emit(self, t: Task, stage: str, status: str, msg: str) -> None:
        t.events.append({"seq": len(t.events), "t": time.time(), "stage": stage, "status": status, "msg": msg})
        if len(t.events) > 2000:  # keep memory bounded on very long downloads
            del t.events[100:600]
            for i, ev in enumerate(t.events):
                ev["seq"] = i

    def progress_for(self, t: Task) -> Callable[[str, str, str], None]:
        def progress(stage: str, status: str, msg: str) -> None:
            if t.cancel.is_set():
                raise Cancelled()
            if status == "start":
                t.stage = stage
            self.emit(t, stage, status, msg)
        return progress

    # ---- lookups
    def get(self, key: str) -> Optional[Task]:
        return self.tasks.get(key)

    def for_job(self, job_id: str) -> Optional[Task]:
        """The most recent task for a job (active one first)."""
        ts = [t for t in self.tasks.values() if t.job_id == job_id]
        ts.sort(key=lambda t: (t.active, t.created_at))
        return ts[-1] if ts else None

    def pending(self) -> list[Task]:
        """Tasks that have no job folder yet (queued, or still downloading)."""
        return [t for t in self.tasks.values() if t.job_id is None and t.status != "done"]


def _overrides(params: dict) -> dict:
    over: dict[str, Any] = {}
    if params.get("llm_backend"):
        over.setdefault("llm", {})["backend"] = params["llm_backend"]
    if params.get("transcribe_backend"):
        over.setdefault("transcribe", {})["backend"] = params["transcribe_backend"]
    if params.get("ignore_quality"):
        over.setdefault("quality", {})["block_on_fail"] = False
    if params.get("layout"):
        over.setdefault("render", {})["layout"] = params["layout"]
    return deep_merge(over, params.get("overrides") or {})


def _run_task(mgr: Manager, t: Task) -> None:
    from ..pipeline import create_job, run_job

    p = t.params
    cfg = mgr.cfg(_overrides(p))
    db = mgr.db(cfg)
    progress = mgr.progress_for(t)
    try:
        if t.kind == "resume":
            job = find_job(cfg.home_path, t.job_id or "")
            t.status = "running"
            run_job(job, cfg, db, from_stage=p.get("from_stage"), progress=progress)
            return
        source, chat = p["source"], p.get("chat")
        from ..fetch import fetch_source, is_url

        if is_url(source):
            t.status, t.stage = "fetching", "fetch"
            progress("fetch", "start", "")
            got = fetch_source(source, cfg, progress, want_chat=not chat)
            progress("fetch", "done", got.title or "")
            source, chat = str(got.video), chat or (str(got.chat) if got.chat else None)
            if got.title:
                t.title = got.title
        sel = cfg.selection
        spec = JobSpec(
            video=source, language=p.get("language") or None, content_type=p.get("content_type") or "podcast",
            max_clips=int(p.get("max_clips") or sel.max_clips),
            min_duration=float(p.get("min_duration") or sel.min_duration),
            max_duration=float(p.get("max_duration") or sel.max_duration),
            speakers=p.get("speakers") or None, chat=chat, direction=p.get("direction") or None,
            rights_confirmed=bool(p.get("rights")),
        )
        if spec.max_duration > sel.hard_cap_duration:
            raise ValueError(f"Clips cannot be longer than {sel.hard_cap_duration:.0f} s.")
        if spec.min_duration >= spec.max_duration:
            raise ValueError("The shortest clip length must be below the longest.")
        job = create_job(Path(source), spec, cfg, db)
        if p.get("source") != source:  # remember the link the file came from
            job.state["source_url"] = p["source"]
            job.save()
        t.job_id, t.status = job.id, "running"
        mgr.emit(t, "job", "created", job.id)
        run_job(job, cfg, db, progress=progress)
    finally:
        db.close()
