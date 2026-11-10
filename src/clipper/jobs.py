"""Job folder layout and job.json state."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel

from .config import ContentType

STAGES = ["ingest", "transcribe", "quality", "candidates", "judge", "snap", "reframe", "captions", "render", "metadata", "report"]
STAGE_ALIASES = {
    "transcript": "quality",  # transcript + quality check (free) before any LLM spend
    "judged": "judge",
    "selection": "snap",
    "boundaries": "snap",
    "clips": "render",
}


def resolve_stage(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    n = STAGE_ALIASES.get(name, name)
    if n not in STAGES:
        raise ValueError(f"unknown stage {name!r}; choose from {', '.join(STAGES + list(STAGE_ALIASES))}")
    return n


class JobSpec(BaseModel):
    video: str
    language: Optional[str] = None
    content_type: ContentType = "podcast"
    max_clips: int = 8
    min_duration: float = 20.0
    max_duration: float = 60.0
    speakers: Optional[int] = None
    rights_confirmed: bool = False


def slugify(s: str, n: int = 40) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return (s or "video")[:n]


class Job:
    def __init__(self, dir: Path):
        self.dir = dir
        self.id = dir.name
        self.state: dict[str, Any] = {}
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    # ---- paths
    @property
    def state_path(self) -> Path:
        return self.dir / "job.json"

    def path(self, name: str) -> Path:
        return self.dir / name

    @property
    def clips_dir(self) -> Path:
        return self.dir / "clips"

    # ---- state
    @property
    def spec(self) -> JobSpec:
        return JobSpec.model_validate(self.state["spec"])

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.state_path)

    def stage_done(self, stage: str) -> bool:
        return self.state.get("stages", {}).get(stage, {}).get("status") == "done"

    def mark(self, stage: str, status: str, seconds: Optional[float] = None, error: Optional[str] = None) -> None:
        st = self.state.setdefault("stages", {})
        st[stage] = {"status": status, "at": time.time(), "seconds": seconds, "error": error}
        self.save()

    def stats(self) -> dict:
        return self.state.setdefault("stats", {})

    def warn(self, msg: str) -> None:
        w = self.state.setdefault("warnings", [])
        if msg not in w:
            w.append(msg)

    def read_json(self, name: str) -> Any:
        return json.loads(self.path(name).read_text(encoding="utf-8"))

    def write_json(self, name: str, data: Any) -> Path:
        p = self.path(name)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(p)
        return p


def jobs_root(home: Path) -> Path:
    return home / "jobs"


def find_job(home: Path, job_ref: str) -> Job:
    p = Path(job_ref)
    if (p / "job.json").is_file():
        return Job(p)
    p = jobs_root(home) / job_ref
    if (p / "job.json").is_file():
        return Job(p)
    matches = [d for d in jobs_root(home).glob(f"{job_ref}*") if (d / "job.json").is_file()]
    if len(matches) == 1:
        return Job(matches[0])
    raise FileNotFoundError(f"job {job_ref!r} not found under {jobs_root(home)}")
