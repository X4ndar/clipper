"""SQLite job history, stage runs, LLM run log, ASR usage and eval runs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_sha256 TEXT,
    status TEXT NOT NULL,
    spec_json TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stage_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    seconds REAL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS llm_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    stage TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    model TEXT NOT NULL,
    params_json TEXT NOT NULL,
    input_sha256 TEXT NOT NULL,
    output_json TEXT,
    stop_reason TEXT,
    usage_json TEXT,
    cost_usd REAL,
    seconds REAL,
    cached INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS llm_runs_input ON llm_runs (input_sha256);
CREATE TABLE IF NOT EXISTS asr_usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    provider TEXT NOT NULL,
    model TEXT,
    audio_sha256 TEXT NOT NULL,
    audio_seconds REAL NOT NULL,
    cost_usd REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS eval_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    gold TEXT NOT NULL,
    candidate_prompt TEXT,
    judge_prompt TEXT,
    candidate_model TEXT,
    judge_model TEXT,
    metrics_json TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class DB:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.conn = sqlite3.connect(str(path), check_same_thread=False, timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # jobs
    def upsert_job(self, job_id: str, source: str, sha: Optional[str], status: str, spec: dict) -> None:
        now = time.time()
        self.conn.execute(
            """INSERT INTO jobs (id, source, source_sha256, status, spec_json, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id) DO UPDATE SET status=excluded.status, spec_json=excluded.spec_json,
               source_sha256=COALESCE(excluded.source_sha256, jobs.source_sha256), updated_at=excluded.updated_at""",
            (job_id, source, sha, status, json.dumps(spec), now, now),
        )
        self.conn.commit()

    def set_job_status(self, job_id: str, status: str) -> None:
        self.conn.execute("UPDATE jobs SET status=?, updated_at=? WHERE id=?", (status, time.time(), job_id))
        self.conn.commit()

    def list_jobs(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)))

    # stages
    def stage_start(self, job_id: str, stage: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO stage_runs (job_id, stage, status, started_at) VALUES (?, ?, 'running', ?)",
            (job_id, stage, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def stage_end(self, run_id: int, status: str, error: Optional[str] = None) -> None:
        now = time.time()
        self.conn.execute(
            "UPDATE stage_runs SET status=?, finished_at=?, seconds=?-started_at, error=? WHERE id=?",
            (status, now, now, error, run_id),
        )
        self.conn.commit()

    # llm runs
    def find_cached_llm(self, input_sha: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            """SELECT * FROM llm_runs WHERE input_sha256=? AND output_json IS NOT NULL AND error IS NULL
               ORDER BY id DESC LIMIT 1""",
            (input_sha,),
        ).fetchone()

    def log_llm(self, **kw: Any) -> int:
        cols = [
            "job_id", "stage", "prompt_version", "model", "params_json", "input_sha256", "output_json",
            "stop_reason", "usage_json", "cost_usd", "seconds", "cached", "error",
        ]
        vals = [kw.get(c) for c in cols]
        cur = self.conn.execute(
            f"INSERT INTO llm_runs ({', '.join(cols)}, created_at) VALUES ({', '.join('?' * len(cols))}, ?)",
            (*vals, time.time()),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def llm_cost_for_job(self, job_id: str) -> float:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_runs WHERE job_id=? AND cached=0", (job_id,)
        ).fetchone()
        return float(row[0])

    # asr usage
    def log_asr(self, job_id: str, provider: str, model: str, audio_sha: str, seconds: float, cost: float) -> None:
        self.conn.execute(
            """INSERT INTO asr_usage (job_id, provider, model, audio_sha256, audio_seconds, cost_usd, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (job_id, provider, model, audio_sha, seconds, cost, time.time()),
        )
        self.conn.commit()

    def asr_totals(self, provider: str) -> tuple[float, float]:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(audio_seconds),0), COALESCE(SUM(cost_usd),0) FROM asr_usage WHERE provider=?",
            (provider,),
        ).fetchone()
        return float(row[0]), float(row[1])

    # eval
    def log_eval(self, job_id: str, gold: str, meta: dict, metrics: dict) -> None:
        self.conn.execute(
            """INSERT INTO eval_runs (job_id, gold, candidate_prompt, judge_prompt, candidate_model, judge_model,
               metrics_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id, gold, meta.get("candidate_prompt"), meta.get("judge_prompt"),
                meta.get("candidate_model"), meta.get("judge_model"), json.dumps(metrics), time.time(),
            ),
        )
        self.conn.commit()

    def list_evals(self) -> list[sqlite3.Row]:
        return list(self.conn.execute("SELECT * FROM eval_runs ORDER BY id"))
