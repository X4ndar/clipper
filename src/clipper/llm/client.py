"""LLM access: Anthropic backend, offline fake backend, response cache, run log and cost guard."""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Type

from pydantic import BaseModel, ValidationError

from ..db import DB

PROMPTS_DIR = Path(__file__).parent / "prompts"


class LLMError(RuntimeError):
    pass


class LLMRefusal(LLMError):
    pass


class BudgetExceeded(LLMError):
    pass


def load_prompt(version: str) -> str:
    p = PROMPTS_DIR / f"{version}.md"
    if not p.is_file():
        raise LLMError(f"prompt {version!r} not found in {PROMPTS_DIR}")
    return p.read_text(encoding="utf-8").strip()


@dataclass
class LLMCall:
    task: str  # "candidates" | "judge"
    model: str
    prompt_version: str
    system: str
    user: str
    schema: Type[BaseModel]
    effort: str = "low"
    max_tokens: int = 16000
    fallbacks: Optional[str] = None
    payload: dict = field(default_factory=dict)  # structured inputs, used by the fake backend

    def params(self) -> dict:
        return {
            "effort": self.effort,
            "max_tokens": self.max_tokens,
            "fallbacks": self.fallbacks,
            "schema": self.schema.__name__,
        }

    def input_hash(self) -> str:
        blob = json.dumps(
            {
                "model": self.model,
                "system": self.system,
                "user": self.user,
                "schema": self.schema.model_json_schema(),
                **self.params(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()


@dataclass
class LLMResponse:
    parsed: BaseModel
    usage: dict
    stop_reason: Optional[str]
    cost_usd: float
    cached: bool
    seconds: float
    model: str


class AnthropicBackend:
    name = "anthropic"

    def __init__(self, client: Any = None):
        self._client = client

    def client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, call: LLMCall) -> tuple[Optional[BaseModel], dict, Optional[str], str]:
        kwargs: dict = dict(
            model=call.model,
            max_tokens=call.max_tokens,
            system=call.system,
            messages=[{"role": "user", "content": call.user}],
            output_format=call.schema,
            output_config={"effort": call.effort},
        )
        c = self.client()
        if call.fallbacks:
            # Server-side refusal fallback (beta). Routes declined requests to a substitute model.
            resp = c.beta.messages.parse(
                **kwargs, fallbacks=call.fallbacks, betas=["server-side-fallback-2026-07-01"]
            )
        else:
            resp = c.messages.parse(**kwargs)
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) is not None else {}
        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            raise LLMRefusal(f"model declined the request: {details}")
        return resp.parsed_output, usage, resp.stop_reason, getattr(resp, "model", call.model)


class ClaudeCodeBackend:
    """Runs each call through the Claude Code CLI (`claude -p`), which uses the Claude subscription
    you are logged in with instead of an API key. No tools, no project settings, no saved session.

    Calls count against the subscription's usage limits, not dollars: cost is logged as $0 and the
    CLI's API-equivalent estimate is kept in the usage record.
    """

    name = "claude_code"
    billed = False

    def __init__(self, cfg=None, runner=None):
        self.timeout_s = getattr(cfg, "claude_code_timeout_s", 600)
        self.executable = getattr(cfg, "claude_code_path", None) or "claude"
        self._run = runner or self._subprocess

    def _subprocess(self, cmd: list[str], stdin: str, cwd: str, env: dict) -> tuple[int, str, str]:
        import subprocess

        try:
            p = subprocess.run(cmd, input=stdin, capture_output=True, text=True, encoding="utf-8",
                               errors="replace", cwd=cwd, env=env, timeout=self.timeout_s)
        except subprocess.TimeoutExpired as e:
            raise LLMError(f"claude CLI timed out after {self.timeout_s}s") from e
        return p.returncode, p.stdout, p.stderr

    def complete(self, call: LLMCall) -> tuple[Optional[BaseModel], dict, Optional[str], str]:
        import os
        import shutil
        import tempfile

        exe = shutil.which(self.executable)
        if not exe:
            raise LLMError("the `claude` command was not found. Install Claude Code and log in with your "
                           "subscription (run `claude`, then /login), or set llm.backend=anthropic.")
        schema = json.dumps(call.schema.model_json_schema(), separators=(",", ":"))
        if any(ch in schema for ch in "&|<>^%!"):
            # On Windows the CLI is a .cmd shim; these characters do not survive cmd.exe quoting.
            raise LLMError("schema contains characters that cannot be passed to the claude CLI")
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        with tempfile.TemporaryDirectory(prefix="clipper-llm-") as td:
            sys_file = Path(td) / "system.md"
            sys_file.write_text(call.system, encoding="utf-8")
            cmd = [exe, "-p", "--model", call.model, "--effort", call.effort, "--output-format", "json",
                   "--system-prompt-file", str(sys_file), "--json-schema", schema, "--tools", "",
                   "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence"]
            code, out, err = self._run(cmd, call.user, td, env)
        try:
            data = json.loads(out)
        except json.JSONDecodeError:
            tail = (err or out).strip()[-400:]
            raise LLMError(f"claude CLI failed (exit {code}): {tail or 'no output'}. If you are not logged in, "
                           "run `claude` and use /login.") from None
        if data.get("is_error") or code != 0:
            raise LLMError(f"claude CLI error: {str(data.get('result') or data.get('subtype'))[:400]}")
        usage = dict(data.get("usage") or {})
        usage = {k: v for k, v in usage.items() if isinstance(v, (int, float, str))}
        usage["api_equivalent_cost_usd"] = data.get("total_cost_usd")
        # modelUsage also lists a small helper model the CLI uses; report the one that wrote the answer.
        mu = data.get("modelUsage") or {}
        model_used = call.model if call.model in mu or not mu else max(mu, key=lambda k: mu[k].get("outputTokens", 0))
        out_obj = data.get("structured_output")
        if out_obj is None:  # older CLI: JSON in the text result
            out_obj = json.loads(data.get("result") or "null")
        parsed = call.schema.model_validate(out_obj) if out_obj is not None else None
        return parsed, usage, "end_turn", model_used


class FakeBackend:
    """Deterministic, offline stand-in for tests and dry runs. Not a quality signal."""

    name = "fake"
    HOOK_WORDS = ("why", "how", "secret", "mistake", "never", "most people", "truth", "actually", "?")
    PAYOFF_WORDS = ("because", "so the", "that's why", "the answer", "turns out", "which means", "result")
    BAD_WORDS = ("welcome", "subscribe", "sponsor", "thanks for", "like and", "today's episode", "see you")

    def complete(self, call: LLMCall) -> tuple[Optional[BaseModel], dict, Optional[str], str]:
        if call.task == "candidates":
            out = self._candidates(call.payload)
        elif call.task == "judge":
            out = self._judge(call.payload)
        else:
            raise LLMError(f"fake backend: unknown task {call.task}")
        # No real tokens are spent, so usage stays zero and cost is $0.
        return call.schema.model_validate(out), {"input_tokens": 0, "output_tokens": 0}, "end_turn", "fake"

    def _candidates(self, p: dict) -> dict:
        """Windows of about the target length, preferring ones that open on a hook-like sentence."""
        sents = p["sentences"]
        lo, hi = p["min_duration"], p["max_duration"]
        target = (lo + hi) / 2
        windows = []
        for i in range(len(sents)):
            j = i
            while j + 1 < len(sents) and sents[j]["end"] - sents[i]["start"] < target:
                j += 1
            dur = sents[j]["end"] - sents[i]["start"]
            if not lo <= dur <= hi * 1.5:
                continue
            first = sents[i]["text"].lower()
            text = " ".join(s["text"] for s in sents[i : j + 1]).lower()
            score = sum(w in first for w in self.HOOK_WORDS) - 2 * sum(w in text for w in self.BAD_WORDS)
            windows.append((score, i, j))
        windows.sort(key=lambda w: (-w[0], w[1]))
        out, used = [], set()
        for _, i, j in windows:
            if len(out) >= p["count"]:
                break
            if used & set(range(i, j + 1)):
                continue
            used |= set(range(i, j + 1))
            out.append({
                "start_sentence_id": sents[i]["id"],
                "end_sentence_id": sents[j]["id"],
                "hook_line": sents[i]["text"][:80],
                "reason": "fake backend: keyword window",
                "content_type": "other",
            })
        return {"candidates": out}

    def _judge(self, p: dict) -> dict:
        clip = p["clip_sentences"]
        text = " ".join(s["text"] for s in clip).lower()
        first = clip[0]["text"].lower()
        hook = 5 + 2 * sum(w in first for w in self.HOOK_WORDS)
        payoff = 5 + 2 * sum(w in text for w in self.PAYOFF_WORDS)
        bad = any(w in text for w in self.BAD_WORDS)
        standalone = 3 if bad else 7
        if re.search(r"\b(he|she|they|it|that)\b", first.split(" ")[0] if first else ""):
            standalone -= 2
        scores = {
            "hook": min(10, hook),
            "standalone_sense": standalone,
            "payoff": min(10, payoff),
            "peak": 6,
            "quotability": 6 + (1 if re.search(r"\d", text) else 0),
            "clean_ending": 7 if clip[-1]["text"].rstrip().endswith((".", "!", "?")) else 4,
        }
        reject = scores["standalone_sense"] < 6 or scores["payoff"] < 6
        words = clip[0]["text"].split()
        return {
            "start_sentence_id": clip[0]["id"],
            "end_sentence_id": clip[-1]["id"],
            "start_word_offset": 0,
            "end_word_offset": 0,
            "scores": scores,
            "reject": reject,
            "reject_reason": "fake backend: missing standalone sense or payoff" if reject else None,
            "rationale": "fake backend: keyword heuristics",
            "confidence": 0.8,
            "hook_text": " ".join(words[:6]) if words else None,
            "title": clip[0]["text"][:90],
            "description": clip[0]["text"][:200],
            "tags": ["shorts", "podcast", "clip"],
            "content_type": "other",
        }


def cost_usd(model: str, usage: dict, prices: dict[str, list[float]]) -> float:
    price = prices.get(model)
    if not price:
        return 0.0
    inp = (usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0) * 1.25
    inp += (usage.get("cache_read_input_tokens") or 0) * 0.1
    return (inp * price[0] + (usage.get("output_tokens") or 0) * price[1]) / 1_000_000


class LLMService:
    """Retries, response cache, SQLite run log and per-job budget, around one backend."""

    def __init__(self, backend, db: DB, cfg, job_id: Optional[str] = None, budget_usd: Optional[float] = None):
        self.backend = backend
        self.db = db
        self.cfg = cfg  # LLMConfig
        self.job_id = job_id
        self.budget_usd = budget_usd if budget_usd is not None else cfg.min_budget_usd
        self._lock = threading.Lock()

    def call(self, call: LLMCall, stage: str) -> LLMResponse:
        h = call.input_hash()
        backend_name = getattr(self.backend, "name", "anthropic")
        if backend_name != "anthropic":
            # Keep offline (fake/test) answers out of the real backend's response cache.
            h = hashlib.sha256(f"{backend_name}:{h}".encode()).hexdigest()
        if self.cfg.cache_responses:
            with self._lock:
                row = self.db.find_cached_llm(h)
            if row is not None:
                try:
                    parsed = call.schema.model_validate_json(row["output_json"])
                    usage = json.loads(row["usage_json"] or "{}")
                    with self._lock:
                        self.db.log_llm(
                            job_id=self.job_id, stage=stage, prompt_version=call.prompt_version,
                            model=call.model, params_json=json.dumps(call.params()), input_sha256=h,
                            output_json=row["output_json"], stop_reason=row["stop_reason"],
                            usage_json=row["usage_json"], cost_usd=0.0, seconds=0.0, cached=1,
                        )
                    return LLMResponse(parsed, usage, row["stop_reason"], 0.0, True, 0.0, call.model)
                except ValidationError:
                    pass  # schema changed since; call again

        with self._lock:
            spent = self.db.llm_cost_for_job(self.job_id) if self.job_id else 0.0
        if spent >= self.budget_usd:
            raise BudgetExceeded(
                f"LLM spend for this job reached ${spent:.3f} (budget ${self.budget_usd:.2f}; raise "
                "llm.max_cost_usd_per_hour or llm.min_budget_usd)."
            )

        last_err: Optional[Exception] = None
        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.monotonic()
            try:
                parsed, usage, stop, model_used = self.backend.complete(call)
                secs = time.monotonic() - t0
                # A refusal fallback bills at the serving model's rates.
                priced = model_used if model_used in self.cfg.prices_per_mtok else call.model
                cost = cost_usd(priced, usage, self.cfg.prices_per_mtok) if getattr(self.backend, "billed", True) else 0.0
                if parsed is None:
                    raise LLMError(f"no parsable output (stop_reason={stop})")
                with self._lock:
                    self.db.log_llm(
                        job_id=self.job_id, stage=stage, prompt_version=call.prompt_version, model=call.model,
                        params_json=json.dumps({**call.params(), "served_by": model_used}), input_sha256=h,
                        output_json=parsed.model_dump_json(), stop_reason=stop, usage_json=json.dumps(usage),
                        cost_usd=cost, seconds=secs, cached=0,
                    )
                return LLMResponse(parsed, usage, stop, cost, False, secs, model_used)
            except LLMRefusal as e:
                self._log_error(call, h, stage, e)
                raise
            except (LLMError, ValidationError, json.JSONDecodeError) as e:
                last_err = e
                self._log_error(call, h, stage, e)
                continue
        raise LLMError(f"{call.task}: failed after {self.cfg.max_retries + 1} attempts: {last_err}")

    def _log_error(self, call: LLMCall, h: str, stage: str, e: Exception) -> None:
        with self._lock:
            self.db.log_llm(
                job_id=self.job_id, stage=stage, prompt_version=call.prompt_version, model=call.model,
                params_json=json.dumps(call.params()), input_sha256=h, cached=0, error=str(e)[:2000],
            )


def get_backend(cfg):
    if cfg.backend == "anthropic":
        return AnthropicBackend()
    if cfg.backend == "claude_code":
        return ClaudeCodeBackend(cfg)
    if cfg.backend == "fake":
        return FakeBackend()
    raise ValueError(f"unknown llm backend {cfg.backend!r}")
