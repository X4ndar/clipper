import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from clipper.config import LLMConfig
from clipper.db import DB
from clipper.llm.client import (
    AnthropicBackend, BudgetExceeded, LLMCall, LLMError, LLMRefusal, LLMService, cost_usd,
)
from clipper.llm.schemas import JudgeOutput, Pass1Output

GOOD_P1 = {"candidates": [{"start_sentence_id": 3, "end_sentence_id": 9, "hook_line": "Why?",
                           "reason": "r", "content_type": "claim"}]}


def test_pass1_schema_validates_and_rejects():
    assert Pass1Output.model_validate(GOOD_P1).candidates[0].end_sentence_id == 9
    bad = {"candidates": [{**GOOD_P1["candidates"][0], "content_type": "meme"}]}
    with pytest.raises(ValidationError):
        Pass1Output.model_validate(bad)
    with pytest.raises(ValidationError):
        Pass1Output.model_validate({"candidates": [{"start_sentence_id": "x"}]})


def test_judge_schema_requires_all_fields():
    with pytest.raises(ValidationError):
        JudgeOutput.model_validate({"start_sentence_id": 1})


def call(**kw):
    base = dict(task="candidates", model="claude-haiku-5-5", prompt_version="candidates_v1", system="sys",
                user="u", schema=Pass1Output)
    base.update(kw)
    return LLMCall(**base)


class ScriptedBackend:
    name = "scripted"

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, c):
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return c.schema.model_validate(out), {"input_tokens": 1000, "output_tokens": 100}, "end_turn", c.model


def svc(tmp_path, backend, budget=None, **cfg):
    return LLMService(backend, DB(tmp_path / "db.sqlite"), LLMConfig(**cfg), job_id="j1", budget_usd=budget)


def test_retry_then_success_and_log(tmp_path):
    b = ScriptedBackend([LLMError("bad json"), GOOD_P1])
    s = svc(tmp_path, b)
    r = s.call(call(), "candidates")
    assert b.calls == 2 and not r.cached
    rows = list(s.db.conn.execute("SELECT error, output_json, prompt_version FROM llm_runs ORDER BY id"))
    assert rows[0]["error"] and rows[1]["output_json"] and rows[1]["prompt_version"] == "candidates_v1"
    assert r.cost_usd == pytest.approx((1000 * 0.10 + 100 * 0.50) / 1e6)


def test_gives_up_after_retries(tmp_path):
    b = ScriptedBackend([LLMError("x")] * 3)
    with pytest.raises(LLMError):
        svc(tmp_path, b, max_retries=2).call(call(), "candidates")


def test_cache_hit_costs_nothing(tmp_path):
    b = ScriptedBackend([GOOD_P1])
    s = svc(tmp_path, b)
    s.call(call(), "candidates")
    r2 = s.call(call(), "candidates")
    assert r2.cached and r2.cost_usd == 0 and b.calls == 1


def test_budget_guard(tmp_path):
    b = ScriptedBackend([GOOD_P1, GOOD_P1])
    s = svc(tmp_path, b, budget=0.0001, cache_responses=False)
    s.call(call(), "candidates")
    with pytest.raises(BudgetExceeded):
        s.call(call(user="other"), "candidates")


def test_refusal_not_retried(tmp_path):
    b = ScriptedBackend([LLMRefusal("no")])
    with pytest.raises(LLMRefusal):
        svc(tmp_path, b).call(call(), "candidates")
    assert b.calls == 1


class FakeMessages:
    def __init__(self, parsed, stop="end_turn"):
        self.parsed, self.stop, self.kwargs = parsed, stop, None

    def parse(self, **kw):
        self.kwargs = kw
        return SimpleNamespace(parsed_output=self.parsed, stop_reason=self.stop, stop_details=None,
                               model=kw["model"], usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 5}))


def test_anthropic_backend_request_shape():
    parsed = Pass1Output.model_validate(GOOD_P1)
    std, beta = FakeMessages(parsed), FakeMessages(parsed)
    client = SimpleNamespace(messages=std, beta=SimpleNamespace(messages=beta))
    be = AnthropicBackend(client)
    be.complete(call(effort="low"))
    assert std.kwargs["output_format"] is Pass1Output
    assert std.kwargs["output_config"] == {"effort": "low"}
    assert "temperature" not in std.kwargs  # current models reject custom sampling params
    be.complete(call(fallbacks="default"))
    assert beta.kwargs["fallbacks"] == "default"
    assert beta.kwargs["betas"] == ["server-side-fallback-2026-07-01"]


def test_anthropic_backend_refusal():
    client = SimpleNamespace(messages=FakeMessages(None, stop="refusal"), beta=None)
    with pytest.raises(LLMRefusal):
        AnthropicBackend(client).complete(call())


def test_cost_formula():
    assert cost_usd("claude-sonnet-5-5", {"input_tokens": 1_000_000, "output_tokens": 100_000},
                    LLMConfig().prices_per_mtok) == pytest.approx(3.0)
    assert cost_usd("unknown", {"input_tokens": 5}, {}) == 0.0
