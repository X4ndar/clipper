"""LLM-facing output schemas. Kept loose (plain ints/strings); semantic checks happen in code."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

ClipKind = Literal["story", "claim", "howto", "funny", "debate", "other"]


class Pass1Candidate(BaseModel):
    start_sentence_id: int
    end_sentence_id: int
    hook_line: str
    reason: str
    content_type: ClipKind


class Pass1Output(BaseModel):
    candidates: list[Pass1Candidate]


class JudgeScores(BaseModel):
    hook: int
    standalone_sense: int
    payoff: int
    peak: int
    quotability: int
    clean_ending: int


class JudgeOutput(BaseModel):
    start_sentence_id: int
    end_sentence_id: int
    start_word_offset: int
    end_word_offset: int
    scores: JudgeScores
    reject: bool
    reject_reason: Optional[str]
    rationale: str
    confidence: float
    hook_text: Optional[str]
    title: str
    description: str
    tags: list[str]
    content_type: ClipKind
