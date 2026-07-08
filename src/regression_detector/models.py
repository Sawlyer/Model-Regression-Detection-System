"""Typed domain models shared across the pipeline."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class FewShotExample(BaseModel):
    email: str
    category: Category
    summary: str


class PromptConfig(BaseModel):
    """A versioned prompt — the 'code' this CI pipeline runs against."""

    version: str
    timestamp: datetime
    model: str
    system_prompt: str
    few_shot: list[FewShotExample] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "PromptConfig":
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class TestCase(BaseModel):
    __test__ = False  # tell pytest this is not a test class

    id: str
    input: str
    expected_category: Category
    ideal_summary: str
    expected_difficulty: Literal["easy", "medium", "hard"]
    notes: str = ""


class GoldenDataset(BaseModel):
    version: str
    cases: list[TestCase]


class ClassifierOutput(BaseModel):
    category: Category
    summary: str


class CaseResult(BaseModel):
    case_id: str
    output: ClassifierOutput | None = None
    error: str | None = None
    category_match: bool = False
    judge_score: int | None = None
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    passed: bool = False


class EvalRun(BaseModel):
    run_id: str
    prompt_version: str
    model: str
    timestamp: datetime
    results: list[CaseResult]
    pass_rate: float
    per_category_accuracy: dict[str, float]
    avg_judge_score: float | None
    avg_latency_ms: float
    total_tokens: int


class RunStatus(str, Enum):
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class CaseFlip(BaseModel):
    case_id: str
    input: str
    expected_category: Category
    baseline_output: ClassifierOutput | None
    current_output: ClassifierOutput | None


class DiffReport(BaseModel):
    baseline_run_id: str
    current_run_id: str
    pass_rate_delta: float
    per_category_delta: dict[str, float]
    regressions: list[CaseFlip]
    improvements: list[CaseFlip]
    status: RunStatus
