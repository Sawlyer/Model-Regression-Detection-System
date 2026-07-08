"""Typed domain models shared across the pipeline."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class Category(str, Enum):
    """The four support-email buckets the classifier must choose from."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class FewShotExample(BaseModel):
    """One labelled example embedded in a prompt to steer the classifier."""

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
        """Load and validate a prompt config from a YAML file."""
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


class TestCase(BaseModel):
    """One human-labelled golden case: an email plus its ground-truth answer."""

    __test__ = False  # tell pytest this is not a test class

    id: str
    input: str
    expected_category: Category
    ideal_summary: str
    expected_difficulty: Literal["easy", "medium", "hard"]
    notes: str = ""


class GoldenDataset(BaseModel):
    """A versioned collection of golden test cases — the evaluation ground truth."""

    version: str
    cases: list[TestCase]


class ClassifierOutput(BaseModel):
    """The structured result the classifier returns for one email."""

    category: Category
    summary: str


class CaseResult(BaseModel):
    """The scored outcome of running one golden case through the pipeline."""

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
    """A complete eval of one prompt over the dataset, with aggregate metrics."""

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
    """Verdict of a run-to-run diff: acceptable, warning, or merge-blocking."""

    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"


class CaseFlip(BaseModel):
    """A case whose pass/fail state changed between two runs (regression or improvement)."""

    case_id: str
    input: str
    expected_category: Category
    baseline_output: ClassifierOutput | None
    current_output: ClassifierOutput | None


class DiffReport(BaseModel):
    """The comparison of a run against its baseline: deltas, flips, and verdict."""

    baseline_run_id: str
    current_run_id: str
    pass_rate_delta: float
    per_category_delta: dict[str, float]
    regressions: list[CaseFlip]
    improvements: list[CaseFlip]
    status: RunStatus
