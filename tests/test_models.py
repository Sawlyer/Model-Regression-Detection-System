from pathlib import Path

import pytest

from regression_detector.models import (
    Category,
    ClassifierOutput,
    PromptConfig,
    TestCase,
)


def test_category_values():
    assert [c.value for c in Category] == ["billing", "technical", "account", "general"]


def test_prompt_config_from_yaml(tmp_path: Path):
    yaml_text = """\
version: v1
timestamp: 2026-07-08T10:00:00Z
model: meta-llama/llama-3.3-70b-instruct:free
system_prompt: |
  You are a support email classifier.
few_shot:
  - email: "I was charged twice"
    category: billing
    summary: "Customer reports a duplicate charge."
"""
    p = tmp_path / "v1.yaml"
    p.write_text(yaml_text)
    cfg = PromptConfig.from_yaml(p)
    assert cfg.version == "v1"
    assert cfg.model.endswith(":free")
    assert cfg.few_shot[0].category is Category.BILLING


def test_classifier_output_rejects_bad_category():
    with pytest.raises(ValueError):
        ClassifierOutput(category="spam", summary="x")


def test_testcase_difficulty_literal():
    with pytest.raises(ValueError):
        TestCase(
            id="tc-001", input="hi", expected_category="general",
            ideal_summary="s", expected_difficulty="impossible",
        )
