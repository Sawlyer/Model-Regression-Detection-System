import json
from pathlib import Path

from typer.testing import CliRunner

from regression_detector.cli import app

runner = CliRunner()
ROOT = Path(__file__).parent.parent
PROMPT_V1 = ROOT / "prompts" / "v1.yaml"
DATASET = ROOT / "data" / "golden_dataset.json"


def _invoke_run(tmp_path: Path, prompt: Path):
    return runner.invoke(app, [
        "run", str(prompt), "--dataset", str(DATASET), "--mock",
        "--db", str(tmp_path / "t.db"), "--report-dir", str(tmp_path / "reports"),
        "--summary-json", str(tmp_path / "summary.json"),
    ])


def test_first_run_is_baseline_exit_zero(tmp_path: Path):
    result = _invoke_run(tmp_path, PROMPT_V1)
    assert result.exit_code == 0, result.output
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "baseline"
    assert Path(summary["report_path"]).exists()


def test_degraded_prompt_triggers_critical_exit_2(tmp_path: Path):
    # baseline with good prompt
    assert _invoke_run(tmp_path, PROMPT_V1).exit_code == 0
    # degraded prompt -> mock forces 'general' -> many regressions
    bad = tmp_path / "v2.yaml"
    bad.write_text(PROMPT_V1.read_text().replace(
        "version: v1", "version: v2").replace(
        "You are a customer support email classifier",
        "DEGRADED You are a customer support email classifier"))
    result = _invoke_run(tmp_path, bad)
    assert result.exit_code == 2, result.output
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == "critical"
    assert summary["regressions"] > 0


def test_list_runs_shows_history(tmp_path: Path):
    _invoke_run(tmp_path, PROMPT_V1)
    result = runner.invoke(app, ["list-runs", "--db", str(tmp_path / "t.db")])
    assert result.exit_code == 0
    assert "v1" in result.output
