"""Typer CLI: run evals, list history, compare runs, regenerate reports."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer

from .alert import build_alert_text, build_slack_payload, send_alert
from .config import Settings
from .dataset import load_dataset
from .diff import compare
from .drift import detect_drift
from .models import PromptConfig, RunStatus
from .provider import MockProvider, OpenRouterProvider
from .report import generate_report
from .runner import run_eval
from .storage import RunStore

app = typer.Typer(help="Model regression detection pipeline.")

_CRITICAL_EXIT = 2


def _make_provider(settings: Settings, mock: bool):
    if mock or not settings.openrouter_api_key:
        if not mock:
            typer.echo("No OPENROUTER_API_KEY found — falling back to MockProvider.")
        return MockProvider()
    return OpenRouterProvider(api_key=settings.openrouter_api_key)


@app.command()
def run(
    prompt_path: Path = typer.Argument(..., help="Path to versioned prompt YAML"),
    dataset: Path = typer.Option(Path("data/golden_dataset.json"), "--dataset"),
    mock: bool = typer.Option(False, "--mock", help="Force the offline mock provider"),
    db: Path | None = typer.Option(None, "--db"),
    report_dir: Path = typer.Option(Path("reports"), "--report-dir"),
    summary_json: Path | None = typer.Option(None, "--summary-json"),
) -> None:
    """Run the golden dataset against a prompt, diff vs previous run, report + alert."""
    settings = Settings.load()
    store = RunStore(db or settings.db_path)
    cfg = PromptConfig.from_yaml(prompt_path)
    ds = load_dataset(dataset)
    provider = _make_provider(settings, mock)

    baseline = store.get_latest_run()
    current = asyncio.run(run_eval(provider, cfg, ds, settings))
    store.save_run(current)

    diff = compare(baseline, current, ds, settings) if baseline else None
    history = store.get_last_n_runs(max(settings.drift_window, 10))
    drift = detect_drift(history, settings.drift_window, settings.drift_threshold)

    report_path = generate_report(
        current, diff, history, drift, report_dir / f"{current.run_id}.html"
    )
    typer.echo(f"Report: {report_path}")

    if diff:
        text = build_alert_text(diff, current, str(report_path))
        payload = build_slack_payload(diff, current, str(report_path))
        send_alert(payload, text, settings.slack_webhook_url)
    else:
        typer.echo(f"Baseline run stored ({current.pass_rate:.0%} pass rate). No diff yet.")

    if drift[0]:
        typer.echo(f"⚠️  Slow drift: {settings.drift_window}-run avg = {drift[1]:.1%}")

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps({
            "run_id": current.run_id,
            "status": diff.status.value if diff else "baseline",
            "pass_rate": current.pass_rate,
            "pass_rate_delta": diff.pass_rate_delta if diff else 0.0,
            "regressions": len(diff.regressions) if diff else 0,
            "improvements": len(diff.improvements) if diff else 0,
            "report_path": str(report_path),
            "drift": drift[0],
        }, indent=2))

    if diff and diff.status is RunStatus.CRITICAL:
        raise typer.Exit(_CRITICAL_EXIT)


@app.command("list-runs")
def list_runs(db: Path | None = typer.Option(None, "--db")) -> None:
    """List stored eval runs, newest first."""
    settings = Settings.load()
    store = RunStore(db or settings.db_path)
    rows = store.list_runs()
    if not rows:
        typer.echo("No runs stored yet.")
        return
    typer.echo(f"{'RUN ID':<22} {'PROMPT':<8} {'TIMESTAMP':<34} PASS RATE")
    for run_id, version, ts, rate in rows:
        typer.echo(f"{run_id:<22} {version:<8} {ts:<34} {rate:.1%}")


@app.command("compare")
def compare_cmd(
    baseline_id: str,
    current_id: str,
    db: Path | None = typer.Option(None, "--db"),
    dataset: Path = typer.Option(Path("data/golden_dataset.json"), "--dataset"),
) -> None:
    """Diff two stored runs by id."""
    settings = Settings.load()
    store = RunStore(db or settings.db_path)
    base, curr = store.get_run(baseline_id), store.get_run(current_id)
    if not base or not curr:
        typer.echo("Run not found.", err=True)
        raise typer.Exit(1)
    diff = compare(base, curr, load_dataset(dataset), settings)
    typer.echo(build_alert_text(diff, curr, "(no report generated)"))
    if diff.status is RunStatus.CRITICAL:
        raise typer.Exit(_CRITICAL_EXIT)


@app.command("report")
def report_cmd(
    run_id: str,
    db: Path | None = typer.Option(None, "--db"),
    out: Path = typer.Option(Path("reports/report.html"), "--out"),
) -> None:
    """Regenerate the HTML report for a stored run (no diff)."""
    settings = Settings.load()
    store = RunStore(db or settings.db_path)
    run_ = store.get_run(run_id)
    if not run_:
        typer.echo("Run not found.", err=True)
        raise typer.Exit(1)
    history = store.get_last_n_runs(max(settings.drift_window, 10))
    drift = detect_drift(history, settings.drift_window, settings.drift_threshold)
    path = generate_report(run_, None, history, drift, out)
    typer.echo(f"Report: {path}")


if __name__ == "__main__":
    app()
