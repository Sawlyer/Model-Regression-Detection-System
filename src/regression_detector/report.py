"""Self-explanatory standalone HTML report with an inline SVG trend chart.

The report is written so a reader who knows nothing about the project can look at
it and immediately understand: what was tested, whether quality dropped, and
exactly which cases changed.
"""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from .models import DiffReport, EvalRun

_env = Environment(
    loader=PackageLoader("regression_detector", "templates"),
    autoescape=select_autoescape(["html"]),
)

_STATUS_ICON = {"ok": "✓", "warning": "!", "critical": "✕", "baseline": "◆"}
_STATUS_LABEL = {
    "ok": "PASS", "warning": "WARNING", "critical": "CRITICAL", "baseline": "BASELINE",
}


def _trend_svg(history: list[EvalRun], width: int = 720, height: int = 240) -> str:
    """history is newest-first; chart draws oldest -> newest, left -> right."""
    runs = list(reversed(history))
    if len(runs) < 2:
        return (
            '<p class="empty">Only one run so far — the trend chart appears '
            "once there are at least two runs to plot.</p>"
        )
    pad_l, pad_r, pad_t, pad_b = 44, 16, 16, 34
    plot_w, plot_h = width - pad_l - pad_r, height - pad_t - pad_b
    n = len(runs)

    def x(i: int) -> float:
        return pad_l + i * plot_w / (n - 1)

    def y(rate: float) -> float:
        return pad_t + (1 - rate) * plot_h

    # horizontal gridlines at 0 / 50 / 100 %
    grid = []
    for frac in (0.0, 0.5, 1.0):
        gy = y(frac)
        grid.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{width - pad_r}" y2="{gy:.1f}" '
            f'class="grid"/>'
            f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" class="axis" text-anchor="end">'
            f'{int(frac * 100)}%</text>'
        )
    pts = [(x(i), y(r.pass_rate)) for i, r in enumerate(runs)]
    poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    # soft area under the line
    area = f"{pad_l},{y(0.0):.1f} " + poly + f" {x(n - 1):.1f},{y(0.0):.1f}"
    dots = []
    for i, ((px, py), r) in enumerate(zip(pts, runs)):
        last = i == n - 1
        cls = "dot-last" if last else "dot"
        dots.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{5 if last else 4}" class="{cls}">'
            f"<title>{r.run_id}: {r.pass_rate:.0%} ({r.prompt_version})</title></circle>"
        )
        if last:
            dots.append(
                f'<text x="{px:.1f}" y="{py - 12:.1f}" class="pointlbl" '
                f'text-anchor="end">{r.pass_rate:.0%}</text>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="trend" role="img" '
        f'aria-label="pass rate over the last {n} runs">'
        f"{''.join(grid)}"
        f'<polygon points="{area}" class="area"/>'
        f'<polyline points="{poly}" class="line"/>'
        f"{''.join(dots)}</svg>"
    )


def _verdict(status: str, diff: DiffReport | None, errored: int) -> tuple[str, str]:
    """Return (headline, plain-language explanation)."""
    if status == "baseline":
        return (
            "Baseline recorded",
            "This is the first run for this dataset — there is nothing to compare "
            "against yet. Future runs are measured against this one.",
        )
    assert diff is not None
    drop = abs(diff.pass_rate_delta) * 100
    n = len(diff.regressions)
    err_note = (
        f" {errored} of them are calls that errored (e.g. API rate limits), which "
        "count as failures."
        if errored
        else ""
    )
    if status == "critical":
        return (
            "Regression detected — merge blocked",
            f"{n} test emails that used to be classified correctly now fail. "
            f"Overall quality dropped {drop:.1f}%, past the critical threshold, so "
            f"this change exits with code 2 and blocks the merge in CI.{err_note}",
        )
    if status == "warning":
        return (
            "Minor drop — review before merging",
            f"Quality dropped {drop:.1f}% ({n} case(s) regressed). Below the "
            f"critical threshold, but worth a look before shipping.{err_note}",
        )
    gain = len(diff.improvements)
    return (
        "No regression — safe to merge",
        f"Quality held or improved versus the previous run"
        + (f" ({gain} case(s) got better)" if gain else "")
        + ". No significant drop detected.",
    )


def generate_report(
    current: EvalRun,
    diff: DiffReport | None,
    history: list[EvalRun],
    drift: tuple[bool, float | None],
    out_path: Path,
    warning_threshold: float = 0.03,
    critical_threshold: float = 0.08,
) -> Path:
    """Render a standalone HTML report for `current` and write it to `out_path`.

    `diff` is None for a baseline (first) run. `history` (newest-first) feeds the
    trend chart, and `drift` is the ``(drifting, average)`` tuple from
    detect_drift. The thresholds are shown so a reader knows what gated the
    verdict. Returns the path written.
    """
    drifting, drift_avg = drift
    status = diff.status.value if diff else "baseline"
    errored = sum(1 for r in current.results if r.error)
    passed = sum(1 for r in current.results if r.passed)
    headline, explanation = _verdict(status, diff, errored)

    categories = []
    for cat, acc in sorted(current.per_category_accuracy.items()):
        delta = diff.per_category_delta.get(cat) if diff else None
        categories.append({"name": cat, "acc": acc, "delta": delta})

    baseline_rate = (current.pass_rate - diff.pass_rate_delta) if diff else None

    html = _env.get_template("report.html.j2").render(
        current=current,
        diff=diff,
        status=status,
        status_icon=_STATUS_ICON[status],
        status_label=_STATUS_LABEL[status],
        headline=headline,
        explanation=explanation,
        pass_pct=current.pass_rate * 100,
        baseline_pct=(baseline_rate * 100) if baseline_rate is not None else None,
        delta_pct=(diff.pass_rate_delta * 100) if diff else None,
        passed=passed,
        total=len(current.results),
        errored=errored,
        regression_count=len(diff.regressions) if diff else 0,
        improvement_count=len(diff.improvements) if diff else 0,
        categories=categories,
        trend_svg=_trend_svg(history),
        n_runs=len(history),
        drift_detected=drifting,
        drift_avg_pct=(drift_avg * 100) if drift_avg is not None else 0.0,
        warning_pct=warning_threshold * 100,
        critical_pct=critical_threshold * 100,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
