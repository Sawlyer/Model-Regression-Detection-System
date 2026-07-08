"""Static HTML diff report with inline SVG trend chart."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from .models import DiffReport, EvalRun

_env = Environment(
    loader=PackageLoader("regression_detector", "templates"),
    autoescape=select_autoescape(["html"]),
)


def _trend_svg(history: list[EvalRun], width: int = 700, height: int = 160) -> str:
    """history is newest-first; chart draws oldest -> newest, left -> right."""
    rates = [r.pass_rate for r in reversed(history)]
    if len(rates) < 2:
        return "<p>Not enough runs for a trend chart yet.</p>"
    pad = 30
    n = len(rates)
    pts = []
    for i, rate in enumerate(rates):
        x = pad + i * (width - 2 * pad) / (n - 1)
        y = pad + (1 - rate) * (height - 2 * pad)
        pts.append(f"{x:.1f},{y:.1f}")
    circles = "".join(
        f'<circle cx="{p.split(",")[0]}" cy="{p.split(",")[1]}" r="3" fill="#4a7dff"/>'
        for p in pts
    )
    labels = (
        f'<text x="4" y="{pad + 4}" font-size="10">100%</text>'
        f'<text x="4" y="{height - pad + 4}" font-size="10">0%</text>'
    )
    return (
        f'<svg width="{width}" height="{height}" role="img" aria-label="pass rate trend">'
        f'<polyline fill="none" stroke="#4a7dff" stroke-width="2" points="{" ".join(pts)}"/>'
        f"{circles}{labels}</svg>"
    )


def generate_report(
    current: EvalRun,
    diff: DiffReport | None,
    history: list[EvalRun],
    drift: tuple[bool, float | None],
    out_path: Path,
) -> Path:
    drifting, drift_avg = drift
    html = _env.get_template("report.html.j2").render(
        current=current,
        diff=diff,
        history=history,
        trend_svg=_trend_svg(history),
        drift_detected=drifting,
        drift_avg=drift_avg or 0.0,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
