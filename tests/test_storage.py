from datetime import datetime, timedelta, timezone
from pathlib import Path

from regression_detector.models import EvalRun
from regression_detector.storage import RunStore


def _run(run_id: str, ts: datetime, pass_rate: float = 0.9) -> EvalRun:
    return EvalRun(run_id=run_id, prompt_version="v1", model="m", timestamp=ts,
                   results=[], pass_rate=pass_rate, per_category_accuracy={"billing": 1.0},
                   avg_judge_score=4.0, avg_latency_ms=120.0, total_tokens=1000)


def test_save_and_get_roundtrip(tmp_path: Path):
    store = RunStore(tmp_path / "t.db")
    t0 = datetime.now(timezone.utc)
    store.save_run(_run("r1", t0))
    loaded = store.get_run("r1")
    assert loaded is not None
    assert loaded.pass_rate == 0.9
    assert loaded.per_category_accuracy == {"billing": 1.0}


def test_get_latest_and_last_n(tmp_path: Path):
    store = RunStore(tmp_path / "t.db")
    t0 = datetime.now(timezone.utc)
    for i in range(5):
        store.save_run(_run(f"r{i}", t0 + timedelta(minutes=i), pass_rate=0.5 + i * 0.1))
    assert store.get_latest_run().run_id == "r4"
    last3 = store.get_last_n_runs(3)
    assert [r.run_id for r in last3] == ["r4", "r3", "r2"]


def test_empty_db(tmp_path: Path):
    store = RunStore(tmp_path / "t.db")
    assert store.get_latest_run() is None
    assert store.get_last_n_runs(7) == []
    assert store.list_runs() == []
