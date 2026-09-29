"""Tests for core.memory, the incident store and recall.

The store is redirected to a temporary directory and the TF-IDF backend is
forced, so retrieval is deterministic and no model download is needed. The
point is the behaviour the app depends on: seeding is idempotent, a
similar-shaped report recalls the right past incident, and a newly recorded
incident is retrievable afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import pytest

import core.memory as memory
from core.agent import build_evidence, template_report
from core.detect import Incident, daily_metrics, detect_anomalies, group_incidents, load_data

SAMPLE_CSV: Path = Path(__file__).resolve().parents[1] / "data" / "sample.csv"

DEMO_TRACKING_TAG: str = "checkout tracking tag broke after app release"
DEMO_LOW_QUALITY_TRAFFIC: str = "low-quality traffic from a new ad campaign"


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A fresh, in-tmp TF-IDF store for each test (skips the model download)."""
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    monkeypatch.setattr(
        memory,
        "_ChromaStore",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("no chroma in tests")),
    )
    memory.reset_store()
    yield
    memory.reset_store()


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    source = load_data(SAMPLE_CSV).copy()
    sessions = source["sessions"].where(source["sessions"] != 0)
    source["conversion_rate"] = source["orders"] / sessions
    return source


@pytest.fixture(scope="module")
def incidents(frame: pd.DataFrame) -> list[Incident]:
    return group_incidents(detect_anomalies(daily_metrics(frame)))


def report_for(incidents: list[Incident], frame: pd.DataFrame, metric: str, onset: str) -> dict[str, Any]:
    """The template report for the incident covering a given metric/onset."""
    break_day = pd.Timestamp(onset).date()
    incident = next(
        incident
        for incident in incidents
        if incident["metric"] == metric
        and incident["start_date"] <= break_day <= incident["end_date"]
    )
    return template_report(incident, build_evidence(frame, incident))


def test_seeding_fills_an_empty_store_once(store: None) -> None:
    """The demo incidents are inserted into an empty store, and only once."""
    assert memory.seed_demo_incidents()["seeded"] == 4
    # Seeding again is a no-op because the store is no longer empty.
    assert memory.seed_demo_incidents()["seeded"] == 0


def test_seeded_incidents_are_marked_demo(store: None) -> None:
    """Every seeded incident carries its cause, date, and the demo flag."""
    memory.seed_demo_incidents()
    report = {"headline": "anything", "metric": "revenue", "likely_driver": "channel=organic"}
    matches = memory.find_similar(report, k=4)

    assert len(matches) == 4
    assert all(match["is_demo"] for match in matches)
    causes = {match["cause"] for match in matches}
    assert DEMO_TRACKING_TAG in causes
    assert DEMO_LOW_QUALITY_TRAFFIC in causes


def test_mobile_organic_report_recalls_the_tracking_tag_incident(
    store: None, frame: pd.DataFrame, incidents: list[Incident]
) -> None:
    """A mobile+organic revenue drop surfaces the historical tracking-tag incident."""
    memory.seed_demo_incidents()
    report = report_for(incidents, frame, "revenue", "2026-08-21")

    matches = memory.find_similar(report, k=1)
    assert matches[0]["cause"] == DEMO_TRACKING_TAG
    assert "organic" in matches[0]["driver"] and "mobile" in matches[0]["driver"]


def test_paid_report_recalls_the_low_quality_traffic_incident(
    store: None, frame: pd.DataFrame, incidents: list[Incident]
) -> None:
    """A paid sessions surge surfaces the historical low-quality-traffic incident."""
    memory.seed_demo_incidents()
    report = report_for(incidents, frame, "sessions", "2026-09-16")

    matches = memory.find_similar(report, k=1)
    assert matches[0]["cause"] == DEMO_LOW_QUALITY_TRAFFIC
    assert matches[0]["metric"] == "sessions"


def test_added_incident_is_retrievable_afterwards(store: None) -> None:
    """A recorded cause comes back when the same pattern is reported again."""
    memory.add_incident(
        headline="Revenue drop on mobile and organic",
        metric="revenue",
        driver="channel=organic, device=mobile",
        cause="a locally recorded root cause",
        date="2026-09-01",
        is_demo=False,
    )

    report = {
        "headline": "Revenue fell on a weekday",
        "metric": "revenue",
        "likely_driver": "channel=organic, device=mobile is the likeliest driver",
    }
    matches = memory.find_similar(report, k=1)

    assert matches[0]["cause"] == "a locally recorded root cause"
    assert matches[0]["is_demo"] is False
    assert matches[0]["date"] == "2026-09-01"


def test_find_similar_on_empty_store_returns_nothing(store: None) -> None:
    """Recall on an empty store yields no matches rather than raising."""
    assert memory.find_similar({"headline": "revenue fell", "metric": "revenue"}) == []