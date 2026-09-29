"""Tests for core.agent (part 1: evidence and the template report).

The narration contract is that numbers come only from the tools, so the scoring
here is deterministic too: for each injected anomaly the template report must
point at the ground-truth segment, quote only tool numbers, and never reach for
a loaded causal word.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from core import rootcause
from core.agent import build_evidence, template_report
from core.detect import Incident, daily_metrics, detect_anomalies, group_incidents, load_data

DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
SAMPLE_CSV: Path = DATA_DIR / "sample.csv"
TRUTH_JSON: Path = DATA_DIR / "anomalies_truth.json"

REPORT_KEYS: set[str] = {
    "headline",
    "severity",
    "what_changed",
    "likely_driver",
    "evidence",
    "recommended_actions",
}

# Words the wording rule forbids: the template may describe behaviour, never
# call out a specific external cause it has not measured.
BANNED_WORDS: list[str] = ["bot", "flood", "bug"]


def truth_entries() -> list[dict[str, object]]:
    payload = json.loads(TRUTH_JSON.read_text(encoding="utf-8"))
    return list(payload["anomalies"])


def ids(entry: dict[str, object]) -> str:
    return str(entry["id"])


def expected_label(segment: dict[str, object]) -> str:
    """The driver label the drilldown would emit for a ground-truth segment."""
    parts = [
        f"{dim}={segment[dim]}"
        for dim in rootcause.SEGMENT_DIMENSIONS
        if dim in segment
    ]
    return ", ".join(parts)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """The long frame the drilldown tools and the app use, augmented with rate."""
    source = load_data(SAMPLE_CSV).copy()
    sessions = source["sessions"].where(source["sessions"] != 0)
    source["conversion_rate"] = source["orders"] / sessions
    return source


@pytest.fixture(scope="module")
def daily(frame: pd.DataFrame) -> pd.DataFrame:
    return daily_metrics(frame)


@pytest.fixture(scope="module")
def incidents(daily: pd.DataFrame) -> list[Incident]:
    return group_incidents(detect_anomalies(daily))


def incident_for(incidents: list[Incident], metric: str, onset: str) -> Incident:
    """The incident whose metric matches and whose span covers the onset."""
    break_day = pd.Timestamp(onset).date()
    for incident in incidents:
        if (
            incident["metric"] == metric
            and incident["start_date"] <= break_day <= incident["end_date"]
        ):
            return incident
    raise AssertionError(f"no {metric} incident covers {onset}")


@pytest.mark.parametrize("entry", truth_entries(), ids=ids)
def test_likely_driver_names_the_ground_truth_segment(
    entry: dict[str, object],
    frame: pd.DataFrame,
    incidents: list[Incident],
) -> None:
    """Each injected anomaly is blamed on the segment it was injected into."""
    metric = str(entry["metric"])
    onset = str(entry["date_start"])
    segment = dict(entry["segment"])  # type: ignore[arg-type]

    incident = incident_for(incidents, metric, onset)
    evidence = build_evidence(frame, incident)
    report = template_report(incident, evidence)

    assert isinstance(report["likely_driver"], str)
    assert expected_label(segment) in report["likely_driver"], (
        f"{metric} on {onset} should be pinned to {expected_label(segment)}, "
        f"got: {report['likely_driver']}"
    )


def test_evidence_is_json_serialisable(frame: pd.DataFrame, incidents: list[Incident]) -> None:
    """The evidence dict and the report survive json.dumps untouched."""
    incident = incident_for(incidents, "revenue", "2026-08-21")
    evidence = build_evidence(frame, incident)

    dumped = json.loads(json.dumps(evidence))
    assert dumped["metric"] == "revenue"
    assert dumped["peak_date"] == "2026-08-21"
    assert dumped["compare"]["metric"] == "revenue"
    assert dumped["drivers"]["drivers"]
    assert len(dumped["ratio"]["effects"]) == 3

    report = template_report(incident, evidence)
    json.dumps(report)


def test_template_report_shape(frame: pd.DataFrame, incidents: list[Incident]) -> None:
    """The report carries exactly the documented keys."""
    incident = incident_for(incidents, "refunds", "2026-09-04")
    report = template_report(incident, build_evidence(frame, incident))

    assert set(report) == REPORT_KEYS
    assert report["severity"] == incident["severity"]
    assert isinstance(report["headline"], str)
    assert isinstance(report["what_changed"], str)
    assert isinstance(report["evidence"], list) and report["evidence"]
    assert isinstance(report["recommended_actions"], list) and report["recommended_actions"]


def test_report_quotes_only_evidence_numbers(frame: pd.DataFrame, incidents: list[Incident]) -> None:
    """Figures in the prose match the evidence, not something the template invented."""
    incident = incident_for(incidents, "revenue", "2026-08-21")
    evidence = build_evidence(frame, incident)
    report = template_report(incident, evidence)

    compare = evidence["compare"]
    driver = evidence["drivers"]["drivers"][0]
    assert compare["current"] is not None
    assert compare["baseline"] is not None
    assert driver["share_of_change"] is not None

    assert f"{compare['current']:,.2f}" in report["what_changed"]
    assert f"{compare['baseline']:,.2f}" in report["what_changed"]
    assert driver["label"] in report["likely_driver"]


@pytest.mark.parametrize("entry", truth_entries(), ids=ids)
def test_report_avoids_banned_wording(
    entry: dict[str, object],
    frame: pd.DataFrame,
    incidents: list[Incident],
) -> None:
    """The template says what the data look like, never what they 'are'."""
    metric = str(entry["metric"])
    onset = str(entry["date_start"])
    incident = incident_for(incidents, metric, onset)
    report = template_report(incident, build_evidence(frame, incident))

    text = " ".join(str(report[key]) for key in REPORT_KEYS if key != "evidence")
    for word in BANNED_WORDS:
        assert word.lower() not in text.lower(), f"forbidden word {word!r} in report text"
    assert any(
        hedge in text for hedge in ("consistent with", "may indicate")
    ), "report should hedge its interpretations"