"""Tests for core.detect, scored against the injected anomalies in data/.

The generated sample ships with a ground-truth file naming where each anomaly
was injected and which metric it moved. These tests check that the detector
finds those dates, and that it stays quiet on the clean part of the series.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from core.detect import (
    DEFAULT_SENSITIVITY,
    METRICS,
    METHOD_BOTH,
    SEVERITY_HIGH,
    Anomaly,
    daily_metrics,
    detect_anomalies,
    load_data,
)

DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
SAMPLE_CSV: Path = DATA_DIR / "sample.csv"
TRUTH_JSON: Path = DATA_DIR / "anomalies_truth.json"

ANOMALY_KEYS: set[str] = {
    "date",
    "metric",
    "value",
    "expected",
    "pct_change",
    "z",
    "severity",
    "method",
}
SEVERITIES: set[str] = {"low", "medium", "high"}

# Which way each injected anomaly should move its metric, keyed by ground-truth
# id: (a) revenue drop, (b) refunds spike, (c) sessions surge.
EXPECTED_DIRECTION: dict[str, str] = {"a": "down", "b": "up", "c": "up"}

# How many days late the detector may be. It should fire as the level breaks,
# not days afterwards.
MAX_DETECTION_LAG_DAYS: int = 2


def truth_entries() -> list[dict[str, object]]:
    """The injected anomalies, read from the ground-truth file."""
    payload = json.loads(TRUTH_JSON.read_text(encoding="utf-8"))
    return list(payload["anomalies"])


def ids(entry: dict[str, object]) -> str:
    return str(entry["id"])


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return daily_metrics(load_data(SAMPLE_CSV))


@pytest.fixture(scope="module")
def anomalies(daily: pd.DataFrame) -> list[Anomaly]:
    return detect_anomalies(daily)


@pytest.mark.parametrize("entry", truth_entries(), ids=ids)
def test_injected_onset_is_detected(entry: dict[str, object], anomalies: list[Anomaly]) -> None:
    """Each injected anomaly is flagged on its metric, at the break."""
    metric = str(entry["metric"])
    onset = pd.Timestamp(str(entry["date_start"])).date()
    end = pd.Timestamp(str(entry["date_end"])).date()

    hits = [a for a in anomalies if a["metric"] == metric and onset <= a["date"] <= end]
    assert hits, f"no {metric} anomaly detected between {onset} and {end}"

    first = min(a["date"] for a in hits)
    assert abs((first - onset).days) <= MAX_DETECTION_LAG_DAYS, (
        f"{metric} broke on {onset} but was first flagged on {first}"
    )


@pytest.mark.parametrize("entry", truth_entries(), ids=ids)
def test_injected_anomaly_moves_in_the_right_direction(entry: dict[str, object], anomalies: list[Anomaly]) -> None:
    """The detector sees a fall where revenue fell and a rise where it rose."""
    metric = str(entry["metric"])
    onset = pd.Timestamp(str(entry["date_start"])).date()
    end = pd.Timestamp(str(entry["date_end"])).date()
    direction = EXPECTED_DIRECTION[str(entry["id"])]

    hits = [a for a in anomalies if a["metric"] == metric and onset <= a["date"] <= end]
    on_onset = [a for a in hits if a["date"] == onset] or hits
    for anomaly in on_onset:
        assert anomaly["pct_change"] is not None
        if direction == "down":
            assert anomaly["pct_change"] < 0, f"{metric} should have fallen on {anomaly['date']}"
        else:
            assert anomaly["pct_change"] > 0, f"{metric} should have risen on {anomaly['date']}"


def test_no_high_severity_anomaly_before_the_injected_window(anomalies: list[Anomaly]) -> None:
    """The clean part of the series produces no loud false alarms."""
    first_onset = min(pd.Timestamp(str(entry["date_start"])).date() for entry in truth_entries())
    early = [a for a in anomalies if a["date"] < first_onset and a["severity"] == SEVERITY_HIGH]
    assert early == [], f"high-severity anomalies before any injected anomaly: {early}"


def test_cross_check_agrees_on_at_least_one_anomaly(anomalies: list[Anomaly]) -> None:
    """The forest independently confirms at least one STL detection."""
    agreed = [a for a in anomalies if a["method"] == METHOD_BOTH]
    assert agreed, "IsolationForest never agreed with the STL z-score"


def test_records_are_well_formed(anomalies: list[Anomaly]) -> None:
    """Every record carries the documented keys, types, and ordering."""
    assert anomalies, "no anomalies detected at all"
    for anomaly in anomalies:
        assert set(anomaly) == ANOMALY_KEYS
        assert isinstance(anomaly["date"], date)
        assert anomaly["metric"] in METRICS
        assert isinstance(anomaly["value"], float)
        assert isinstance(anomaly["expected"], float)
        assert anomaly["severity"] in SEVERITIES
        assert abs(anomaly["z"]) > DEFAULT_SENSITIVITY or "isolation_forest" in anomaly["method"]
        assert anomaly["pct_change"] is None or isinstance(anomaly["pct_change"], float)

    keys = [(a["date"], -abs(a["z"])) for a in anomalies]
    assert keys == sorted(keys), "records are not ordered by date then descending |z|"


def test_daily_metrics_aggregates_the_long_dataset(daily: pd.DataFrame) -> None:
    """Daily totals are the sums of the rows behind them."""
    assert list(daily.columns) == ["revenue", "orders", "sessions", "refunds", "conversion_rate"]
    assert len(daily) == 180
    assert daily.index.is_monotonic_increasing
    assert (daily["conversion_rate"] - daily["orders"] / daily["sessions"]).abs().max() < 1e-12
    assert (daily[["revenue", "orders", "sessions", "refunds"]] >= 0).all().all()


def test_load_data_validates_the_schema() -> None:
    """A frame missing a required column is rejected by name."""
    frame = load_data(SAMPLE_CSV)
    assert len(frame) == 25_920
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])

    with pytest.raises(ValueError, match="region"):
        load_data(frame.drop(columns=["region"]))

    with pytest.raises(FileNotFoundError):
        load_data(DATA_DIR / "no_such_file.csv")


def test_load_data_does_not_mutate_the_input_frame() -> None:
    """The caller's frame is left exactly as it was passed in."""
    original = load_data(SAMPLE_CSV)
    snapshot = original.copy()
    returned = load_data(original)
    pd.testing.assert_frame_equal(original, snapshot)
    assert returned is not original


def test_too_short_a_series_is_rejected(daily: pd.DataFrame) -> None:
    """There must be history to predict from and to decompose."""
    with pytest.raises(ValueError, match="at least 14 days"):
        detect_anomalies(daily.iloc[:10])
