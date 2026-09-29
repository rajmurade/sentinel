"""Tests for core.rootcause, scored against the injected anomalies in data/.

The ground-truth file names, for each injected anomaly, the metric that moved
and the segment that caused it. These tests check that the drilldown tools
surface that segment, that the arithmetic is self-consistent, and that every
payload survives a JSON round trip so it can be handed to the agent.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from core.detect import load_data
from core.rootcause import (
    COVERAGE_THRESHOLD,
    SEGMENT_DIMENSIONS,
    baseline_dates,
    compare_windows,
    contribution_by_dimension,
    ratio_decomposition,
    top_drivers,
)

DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
SAMPLE_CSV: Path = DATA_DIR / "sample.csv"
TRUTH_JSON: Path = DATA_DIR / "anomalies_truth.json"

# (a) revenue falls, (b) refunds rise, (c) sessions rise.
EXPECTED_DIRECTION: dict[str, str] = {"a": "down", "b": "up", "c": "up"}


def truth_entries() -> list[dict[str, object]]:
    payload = json.loads(TRUTH_JSON.read_text(encoding="utf-8"))
    return list(payload["anomalies"])


def truth_by_id() -> dict[str, dict[str, object]]:
    return {str(entry["id"]): entry for entry in truth_entries()}


def ids(entry: dict[str, object]) -> str:
    return str(entry["id"])


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return load_data(SAMPLE_CSV)


def test_anomaly_a_drilldown_points_at_mobile_organic(df: pd.DataFrame) -> None:
    """The revenue break is reported as mobile organic traffic."""
    entry = truth_by_id()["a"]
    result = top_drivers(df, str(entry["metric"]), str(entry["date_start"]), k=3)
    top = result["drivers"][0]

    assert top["segment"] == {"channel": "organic", "device": "mobile"}
    assert set(top["segment"]) >= set(entry["segment"])  # type: ignore[arg-type]
    assert top["change"] < 0, "a revenue drop should show a negative change"


def test_anomaly_c_drilldown_points_at_paid(df: pd.DataFrame) -> None:
    """The sessions surge is reported against the paid channel."""
    entry = truth_by_id()["c"]
    result = top_drivers(df, str(entry["metric"]), str(entry["date_start"]), k=3)
    top = result["drivers"][0]

    assert top["segment"] == {"channel": "paid"}
    assert top["change"] > 0, "a sessions surge should show a positive change"


@pytest.mark.parametrize("entry", truth_entries(), ids=ids)
def test_top_driver_matches_the_injected_segment(df: pd.DataFrame, entry: dict[str, object]) -> None:
    """Every injected anomaly is traced to the segment that actually caused it."""
    metric = str(entry["metric"])
    result = top_drivers(df, metric, str(entry["date_start"]), k=3)
    drivers = result["drivers"]
    assert drivers, f"no drivers found for the {metric} anomaly"

    injected = {str(key): str(value) for key, value in dict(entry["segment"]).items()}  # type: ignore[arg-type]
    assert drivers[0]["segment"] == injected

    top = drivers[0]
    assert top["share_of_change"] >= COVERAGE_THRESHOLD, "the top driver should account for the change"
    assert top["rank"] == 1
    assert set(top["dimensions"]) == set(injected)

    falling = EXPECTED_DIRECTION[str(entry["id"])] == "down"
    assert (top["change"] or 0.0) < 0 if falling else (top["change"] or 0.0) > 0


def test_top_drivers_cuts_every_dimension_and_pair(df: pd.DataFrame) -> None:
    """Ten cuts are tried and the ranks come back in order."""
    result = top_drivers(df, "revenue", "2026-08-21", k=3)
    assert result["cuts_searched"] == len(SEGMENT_DIMENSIONS) + 6
    assert [driver["rank"] for driver in result["drivers"]] == [1, 2, 3]

    seen = {tuple(driver["dimensions"]) for driver in result["drivers"]}
    assert len(seen) == 3, "a cut was reported twice"


def test_contributions_sum_to_the_total_change(df: pd.DataFrame) -> None:
    """The segments of a dimension add up to the change the store recorded."""
    result = contribution_by_dimension(df, "revenue", "2026-08-21", "channel")
    shares = [segment["share_of_change"] or 0.0 for segment in result["segments"]]
    assert sum(shares) == pytest.approx(1.0)

    changes = sum(segment["change"] or 0.0 for segment in result["segments"])
    assert changes == pytest.approx(result["total_change"])
    assert sum(segment["current"] or 0.0 for segment in result["segments"]) == pytest.approx(
        result["current_total"]
    )


def test_contribution_shares_are_signed(df: pd.DataFrame) -> None:
    """A channel that rose while the store fell is not credited with the fall."""
    result = contribution_by_dimension(df, "revenue", "2026-08-21", "channel")
    by_label = {segment["label"]: segment for segment in result["segments"]}

    assert by_label["channel=organic"]["change"] < 0
    assert by_label["channel=organic"]["share_of_change"] > 1, "organic holds more than the whole drop"
    assert by_label["channel=paid"]["change"] > 0
    assert by_label["channel=paid"]["share_of_change"] < 0, "paid rose, so it cannot explain a fall"


def test_ratio_decomposition_adds_up_exactly(df: pd.DataFrame) -> None:
    """Sessions, conversion, and AOV account for the revenue change with no remainder."""
    for day in ("2026-08-21", "2026-09-16", "2026-09-04"):
        result = ratio_decomposition(df, day)
        total = sum(effect["effect"] for effect in result["effects"])  # type: ignore[misc]
        assert total == pytest.approx(result["revenue_change"]), f"effects do not add up on {day}"
        assert [effect["factor"] for effect in result["effects"]] == ["sessions", "conversion", "aov"]


def test_ratio_decomposition_explains_the_conversion_collapse(df: pd.DataFrame) -> None:
    """On the paid bot-flood day, the sessions surge and the conversion collapse cancel."""
    result = ratio_decomposition(df, "2026-09-16")
    effects = {effect["factor"]: effect["effect"] for effect in result["effects"]}  # type: ignore[misc]

    assert effects["sessions"] > 0, "sessions surged"
    assert effects["conversion"] < 0, "conversion collapsed"

    # Each effect dwarfs the net movement, which is the point of the anomaly:
    # revenue barely moved because the two large effects offset one another.
    net = effects["sessions"] + effects["conversion"]
    assert abs(net) < 0.05 * abs(effects["sessions"])
    assert abs(effects["conversion"]) > 0.9 * abs(effects["sessions"])


def test_baselines_are_weekday_aligned(df: pd.DataFrame) -> None:
    """A day is compared against the same weekday, never a mixed bag of days."""
    for day_text in ("2026-08-21", "2026-09-16", "2026-04-08"):
        day = date.fromisoformat(day_text)
        matched = baseline_dates(day)
        assert len(matched) == 2
        assert all((day - each).days % 7 == 0 for each in matched)
        assert all(each < day for each in matched)

    result = compare_windows(df, "revenue", "2026-08-21")
    observed = result["baseline_observations"]
    assert [each["date"] for each in observed] == result["baseline_dates"]

    recorded = df.loc[df["date"] == pd.Timestamp("2026-08-21"), "revenue"].sum()
    assert result["current"] == pytest.approx(recorded)
    assert result["change"] == pytest.approx(result["current"] - result["baseline"])
    assert len(observed) == result["baseline_window_days"] / 7


def test_compare_windows_reports_a_falling_metric(df: pd.DataFrame) -> None:
    """The store-level view agrees with the sign of the anomaly."""
    falling = compare_windows(df, "revenue", "2026-08-21")
    assert falling["change"] < 0
    assert falling["pct_change"] < 0

    rising = compare_windows(df, "refunds", "2026-09-04")
    assert rising["change"] > 0


@pytest.mark.parametrize(
    "call",
    [
        lambda df: compare_windows(df, "revenue", "2026-08-21"),
        lambda df: contribution_by_dimension(df, "revenue", "2026-08-21", "channel"),
        lambda df: top_drivers(df, "revenue", "2026-08-21"),
        lambda df: ratio_decomposition(df, "2026-08-21"),
    ],
    ids=["compare_windows", "contribution_by_dimension", "top_drivers", "ratio_decomposition"],
)
def test_every_payload_survives_a_json_round_trip(df: pd.DataFrame, call: object) -> None:
    """The agent receives these as tool output, so they must be valid JSON.

    ``allow_nan=False`` is the point of the test: Python would happily write
    ``NaN``, which is not JSON and would break a strict client.
    """
    payload = call(df)  # type: ignore[operator]
    encoded = json.dumps(payload, allow_nan=False)
    assert json.loads(encoded) == payload


def test_unknown_dimension_is_rejected(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Unknown dimension"):
        contribution_by_dimension(df, "revenue", "2026-08-21", "colour")


def test_unknown_metric_is_rejected(df: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="margin"):
        compare_windows(df, "margin", "2026-08-21")


def test_a_day_with_no_data_is_rejected(df: pd.DataFrame) -> None:
    missing = (date(2026, 4, 1) + timedelta(days=365)).isoformat()
    with pytest.raises(ValueError, match="No revenue recorded"):
        ratio_decomposition(df, missing)


def test_baseline_window_must_cover_a_whole_week() -> None:
    with pytest.raises(ValueError, match="at least 7 days"):
        baseline_dates(date(2026, 8, 21), baseline_window=3)
