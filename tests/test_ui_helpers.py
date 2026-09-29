"""Tests for core.ui_helpers, the display layer of the Streamlit app.

Everything here must work without streamlit running: the helpers are pure
functions that turn numbers or drilldown payloads into figures, tables, and
HTML fragments.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from core import rootcause
from core.detect import METRICS, daily_metrics, load_data
from core.ui_helpers import (
    badge_html,
    contribution_chart,
    drivers_dataframe,
    flagged_dates_by_metric,
    format_pct,
    metric_chart,
    waterfall_chart,
)

DATA_DIR: Path = Path(__file__).resolve().parents[1] / "data"
SAMPLE_CSV: Path = DATA_DIR / "sample.csv"


@pytest.fixture(scope="module")
def daily() -> pd.DataFrame:
    return daily_metrics(load_data(SAMPLE_CSV))


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    source = load_data(SAMPLE_CSV).copy()
    sessions = source["sessions"].where(source["sessions"] != 0)
    source["conversion_rate"] = source["orders"] / sessions
    return source


def test_badge_html_marks_the_severity() -> None:
    html = badge_html("high")
    assert "HIGH" in html
    assert "background-color" in html
    assert "border-radius" in html


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.1234, "+12.3%"), (-0.05, "-5.0%"), (1.0, "+100.0%"), (None, "n/a")],
)
def test_format_pct(value: float | None, expected: str) -> None:
    assert format_pct(value) == expected


def test_metric_chart_draws_a_flag_marker(daily: pd.DataFrame) -> None:
    plain = metric_chart(daily, "revenue", [])
    assert len(plain.data) == 1

    flagged = metric_chart(daily, "revenue", [date(2026, 9, 4), date(2026, 9, 4)])
    assert len(flagged.data) == 2
    # Duplicate dates must not produce duplicate markers.
    assert len(flagged.data[1].x) == 1


def test_contribution_chart_builds_from_a_drilldown_payload(frame: pd.DataFrame) -> None:
    payload = rootcause.contribution_by_dimension(frame, "revenue", "2026-09-16", "channel")
    figure = contribution_chart(payload)
    assert len(figure.data) == 1
    assert len(figure.data[0].x) == len(payload["segments"])


def test_drivers_dataframe_ends_on_a_total_row(frame: pd.DataFrame) -> None:
    payload = rootcause.top_drivers(frame, "revenue", "2026-09-16", k=3)
    table = drivers_dataframe(payload)
    assert table.shape[0] == 4
    assert list(table["segment"])[-1] == "TOTAL"
    assert list(table.columns) == [
        "rank",
        "segment",
        "current",
        "baseline",
        "change",
        "pct_change",
        "share",
    ]


def test_waterfall_chart_builds_from_the_ratio_decomposition(frame: pd.DataFrame) -> None:
    payload = rootcause.ratio_decomposition(frame, "2026-09-16")
    figure = waterfall_chart(payload)
    assert len(figure.data) == 1
    assert list(figure.data[0].x) == ["Baseline", "Sessions", "Conversion", "Aov", "Current"]


def test_flagged_dates_by_metric_initialises_every_metric() -> None:
    incidents: list[dict[str, Any]] = [
        {"metric": "revenue", "records": [{"date": date(2026, 9, 4)}]}
    ]
    flagged = flagged_dates_by_metric(incidents)
    assert set(flagged) == set(METRICS)
    assert flagged["revenue"] == [date(2026, 9, 4)]
    assert flagged["sessions"] == []