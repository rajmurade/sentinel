"""Root-cause drilldown over the long-format metrics dataset.

Pure functions, no model calls. Everything the narration agent is allowed to
say about causes has to come from here, so every function returns plain
JSON-serialisable data with the arithmetic spelled out: the current value, the
baseline it is compared against, the difference, and the share of the total
change. Nothing is rounded or estimated on the way out.

**Baselines are weekday-aligned.** A day is compared against the same weekday in
each of the preceding weeks, not against every day in the window. A Saturday is
compared with previous Saturdays, so a Tuesday is never measured against a
weekend-heavy average and made to look broken.

Contribution is signed. A segment's share is its change divided by the total
change, so a share above 1 means the segment moved further than the net total,
which is how one segment falling while another rises becomes visible. Ranking
uses the signed share rather than its absolute value, so a segment moving
*against* the direction of the total change is never offered as a cause of it.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from itertools import combinations
from typing import TypedDict

import pandas as pd

SEGMENT_DIMENSIONS: list[str] = ["region", "channel", "device", "product_category"]
DATE_COLUMN: str = "date"
REVENUE_COLUMN: str = "revenue"
ORDERS_COLUMN: str = "orders"
SESSIONS_COLUMN: str = "sessions"

DEFAULT_BASELINE_WINDOW: int = 14
DEFAULT_TOP_K: int = 3
WEEK_DAYS: int = 7
# Share of the total change at which a segment counts as explaining the change
# rather than merely contributing to it.
COVERAGE_THRESHOLD: float = 0.9


class SegmentChange(TypedDict):
    """One segment's movement and its share of the total movement."""

    segment: dict[str, str]
    label: str
    current: float | None
    baseline: float | None
    change: float | None
    pct_change: float | None
    share_of_change: float | None


class Driver(SegmentChange):
    """A segment found by a particular cut of the data."""

    dimensions: list[str]
    rank: int


def _as_date(value: str | date | pd.Timestamp) -> date:
    """Normalise anything date-like to a plain ``date``."""
    return pd.Timestamp(value).date()


def _as_float(value: float | None) -> float | None:
    """Coerce to a JSON-safe float, mapping NaN and infinity to ``None``."""
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")


def baseline_dates(day: date, baseline_window: int = DEFAULT_BASELINE_WINDOW) -> list[date]:
    """The same weekday in each preceding week back to ``baseline_window`` days.

    With the default 14-day window this is two earlier occurrences of the same
    weekday, so the comparison is like for like.
    """
    if baseline_window < WEEK_DAYS:
        raise ValueError(
            f"baseline_window must be at least {WEEK_DAYS} days to match weekdays, got {baseline_window}"
        )
    offsets = range(WEEK_DAYS, baseline_window + 1, WEEK_DAYS)
    return sorted(day - timedelta(days=offset) for offset in offsets)


def _daily_totals(df: pd.DataFrame, metric: str, days: list[date]) -> pd.Series:
    """Store-wide total of a metric for each of the given days."""
    wanted = [pd.Timestamp(day) for day in days]
    rows = df[df[DATE_COLUMN].isin(wanted)]
    if rows.empty:
        return pd.Series(dtype=float, name=metric)
    return rows.groupby(DATE_COLUMN)[metric].sum()


def _segment_changes(
    df: pd.DataFrame,
    metric: str,
    day: date,
    dimensions: list[str],
    baseline: list[date],
) -> tuple[list[SegmentChange], float, float, float]:
    """Per-segment current value, baseline, change, and share of the change.

    Returns the segment rows, the current total, the baseline total, and the
    total change. Totals are summed over the segments actually present, so the
    segment changes add up to the total change exactly.
    """
    wanted = [pd.Timestamp(each) for each in [day, *baseline]]
    rows = df[df[DATE_COLUMN].isin(wanted)]
    if rows.empty:
        return [], 0.0, 0.0, 0.0

    pivot = rows.groupby([*dimensions, DATE_COLUMN], as_index=False)[metric].sum()
    current = pivot[pivot[DATE_COLUMN] == pd.Timestamp(day)].set_index(dimensions)[metric]
    earlier = pivot[pivot[DATE_COLUMN] != pd.Timestamp(day)].groupby(dimensions)[metric].mean()

    merged = pd.concat([current.rename("current"), earlier.rename("baseline")], axis=1).fillna(0.0)

    current_total = float(merged["current"].sum())
    baseline_total = float(merged["baseline"].sum())
    total_change = current_total - baseline_total

    segments: list[SegmentChange] = []
    for key, row in merged.iterrows():
        segment = dict(zip(dimensions, key if isinstance(key, tuple) else (key,)))
        change = float(row["current"]) - float(row["baseline"])
        segments.append(
            SegmentChange(
                segment=segment,
                label=", ".join(f"{name}={value}" for name, value in segment.items()),
                current=_as_float(row["current"]),
                baseline=_as_float(row["baseline"]),
                change=_as_float(change),
                pct_change=_as_float(change / float(row["baseline"])) if float(row["baseline"]) != 0 else None,
                share_of_change=_as_float(change / total_change) if total_change != 0 else None,
            )
        )
    return segments, current_total, baseline_total, total_change


def compare_windows(
    df: pd.DataFrame,
    metric: str,
    date: str | date | pd.Timestamp,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
) -> dict[str, object]:
    """Compare a metric on one day against the same weekday in recent weeks.

    The store-level view that the per-dimension and per-factor cuts hang off, so
    a caller can check the headline change before trusting a drilldown.
    """
    _require_columns(df, [DATE_COLUMN, metric])
    day = _as_date(date)
    baseline = baseline_dates(day, baseline_window)
    totals = _daily_totals(df, metric, [day, *baseline])

    current = _as_float(totals.get(pd.Timestamp(day)))
    if current is None:
        raise ValueError(f"No {metric} recorded on {day.isoformat()}")
    baseline_total = _as_float(totals.drop(index=pd.Timestamp(day)).mean())
    change = current - baseline_total if baseline_total is not None else None

    return {
        "metric": metric,
        "date": day.isoformat(),
        "baseline_window_days": baseline_window,
        "baseline_dates": [each.isoformat() for each in baseline],
        "current": current,
        "baseline": baseline_total,
        "change": _as_float(change),
        "pct_change": _as_float(change / baseline_total) if change is not None and baseline_total else None,
        "baseline_observations": [
            {"date": each.isoformat(), "value": _as_float(totals.get(pd.Timestamp(each)))} for each in baseline
        ],
    }


def contribution_by_dimension(
    df: pd.DataFrame,
    metric: str,
    date: str | date | pd.Timestamp,
    dim: str,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
) -> dict[str, object]:
    """Each segment of one dimension: its change and its share of the total change."""
    _require_columns(df, [DATE_COLUMN, metric])
    if dim not in SEGMENT_DIMENSIONS:
        raise ValueError(f"Unknown dimension {dim!r}; expected one of {', '.join(SEGMENT_DIMENSIONS)}")

    day = _as_date(date)
    baseline = baseline_dates(day, baseline_window)
    segments, current_total, baseline_total, total_change = _segment_changes(df, metric, day, [dim], baseline)
    segments.sort(key=lambda segment: segment["change"] or 0.0, reverse=True)

    return {
        "metric": metric,
        "date": day.isoformat(),
        "dimension": dim,
        "baseline_window_days": baseline_window,
        "baseline_dates": [each.isoformat() for each in baseline],
        "current_total": _as_float(current_total),
        "baseline_total": _as_float(baseline_total),
        "total_change": _as_float(total_change),
        "segments": segments,
    }


def _rankable(share: float | None) -> float:
    """Sort key for a share that may be missing, treating unknown as worst."""
    return float(share) if share is not None else float("-inf")


def _driver_sort_key(candidate: Driver) -> tuple[int, int, float, str]:
    """Rank the smallest explanation that accounts for the change.

    A coarse cut always concentrates more of a net change than a fine one: every
    channel of ``device=mobile`` sits inside that segment, so it absorbs the
    rising and falling parts of the same movement and its share runs above 1.
    Ranking purely on share therefore hands back the vaguest cut available.

    So candidates that account for essentially the whole change are ranked first,
    and among those the most specific cut wins, which is the useful answer. Only
    once those are exhausted does ranking fall back to share, so the remaining
    slots report whatever smaller movements exist.
    """
    share = candidate["share_of_change"] or 0.0
    sufficient = share >= COVERAGE_THRESHOLD
    return (
        0 if sufficient else 1,
        -len(candidate["dimensions"]) if sufficient else 0,
        -share,
        candidate["label"],
    )


def _is_generalisation(segment: dict[str, str], accepted: list[Driver]) -> bool:
    """True when a segment is a coarser view of a driver already selected.

    A driver whose every filter already appears in an accepted driver says
    nothing the accepted one has not already said, so it is skipped. A
    refinement is kept: narrowing a cause is new information.
    """
    for driver in accepted:
        if len(segment) >= len(driver["segment"]):
            continue
        if all(driver["segment"].get(dimension) == value for dimension, value in segment.items()):
            return True
    return False


def top_drivers(
    df: pd.DataFrame,
    metric: str,
    date: str | date | pd.Timestamp,
    k: int = DEFAULT_TOP_K,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
) -> dict[str, object]:
    """Search every single dimension and every pair, and rank the segments.

    Ten cuts are tried: four dimensions alone and six pairs. Each cut is scored
    by the share of the total change its worst segment accounts for, and the
    smallest segment accounting for the whole change is reported first. Cuts
    that merely restate an already-selected driver in fewer words are skipped.
    """
    _require_columns(df, [DATE_COLUMN, metric])
    day = _as_date(date)
    baseline = baseline_dates(day, baseline_window)
    cuts = [[dimension] for dimension in SEGMENT_DIMENSIONS] + [
        list(pair) for pair in combinations(SEGMENT_DIMENSIONS, 2)
    ]

    candidates: list[Driver] = []
    for dimensions in cuts:
        segments, _, _, _ = _segment_changes(df, metric, day, dimensions, baseline)
        if not segments:
            continue
        best = max(segments, key=lambda segment: _rankable(segment["share_of_change"]))
        candidates.append(
            Driver(
                dimensions=dimensions,
                segment=best["segment"],
                label=best["label"],
                current=best["current"],
                baseline=best["baseline"],
                change=best["change"],
                pct_change=best["pct_change"],
                share_of_change=best["share_of_change"],
                rank=0,
            )
        )

    candidates.sort(key=_driver_sort_key)
    drivers: list[Driver] = []
    for candidate in candidates:
        if len(drivers) >= k:
            break
        if _is_generalisation(candidate["segment"], drivers):
            continue
        candidate["rank"] = len(drivers) + 1
        drivers.append(candidate)

    totals = _daily_totals(df, metric, [day, *baseline])
    current_total = _as_float(totals.get(pd.Timestamp(day)))
    baseline_total = _as_float(totals.drop(index=pd.Timestamp(day)).mean())

    return {
        "metric": metric,
        "date": day.isoformat(),
        "baseline_window_days": baseline_window,
        "baseline_dates": [each.isoformat() for each in baseline],
        "cuts_searched": len(cuts),
        "current_total": current_total,
        "baseline_total": baseline_total,
        "total_change": _as_float(current_total - baseline_total)
        if current_total is not None and baseline_total is not None
        else None,
        "drivers": drivers,
    }


def ratio_decomposition(
    df: pd.DataFrame,
    date: str | date | pd.Timestamp,
    baseline_window: int = DEFAULT_BASELINE_WINDOW,
) -> dict[str, object]:
    """Split the revenue change into sessions, conversion, and AOV effects.

    Revenue is sessions x conversion x AOV, so the change is split by
    substituting one factor at a time. The three effects sum to the revenue
    change exactly, with no residual term.
    """
    _require_columns(df, [DATE_COLUMN, SESSIONS_COLUMN, ORDERS_COLUMN, REVENUE_COLUMN])
    day = _as_date(date)
    baseline = baseline_dates(day, baseline_window)
    totals = _daily_totals(df, REVENUE_COLUMN, [day, *baseline])

    if pd.Timestamp(day) not in totals.index:
        raise ValueError(f"No revenue recorded on {day.isoformat()}")
    sessions_now = float(_daily_totals(df, SESSIONS_COLUMN, [day]).get(pd.Timestamp(day), 0.0))
    orders_now = float(_daily_totals(df, ORDERS_COLUMN, [day]).get(pd.Timestamp(day), 0.0))
    revenue_now = float(totals.loc[pd.Timestamp(day)])

    sessions_before = float(_daily_totals(df, SESSIONS_COLUMN, baseline).mean())
    orders_before = float(_daily_totals(df, ORDERS_COLUMN, baseline).mean())
    revenue_before = float(totals.drop(index=pd.Timestamp(day)).mean())

    conversion_now = orders_now / sessions_now if sessions_now else 0.0
    conversion_before = orders_before / sessions_before if sessions_before else 0.0
    aov_now = revenue_now / orders_now if orders_now else 0.0
    aov_before = revenue_before / orders_before if orders_before else 0.0

    sessions_effect = (sessions_now - sessions_before) * conversion_before * aov_before
    conversion_effect = sessions_now * (conversion_now - conversion_before) * aov_before
    aov_effect = sessions_now * conversion_now * (aov_now - aov_before)
    revenue_change = revenue_now - revenue_before

    def _effect(factor: str, effect: float) -> dict[str, object]:
        return {
            "factor": factor,
            "effect": _as_float(effect),
            "share_of_change": _as_float(effect / revenue_change) if revenue_change != 0 else None,
            "pct_of_baseline_revenue": _as_float(effect / revenue_before) if revenue_before != 0 else None,
        }

    return {
        "date": day.isoformat(),
        "baseline_window_days": baseline_window,
        "baseline_dates": [each.isoformat() for each in baseline],
        "current": {
            "sessions": _as_float(sessions_now),
            "orders": _as_float(orders_now),
            "conversion_rate": _as_float(conversion_now),
            "aov": _as_float(aov_now),
            "revenue": _as_float(revenue_now),
        },
        "baseline": {
            "sessions": _as_float(sessions_before),
            "orders": _as_float(orders_before),
            "conversion_rate": _as_float(conversion_before),
            "aov": _as_float(aov_before),
            "revenue": _as_float(revenue_before),
        },
        "revenue_change": _as_float(revenue_change),
        "effects": [
            _effect("sessions", sessions_effect),
            _effect("conversion", conversion_effect),
            _effect("aov", aov_effect),
        ],
    }
