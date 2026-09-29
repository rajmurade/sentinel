"""Anomaly detection for daily e-commerce metrics.

Pipeline: load a long-format dataset, aggregate it to one row per day, then flag
days where a metric leaves its seasonally adjusted expectation.

**Expected value.** An STL decomposition supplies the weekly seasonal component.
The level underneath it is estimated from the *trailing* window rather than from
a fit over the whole series, because a decomposition that sees a sustained level
shift absorbs it into its own trend. On this dataset a revenue break that costs
6.9% store-wide leaves a residual of almost nothing when the trend is fitted to
the whole series, so the break is invisible. Estimating the level from the days
*before* the break makes the break visible, which is the whole point.

**Two independent checks.**

* STL + robust z (primary). The residual against the expected value is centred
  and scaled by the median and the median absolute deviation of the trailing
  window, which the anomalies themselves cannot drag around the way a mean and
  standard deviation would. A day is flagged when ``|z|`` exceeds
  ``sensitivity``.
* IsolationForest (cross-check). A forest is fitted to the per-day vector of
  standardised residuals across all metrics, so it also catches days where
  several metrics move together even when no single one is extreme.

The forest votes on *days*, not metrics. On a day it flags, the metric with the
largest ``|z|`` carries the record, and ``method`` records which checks agreed.

The trailing window means the first ``window`` days of a series cannot be scored:
there is no history behind them to compare against.

Every number in an :class:`Anomaly` comes from the data or from a model fitted to
it. Nothing here is estimated for flavour or rounded to flatter.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL

REQUIRED_COLUMNS: list[str] = [
    "date",
    "region",
    "channel",
    "device",
    "product_category",
    "sessions",
    "orders",
    "revenue",
    "refunds",
]
SUM_COLUMNS: list[str] = ["sessions", "orders", "revenue", "refunds"]

# Checked in this order, so reports list revenue first.
METRICS: list[str] = ["revenue", "orders", "sessions", "conversion_rate", "refunds"]

SEASONAL_PERIOD: int = 7
TRAILING_WINDOW: int = 28
MAD_TO_SIGMA: float = 1.4826  # makes the MAD an estimate of sigma for normal data
MEDIUM_SEVERITY_MULTIPLE: float = 1.5
HIGH_SEVERITY_MULTIPLE: float = 2.5
SEVERITY_LOW: str = "low"
SEVERITY_MEDIUM: str = "medium"
SEVERITY_HIGH: str = "high"

METHOD_STL: str = "stl_mad"
METHOD_FOREST: str = "isolation_forest"
METHOD_BOTH: str = "stl_mad+isolation_forest"

ISOLATION_FOREST_TREES: int = 200
ISOLATION_FOREST_SEED: int = 0
DEFAULT_SENSITIVITY: float = 3.5
DEFAULT_CONTAMINATION: float = 0.02


class Anomaly(TypedDict):
    """One detected anomaly.

    ``value`` is what the metric did, ``expected`` is what the seasonally
    adjusted model predicted, and ``pct_change`` is the gap between them. All
    three come straight from the data.
    """

    date: date
    metric: str
    value: float
    expected: float
    pct_change: float | None
    z: float
    severity: str
    method: str


def load_data(path_or_df: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a long-format dataset, validate its schema, and parse the dates.

    Accepts either a path to a CSV or an already-loaded frame. An input frame is
    copied, never modified in place.
    """
    if isinstance(path_or_df, pd.DataFrame):
        frame = path_or_df.copy()
    else:
        path = Path(path_or_df)
        if not path.is_file():
            raise FileNotFoundError(f"No dataset found at {path}")
        frame = pd.read_csv(path)

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no rows")

    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    for column in SUM_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    return frame.sort_values("date").reset_index(drop=True)


def daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the long dataset to one row per day, indexed by date.

    Sums are additive, so the additive columns are summed. ``conversion_rate``
    is a ratio and is recomputed from the daily totals rather than averaged,
    which would weight a quiet day the same as a busy one.
    """
    daily = (
        df.groupby("date", as_index=True)
        .agg(
            revenue=("revenue", "sum"),
            orders=("orders", "sum"),
            sessions=("sessions", "sum"),
            refunds=("refunds", "sum"),
        )
        .sort_index()
    )
    daily["conversion_rate"] = daily["orders"] / daily["sessions"].replace(0, np.nan)
    return daily


def stl_seasonal(series: pd.Series, period: int = SEASONAL_PERIOD) -> np.ndarray:
    """Weekly seasonal component of a daily series, from a robust STL fit."""
    values = series.to_numpy(dtype=float)
    if len(values) < 2 * period:
        raise ValueError(f"Need at least {2 * period} days to decompose, got {len(values)}")
    if not np.isfinite(values).all():
        raise ValueError("Cannot decompose a series containing missing or infinite values")
    return np.asarray(STL(values, period=period, robust=True).fit().seasonal, dtype=float)


def expected_values(series: pd.Series, seasonal: np.ndarray, window: int = TRAILING_WINDOW) -> np.ndarray:
    """Value each day is expected to take, given only the days before it.

    A straight line is fitted to the ``window`` preceding days with the seasonal
    component removed, then extrapolated one step and the seasonal component for
    the current day added back. The fit never sees the day it predicts, so a
    sustained level shift stays visible instead of being absorbed.

    The first ``window`` days are ``nan``: they have no history to predict from.
    """
    values = series.to_numpy(dtype=float)
    if window < 1:
        raise ValueError(f"Trailing window must be at least 1 day, got {window}")
    deseasonalised = values - seasonal
    expected = np.full(len(values), np.nan)
    offsets = np.arange(window, dtype=float)
    for position in range(window, len(values)):
        slope, intercept = np.polyfit(offsets, deseasonalised[position - window:position], 1)
        expected[position] = intercept + slope * window + seasonal[position]
    return expected


def median_absolute_deviation(values: np.ndarray) -> float:
    """Median distance from the median, the robust stand-in for a spread."""
    return float(np.median(np.abs(values - np.median(values))))


def robust_z_scores(residuals: np.ndarray, window: int = TRAILING_WINDOW) -> np.ndarray:
    """Standardise residuals by the median and MAD of the trailing window.

    Both the centre and the spread come from the same trailing window as the
    prediction, so a day is compared against how the metric has been behaving
    lately rather than against an average that recent anomalies have already
    shifted. Days with too little history, or no spread at all to measure
    against, score zero.
    """
    series = pd.Series(residuals)
    centre = series.rolling(window, min_periods=window).median().to_numpy()
    spread = MAD_TO_SIGMA * series.rolling(window, min_periods=window).apply(median_absolute_deviation, raw=True).to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (residuals - centre) / spread
    return np.where(np.isfinite(z), z, 0.0)


def isolation_forest_flags(residual_matrix: np.ndarray, contamination: float = DEFAULT_CONTAMINATION) -> np.ndarray:
    """True for each day the forest treats as an outlier.

    The input is one column per metric, already standardised, so the forest sees
    the seasonally adjusted metric vector rather than the raw trend and weekday
    swing that dominate it.
    """
    scaled = StandardScaler().fit_transform(residual_matrix)
    forest = IsolationForest(
        n_estimators=ISOLATION_FOREST_TREES,
        contamination=contamination,
        random_state=ISOLATION_FOREST_SEED,
    )
    forest.fit(scaled)
    return np.asarray(forest.predict(scaled) == -1, dtype=bool)


def severity_for(z: float, sensitivity: float) -> str:
    """Band an absolute z-score into low, medium, or high.

    The bands are multiples of the sensitivity, so they hold at any setting: a
    day only has to clear the bar to be flagged, and clearing it by half again
    is what makes something medium.
    """
    magnitude = abs(z)
    if magnitude >= HIGH_SEVERITY_MULTIPLE * sensitivity:
        return SEVERITY_HIGH
    if magnitude >= MEDIUM_SEVERITY_MULTIPLE * sensitivity:
        return SEVERITY_MEDIUM
    return SEVERITY_LOW


def detect_anomalies(
    daily: pd.DataFrame,
    sensitivity: float = DEFAULT_SENSITIVITY,
    window: int = TRAILING_WINDOW,
    period: int = SEASONAL_PERIOD,
    contamination: float = DEFAULT_CONTAMINATION,
) -> list[Anomaly]:
    """Flag anomalous days in a daily metrics table.

    Returns one record per flagged ``(date, metric)`` pair, ordered by date and
    then by descending ``|z|``. The first ``window`` days are never flagged, as
    they carry no history to be compared against.
    """
    missing = [metric for metric in METRICS if metric not in daily.columns]
    if missing:
        raise ValueError(f"Daily metrics are missing columns: {', '.join(missing)}")

    seasonal = {metric: stl_seasonal(daily[metric], period) for metric in METRICS}
    expected = {metric: expected_values(daily[metric], seasonal[metric], window) for metric in METRICS}
    z_scores = {
        metric: robust_z_scores(daily[metric].to_numpy(dtype=float) - expected[metric], window)
        for metric in METRICS
    }
    forest_flags = isolation_forest_flags(
        np.column_stack([z_scores[metric] for metric in METRICS]), contamination
    )

    anomalies: list[Anomaly] = []
    for position, day in enumerate(daily.index):
        day_z = {metric: float(z_scores[metric][position]) for metric in METRICS}
        loudest = max(METRICS, key=lambda metric: abs(day_z[metric]))

        for metric in METRICS:
            z = day_z[metric]
            by_score = abs(z) > sensitivity
            by_forest = bool(forest_flags[position]) and metric == loudest
            if not by_score and not by_forest:
                continue

            if by_score and by_forest:
                method = METHOD_BOTH
            elif by_score:
                method = METHOD_STL
            else:
                method = METHOD_FOREST

            value = float(daily[metric].iloc[position])
            baseline = float(expected[metric][position])
            anomalies.append(
                Anomaly(
                    date=pd.Timestamp(day).date(),
                    metric=metric,
                    value=value,
                    expected=baseline,
                    pct_change=(value - baseline) / baseline if baseline != 0 else None,
                    z=z,
                    severity=severity_for(z, sensitivity),
                    method=method,
                )
            )

    return sorted(anomalies, key=lambda anomaly: (anomaly["date"], -abs(anomaly["z"])))


# How far apart two flagged days on the same metric may be and still be treated
# as one story. A gap of two days lets a metric stutter through a loud weekday
# without spawning a separate incident for it.
INCIDENT_MAX_GAP_DAYS: int = 2

# Order that decides which of several incidents is the most important.
SEVERITY_RANKING: dict[str, int] = {SEVERITY_HIGH: 0, SEVERITY_MEDIUM: 1, SEVERITY_LOW: 2}


class Incident(TypedDict):
    """A run of anomalous days on one metric, presented as one story.

    Days recorded within ``INCIDENT_MAX_GAP_DAYS`` of each other are grouped
    together. The peak day is the member that moved furthest from its seasonal
    expectation, by |pct_change| and then |z|.
    """

    metric: str
    start_date: date
    end_date: date
    days: int
    peak_date: date
    peak_value: float
    peak_expected: float
    peak_pct_change: float | None
    peak_z: float
    severity: str
    methods: list[str]
    records: list[Anomaly]


def _peak_key(anomaly: Anomaly) -> tuple[float, float]:
    """How loud one anomaly is: peak change first, then peak z-score."""
    return (abs(anomaly["pct_change"] or 0.0), abs(anomaly["z"]))


def _incident_severity(members: list[Anomaly]) -> str:
    """An incident's severity is its most severe day's severity."""
    return min(members, key=lambda anomaly: SEVERITY_RANKING[anomaly["severity"]])["severity"]


def group_incidents(anomalies: list[Anomaly], max_gap_days: int = INCIDENT_MAX_GAP_DAYS) -> list[Incident]:
    """Cluster the flagged days per metric into incidents.

    A flagged day joins the previous incident on its metric when it falls within
    ``max_gap_days`` of it, otherwise it starts a new one. Each cluster becomes
    an incident carrying its loudest day as the peak plus the union of detection
    methods across the run.

    Incidents are returned most severe first; ties are broken by how far the
    peak moved from expectation.
    """
    clusters_by_metric: dict[str, list[list[Anomaly]]] = {metric: [] for metric in METRICS}
    for anomaly in sorted(anomalies, key=lambda item: item["date"]):
        clusters = clusters_by_metric[anomaly["metric"]]
        if clusters and (anomaly["date"] - clusters[-1][-1]["date"]).days <= max_gap_days:
            clusters[-1].append(anomaly)
        else:
            clusters.append([anomaly])

    incidents: list[Incident] = []
    for clusters in clusters_by_metric.values():
        for members in clusters:
            peak = max(members, key=_peak_key)
            incidents.append(
                Incident(
                    metric=members[0]["metric"],
                    start_date=min(member["date"] for member in members),
                    end_date=max(member["date"] for member in members),
                    days=len(members),
                    peak_date=peak["date"],
                    peak_value=peak["value"],
                    peak_expected=peak["expected"],
                    peak_pct_change=peak["pct_change"],
                    peak_z=peak["z"],
                    peak_method=peak["method"],
                    severity=_incident_severity(members),
                    methods=sorted({member["method"] for member in members}),
                    records=sorted(members, key=lambda member: member["date"]),
                )
            )

    incidents.sort(
        key=lambda incident: (
            SEVERITY_RANKING[incident["severity"]],
            -abs(incident["peak_pct_change"] or 0.0),
            -abs(incident["peak_z"]),
        )
    )
    return incidents
