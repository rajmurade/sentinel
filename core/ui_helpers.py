"""Display helpers for the Streamlit UI.

Pure functions only — no streamlit imports, so this module stays importable and
testable outside the app. Everything in here turns a number or a drilldown
payload into a figure, a styled fragment of HTML, or a display-ready table.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from core.detect import METRICS

SEVERITY_COLORS: dict[str, str] = {
    "high": "#c0392b",
    "medium": "#d68910",
    "low": "#2471a3",
}

COLOR_FLAGGED: str = "#c0392b"
COLOR_LINE: str = "#1f6feb"
COLOR_RISE: str = "#1e8449"
COLOR_FALL: str = "#c0392b"
COLOR_TOTAL: str = "#5d6d7e"


def badge_html(severity: str) -> str:
    """A rounded severity pill for the incident card."""
    background = SEVERITY_COLORS[severity]
    return (
        f'<span style="background-color:{background};color:white;'
        f'padding:1px 9px;border-radius:11px;font-size:0.85em;'
        f'font-weight:600;white-space:nowrap;">{severity.upper()}</span>'
    )


def format_pct(value: float | None) -> str:
    """A signed percentage from a fraction, or a dash when it is missing."""
    if value is None:
        return "n/a"
    return f"{value:+.1%}"


def _compact(value: float | None, width: int = 0) -> str:
    """A number with thousands separators, or a dash when it is missing."""
    if value is None:
        return "n/a"
    return f"{value:,.{width}f}"


def format_value(value: float | None, decimals: int = 2) -> str:
    """Display-ready number (thousands separators, fixed decimals)."""
    return _compact(value, decimals)


def _signed(value: float | None) -> str:
    """A signed number with thousands separators, or a dash when missing."""
    if value is None:
        return "n/a"
    return f"{value:+,.0f}"


def metric_title(metric: str) -> str:
    """'conversion_rate' -> 'Conversion Rate'."""
    return metric.replace("_", " ").title()


def _hover(message: str) -> str:
    return f"%{{x|%Y-%m-%d}}<br>{message}<extra></extra>"


def metric_chart(daily: pd.DataFrame, metric: str, flagged_dates: list[Any], height: int = 240) -> go.Figure:
    """One line chart per metric, red markers where days were flagged.

    ``daily`` is the index-by-date metrics table. ``flagged_dates`` are the
    ``date`` objects of every flagged day for this metric, taken from the
    incident records, so the markers line up with the values they flag.
    """
    series = daily[metric].dropna()
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines",
            line={"color": COLOR_LINE, "width": 2},
            hovertemplate=_hover(f"{metric}: %{{y:,.2f}}"),
        )
    )

    unique_dates = list(dict.fromkeys(flagged_dates))
    if unique_dates:
        stamp = pd.to_datetime(unique_dates)
        figure.add_trace(
            go.Scatter(
                x=stamp,
                y=daily.loc[stamp, metric],
                mode="markers",
                marker={"color": COLOR_FLAGGED, "size": 9, "symbol": "circle"},
                hovertemplate=_hover(f"flagged {metric}: %{{y:,.2f}}"),
            )
        )

    figure.update_layout(
        title=metric_title(metric),
        height=height,
        showlegend=False,
        margin={"l": 60, "r": 16, "t": 40, "b": 30},
        yaxis_title=metric,
    )
    return figure


def contribution_chart(payload: dict[str, Any], height: int = 320) -> go.Figure:
    """A signed bar of every segment's contribution to the total change."""
    segments = payload["segments"]
    changes = [(segment["change"] or 0.0) for segment in segments]
    labels = [segment["label"] for segment in segments]
    colors = [COLOR_FALL if change < 0 else COLOR_RISE for change in changes]

    figure = go.Figure(
        go.Bar(
            x=labels,
            y=changes,
            marker_color=colors,
            text=[_signed(change) for change in changes],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:+,.0f}<extra></extra>",
        )
    )
    net_change = payload.get("total_change")
    if isinstance(net_change, float):
        figure.add_hline(
            y=net_change,
            line={"dash": "dash", "color": COLOR_TOTAL},
            annotation_text=f"net {_signed(net_change)}",
            annotation_position="bottom right",
        )

    figure.update_layout(
        title=f"Change by {metric_title(str(payload['dimension']))} ({payload['metric']})",
        height=height,
        margin={"l": 60, "r": 16, "t": 40, "b": 60},
        yaxis_title="change vs baseline",
    )
    return figure


def drivers_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    """The top drivers as a display-only table, totals on the last row."""
    rows: list[dict[str, str]] = []
    for driver in payload["drivers"]:
        rows.append(
            {
                "rank": str(driver["rank"]),
                "segment": driver["label"],
                "current": _compact(driver["current"], 2),
                "baseline": _compact(driver["baseline"], 2),
                "change": _compact(driver["change"], 2),
                "pct_change": format_pct(driver["pct_change"]),
                "share": format_pct(driver["share_of_change"]),
            }
        )

    baseline_total = payload["baseline_total"]
    current_total = payload["current_total"]
    total_change = payload["total_change"]
    total_pct: float | None = None
    if isinstance(baseline_total, float) and isinstance(current_total, float) and baseline_total != 0:
        total_pct = (current_total - baseline_total) / baseline_total
    rows.append(
        {
            "rank": "",
            "segment": "TOTAL",
            "current": _compact(current_total if isinstance(current_total, float) else None, 2),
            "baseline": _compact(baseline_total if isinstance(baseline_total, float) else None, 2),
            "change": _compact(total_change if isinstance(total_change, float) else None, 2),
            "pct_change": format_pct(total_pct),
            "share": "",
        }
    )

    return pd.DataFrame(rows, columns=["rank", "segment", "current", "baseline", "change", "pct_change", "share"])


def waterfall_chart(payload: dict[str, Any], height: int = 320) -> go.Figure:
    """Baseline revenue, the sessions/conversion/AOV effects, and the current total."""
    baseline = payload["baseline"]["revenue"]
    current = payload["current"]["revenue"]
    effects = payload["effects"]

    x_labels = ["Baseline", *(effect["factor"].title() for effect in effects), "Current"]
    y_values: list[float | None] = [baseline, *(effect["effect"] for effect in effects), current]
    measures = ["absolute", "relative", "relative", "relative", "total"]

    figure = go.Figure(
        go.Waterfall(
            x=x_labels,
            y=y_values,
            measure=measures,
            text=[_signed(value) for value in y_values],
            textposition="outside",
            connector={"line": {"color": COLOR_TOTAL}},
            increasing={"marker": {"color": COLOR_RISE}},
            decreasing={"marker": {"color": COLOR_FALL}},
            totals={"marker": {"color": COLOR_TOTAL}},
            hovertemplate="%{x}<br>%{y:+,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Revenue change breakdown",
        height=height,
        margin={"l": 60, "r": 16, "t": 40, "b": 40},
        yaxis_title="revenue",
    )
    return figure


def friendly_error(exc: Exception) -> str:
    """A human-readable line for whatever went wrong during a run."""
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return f"Problem with the dataset: {exc}"
    if isinstance(exc, FileNotFoundError):
        return f"Dataset not found: {exc}"
    return f"Something went wrong running the watchdog: {exc}"


def flagged_dates_by_metric(incidents: list[dict[str, Any]]) -> dict[str, list[Any]]:
    """Every flagged day per metric, flattened from the incident records."""
    flagged: dict[str, list[Any]] = {metric: [] for metric in METRICS}
    for incident in incidents:
        for record in incident["records"]:
            flagged[incident["metric"]].append(record["date"])
    return flagged


def pct_change_direction(pct_change: float | None) -> str:
    """'Down' or 'Up' for an incident's move, matching the sign of the change."""
    if pct_change is None:
        return "n/a"
    return "Down" if pct_change < 0 else "Up"