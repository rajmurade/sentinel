"""Incident narration.

Part 1: tool evidence and the deterministic template report, no LLM.

``build_evidence`` collects everything the narration is allowed to say — the
store-level comparison, the top drivers, the revenue split — from the pure
drilldown tools in :mod:`core.rootcause`. ``template_report`` turns that
evidence into a report with a strict rule: **every number in the text comes
straight from the evidence dict**. There are no invented, interpolated, or
estimated figures.

The wording rule for interpretation: a behavioural reading is always couched in
"consistent with" or "may indicate" (a traffic surge with collapsing conversion
is "consistent with low-quality traffic"; it is never called a bot flood or a
tracking bug).

The LLM narration layer is built on top of this in part 2. Rule 4 in
``CONTEXT.md`` applies there: if the LLM call fails, times out, or has no API
key, ``template_report`` is what the app falls back to, so it must never raise.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from core import rootcause
from core.detect import Incident


def _number(value: float | None) -> str:
    """A number with thousands separators, or a plain hedge when missing."""
    if value is None:
        return "no recorded value"
    return f"{value:,.2f}"


def _percent(value: float | None) -> str:
    """A signed percentage from a fraction, hedged when missing."""
    if value is None:
        return "unknown"
    return f"{value:+.1%}"


def _direction(pct_change: float | None) -> str:
    """'rose' or 'fell' from the sign of a change, hedged when missing."""
    if pct_change is None:
        return "moved"
    return "fell" if pct_change < 0 else "rose"


def _metric_title(metric: str) -> str:
    """'conversion_rate' -> 'Conversion Rate'."""
    return metric.replace("_", " ").title()


def build_evidence(df: pd.DataFrame, incident: Incident) -> dict[str, Any]:
    """Assemble every tool number available for one incident.

    Runs the store-level comparison, the top-driver search, and the revenue
    ratio decomposition from :mod:`core.rootcause` for the incident's metric on
    its peak day. ``df`` must be a long-format frame carrying the metric column
    (the app augments ``conversion_rate`` before handing the frame over).

    The result is plain JSON-serialisable data; the narration is not allowed to
    print anything that is not in here.
    """
    peak = incident["peak_date"].isoformat()
    return {
        "metric": incident["metric"],
        "peak_date": peak,
        "compare": rootcause.compare_windows(df, incident["metric"], peak),
        "drivers": rootcause.top_drivers(
            df, incident["metric"], peak, k=rootcause.DEFAULT_TOP_K
        ),
        "ratio": rootcause.ratio_decomposition(df, peak),
    }


def _evidence_lines(evidence: dict[str, Any]) -> list[str]:
    """The evidence[] rows: every number a report can quote, spelled out."""
    metric = evidence["metric"]
    peak = evidence["peak_date"]
    compare = evidence["compare"]
    lines = [
        f"{metric} on {peak}: observed {_number(compare['current'])} vs "
        f"{_number(compare['baseline'])} on the same weekday over the preceding "
        f"{compare['baseline_window_days']} days ({_percent(compare['pct_change'])})."
    ]

    for driver in evidence["drivers"]["drivers"]:
        lines.append(
            f"Driver {driver['rank']}: {driver['label']}, current "
            f"{_number(driver['current'])}, baseline {_number(driver['baseline'])}, "
            f"change {_number(driver['change'])} ({_percent(driver['pct_change'])}), "
            f"share of the total change {_percent(driver['share_of_change'])}."
        )

    effects = evidence["ratio"]["effects"]
    lines.append(
        f"Revenue on {peak} changed {_number(evidence['ratio']['revenue_change'])}: "
        f"sessions {_number(effects[0]['effect'])}, conversion "
        f"{_number(effects[1]['effect'])}, aov {_number(effects[2]['effect'])}."
    )
    return lines


def _recommended_actions(incident: Incident, evidence: dict[str, Any]) -> list[str]:
    """Deterministic checks, each tied to evidence and couched in hedged language."""
    metric = incident["metric"]
    peak = evidence["peak_date"]
    compare = evidence["compare"]
    effects = evidence["ratio"]["effects"]
    drivers = evidence["drivers"]["drivers"]
    driver_label = drivers[0]["label"] if drivers else "the affected segment"

    actions: list[str] = []

    if metric == "refunds":
        actions.append(
            f"A spike in refunds concentrated in {driver_label} may indicate a product "
            f"or fulfilment problem; inspect that flow around {peak}."
        )

    sessions_effect = effects[0]["effect"] or 0.0
    conversion_effect = effects[1]["effect"] or 0.0
    if metric == "revenue" and sessions_effect > 0 and conversion_effect < 0:
        actions.append(
            f"Session growth with collapsing conversion is consistent with low-quality "
            f"traffic; review the funnel behind {driver_label} around {peak}."
        )

    if compare["pct_change"] is not None:
        actions.append(
            f"A move of {_percent(compare['pct_change'])} against the weekday baseline "
            f"may indicate a lasting break rather than noise; double-check the {metric} "
            f"feed for {peak} before reacting."
        )

    if incident["severity"] == "high":
        actions.append(
            "Treat this as urgent: confirm the peak-day numbers before they propagate "
            "into downstream reporting."
        )

    actions.append(
        "Watch the following week: a lasting break and a one-off artefact need "
        "different responses."
    )
    return actions


def template_report(incident: Incident, evidence: dict[str, Any]) -> dict[str, str | list[str]]:
    """The deterministic report, built only from ``incident`` metadata and ``evidence``.

    The headline and severity come from the detection output carried by the
    incident; every attributed number in the text is read from ``evidence``.
    This function must never raise on malformed evidence — it is the fallback
    the app relies on when the LLM is unavailable (rule 4).
    """
    metric = evidence["metric"]
    peak = evidence["peak_date"]
    compare = evidence["compare"]

    if compare["pct_change"] is not None:
        what_changed = (
            f"On {peak} {_metric_title(metric)} {_direction(compare['pct_change'])} from "
            f"{_number(compare['baseline'])} to {_number(compare['current'])} against the "
            f"same weekday over the preceding {compare['baseline_window_days']} days — "
            f"{_percent(compare['pct_change'])}."
        )
    else:
        what_changed = (
            f"On {peak} {_metric_title(metric)} went from {_number(compare['baseline'])} "
            f"to {_number(compare['current'])} against the same weekday over the preceding "
            f"{compare['baseline_window_days']} days."
        )

    drivers = evidence["drivers"]["drivers"]
    if not drivers:
        likely_driver = "No single segment in the dimension cuts explained most of this change."
    else:
        driver = drivers[0]
        parts = [f"{driver['label']} is the likeliest driver"]
        if driver["share_of_change"] is not None:
            parts.append(
                f"it accounted for {_percent(driver['share_of_change'])} of the total change"
            )
        if driver["pct_change"] is not None:
            parts.append(
                f"moving {_percent(driver['pct_change'])} against its weekday baseline"
            )
        head, *tail = parts
        likely_driver = head + (": " + ", ".join(tail) if tail else "") + "."

    return {
        "headline": f"{_metric_title(metric)} {_direction(compare['pct_change'])} on {peak}",
        "severity": incident["severity"],
        "what_changed": what_changed,
        "likely_driver": likely_driver,
        "evidence": _evidence_lines(evidence),
        "recommended_actions": _recommended_actions(incident, evidence),
    }