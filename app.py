"""Sentinel — AI metric watchdog UI.

Runs the detection and root-cause drilldown pipeline over the sample dataset or
an uploaded CSV, and renders one chart per metric plus incident cards with a
drilldown panel. The LLM narration, memory, and alert steps are intentionally
not wired in here yet.
"""

from __future__ import annotations

import io
import runpy
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core import rootcause, ui_helpers
from core.agent import build_evidence, template_report
from core.detect import (
    DEFAULT_SENSITIVITY,
    METRICS,
    Incident,
    daily_metrics,
    detect_anomalies,
    group_incidents,
    load_data,
)

ROOT: Path = Path(__file__).resolve().parent
SAMPLE_CSV: Path = ROOT / "data" / "sample.csv"
GENERATOR: Path = ROOT / "data" / "generate.py"

DIM_LABELS: dict[str, str] = {
    "region": "By region",
    "channel": "By channel",
    "device": "By device",
    "product_category": "By category",
}

st.set_page_config(page_title="Sentinel", layout="wide")


def ensure_sample_data() -> None:
    """Generate the sample dataset on first launch when it is missing."""
    if SAMPLE_CSV.is_file():
        return
    with redirect_stdout(io.StringIO()):
        runpy.run_path(str(GENERATOR), run_name="__main__")


@st.cache_data(show_spinner=False)
def load_dataset(source: str | bytes) -> pd.DataFrame:
    """Read the sample CSV or uploaded bytes and validate them."""
    if isinstance(source, bytes):
        if not source:
            raise ValueError("The uploaded file is empty")
        return load_data(pd.read_csv(io.BytesIO(source)))
    return load_data(source)


@st.cache_data(show_spinner=False)
def run_watchdog(daily: pd.DataFrame, sensitivity: float) -> list[Incident]:
    """Detect anomalies and group them into incidents."""
    return group_incidents(detect_anomalies(daily, sensitivity=sensitivity))


@st.cache_data(show_spinner=False)
def drilldown_dimension(frame: pd.DataFrame, metric: str, day: str, dim: str) -> dict[str, Any]:
    return rootcause.contribution_by_dimension(frame, metric, day, dim)


@st.cache_data(show_spinner=False)
def drilldown_drivers(frame: pd.DataFrame, metric: str, day: str) -> dict[str, Any]:
    return rootcause.top_drivers(frame, metric, day, k=3)


@st.cache_data(show_spinner=False)
def drilldown_ratio(frame: pd.DataFrame, day: str) -> dict[str, Any]:
    return rootcause.ratio_decomposition(frame, day)


@st.cache_data(show_spinner=False)
def incident_report(frame: pd.DataFrame, incident: Incident) -> dict[str, Any]:
    """Deterministic narrative from the drilldown tools (no LLM wired yet)."""
    return template_report(incident, build_evidence(frame, incident))


def augment_conversion_rate(frame: pd.DataFrame) -> pd.DataFrame:
    """Add per-segment conversion_rate (orders / sessions) for drilldowns."""
    copy = frame.copy()
    sessions = copy["sessions"].where(copy["sessions"] != 0)
    copy["conversion_rate"] = copy["orders"] / sessions
    return copy


def render_incident(incident: Incident, frame: pd.DataFrame) -> None:
    """One incident card; expanding it opens the root-cause drilldown."""
    metric = incident["metric"]
    direction = ui_helpers.pct_change_direction(incident["peak_pct_change"])
    peak_change = ui_helpers.format_pct(incident["peak_pct_change"])
    headline = (
        f"{direction} | {incident['start_date']} → {incident['end_date']} "
        f"| {incident['days']} day{'s' if incident['days'] != 1 else ''} | peak {peak_change}"
    )
    title = f"{ui_helpers.metric_title(metric)} — {headline}"

    left, pill = st.columns([0.06, 0.94])
    with left:
        st.markdown(ui_helpers.badge_html(incident["severity"]), unsafe_allow_html=True)
    with pill:
        with st.expander(title):
            peak = incident["peak_date"].isoformat()
            st.caption(
                f"Peak day {incident['peak_date']}: observed "
                f"{ui_helpers.format_value(incident['peak_value'])} vs expected "
                f"{ui_helpers.format_value(incident['peak_expected'])} "
                f"(z = {incident['peak_z']:.2f}, method {incident['peak_method']})."
            )

            tab_names = ["Report"]
            tab_names.extend(DIM_LABELS[dim] for dim in rootcause.SEGMENT_DIMENSIONS)
            tab_names.append("Top drivers")
            if metric == "revenue":
                tab_names.append("Revenue split")
            tabs = st.tabs(tab_names)

            for index, name in enumerate(tab_names):
                with tabs[index]:
                    if name == "Report":
                        report = incident_report(frame, incident)
                        st.markdown(f"**{report['headline']}**")
                        st.caption(f"Severity: {report['severity'].upper()}")
                        st.markdown(report["what_changed"])
                        st.markdown(report["likely_driver"])
                        st.markdown("**Evidence**")
                        for line in report["evidence"]:
                            st.markdown(f"- {line}")
                        st.markdown("**Recommended checks**")
                        for action in report["recommended_actions"]:
                            st.markdown(f"- {action}")
                    elif name in DIM_LABELS.values():
                        dim = next(key for key, label in DIM_LABELS.items() if label == name)
                        payload = drilldown_dimension(frame, metric, peak, dim)
                        st.plotly_chart(ui_helpers.contribution_chart(payload), width="stretch")
                    elif name == "Top drivers":
                        payload = drilldown_drivers(frame, metric, peak)
                        st.dataframe(ui_helpers.drivers_dataframe(payload), width="stretch")
                        st.caption(
                            "Ranked by how much of the total change the segment explains; "
                            "the most specific cut that accounts for the change comes first."
                        )
                    else:
                        payload = drilldown_ratio(frame, peak)
                        st.plotly_chart(ui_helpers.waterfall_chart(payload), width="stretch")
                        st.caption("Revenue = sessions × conversion × AOV; the three effects sum to the net change.")


def main() -> None:
    ensure_sample_data()

    with st.sidebar:
        st.header("Sentinel")
        st.caption("AI metric watchdog")
        data_source = st.radio("Data source", ["Sample dataset", "Upload a CSV"])
        uploaded = None
        if data_source == "Upload a CSV":
            uploaded = st.file_uploader("CSV file", type=["csv"])
        sensitivity = st.slider(
            "Sensitivity",
            min_value=1.0,
            max_value=7.0,
            value=DEFAULT_SENSITIVITY,
            step=0.1,
            help="A day must clear this many robust z-scores to be flagged.",
        )
        run_clicked = st.button("Run watchdog", type="primary", width="stretch")

    source: str | bytes = str(SAMPLE_CSV) if uploaded is None else uploaded.getvalue()

    if run_clicked or "incidents" not in st.session_state:
        try:
            frame = augment_conversion_rate(load_dataset(source))
            daily = daily_metrics(frame)
            st.session_state.frame = frame
            st.session_state.daily = daily
            st.session_state.incidents = run_watchdog(daily, sensitivity)
        except Exception as exc:  # noqa: BLE001 - surface any run error to the user
            st.error(ui_helpers.friendly_error(exc))

    if "incidents" not in st.session_state:
        return

    incidents: list[Incident] = st.session_state.incidents
    if not incidents:
        st.info("No anomalies cleared the sensitivity threshold.")
        return

    flagged = ui_helpers.flagged_dates_by_metric(incidents)
    st.title("Metric watchdog")
    for metric in METRICS:
        st.plotly_chart(ui_helpers.metric_chart(st.session_state.daily, metric, flagged[metric]), width="stretch")

    st.subheader("Incidents")
    for incident in incidents:
        render_incident(incident, st.session_state.frame)


if __name__ == "__main__":
    main()