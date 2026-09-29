"""Sentinel — AI metric watchdog UI.

Runs the detection and root-cause drilldown pipeline over the sample dataset or
an uploaded CSV, and renders one chart per metric plus incident cards with a
drilldown panel. Narration is deterministic for now; the LLM is not wired in.
Incident memory and alert delivery are wired in, and both degrade cleanly when
their configuration is missing.
"""

from __future__ import annotations

import io
import runpy
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from core import alerts, memory, rootcause, ui_helpers
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

TEST_ALERT_TEXT: str = (
    "HEADLINE: Sentinel test alert\n"
    "SEVERITY: info\n"
    "WHAT CHANGED: nothing — this is a test of the alert channel.\n"
    "LIKELY DRIVER: none, this is a connectivity check."
)

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


def seed_memory() -> None:
    """Seed the demo incidents once, on first start, never failing the app."""
    try:
        memory.seed_demo_incidents()
    except Exception as exc:  # noqa: BLE001 - memory is optional, degrade the run
        st.warning(f"Incident memory is unavailable: {ui_helpers.friendly_error(exc)}")


def similar_past_incidents(report: dict[str, Any], metric: str) -> list[dict[str, Any]]:
    """Past incidents showing a similar pattern, or an empty list on failure."""
    try:
        return memory.find_similar({**report, "metric": metric})
    except Exception as exc:  # noqa: BLE001 - memory is optional, degrade the run
        st.warning(f"Could not reach incident memory: {ui_helpers.friendly_error(exc)}")
        return []


def deliver_alert(report_text: str) -> tuple[bool, str]:
    """Send the summary to every configured channel, reporting each outcome."""
    webhook_ok, webhook_message = alerts.send_webhook(report_text)
    email_ok, email_message = alerts.send_email(report_text)
    return (webhook_ok or email_ok), f"{webhook_message}. Email: {email_message}"


def show_alert_result(ok: bool, message: str) -> None:
    """Render a delivery outcome, with a nudge when nothing is configured."""
    if ok:
        st.success(message)
    elif "not configured" in message:
        st.info(f"{message}. Set WEBHOOK_URL in .env to enable alerts.")
    else:
        st.warning(message)


def render_recent_alerts(limit: int = 5) -> None:
    """The last few delivery attempts, read straight from the alert log."""
    entries = alerts.recent_alerts(limit)
    if not entries:
        return
    st.sidebar.markdown("**Recent alerts**")
    for entry in entries:
        st.sidebar.caption(
            f"{entry['timestamp']} · {entry['channel']} · {entry['status']} · {entry['headline']}"
        )


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

                        st.markdown("**Similar past incidents**")
                        st.caption("Same-shaped pattern in earlier incidents, not necessarily the same cause.")
                        matches = similar_past_incidents(report, metric)
                        if not matches:
                            st.caption("No similar past incidents recorded yet.")
                        for match in matches:
                            demo_tag = " · demo data" if match["is_demo"] else ""
                            st.markdown(
                                f"- {match['date']} — {match['headline']} · "
                                f"similarity {match['similarity']:.0%} · "
                                f"recorded cause: {match['cause']}{demo_tag}"
                            )

                        with st.form(key=f"cause_form_{metric}_{peak}"):
                            cause_input = st.text_input(
                                "Mark cause / resolve",
                                placeholder="What actually caused this? (stored for future recall)",
                            )
                            submitted = st.form_submit_button("Save cause")
                        if submitted:
                            cause_text = cause_input.strip()
                            if not cause_text:
                                st.warning("Enter a cause before saving.")
                            else:
                                try:
                                    memory.add_incident(
                                        headline=report["headline"],
                                        metric=metric,
                                        driver=report["likely_driver"],
                                        cause=cause_text,
                                        date=peak,
                                        is_demo=False,
                                    )
                                    st.success("Cause recorded — it will inform future similar incidents.")
                                except Exception as exc:  # noqa: BLE001
                                    st.error(f"Could not save the cause: {ui_helpers.friendly_error(exc)}")

                        if st.button("Send alert", key=f"send_alert_{metric}_{peak}"):
                            show_alert_result(*deliver_alert(alerts.format_alert(incident, report)))
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
    seed_memory()

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
        if st.button("Send test alert", width="stretch"):
            show_alert_result(*deliver_alert(TEST_ALERT_TEXT))
        render_recent_alerts()

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