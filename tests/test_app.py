"""App-level checks for the deployment contract.

These run the real Streamlit app through ``AppTest`` with no secrets configured,
which is exactly how it runs on Hugging Face Spaces. Anything that must degrade
gracefully when a secret is missing is asserted here.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import core.alerts as alerts
from app import DEMO_MODE_DEFAULT

APP_PATH: Path = Path(__file__).resolve().parents[1] / "app.py"
SECRET_VARS: list[str] = [
    "WEBHOOK_URL",
    "SMTP_HOST",
    "SMTP_USER",
    "SMTP_PASSWORD",
    "ALERT_TO",
    "LLM_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
]


@pytest.fixture(scope="module")
def app() -> Any:
    """The app, run once with no secrets configured.

    Module scope: Streamlit re-runs are the slow part of this file, and one
    render is enough to assert on every element the tests below inspect.
    """
    from streamlit.testing.v1 import AppTest

    for name in SECRET_VARS:
        os.environ.pop(name, None)
    for name in ("WEBHOOK_URL", "SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "ALERT_TO"):
        if hasattr(alerts, name):
            setattr(alerts, name, "")

    test = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    assert not test.exception, [e.value for e in test.exception]
    return test


def test_demo_mode_is_on_by_default() -> None:
    """The Space must show something useful before anyone touches a control."""
    assert DEMO_MODE_DEFAULT is True


def test_app_runs_with_no_secrets(app: Any) -> None:
    """Charts and incidents render with no keys and no webhook configured."""
    assert app.title[0].value == "Metric watchdog"
    assert app.subheader[0].value.startswith("Incidents (")
    assert len(app.expander) > 0
    # One chart per metric (METRICS) plus the per-dimension drilldown charts that
    # Streamlit renders eagerly inside every tab.
    assert len(app.get("plotly_chart")) >= 5
    assert not app.error


def test_narration_uses_the_template_fallback(app: Any) -> None:
    """With no LLM key, the Report tab still shows a grounded narrative."""
    report_tab = app.tabs[0]
    text = " ".join(element.value for element in report_tab.markdown)
    assert "Evidence" in text
    assert "Recommended checks" in text


def test_similar_past_incidents_are_labelled_as_examples(app: Any) -> None:
    """Seeded history is marked as demo data, never presented as real."""
    report_tab = app.tabs[0]
    rendered = " ".join(element.value for element in report_tab.markdown)
    assert "demo data" in rendered
    assert "not necessarily the same cause" in " ".join(c.value for c in report_tab.caption)


def test_mark_cause_form_is_present(app: Any) -> None:
    """Each incident offers a way to record what actually happened."""
    labels = [field.label for field in app.tabs[0].text_input]
    assert "Mark cause / resolve" in labels


def test_send_alert_button_reports_not_configured(app: Any) -> None:
    """Alerting is available but honest: with no webhook it says so, not crash."""
    button = next(b for b in app.button if b.label == "Send alert")
    button.click().run()

    assert not app.exception
    combined = " ".join(i.value for i in app.info) + " ".join(w.value for w in app.warning)
    assert "not configured" in combined
    assert "WEBHOOK_URL" in combined


def test_turning_off_demo_mode_without_a_file_prompts_for_upload() -> None:
    """No file and no demo means a clear prompt rather than a broken run."""
    from streamlit.testing.v1 import AppTest

    test = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    test.checkbox[0].set_value(False).run()

    assert not test.exception, [e.value for e in test.exception]
    assert any("Upload a CSV" in i.value for i in test.info)
    assert len(test.expander) == 0


def test_demo_mode_reload_renders_incidents() -> None:
    """Switching demo mode back on re-renders the bundled dataset."""
    from streamlit.testing.v1 import AppTest

    test = AppTest.from_file(str(APP_PATH), default_timeout=300).run()
    test.checkbox[0].set_value(False).run()
    test.checkbox[0].set_value(True).run()

    assert not test.exception, [e.value for e in test.exception]
    assert len(test.expander) > 0
