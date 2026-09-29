"""Tests for core.alerts.

Webhook delivery is tested with ``requests`` mocked, so nothing leaves the
machine. The log is redirected to a temporary file and the module-level config
is monkeypatched, which is what makes the "never configured" and "never raises"
paths easy to exercise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

import pytest

import core.alerts as alerts
from core.agent import build_evidence, template_report
from core.detect import (
    Incident,
    daily_metrics,
    detect_anomalies,
    group_incidents,
    load_data,
)

SAMPLE_CSV: Path = Path(__file__).resolve().parents[1] / "data" / "sample.csv"

DISCORD_URL: str = "https://discord.com/api/webhooks/123/secret-token"
SLACK_URL: str = "https://hooks.slack.com/services/T000/B000/secret-token"

EVIDENCE_LINE: str = "Revenue moved -21.4% against its weekday baseline."
ACTION_LINE: str = "Compare organic sessions on mobile against the previous week."


class FakeResponse:
    """Just enough of a ``requests.Response`` for the send path."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


@pytest.fixture
def log_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Send every attempt to a throwaway log, with all config cleared."""
    path = tmp_path / "alerts_log.jsonl"
    monkeypatch.setattr(alerts, "ALERT_LOG_PATH", path)
    monkeypatch.setattr(alerts, "WEBHOOK_URL", "")
    monkeypatch.setattr(alerts, "SMTP_HOST", "")
    monkeypatch.setattr(alerts, "SMTP_USER", "")
    monkeypatch.setattr(alerts, "SMTP_PASSWORD", "")
    monkeypatch.setattr(alerts, "ALERT_TO", "")
    yield path


@pytest.fixture(scope="module")
def report() -> dict[str, Any]:
    """A real template report, so the formatting test uses the actual shape."""
    frame = load_data(SAMPLE_CSV)
    metrics = daily_metrics(frame)
    incident: Incident = group_incidents(detect_anomalies(metrics))[0]
    return template_report(incident, build_evidence(frame, incident))


def read_log(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_format_alert_contains_every_required_section(report: dict[str, Any]) -> None:
    """The summary carries severity, narrative, two evidence lines and actions."""
    text = alerts.format_alert(report=report, incident={"severity": report["severity"]})

    assert f"HEADLINE: {report['headline']}" in text
    assert f"SEVERITY: {report['severity']}" in text
    assert report["what_changed"] in text
    assert report["likely_driver"] in text
    assert "EVIDENCE:" in text
    assert "RECOMMENDED ACTIONS:" in text
    assert text.index("EVIDENCE:") < text.index("RECOMMENDED ACTIONS:")


def test_format_alert_caps_evidence_at_two_lines() -> None:
    """Only the top two evidence lines are included, never the whole list."""
    text = alerts.format_alert(
        incident={"severity": "high"},
        report={
            "headline": "Revenue fell",
            "severity": "high",
            "what_changed": "Revenue fell.",
            "likely_driver": "organic mobile.",
            "evidence": [EVIDENCE_LINE, ACTION_LINE, "a third line that must not appear"],
            "recommended_actions": [],
        },
    )

    assert EVIDENCE_LINE in text
    assert ACTION_LINE in text
    assert "a third line that must not appear" not in text


def test_discord_webhook_payload_uses_content_key(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Discord URL posts ``{"content": ...}``."""
    monkeypatch.setattr(alerts, "WEBHOOK_URL", DISCORD_URL)
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, str], timeout: int) -> FakeResponse:
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(204)

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    ok, message = alerts.send_webhook("HEADLINE: Revenue fell on 2026-08-21")

    assert ok is True
    assert "204" in message
    assert captured["url"] == DISCORD_URL
    assert captured["json"] == {"content": "HEADLINE: Revenue fell on 2026-08-21"}
    assert captured["timeout"] == alerts.REQUEST_TIMEOUT


def test_slack_webhook_payload_uses_text_key(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Slack URL posts ``{"text": ...}``."""
    monkeypatch.setattr(alerts, "WEBHOOK_URL", SLACK_URL)
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, str], timeout: int) -> FakeResponse:
        captured.update({"url": url, "json": json, "timeout": timeout})
        return FakeResponse(200)

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    ok, _ = alerts.send_webhook("HEADLINE: Sessions rose on 2026-09-16")

    assert ok is True
    assert captured["json"] == {"text": "HEADLINE: Sessions rose on 2026-09-16"}


def test_discord_body_is_truncated_to_1900_chars(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Discord messages are capped at 1900 characters."""
    monkeypatch.setattr(alerts, "WEBHOOK_URL", DISCORD_URL)
    captured: dict[str, str] = {}

    def fake_post(url: str, json: dict[str, str], timeout: int) -> FakeResponse:
        captured.update(json)
        return FakeResponse(204)

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    alerts.send_webhook("HEADLINE: long\n" + ("x" * 5000))

    assert len(captured["content"]) == alerts.DISCORD_MAX_CHARS == 1900


def test_slack_body_is_not_truncated(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Slack has a much higher limit, so the full body goes out."""
    monkeypatch.setattr(alerts, "WEBHOOK_URL", SLACK_URL)
    captured: dict[str, str] = {}

    def fake_post(url: str, json: dict[str, str], timeout: int) -> FakeResponse:
        captured.update(json)
        return FakeResponse(200)

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    alerts.send_webhook("HEADLINE: long\n" + ("x" * 5000))

    assert len(captured["text"]) > alerts.DISCORD_MAX_CHARS


def test_missing_webhook_returns_not_configured(log_path: Path) -> None:
    """An unset webhook is a friendly no-op, not a crash."""
    ok, message = alerts.send_webhook("HEADLINE: anything")

    assert ok is False
    assert "not configured" in message


def test_unknown_webhook_url_returns_not_configured(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A URL that is neither Discord nor Slack is refused before any request."""
    monkeypatch.setattr(alerts, "WEBHOOK_URL", "https://example.com/hook")

    def explode(*_args: Any, **_kwargs: Any) -> FakeResponse:
        raise AssertionError("no request should be made for an unknown host")

    monkeypatch.setattr(alerts.requests, "post", explode)

    ok, message = alerts.send_webhook("HEADLINE: anything")

    assert ok is False
    assert "not configured" in message


def test_failed_request_is_reported_not_raised(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A transport error comes back as ``(False, message)``."""
    monkeypatch.setattr(alerts, "WEBHOOK_URL", DISCORD_URL)

    def fake_post(url: str, json: dict[str, str], timeout: int) -> FakeResponse:
        raise TimeoutError("connection timed out")

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    ok, message = alerts.send_webhook("HEADLINE: anything")

    assert ok is False
    assert "timed out" in message


def test_missing_smtp_returns_not_configured(log_path: Path) -> None:
    """Email is skipped cleanly when SMTP is not configured."""
    ok, message = alerts.send_email("HEADLINE: anything")

    assert ok is False
    assert "not configured" in message


def test_log_line_records_timestamp_channel_status_and_headline(log_path: Path) -> None:
    """Every attempt, successful or not, appends one JSON line to the log."""
    alerts.send_webhook("HEADLINE: Revenue fell on 2026-08-21")
    alerts.send_webhook("HEADLINE: Sessions rose on 2026-09-16")

    entries = read_log(log_path)
    assert len(entries) == 2
    for entry in entries:
        assert entry["channel"] == "webhook"
        assert entry["status"] == "failed"
        assert entry["timestamp"].endswith("+00:00")
    assert entries[0]["headline"] == "Revenue fell on 2026-08-21"
    assert entries[1]["headline"] == "Sessions rose on 2026-09-16"


def test_log_never_contains_the_webhook_url_or_password(
    log_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Secrets are redacted from the log and the returned message."""
    monkeypatch.setattr(alerts, "WEBHOOK_URL", DISCORD_URL)
    monkeypatch.setattr(alerts, "SMTP_PASSWORD", "hunter2")

    def fake_post(url: str, json: dict[str, str], timeout: int) -> FakeResponse:
        raise ConnectionError(f"could not reach {url}")

    monkeypatch.setattr(alerts.requests, "post", fake_post)

    _, message = alerts.send_webhook("HEADLINE: anything")

    assert DISCORD_URL not in message
    assert "hunter2" not in message
    raw = log_path.read_text(encoding="utf-8")
    assert DISCORD_URL not in raw
    assert "hunter2" not in raw
    assert "[redacted]" in raw


def test_recent_alerts_returns_the_tail_in_order(log_path: Path) -> None:
    """``recent_alerts`` returns the newest entries, oldest first."""
    for index in range(4):
        alerts.send_webhook(f"HEADLINE: incident {index}")

    entries = alerts.recent_alerts(limit=2)

    assert [entry["headline"] for entry in entries] == ["incident 2", "incident 3"]


def test_recent_alerts_is_empty_without_a_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No log file means no entries, not an error."""
    monkeypatch.setattr(alerts, "ALERT_LOG_PATH", tmp_path / "missing.jsonl")

    assert alerts.recent_alerts() == []
