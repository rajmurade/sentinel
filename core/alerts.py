"""Alert delivery.

POSTs the incident summary to a Discord or Slack webhook and, optionally,
emails it over SMTP. Every function returns ``(ok, message)`` and never raises:
an unset webhook, a missing SMTP host, a timeout or a rejected request all
degrade to a friendly message (CONTEXT.md rules 3 and 4).

Every send attempt, successful or not, is appended to ``alerts_log.jsonl``.
The log and the returned messages are scrubbed of the webhook URL and the SMTP
password, so a secret never reaches the screen, the log or a commit.
"""

import json
import os
import smtplib
import ssl
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from core.detect import Incident

load_dotenv()

WEBHOOK_URL: str = os.getenv("WEBHOOK_URL", "")
SMTP_HOST: str = os.getenv("SMTP_HOST", "")
SMTP_PORT: int = 587
SMTP_USER: str = os.getenv("SMTP_USER", "")
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
ALERT_TO: str = os.getenv("ALERT_TO", "")

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
ALERT_LOG_PATH: Path = PROJECT_ROOT / "alerts_log.jsonl"

REQUEST_TIMEOUT: int = 10
DISCORD_MAX_CHARS: int = 1900
EMAIL_SUBJECT: str = "Sentinel incident alert"

try:
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
except ValueError:
    SMTP_PORT = 587


def format_alert(incident: Incident, report: dict[str, Any]) -> str:
    """The plain-text incident summary: severity, narrative, evidence, actions."""
    severity = report.get("severity") or incident.get("severity", "unknown")
    lines: list[str] = [
        f"HEADLINE: {report.get('headline', '')}",
        f"SEVERITY: {severity}",
        f"WHAT CHANGED: {report.get('what_changed', '')}",
        f"LIKELY DRIVER: {report.get('likely_driver', '')}",
        "EVIDENCE:",
    ]
    for line in list(report.get("evidence", []))[:2]:
        lines.append(f"- {line}")
    lines.append("RECOMMENDED ACTIONS:")
    for action in report.get("recommended_actions", []):
        lines.append(f"- {action}")
    return "\n".join(lines)


def send_webhook(report_text: str) -> tuple[bool, str]:
    """Post the summary to the configured Discord or Slack webhook."""
    if not WEBHOOK_URL:
        return _log_attempt("webhook", False, report_text, "not configured: WEBHOOK_URL is not set")

    kind = _webhook_kind(WEBHOOK_URL)
    if kind == "discord":
        payload: dict[str, str] = {"content": report_text[:DISCORD_MAX_CHARS]}
    elif kind == "slack":
        payload = {"text": report_text}
    else:
        return _log_attempt(
            "webhook", False, report_text, "not configured: WEBHOOK_URL is not a Discord or Slack URL"
        )

    try:
        response = requests.post(WEBHOOK_URL, json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - delivery must never break the app
        return _log_attempt("webhook", False, report_text, f"request failed: {exc}")

    ok = 200 <= response.status_code < 300
    return _log_attempt("webhook", ok, report_text, f"{kind} webhook returned {response.status_code}")


def send_email(report_text: str) -> tuple[bool, str]:
    """Email the summary over SMTP with STARTTLS; skipped when not configured."""
    if not SMTP_HOST or not ALERT_TO:
        return _log_attempt(
            "email", False, report_text, "not configured: SMTP_HOST and ALERT_TO are not both set"
        )

    message = EmailMessage()
    message["Subject"] = EMAIL_SUBJECT
    message["From"] = SMTP_USER or ALERT_TO
    message["To"] = ALERT_TO
    message.set_content(report_text)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=REQUEST_TIMEOUT) as server:
            server.starttls(context=ssl.create_default_context())
            if SMTP_USER and SMTP_PASSWORD:
                server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(message)
    except Exception as exc:  # noqa: BLE001 - delivery must never break the app
        return _log_attempt("email", False, report_text, f"smtp failed: {exc}")

    return _log_attempt("email", True, report_text, "email sent")


def recent_alerts(limit: int = 5) -> list[dict[str, Any]]:
    """The most recent log entries, oldest first; empty when nothing was sent."""
    try:
        lines = ALERT_LOG_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001 - a missing or unreadable log is not an error
        return []
    entries: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except Exception:  # noqa: BLE001 - skip a half-written line
            continue
    return entries


def _webhook_kind(url: str) -> str:
    """``"discord"``, ``"slack"``, or ``"unknown"`` based on the webhook host."""
    if "discord.com/api/webhooks" in url:
        return "discord"
    if "hooks.slack.com" in url:
        return "slack"
    return "unknown"


def _redact(text: str) -> str:
    """Remove the webhook URL and the SMTP password from anything we surface."""
    safe = text
    for secret in (WEBHOOK_URL, SMTP_PASSWORD):
        if secret:
            safe = safe.replace(secret, "[redacted]")
    return safe


def _headline_from_text(report_text: str) -> str:
    """The ``HEADLINE:`` line, or the first non-empty line as a fallback."""
    for line in report_text.splitlines():
        if line.startswith("HEADLINE: "):
            return line.removeprefix("HEADLINE: ").strip()
    for line in report_text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _log_attempt(
    channel: str, ok: bool, report_text: str, message: str
) -> tuple[bool, str]:
    """Append the attempt to ``alerts_log.jsonl`` and return ``(ok, message)``."""
    safe_message = _redact(message)
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "channel": channel,
        "status": "sent" if ok else "failed",
        "headline": _headline_from_text(report_text),
        "message": safe_message,
    }
    try:
        ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with ALERT_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except Exception:  # noqa: BLE001 - writing the log must never break the send
        pass
    return ok, safe_message
