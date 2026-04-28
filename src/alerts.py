"""
Alerting & Webhooks.
Sends alerts when high-risk batches detected, model drift occurs, or queue builds up.
Uses email (smtplib) and/or webhook URL (HTTP POST).
Configure via environment variables.
"""
import os
import json
import requests
from datetime import datetime


WEBHOOK_URL  = os.getenv("ALERT_WEBHOOK_URL", "")   # Slack/Teams/Discord webhook
ALERT_EMAIL  = os.getenv("ALERT_EMAIL", "")          # Recipient email
SMTP_HOST    = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER    = os.getenv("SMTP_USER", "")
SMTP_PASS    = os.getenv("SMTP_PASS", "")


def send_webhook(title: str, message: str, level: str = "info") -> bool:
    """Send alert to webhook URL (Slack/Teams/Discord compatible)."""
    if not WEBHOOK_URL:
        return False
    colors = {"info": "#1f77b4", "warning": "#ffc107", "critical": "#dc3545"}
    payload = {
        "text": f"*[CrediSense AI] {title}*\n{message}",
        "attachments": [{
            "color": colors.get(level, "#1f77b4"),
            "text": message,
            "footer": f"CrediSense AI | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        }]
    }
    try:
        resp = requests.post(WEBHOOK_URL, json=payload, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def send_email_alert(subject: str, body: str) -> bool:
    """Send email alert via SMTP."""
    if not (ALERT_EMAIL and SMTP_USER and SMTP_PASS):
        return False
    try:
        import smtplib
        from email.mime.text import MIMEText
        msg = MIMEText(body)
        msg["Subject"] = f"[CrediSense AI] {subject}"
        msg["From"]    = SMTP_USER
        msg["To"]      = ALERT_EMAIL
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        return True
    except Exception:
        return False


def alert(title: str, message: str, level: str = "info") -> dict:
    """Send alert via all configured channels."""
    webhook_sent = send_webhook(title, message, level)
    email_sent   = send_email_alert(title, message)
    return {"webhook": webhook_sent, "email": email_sent,
            "any_sent": webhook_sent or email_sent}


def check_and_alert_drift(psi: float) -> dict | None:
    """Auto-alert if PSI exceeds threshold."""
    if psi >= 0.2:
        return alert(
            "Model Drift Detected",
            f"PSI = {psi:.4f} — Significant drift detected. Consider retraining.",
            level="critical"
        )
    elif psi >= 0.1:
        return alert(
            "Model Drift Warning",
            f"PSI = {psi:.4f} — Moderate drift. Monitor closely.",
            level="warning"
        )
    return None


def check_and_alert_queue(pending: int, threshold: int = 20) -> dict | None:
    """Alert if HITL queue builds up."""
    if pending >= threshold:
        return alert(
            "HITL Queue Backlog",
            f"{pending} cases pending manual review. Please process the queue.",
            level="warning"
        )
    return None
