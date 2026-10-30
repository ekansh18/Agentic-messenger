"""Notification helpers: SMTP email, Pushover push, with a console fallback.

``send_message`` picks the best available channel automatically:
email (if SMTP is configured) -> Pushover (if configured) -> console print.
"""

import os
import smtplib
from email.message import EmailMessage

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

PUSHOVER_URL = "https://api.pushover.net/1/messages.json"


def email_configured() -> bool:
    return all(os.getenv(k) for k in ("EMAIL_ADDRESS", "EMAIL_SMTP_SERVER", "EMAIL_APP_PASSWORD"))


def push_configured() -> bool:
    return all(os.getenv(k) for k in ("PUSHOVER_USER", "PUSHOVER_TOKEN"))


def send_email(subject: str, text_body: str, html_body: str) -> None:
    """Send an email to yourself via SMTP (STARTTLS on port 587)."""
    address = os.environ["EMAIL_ADDRESS"]
    msg = EmailMessage()
    msg["From"] = address
    msg["To"] = address
    msg["Subject"] = subject
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")

    with smtplib.SMTP(os.environ["EMAIL_SMTP_SERVER"], 587, timeout=30) as server:
        server.starttls()
        server.login(address, os.environ["EMAIL_APP_PASSWORD"])
        server.send_message(msg)


def push(message: str) -> None:
    """Send a Pushover push notification."""
    print(f"Push: {message}")
    payload = {
        "user": os.environ["PUSHOVER_USER"],
        "token": os.environ["PUSHOVER_TOKEN"],
        "message": message,
    }
    requests.post(PUSHOVER_URL, data=payload, timeout=30).raise_for_status()


def send_message(subject: str, text_body: str, html_body: str) -> str:
    """Deliver a message via the best configured channel. Returns the channel used."""
    if email_configured():
        send_email(subject, text_body, html_body)
        return "email"
    if push_configured():
        push(f"Subject: {subject}\n\n{text_body}")
        return "push"
    print(f"[console] Subject: {subject}\n\n{text_body}\n")
    return "console"
