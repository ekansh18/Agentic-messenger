"""Config, messenger and tool wiring: pure logic, no LLM involved."""

import re
from pathlib import Path

import pytest

from agentic_lab import config, messenger
from agentic_lab.tools import push_tool, send_email_tool

ROOT = Path(__file__).resolve().parent.parent


# ---------- config ----------

def test_require_env_passes_when_set(monkeypatch):
    monkeypatch.setenv("FOO", "1")
    config.require_env("FOO")


def test_require_env_exits_listing_all_missing():
    with pytest.raises(SystemExit) as exc:
        config.require_env("MISSING_A", "MISSING_B")
    assert "MISSING_A" in str(exc.value) and "MISSING_B" in str(exc.value)


def test_every_env_var_read_by_code_is_documented_in_env_example():
    """Guards against the classic bug: code reads a variable that .env.example never mentions."""
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (ROOT / ".env.example").read_text(), re.M))
    used = set()
    for path in [*(ROOT / "agentic_lab").glob("*.py"), *(ROOT / "examples").glob("*.py")]:
        src = path.read_text()
        used |= set(re.findall(r'getenv\(\s*"([A-Z][A-Z0-9_]+)"', src))
        used |= set(re.findall(r'environ\[\s*"([A-Z][A-Z0-9_]+)"', src))
        used |= set(re.findall(r'"([A-Z][A-Z0-9_]*(?:_API_KEY|_KEY))"', src))  # names passed to require_env / tables
    assert {"OPENAI_API_KEY", "MODEL_NAME", "EMAIL_ADDRESS", "PUSHOVER_TOKEN", "GROQ_MODEL"} <= used, "scanner is broken"
    assert used <= documented, f"undocumented env vars: {sorted(used - documented)}"


# ---------- messenger channel selection ----------

def test_channel_falls_back_to_console(capsys):
    assert messenger.send_message("Subj", "Body", "<p>Body</p>") == "console"
    out = capsys.readouterr().out
    assert "Subj" in out and "Body" in out


def test_partial_email_config_is_not_enough(monkeypatch):
    monkeypatch.setenv("EMAIL_ADDRESS", "a@b.c")  # server + password missing
    assert not messenger.email_configured()
    assert messenger.send_message("s", "t", "h") == "console"


class FakeSMTP:
    instances: list["FakeSMTP"] = []

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.started_tls = False
        self.login_args = None
        self.sent = None
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        self.started_tls = True

    def login(self, user, password):
        self.login_args = (user, password)

    def send_message(self, msg):
        self.sent = msg


def test_email_sent_over_smtp_with_tls_login_and_both_bodies(monkeypatch):
    FakeSMTP.instances.clear()
    monkeypatch.setattr(messenger.smtplib, "SMTP", FakeSMTP)
    monkeypatch.setenv("EMAIL_ADDRESS", "me@example.com")
    monkeypatch.setenv("EMAIL_SMTP_SERVER", "smtp.example.com")
    monkeypatch.setenv("EMAIL_APP_PASSWORD", "app-pass")

    assert messenger.send_message("Hello", "plain body", "<b>html body</b>") == "email"

    smtp = FakeSMTP.instances[0]
    assert (smtp.host, smtp.port) == ("smtp.example.com", 587)
    assert smtp.started_tls
    assert smtp.login_args == ("me@example.com", "app-pass")
    msg = smtp.sent
    assert msg["From"] == msg["To"] == "me@example.com"
    assert msg["Subject"] == "Hello"
    assert msg.get_body(("plain",)).get_content().strip() == "plain body"
    assert "<b>html body</b>" in msg.get_body(("html",)).get_content()


class FakeResponse:
    def raise_for_status(self):
        pass


def test_push_posts_expected_payload_when_email_absent(monkeypatch):
    calls = []
    monkeypatch.setattr(messenger.requests, "post", lambda url, data, timeout: calls.append((url, data)) or FakeResponse())
    monkeypatch.setenv("PUSHOVER_USER", "u123")
    monkeypatch.setenv("PUSHOVER_TOKEN", "a456")

    assert messenger.send_message("S", "T", "<p>T</p>") == "push"
    url, data = calls[0]
    assert url == messenger.PUSHOVER_URL
    assert data["user"] == "u123" and data["token"] == "a456"
    assert "S" in data["message"] and "T" in data["message"]


def test_email_takes_priority_over_push(monkeypatch):
    monkeypatch.setattr(messenger.smtplib, "SMTP", FakeSMTP)
    for k, v in {"EMAIL_ADDRESS": "a@b.c", "EMAIL_SMTP_SERVER": "s", "EMAIL_APP_PASSWORD": "p",
                 "PUSHOVER_USER": "u", "PUSHOVER_TOKEN": "t"}.items():
        monkeypatch.setenv(k, v)
    assert messenger.send_message("s", "t", "h") == "email"


# ---------- function tools ----------

def test_send_email_tool_schema_exposes_three_string_params():
    schema = send_email_tool.params_json_schema
    assert set(schema["properties"]) == {"subject", "text_body", "html_body"}
    assert all(p["type"] == "string" for p in schema["properties"].values())
    assert set(schema["required"]) == {"subject", "text_body", "html_body"}
    assert send_email_tool.name == "send_email_tool"
    assert "email" in send_email_tool.description.lower()


def test_push_tool_schema():
    assert push_tool.params_json_schema["properties"].keys() == {"message"}
