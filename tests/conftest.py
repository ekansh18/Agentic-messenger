"""Shared fixtures: no network, no real keys, no tracing export."""

import socket

import pytest
from agents import set_tracing_disabled
from agents.models.multi_provider import MultiProvider

from tests.fakes import FakeModel

CREDENTIAL_VARS = [
    "OPENAI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "EMAIL_ADDRESS",
    "EMAIL_SMTP_SERVER",
    "EMAIL_APP_PASSWORD",
    "PUSHOVER_USER",
    "PUSHOVER_TOKEN",
    "MODEL_NAME",
]


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Start every test with no credentials, a dummy OpenAI key, and tracing off."""
    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    # load_dotenv(override=True) runs at import time; make sure a developer's real .env can't leak in
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    set_tracing_disabled(True)


@pytest.fixture
def fake_llm(monkeypatch):
    """Route every string model name (e.g. "gpt-5.4-mini") to a FakeModel driven by ``respond``.

    Usage: ``requests = fake_llm(respond)``; ``requests`` collects every model call made.
    """

    def install(respond):
        record: list = []
        model = FakeModel(respond, record)
        monkeypatch.setattr(MultiProvider, "get_model", lambda self, name=None: model)
        return record

    return install


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Fail loudly if any test (or a regression in the code) tries to open a real connection."""

    def blocked(self, *args, **kwargs):
        raise RuntimeError(f"Network access attempted in a test: {args}")

    monkeypatch.setattr(socket.socket, "connect", blocked)
