"""Central configuration: environment loading and model names."""

import os

from dotenv import load_dotenv

load_dotenv(override=True)

# Default OpenAI model; override with MODEL_NAME in .env
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.4-mini")

# OpenAI-compatible endpoints for non-OpenAI providers (used in lab 3)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "moonshotai/kimi-k2.6")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def require_env(*names: str) -> None:
    """Exit with a clear message if required environment variables are missing."""
    missing = [n for n in names if not os.getenv(n)]
    if missing:
        raise SystemExit(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill them in."
        )
