"""Shared sales-agent definitions (ComplAI) used by the orchestration and model labs."""

from agents import Agent

from .config import MODEL_NAME

INTRO = """
You are a sales agent working for ComplAI,
a company that provides a SaaS tool for ensuring SOC2 compliance and preparing for audits, powered by AI.
You write emails.
"""

STYLES = {
    "Professional Sales Agent": "Your email style is professional, serious, with gravitas and credibility.",
    "Humorous Sales Agent": "Your email style is witty, engaging, and humorous.",
    "Executive Sales Agent": "Your email style is concise, to the point, in the style of a busy senior executive.",
}

EMAIL_TOOL_DESCRIPTION = "Use this tool to write a sales email. In the input, just instruct it to write a sales email."

MANAGER_TASK = """
Follow these steps:

1. Generate Drafts: Use each of the three sales_email_writer tools to generate different email drafts.
Just instruct each to write a sales email; no further details are needed.
Do not proceed until all three drafts are ready, one from each tool.

2. Evaluate and Select: Review the drafts and choose the single best email using your judgment of which one is most effective.

3. Use your tool to send the best email (and only the best email) to the user. Only send 1 email.
"""


def build_style_agents(model=MODEL_NAME) -> list[Agent]:
    """Three sales agents that differ only in writing style."""
    return [Agent(name=name, instructions=INTRO + style, model=model) for name, style in STYLES.items()]
