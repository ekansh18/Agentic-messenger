"""Lab 2: multi-agent orchestration, by code and by LLM (tools and handoffs).

Run:  python -m examples.lab2_orchestration [code|tools|handoffs|all]
"""

import argparse
import asyncio

from agents import Agent, ModelSettings, Runner, trace

from agentic_lab.config import MODEL_NAME, require_env
from agentic_lab.sales import EMAIL_TOOL_DESCRIPTION, MANAGER_TASK, build_style_agents
from agentic_lab.tools import send_email_tool

MESSAGE = "Write a cold sales email"


async def generate_drafts(agents: list[Agent]) -> str:
    results = await asyncio.gather(*(Runner.run(a, MESSAGE) for a in agents))
    outputs = [r.final_output for r in results]
    return "Cold sales emails:\n\n" + "\n\nEmail:\n\n".join(outputs)


async def by_code() -> None:
    """Deterministic orchestration: parallel drafts, then a picker agent that sends the winner."""
    print("\n=== Orchestrating by code ===")
    agents = build_style_agents()

    picker = Agent(
        name="Sales Sender",
        instructions=(
            "You pick the best cold sales email from the given options. "
            "Imagine you are a customer and pick the one you are most likely to respond to. "
            "Then use your tool to send the email."
        ),
        model=MODEL_NAME,
        tools=[send_email_tool],
        model_settings=ModelSettings(tool_choice="required"),
    )

    with trace("Sales selection workflow with sending"):
        emails = await generate_drafts(agents)
        response = await Runner.run(picker, emails)
    print(f"Final response:\n{response.final_output}")


def build_writer_tools() -> list:
    agents = build_style_agents()
    return [
        agent.as_tool(tool_name=f"sales_email_writer_{i}", tool_description=EMAIL_TOOL_DESCRIPTION)
        for i, agent in enumerate(agents, start=1)
    ]


async def by_tools() -> None:
    """LLM orchestration: a Sales Manager calls writer agents as tools (A -> B -> A)."""
    print("\n=== Orchestrating by LLM: agents as tools ===")
    manager = Agent(
        name="Sales Manager",
        instructions=(
            "You are a Sales Manager at ComplAI. "
            "Your goal is to find the single best cold sales email using the sales_email_writer tools."
        ),
        tools=build_writer_tools() + [send_email_tool],
        model=MODEL_NAME,
    )
    with trace("Sales manager"):
        result = await Runner.run(manager, MANAGER_TASK)
    print(result.final_output)


async def by_handoffs() -> None:
    """LLM orchestration with a handoff (A -> B): control passes to the sender agent."""
    print("\n=== Orchestrating by LLM: handoffs ===")
    sender = Agent(
        name="Email Sender",
        instructions=(
            "You pick the best cold sales email from the given options. "
            "Imagine you are a customer and pick the one you are most likely to respond to. "
            "Then use your tool to send the email."
        ),
        model=MODEL_NAME,
        tools=[send_email_tool],
        model_settings=ModelSettings(tool_choice="required"),
        handoff_description="Choose the best of the drafted emails and send it",
    )
    manager = Agent(
        name="Sales Manager",
        instructions=(
            "You are a Sales Manager at ComplAI. "
            "You get your sales team to draft emails, then hand off to the Email Sender."
        ),
        tools=build_writer_tools(),
        handoffs=[sender],
        model=MODEL_NAME,
    )
    task = (
        "1. Use each of the three sales_email_writer tools to draft an email. "
        "Do not proceed until all three drafts are ready.\n"
        "2. Hand off to the Email Sender to choose and send the best email."
    )
    with trace("Sales manager with handoff"):
        result = await Runner.run(manager, task)
    print(result.final_output)


async def main(mode: str) -> None:
    require_env("OPENAI_API_KEY")
    if mode in ("code", "all"):
        await by_code()
    if mode in ("tools", "all"):
        await by_tools()
    if mode in ("handoffs", "all"):
        await by_handoffs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", default="all", choices=["code", "tools", "handoffs", "all"])
    asyncio.run(main(parser.parse_args().mode))
