"""Lab 3: multiple model providers, structured outputs, and guardrails.

Run:  python -m examples.lab3_models_guardrails
Optional extras: --sandbox (needs ./code with a buggy file) and --mcp (Context7 MCP server).

Gemini, OpenRouter and Groq keys are optional; providers without a key are skipped.
"""

import argparse
import asyncio
import os
from pathlib import Path

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OpenAIChatCompletionsModel,
    Runner,
    output_guardrail,
    trace,
)
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from agentic_lab.config import (
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    GROQ_BASE_URL,
    GROQ_MODEL,
    MODEL_NAME,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    require_env,
)
from agentic_lab.sales import INTRO
from agentic_lab.tools import send_email_tool

INSTRUCTIONS = (
    INTRO.replace("You write emails.", "")
    + "You write compelling sales emails that are likely to get a response."
)
DESCRIPTION = "Use this tool to write a sales email. In the input, just instruct it to write a sales email."

PROVIDERS = [
    ("Gemini", "GOOGLE_API_KEY", GEMINI_BASE_URL, GEMINI_MODEL),
    ("OpenRouter", "OPENROUTER_API_KEY", OPENROUTER_BASE_URL, OPENROUTER_MODEL),
    ("Groq", "GROQ_API_KEY", GROQ_BASE_URL, GROQ_MODEL),
]


def build_provider_models() -> dict[str, OpenAIChatCompletionsModel]:
    """Any OpenAI-compatible endpoint works: base URL -> async client -> model object."""
    models = {}
    for name, key_var, base_url, model in PROVIDERS:
        key = os.getenv(key_var)
        if not key:
            print(f"{key_var} not set; skipping {name}")
            continue
        client = AsyncOpenAI(base_url=base_url, api_key=key)
        models[name] = OpenAIChatCompletionsModel(model=model, openai_client=client)
    return models


class EmailReview(BaseModel):
    is_professional: bool = Field(description="Whether the email is professional and appropriate")
    number_of_sentences: int = Field(
        description="The number of sentences in the body of the email, not including the greeting and signature"
    )
    contains_placeholders: bool = Field(description="Whether the email contains placeholders for personalization")


checker = Agent(
    name="Checker",
    instructions="You review potential sales emails",
    model=MODEL_NAME,
    output_type=EmailReview,
)


@output_guardrail
async def email_guardrail(ctx, agent, message):
    result = await Runner.run(checker, message, context=ctx.context)
    review = result.final_output
    is_problem = review.contains_placeholders or not review.is_professional
    return GuardrailFunctionOutput(output_info={"review": review}, tripwire_triggered=is_problem)


async def multi_model_manager(models: dict[str, OpenAIChatCompletionsModel]) -> None:
    print("\n=== Sales manager across different models ===")
    if not models:
        print("No alternative providers configured; skipping.")
        return
    tools = []
    for i, (name, model) in enumerate(models.items(), start=1):
        agent = Agent(name=f"{name} Sales Agent", instructions=INSTRUCTIONS, model=model)
        tools.append(agent.as_tool(tool_name=f"sales_agent{i}", tool_description=DESCRIPTION))

    manager = Agent(
        name="Sales Manager",
        instructions=(
            "You are a Sales Manager at ComplAI. "
            "Your goal is to find the single best cold sales email using the sales_agent tools."
        ),
        tools=tools + [send_email_tool],
        model=MODEL_NAME,
    )
    task = (
        "Follow these steps:\n"
        "1. Use each sales_agent tool once to draft an email. Just instruct each to write a sales email. "
        "Do not proceed until all drafts are ready.\n"
        "2. Choose the single best email.\n"
        "3. Use your tool to send the best email (and only the best email). Only send 1 email."
    )
    with trace("Sales Manager across different models"):
        result = await Runner.run(manager, task)
    print(result.final_output)


async def structured_outputs() -> None:
    print("\n=== Structured outputs ===")
    email = """
Hi [first_name],

I'm hitting you up to see if you'd like to buy our product. It's really great. You'll miss out if you don't buy it.

Laters.

Ed
"""
    result = await Runner.run(checker, email)
    print(result.final_output)


async def guardrails(models: dict[str, OpenAIChatCompletionsModel]) -> None:
    print("\n=== Guardrails ===")
    model = next(iter(models.values()), MODEL_NAME)
    cowboy_instructions = INSTRUCTIONS + "\nSpeak like a cowboy"

    print("--- framework output guardrail ---")
    guarded = Agent(name="Cowboy", instructions=cowboy_instructions, model=model, output_guardrails=[email_guardrail])
    try:
        result = await Runner.run(guarded, "Write a cold sales email")
        print(result.final_output)
    except Exception as exc:  # OutputGuardrailTripwireTriggered
        print(f"Guardrail tripped: {type(exc).__name__}")

    print("--- explicit check (works in any framework) ---")
    simple = Agent(name="Simple Cowboy", instructions=cowboy_instructions, model=model)
    email = (await Runner.run(simple, "Write a cold sales email")).final_output
    print(email)
    review = (await Runner.run(checker, email)).final_output
    if not review.is_professional or review.contains_placeholders:
        print("The email is not professional or has placeholders and will not be sent")
    else:
        print("Email is good")


def build_sandbox_agent(code_dir: Path, output_dir: Path):
    """Build (but do not run) the SandboxAgent and its RunConfig. Returns ``(agent, run_config)``."""
    from agents.run import RunConfig
    from agents.sandbox import Manifest, SandboxAgent, SandboxPathGrant, SandboxRunConfig
    from agents.sandbox.capabilities import Capabilities
    from agents.sandbox.entries import LocalDir
    from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

    instructions = f"""
You are a software engineer that fixes bugs.
Review files in the sandbox code directory.

Write the fixed version of the file to this host output directory:
{output_dir}

Use full file paths when writing output.
Respond with a summary of what you did.
"""
    manifest = Manifest(
        entries={"code": LocalDir(src=code_dir)},
        extra_path_grants=[SandboxPathGrant(path=str(output_dir))],
    )
    agent = SandboxAgent(
        name="Engineer",
        instructions=instructions,
        model=MODEL_NAME,
        default_manifest=manifest,
        capabilities=Capabilities.default(),
    )
    run_config = RunConfig(
        sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()),
        workflow_name="Sandbox coding example",
    )
    return agent, run_config


async def sandbox_demo() -> None:
    """Optional: SandboxAgent that fixes a bug in ./code and writes results to ./output."""
    print("\n=== Sandbox agent ===")
    code_dir = Path("code").resolve()
    output_dir = Path("output").resolve()
    if not code_dir.exists():
        print(f"Create {code_dir} with a buggy file first; skipping.")
        return
    output_dir.mkdir(exist_ok=True)

    agent, run_config = build_sandbox_agent(code_dir, output_dir)
    result = await Runner.run(agent, "Fix the bug in the code", run_config=run_config)
    print(result.final_output)


async def mcp_demo() -> None:
    """Optional: answer a docs question through the Context7 MCP server."""
    from agents.mcp import MCPServerStreamableHttp

    print("\n=== MCP (Context7) ===")
    task = (
        "In the SandboxAgents feature of the OpenAI Agents SDK, what is the role of the Manifest object? "
        "Always be accurate. If you don't know the answer, say so."
    )
    params = {"url": "https://mcp.context7.com/mcp", "timeout": 60}
    async with MCPServerStreamableHttp(name="Context7", params=params) as server:
        agent = Agent(
            name="Expert",
            instructions="Use Context7 to answer the question",
            mcp_servers=[server],
            model=MODEL_NAME,
        )
        result = await Runner.run(agent, task)
    print(result.final_output)


async def main(args: argparse.Namespace) -> None:
    require_env("OPENAI_API_KEY")
    models = build_provider_models()
    await multi_model_manager(models)
    await structured_outputs()
    await guardrails(models)
    if args.sandbox:
        await sandbox_demo()
    if args.mcp:
        await mcp_demo()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sandbox", action="store_true", help="run the optional SandboxAgent demo")
    parser.add_argument("--mcp", action="store_true", help="run the optional MCP demo")
    asyncio.run(main(parser.parse_args()))
