"""Lab 1: a first agent, tracing, streaming, a function tool, and memory (sessions).

Run:  python -m examples.lab1_basics
"""

import asyncio

from agents import Agent, Runner, SQLiteSession, trace
from openai.types.responses import ResponseTextDeltaEvent

from agentic_lab.config import MODEL_NAME, require_env
from agentic_lab.tools import push_tool


async def simple_agent() -> None:
    print("\n=== Part 1: simple agent, tracing, streaming ===")
    agent = Agent(name="Jokester", instructions="You are a joke teller", model=MODEL_NAME)

    with trace("Telling a joke"):
        result = await Runner.run(agent, "Tell a joke about Autonomous AI Agents")
    print(result.final_output)

    print("\n--- streaming ---")
    streamed = Runner.run_streamed(agent, input="Please tell me 5 jokes about AI Agents.")
    async for event in streamed.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)
    print()


async def tool_agent() -> None:
    print("\n=== Part 2: function tool ===")
    notifier = Agent(
        name="Notifier",
        model=MODEL_NAME,
        instructions="You notify the user upon request",
        tools=[push_tool],
    )
    with trace("Pizza has arrived"):
        result = await Runner.run(notifier, "Notify the user that the pizza is here")
    print(result.final_output)


async def memory() -> None:
    print("\n=== Part 3: memory ===")
    agent = Agent(name="Assistant", model=MODEL_NAME)

    print("--- no memory: each run starts fresh ---")
    r = await Runner.run(agent, "Hi there. My name is Ed.")
    print(r.final_output)
    r = await Runner.run(agent, "What's my name?")
    print(r.final_output)

    print("--- approach 1: pass the history manually ---")
    r = await Runner.run(agent, "Hi there. My name is Ed.")
    next_input = r.to_input_list() + [{"role": "user", "content": "What's my name?"}]
    r = await Runner.run(agent, next_input)
    print(r.final_output)

    print("--- approach 2: SQLiteSession ---")
    # In-memory by default; use SQLiteSession("12346", "memory.db") to persist on disk.
    session = SQLiteSession("12346")
    r = await Runner.run(agent, "Hi there. My name is Ed.", session=session)
    print(r.final_output)
    r = await Runner.run(agent, "What's my name?", session=session)
    print(r.final_output)


async def main() -> None:
    require_env("OPENAI_API_KEY")
    await simple_agent()
    await tool_agent()
    await memory()


if __name__ == "__main__":
    asyncio.run(main())
