"""Offline test doubles for the OpenAI Agents SDK.

``FakeModel`` implements the SDK's ``Model`` interface with a user-supplied ``respond`` function, so the real
agent loop (tool calls, handoffs, structured outputs, sessions, streaming) runs with zero network access.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from agents import Model, ModelResponse, Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseTextDeltaEvent,
)


def text_msg(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id=f"msg_{uuid.uuid4().hex[:8]}",
        type="message",
        role="assistant",
        status="completed",
        content=[ResponseOutputText(text=text, type="output_text", annotations=[])],
    )


def tool_call(name: str, args: dict[str, Any] | None = None, call_id: str | None = None) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{uuid.uuid4().hex[:8]}",
        call_id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        name=name,
        arguments=json.dumps(args or {}),
        type="function_call",
    )


def _as_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump()
    return {"value": str(item)}


def input_items(input: str | list) -> list[dict[str, Any]]:
    if isinstance(input, str):
        return [{"role": "user", "content": input}]
    return [_as_dict(i) for i in input]


def input_text(input: str | list) -> str:
    """Flatten the whole input (history included) to a string for substring checks."""
    return json.dumps(input_items(input), default=str)


def tool_outputs(input: str | list) -> list[dict[str, Any]]:
    return [i for i in input_items(input) if i.get("type") == "function_call_output"]


def has_output_for(input: str | list, call_id: str) -> bool:
    return any(o.get("call_id") == call_id for o in tool_outputs(input))


class Request:
    """What the model was asked, handed to the ``respond`` callback."""

    def __init__(self, system: str | None, input: str | list, tools: list, handoffs: list, output_schema: Any):
        self.system = system or ""
        self.input = input
        self.tools = tools
        self.handoffs = handoffs
        self.output_schema = output_schema

    @property
    def tool_names(self) -> list[str]:
        return [getattr(t, "name", "") for t in self.tools]

    @property
    def handoff_names(self) -> list[str]:
        return [h.tool_name for h in self.handoffs]


Responder = Callable[[Request], list]


class FakeModel(Model):
    def __init__(self, respond: Responder, record: list[Request] | None = None):
        self.respond = respond
        self.requests: list[Request] = record if record is not None else []

    async def get_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        *,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    ) -> ModelResponse:
        req = Request(system_instructions, input, tools, handoffs, output_schema)
        self.requests.append(req)
        return ModelResponse(output=self.respond(req), usage=Usage(), response_id=None)

    async def stream_response(
        self,
        system_instructions,
        input,
        model_settings,
        tools,
        output_schema,
        handoffs,
        tracing,
        *,
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    ) -> AsyncIterator:
        req = Request(system_instructions, input, tools, handoffs, output_schema)
        self.requests.append(req)
        output = self.respond(req)
        seq = 0
        for item in output:
            if isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    # emit the text in two chunks so streaming consumers see multiple deltas
                    text = part.text
                    for chunk in (text[: len(text) // 2], text[len(text) // 2 :]):
                        seq += 1
                        yield ResponseTextDeltaEvent(
                            content_index=0,
                            delta=chunk,
                            item_id=item.id,
                            logprobs=[],
                            output_index=0,
                            sequence_number=seq,
                            type="response.output_text.delta",
                        )
        yield ResponseCompletedEvent(
            type="response.completed",
            sequence_number=seq + 1,
            response=Response(
                id="resp_fake",
                created_at=0,
                model="fake",
                object="response",
                output=output,
                tool_choice="auto",
                tools=[],
                parallel_tool_calls=False,
            ),
        )
