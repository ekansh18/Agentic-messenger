"""Labs 1 and 2 end to end against a scripted fake LLM: real agent loop, no network."""

import pytest

from examples import lab1_basics, lab2_orchestration
from tests.fakes import has_output_for, input_text, text_msg, tool_call

SEND_CALL_ID = "send_email_call"


# ---------- Lab 1 ----------

async def test_simple_agent_prints_answer_and_streams_deltas(fake_llm, capsys):
    fake_llm(lambda req: [text_msg("Why did the agent cross the road? To fetch the other side.")])
    await lab1_basics.simple_agent()
    out = capsys.readouterr().out
    # printed once from the non-streaming run and once re-assembled from the streamed deltas
    assert out.count("To fetch the other side.") == 2


async def test_tool_agent_calls_push_tool_and_reports(fake_llm, capsys):
    def respond(req):
        if "push_tool" in input_text(req.input) and '"function_call_output"' in input_text(req.input):
            return [text_msg("Notified the user.")]
        assert "push_tool" in req.tool_names
        return [tool_call("push_tool", {"message": "Pizza is here"})]

    fake_llm(respond)
    await lab1_basics.tool_agent()
    out = capsys.readouterr().out
    assert "[console] Push: Pizza is here" in out  # push_tool really executed (console fallback)
    assert "Notified the user." in out


async def test_memory_section_shows_forgetting_without_session_and_recall_with_one(fake_llm, capsys):
    """The fake only 'knows' the name if it appears in the history it is sent, like a real stateless LLM."""

    def respond(req):
        text = input_text(req.input)
        if "What's my name?" in text:
            return [text_msg("Your name is Ed." if "My name is Ed" in text else "I don't know your name.")]
        return [text_msg("Nice to meet you, Ed.")]

    fake_llm(respond)
    await lab1_basics.memory()
    out = capsys.readouterr().out.splitlines()

    def answer_after(marker):
        return out[next(i for i, l in enumerate(out) if marker in l) :]

    no_memory = "\n".join(answer_after("no memory")[:4])
    assert "I don't know your name." in no_memory
    manual = "\n".join(answer_after("approach 1")[:3])
    assert "Your name is Ed." in manual
    session = "\n".join(answer_after("approach 2")[:4])
    assert "Your name is Ed." in session


# ---------- Lab 2 helpers ----------

def sales_llm(req):
    """Fake LLM covering every agent in lab 2, dispatching on the system prompt."""
    system, text = req.system, input_text(req.input)

    if "Sales Manager" in system:
        done = [n for n in range(1, 4) if f"writer_{n}_done" in text]
        if not text.count("function_call_output"):
            return [tool_call(f"sales_email_writer_{n}", {"input": "Write a sales email"}, f"writer_{n}_done") for n in range(1, 4)]
        if "send_email_tool" in req.tool_names and not has_output_for(req.input, SEND_CALL_ID):
            assert len(done) == 3, "manager must collect all three drafts before sending"
            return [tool_call("send_email_tool", {"subject": "Best email", "text_body": "t", "html_body": "<p>t</p>"}, SEND_CALL_ID)]
        if req.handoffs and not has_output_for(req.input, SEND_CALL_ID):
            return [tool_call(req.handoff_names[0], {})]
        return [text_msg("Manager done.")]

    if "pick the best cold sales email" in system:
        if not has_output_for(req.input, SEND_CALL_ID):
            return [tool_call("send_email_tool", {"subject": "Picked", "text_body": "winner", "html_body": "<p>winner</p>"}, SEND_CALL_ID)]
        return [text_msg("Sent the winner.")]

    style = "serious" if "professional" in system else "humorous" if "witty" in system else "executive"
    return [text_msg(f"Draft written in a {style} style.")]


async def test_by_code_generates_three_drafts_then_picker_sends_one_email(fake_llm, capsys):
    requests = fake_llm(sales_llm)
    await lab2_orchestration.by_code()
    out = capsys.readouterr().out
    assert out.count("[console] Subject:") == 1  # exactly one email "sent"
    assert "Subject: Picked" in out
    # the picker saw all three styles in one prompt
    picker_input = next(input_text(r.input) for r in requests if "pick the best" in r.system and "serious" in input_text(r.input))
    assert all(s in picker_input for s in ("serious", "humorous", "executive"))
    assert "Sent the winner." in out


async def test_by_tools_manager_calls_all_writers_as_tools_and_sends_once(fake_llm, capsys):
    requests = fake_llm(sales_llm)
    await lab2_orchestration.by_tools()
    out = capsys.readouterr().out
    assert out.count("[console] Subject:") == 1
    assert "Manager done." in out
    manager_req = next(r for r in requests if "Sales Manager" in r.system)
    assert manager_req.tool_names == ["sales_email_writer_1", "sales_email_writer_2", "sales_email_writer_3", "send_email_tool"]


async def test_by_handoffs_transfers_control_to_email_sender(fake_llm, capsys):
    requests = fake_llm(sales_llm)
    await lab2_orchestration.by_handoffs()
    out = capsys.readouterr().out
    assert out.count("[console] Subject:") == 1
    assert "Sent the winner." in out  # final output came from the sender agent, not the manager
    manager_req = next(r for r in requests if "Sales Manager" in r.system)
    assert manager_req.handoff_names == ["transfer_to_email_sender"]
    assert "send_email_tool" not in manager_req.tool_names  # manager cannot send; it must hand off


async def test_main_dispatches_modes(fake_llm, capsys):
    fake_llm(sales_llm)
    await lab2_orchestration.main("tools")
    out = capsys.readouterr().out
    assert "agents as tools" in out and "by code" not in out.lower().replace("orchestrating by llm", "")


async def test_main_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(SystemExit):
        await lab2_orchestration.main("all")
