"""Lab 3 (providers, structured outputs, guardrails) and Lab 4 (deep research), offline."""

import json
import sys

import pytest
from agents import OpenAIChatCompletionsModel
from agents.exceptions import OutputGuardrailTripwireTriggered
from pydantic import ValidationError

from agentic_lab import deep_research as dr
from examples import lab3_models_guardrails as lab3
from examples import lab4_deep_research as lab4
from tests.fakes import FakeModel, has_output_for, input_text, text_msg, tool_call


# ---------- Lab 3: providers ----------

def test_providers_without_keys_are_skipped(capsys):
    assert lab3.build_provider_models() == {}
    out = capsys.readouterr().out
    for var in ("GOOGLE_API_KEY", "OPENROUTER_API_KEY", "GROQ_API_KEY"):
        assert var in out


def test_providers_with_keys_build_chat_completions_models_on_the_right_endpoints(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "g")
    monkeypatch.setenv("GROQ_API_KEY", "q")
    models = lab3.build_provider_models()
    assert set(models) == {"Gemini", "Groq"}  # OpenRouter has no key -> skipped
    assert all(isinstance(m, OpenAIChatCompletionsModel) for m in models.values())
    assert "generativelanguage.googleapis.com" in str(models["Gemini"]._client.base_url)
    assert "api.groq.com" in str(models["Groq"]._client.base_url)
    assert models["Gemini"].model == lab3.GEMINI_MODEL


def test_model_names_are_overridable_from_env(monkeypatch):
    import importlib

    from agentic_lab import config

    monkeypatch.setenv("MODEL_NAME", "my-model")
    monkeypatch.setenv("GROQ_MODEL", "my-groq")
    reloaded = importlib.reload(config)
    try:
        assert reloaded.MODEL_NAME == "my-model" and reloaded.GROQ_MODEL == "my-groq"
    finally:
        monkeypatch.delenv("MODEL_NAME")
        monkeypatch.delenv("GROQ_MODEL")
        importlib.reload(config)


# ---------- Lab 3: structured outputs ----------

def test_email_review_schema_fields():
    schema = lab3.EmailReview.model_json_schema()
    assert set(schema["properties"]) == {"is_professional", "number_of_sentences", "contains_placeholders"}
    with pytest.raises(ValidationError):
        lab3.EmailReview(is_professional="maybe", number_of_sentences="x", contains_placeholders=None)


def checker_llm(req):
    """Fake reviewer: flags '[placeholder]' emails and unprofessional slang."""
    if req.output_schema is not None and "review potential sales emails" in req.system:
        text = input_text(req.input)
        return [text_msg(json.dumps({
            "is_professional": "Laters" not in text and "Howdy" not in text,
            "number_of_sentences": 3,
            "contains_placeholders": "[first_name]" in text,
        }))]
    return [text_msg("Hi Jane, ComplAI automates SOC2 audits. Want a demo? Best, Sam")]


async def test_structured_output_returns_typed_object(fake_llm, capsys):
    fake_llm(checker_llm)
    await lab3.structured_outputs()
    out = capsys.readouterr().out
    assert "is_professional=False" in out and "contains_placeholders=True" in out


async def test_checker_passes_a_clean_email(fake_llm):
    from agents import Runner

    fake_llm(checker_llm)
    review = (await Runner.run(lab3.checker, "Dear Jane, we help with SOC2.")).final_output
    assert isinstance(review, lab3.EmailReview)
    assert review.is_professional and not review.contains_placeholders


# ---------- Lab 3: guardrails ----------

async def test_output_guardrail_trips_on_placeholder_or_unprofessional_output(fake_llm, capsys):
    from agents import Agent, Runner

    fake_llm(checker_llm)
    bad_model = FakeModel(lambda req: [text_msg("Howdy [first_name], buy our thing, partner!")])
    agent = Agent(name="Cowboy", instructions="x", model=bad_model, output_guardrails=[lab3.email_guardrail])
    with pytest.raises(OutputGuardrailTripwireTriggered):
        await Runner.run(agent, "Write a cold sales email")


async def test_output_guardrail_lets_clean_output_through(fake_llm):
    from agents import Agent, Runner

    fake_llm(checker_llm)
    good_model = FakeModel(lambda req: [text_msg("Dear Jane, ComplAI makes SOC2 audits painless.")])
    agent = Agent(name="Cowboy", instructions="x", model=good_model, output_guardrails=[lab3.email_guardrail])
    result = await Runner.run(agent, "Write a cold sales email")
    assert "SOC2" in result.final_output


async def test_guardrails_demo_covers_both_framework_and_explicit_checks(fake_llm, capsys):
    fake_llm(checker_llm)
    cowboy = FakeModel(lambda req: [text_msg("Howdy partner! Try ComplAI. Laters.")])
    await lab3.guardrails({"Fake": cowboy})
    out = capsys.readouterr().out
    assert "Guardrail tripped: OutputGuardrailTripwireTriggered" in out
    assert "will not be sent" in out


# ---------- Lab 3: multi-model manager ----------

async def test_multi_model_manager_uses_one_tool_per_provider_and_sends_once(fake_llm, capsys):
    send_id = "send_call"

    def manager_llm(req):
        text = input_text(req.input)
        if "Sales Manager" in req.system:
            if "function_call_output" not in text:
                return [tool_call(f"sales_agent{i}", {"input": "Write a sales email"}) for i in (1, 2)]
            if not has_output_for(req.input, send_id):
                return [tool_call("send_email_tool", {"subject": "S", "text_body": "t", "html_body": "<p>t</p>"}, send_id)]
            return [text_msg("Sent the best one.")]
        return [text_msg("A draft.")]

    requests = fake_llm(manager_llm)
    providers = {
        "Gemini": FakeModel(lambda r: [text_msg("Gemini draft")]),
        "Groq": FakeModel(lambda r: [text_msg("Groq draft")]),
    }
    await lab3.multi_model_manager(providers)
    out = capsys.readouterr().out
    assert out.count("[console] Subject:") == 1 and "Sent the best one." in out
    manager = next(r for r in requests if "Sales Manager" in r.system)
    assert manager.tool_names == ["sales_agent1", "sales_agent2", "send_email_tool"]
    # each provider's own model actually served its tool call
    assert providers["Gemini"].requests and providers["Groq"].requests


async def test_multi_model_manager_skips_when_no_providers(capsys):
    await lab3.multi_model_manager({})
    assert "skipping" in capsys.readouterr().out.lower()


# ---------- Lab 3: optional extras ----------

def test_sandbox_agent_builds_with_manifest_and_output_grant(tmp_path):
    code, out = tmp_path / "code", tmp_path / "output"
    code.mkdir(), out.mkdir()
    agent, run_config = lab3.build_sandbox_agent(code, out)
    assert agent.name == "Engineer"
    assert str(out) in agent.instructions
    assert "code" in agent.default_manifest.entries
    assert run_config.workflow_name == "Sandbox coding example"


async def test_sandbox_demo_skips_cleanly_without_code_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    await lab3.sandbox_demo()
    assert "skipping" in capsys.readouterr().out.lower()


# ---------- Lab 4: deep research ----------

def test_plan_and_report_schemas():
    assert set(dr.WebSearchPlan.model_json_schema()["properties"]) == {"searches"}
    assert set(dr.WebSearchItem.model_json_schema()["properties"]) == {"reason", "query"}
    assert set(dr.ReportData.model_json_schema()["properties"]) == {"short_summary", "markdown_report", "follow_up_questions"}


def test_agent_configuration():
    assert dr.planner_agent.output_type is dr.WebSearchPlan
    assert dr.writer_agent.output_type is dr.ReportData
    assert dr.search_agent.model_settings.tool_choice == "required"
    assert [type(t).__name__ for t in dr.search_agent.tools] == ["WebSearchTool"]
    assert [t.name for t in dr.email_agent.tools] == ["send_email_tool"]
    assert str(dr.HOW_MANY_SEARCHES) in dr.planner_agent.instructions


def research_llm(searches=3):
    send_id = "research_send"

    def respond(req):
        if "come up with a set of web searches" in req.system:
            return [text_msg(json.dumps({"searches": [{"reason": f"r{i}", "query": f"q{i}"} for i in range(searches)]}))]
        if "Search term:" in input_text(req.input) and "research assistant" in req.system:
            q = [w for w in input_text(req.input).split("Search term: ")[1].split("\\n")[0:1]][0]
            return [text_msg(f"Summary for {q}")]
        if "senior researcher" in req.system:
            seen = input_text(req.input)
            return [text_msg(json.dumps({
                "short_summary": "Short summary.",
                "markdown_report": "# Report\n" + " | ".join(f"saw q{i}" for i in range(searches) if f"Summary for q{i}" in seen),
                "follow_up_questions": ["What next?"],
            }))]
        if "provided with a detailed report" in req.system:
            if not has_output_for(req.input, send_id):
                return [tool_call("send_email_tool", {"subject": "Report", "text_body": "t", "html_body": "<h1>Report</h1>"}, send_id)]
            return [text_msg("emailed")]
        raise AssertionError(f"unexpected agent: {req.system[:60]}")

    return respond


async def test_pipeline_runs_all_searches_in_parallel_and_feeds_writer(fake_llm, capsys):
    requests = fake_llm(research_llm(searches=3))
    report = await dr.run_research("What are AI agents?", send_email=True)
    assert report.short_summary == "Short summary."
    # writer received every search summary -> proves fan-out results were collected and passed on
    assert "saw q0 | saw q1 | saw q2" in report.markdown_report
    # planner got the query, search agent ran once per planned search, email sent exactly once
    assert sum("Search term:" in input_text(r.input) for r in requests) == 3
    assert capsys.readouterr().out.count("[console] Subject:") == 1


async def test_pipeline_can_skip_email(fake_llm, capsys):
    fake_llm(research_llm(searches=2))
    await dr.run_research("q", send_email=False)
    assert "[console] Subject:" not in capsys.readouterr().out


async def test_cli_writes_report_file_and_prints_followups(fake_llm, tmp_path, capsys):
    fake_llm(research_llm(searches=2))
    out_file = tmp_path / "report.md"
    await lab4.main("Topic", send_email=False, output=out_file)
    assert out_file.read_text().startswith("# Report")
    out = capsys.readouterr().out
    assert "Short summary." in out and "What next?" in out and "report.md" in out


async def test_cli_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(SystemExit):
        await lab4.main("Topic", False, None)
