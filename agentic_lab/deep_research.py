"""Deep research pipeline: planner -> parallel searches -> writer -> emailer.

Orchestrated by code with one ``Runner.run`` per step and structured outputs throughout.
"""

import asyncio

from agents import Agent, Runner, WebSearchTool, trace
from agents.model_settings import ModelSettings
from pydantic import BaseModel, Field

from .config import MODEL_NAME
from .tools import send_email_tool

HOW_MANY_SEARCHES = 5


class WebSearchItem(BaseModel):
    reason: str = Field(description="Your reasoning for why this search is important to the query.")
    query: str = Field(description="The search term to use for the web search.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform to best answer the query.")


class ReportData(BaseModel):
    short_summary: str = Field(description="A short 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final report")
    follow_up_questions: list[str] = Field(description="Suggested topics to research further")


search_agent = Agent(
    name="Search Agent",
    instructions=(
        "You are a research assistant. Given a search term, you search the web for that term and "
        "produce a concise summary of the results. The summary must be 2-3 paragraphs and less than 300 words. "
        "Capture the main points and be succinct. Reply only with the summary."
    ),
    tools=[WebSearchTool()],  # OpenAI hosted tool: billed per call
    model=MODEL_NAME,
    model_settings=ModelSettings(tool_choice="required"),
)

planner_agent = Agent(
    name="Planner Agent",
    instructions=(
        "You are a research assistant. Given a user query, come up with a set of web searches "
        f"to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for."
    ),
    model=MODEL_NAME,
    output_type=WebSearchPlan,
)

writer_agent = Agent(
    name="Writer Agent",
    instructions=(
        "You are a senior researcher tasked with writing a cohesive report for a research query. "
        "You will be provided with the original query, and some research. "
        "Generate a comprehensive report based on the research and the query. "
        "The final output should be in markdown format, and it should be lengthy and detailed. "
        "Aim for 5-10 pages of content, at least 1000 words."
    ),
    model=MODEL_NAME,
    output_type=ReportData,
)

email_agent = Agent(
    name="Email Agent",
    instructions=(
        "You are provided with a detailed report. Use your tool to send an email, converting the report into "
        "a clean, well presented HTML email with an appropriate subject line."
    ),
    tools=[send_email_tool],
    model=MODEL_NAME,
)


async def plan_searches(query: str) -> WebSearchPlan:
    print("Planning searches...")
    result = await Runner.run(planner_agent, f"Query: {query}")
    print(f"Will perform {len(result.final_output.searches)} searches")
    return result.final_output


async def search(item: WebSearchItem) -> str:
    message = f"Search term: {item.query}\nReason for searching: {item.reason}"
    result = await Runner.run(search_agent, message)
    return str(result.final_output)


async def perform_searches(plan: WebSearchPlan) -> list[str]:
    results = await asyncio.gather(*(search(item) for item in plan.searches))
    print("Finished searching")
    return list(results)


async def write_report(query: str, search_results: list[str]) -> ReportData:
    print("Thinking about report...")
    message = f"Original query: {query}\nSummarized search results: {search_results}"
    result = await Runner.run(writer_agent, message)
    print("Finished writing report")
    return result.final_output


async def send_report_email(report: ReportData) -> None:
    print("Writing email...")
    await Runner.run(email_agent, report.markdown_report)
    print("Email step complete")


async def run_research(query: str, send_email: bool = True) -> ReportData:
    """Run the full pipeline and return the report."""
    with trace("Research trace"):
        plan = await plan_searches(query)
        results = await perform_searches(plan)
        report = await write_report(query, results)
        if send_email:
            await send_report_email(report)
    return report
