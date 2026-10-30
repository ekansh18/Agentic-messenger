"""Lab 4: Deep Research agent (planner -> parallel web searches -> writer -> emailer).

Run:  python -m examples.lab4_deep_research "Most popular AI Agent frameworks in 2026"
      python -m examples.lab4_deep_research "..." --no-email --output report.md

Note: WebSearchTool is an OpenAI hosted tool and is billed per call (about 5 searches per run).
"""

import argparse
import asyncio
from pathlib import Path

from agentic_lab.config import require_env
from agentic_lab.deep_research import run_research

DEFAULT_QUERY = "Most popular AI Agent frameworks in 2026"


async def main(query: str, send_email: bool, output: Path | None) -> None:
    require_env("OPENAI_API_KEY")
    print(f"Starting research: {query}")
    report = await run_research(query, send_email=send_email)

    print(f"\nSummary: {report.short_summary}\n")
    if output:
        output.write_text(report.markdown_report, encoding="utf-8")
        print(f"Report written to {output}")
    else:
        print(report.markdown_report)

    print("\nFollow-up questions:")
    for q in report.follow_up_questions:
        print(f"  - {q}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--no-email", action="store_true", help="do not send the report by email/push")
    parser.add_argument("--output", type=Path, help="write the markdown report to this file")
    args = parser.parse_args()
    asyncio.run(main(args.query, not args.no_email, args.output))
