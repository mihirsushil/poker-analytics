"""CLI entrypoint for Phase 3: run the tool-selection eval against one or more live models.

NOT executed as part of this project's test suite or CI - this makes real, billed API calls.
Requires ANTHROPIC_API_KEY and/or OPENAI_API_KEY in the environment depending on --models.

    python -m poker_analytics.eval.run --models claude-sonnet-5,claude-haiku-4-5 \\
        --scenarios eval/scenarios
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .aggregate import aggregate
from .clients import make_client
from .loader import load_scenarios
from .scorer import score


def run_eval(model_name: str, scenarios_dir: Path) -> None:
    client = make_client(model_name)
    scenarios = load_scenarios(scenarios_dir)
    results = [score(s, client.select_tool(s.prompt)) for s in scenarios]
    report = aggregate(results)

    print(f"\n=== {model_name} ===")
    print(f"Overall accuracy: {report.correct}/{report.total} ({report.overall_accuracy:.1%})")
    print(f"False-positive rate on negative scenarios: {report.false_positives}/{report.negative_total} ({report.false_positive_rate:.1%})")
    print("Per-tool accuracy:")
    for tool, acc in sorted(report.per_tool_accuracy.items()):
        print(f"  {tool}: {acc:.1%}")

    misses = [r for r in results if not r.correct]
    if misses:
        print(f"\n{len(misses)} miss(es):")
        for r in misses:
            print(f"  [{r.scenario.id}] {r.reason}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="comma-separated model names, e.g. claude-sonnet-5,gpt-4o-mini")
    parser.add_argument("--scenarios", default="eval/scenarios", help="directory of scenario YAML files")
    args = parser.parse_args(argv)

    scenarios_dir = Path(args.scenarios)
    for model_name in args.models.split(","):
        run_eval(model_name.strip(), scenarios_dir)


if __name__ == "__main__":
    sys.exit(main())
