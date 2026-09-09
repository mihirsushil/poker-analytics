"""Scores a single model ToolCall against the Scenario that produced it."""
from __future__ import annotations

from .schema import Scenario, ScenarioResult, ToolCall


def score(scenario: Scenario, actual: ToolCall) -> ScenarioResult:
    if scenario.expected_tool is None:
        # Negative scenario: correct iff the model called no tool at all.
        if actual.tool_name is None:
            return ScenarioResult(scenario, actual, correct=True, false_positive=False, reason="no tool called, as expected")
        return ScenarioResult(
            scenario, actual, correct=False, false_positive=True,
            reason=f"expected no tool call, model called {actual.tool_name!r}",
        )

    if actual.tool_name != scenario.expected_tool:
        return ScenarioResult(
            scenario, actual, correct=False, false_positive=False,
            reason=f"expected tool {scenario.expected_tool!r}, model called {actual.tool_name!r}",
        )

    for key, expected_value in scenario.expected_args.items():
        if key not in actual.args:
            return ScenarioResult(
                scenario, actual, correct=False, false_positive=False,
                reason=f"missing expected arg {key!r}",
            )
        if actual.args[key] != expected_value:
            return ScenarioResult(
                scenario, actual, correct=False, false_positive=False,
                reason=f"arg {key!r} = {actual.args[key]!r}, expected {expected_value!r}",
            )

    for key in scenario.unexpected_args_absent:
        if actual.args.get(key) is not None:
            return ScenarioResult(
                scenario, actual, correct=False, false_positive=False,
                reason=f"arg {key!r} should be absent/None, got {actual.args.get(key)!r}",
            )

    return ScenarioResult(scenario, actual, correct=True, false_positive=False, reason="match")
