"""Aggregates a list of per-scenario ScenarioResults into summary metrics."""
from __future__ import annotations

from dataclasses import dataclass, field

from .schema import ScenarioResult


@dataclass(frozen=True)
class EvalReport:
    overall_accuracy: float
    per_tool_accuracy: dict[str, float]
    false_positive_rate: float
    total: int
    correct: int
    negative_total: int
    false_positives: int


def aggregate(results: list[ScenarioResult]) -> EvalReport:
    if not results:
        raise ValueError("cannot aggregate an empty result set")

    total = len(results)
    correct = sum(1 for r in results if r.correct)

    negatives = [r for r in results if r.scenario.expected_tool is None]
    false_positives = sum(1 for r in negatives if r.false_positive)
    false_positive_rate = false_positives / len(negatives) if negatives else 0.0

    per_tool_hits: dict[str, int] = {}
    per_tool_total: dict[str, int] = {}
    for r in results:
        tool = r.scenario.expected_tool
        if tool is None:
            continue
        per_tool_total[tool] = per_tool_total.get(tool, 0) + 1
        if r.correct:
            per_tool_hits[tool] = per_tool_hits.get(tool, 0) + 1

    per_tool_accuracy = {
        tool: per_tool_hits.get(tool, 0) / count for tool, count in per_tool_total.items()
    }

    return EvalReport(
        overall_accuracy=correct / total,
        per_tool_accuracy=per_tool_accuracy,
        false_positive_rate=false_positive_rate,
        total=total,
        correct=correct,
        negative_total=len(negatives),
        false_positives=false_positives,
    )
