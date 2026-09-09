"""Data shapes for the Phase 3 tool-selection eval. See eval/README.md for the
scenario YAML schema this mirrors."""
from __future__ import annotations

from dataclasses import dataclass, field

VALID_CATEGORIES = frozenset({"single_tool", "argument_inference", "ambiguous", "negative"})


@dataclass(frozen=True)
class Scenario:
    id: str
    category: str
    prompt: str
    expected_tool: str | None
    expected_args: dict = field(default_factory=dict)
    unexpected_args_absent: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"{self.id}: unknown category {self.category!r}")
        if self.category == "negative" and self.expected_tool is not None:
            raise ValueError(f"{self.id}: category 'negative' requires expected_tool: null")
        if self.expected_tool is None and self.category != "negative":
            raise ValueError(f"{self.id}: expected_tool: null is only valid for category 'negative'")


@dataclass(frozen=True)
class ToolCall:
    """What a model actually did in response to a scenario's prompt. `tool_name is None`
    means the model answered without calling any tool."""

    tool_name: str | None
    args: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    actual: ToolCall
    correct: bool
    false_positive: bool
    reason: str
