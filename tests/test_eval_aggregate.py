"""Unit tests for eval report aggregation - synthetic ScenarioResults, no live model calls."""
import pytest

from poker_analytics.eval.aggregate import aggregate
from poker_analytics.eval.schema import Scenario, ScenarioResult, ToolCall


def make_result(expected_tool, correct, false_positive=False, scenario_id="s") -> ScenarioResult:
    scenario = Scenario(
        id=scenario_id,
        category="negative" if expected_tool is None else "single_tool",
        prompt="irrelevant",
        expected_tool=expected_tool,
        expected_args={},
    )
    actual = ToolCall(tool_name=expected_tool if correct else "wrong_tool", args={})
    return ScenarioResult(scenario, actual, correct=correct, false_positive=false_positive, reason="test")


def test_aggregate_rejects_empty_results():
    with pytest.raises(ValueError):
        aggregate([])


def test_overall_accuracy_is_fraction_correct():
    results = [
        make_result("query_hands", correct=True, scenario_id="a"),
        make_result("query_hands", correct=False, scenario_id="b"),
        make_result("compute_stats", correct=True, scenario_id="c"),
        make_result("compute_stats", correct=True, scenario_id="d"),
    ]
    report = aggregate(results)
    assert report.total == 4
    assert report.correct == 3
    assert report.overall_accuracy == pytest.approx(0.75)


def test_per_tool_accuracy_is_isolated_per_tool():
    results = [
        make_result("query_hands", correct=True, scenario_id="a"),
        make_result("query_hands", correct=True, scenario_id="b"),
        make_result("query_hands", correct=False, scenario_id="c"),
        make_result("find_leaks", correct=False, scenario_id="d"),
    ]
    report = aggregate(results)
    assert report.per_tool_accuracy["query_hands"] == pytest.approx(2 / 3)
    assert report.per_tool_accuracy["find_leaks"] == pytest.approx(0.0)


def test_false_positive_rate_only_considers_negative_scenarios():
    results = [
        make_result(None, correct=True, false_positive=False, scenario_id="neg-ok"),
        make_result(None, correct=False, false_positive=True, scenario_id="neg-bad"),
        make_result("query_hands", correct=False, scenario_id="pos-miss"),  # not a negative scenario
    ]
    report = aggregate(results)
    assert report.negative_total == 2
    assert report.false_positives == 1
    assert report.false_positive_rate == pytest.approx(0.5)


def test_negative_scenarios_excluded_from_per_tool_accuracy():
    results = [
        make_result(None, correct=True, scenario_id="neg"),
        make_result("query_hands", correct=True, scenario_id="pos"),
    ]
    report = aggregate(results)
    assert None not in report.per_tool_accuracy
    assert set(report.per_tool_accuracy) == {"query_hands"}
