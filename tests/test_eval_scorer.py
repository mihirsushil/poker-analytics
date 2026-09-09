"""Unit tests for the eval scorer - no live model calls, all synthetic ToolCalls."""
from poker_analytics.eval.schema import Scenario, ToolCall
from poker_analytics.eval.scorer import score


def make_scenario(**overrides) -> Scenario:
    defaults = dict(
        id="test-scenario",
        category="single_tool",
        prompt="irrelevant for scoring tests",
        expected_tool="query_hands",
        expected_args={},
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def test_exact_tool_and_arg_match_is_correct():
    scenario = make_scenario(expected_args={"position": "BTN"})
    actual = ToolCall(tool_name="query_hands", args={"position": "BTN", "limit": 50})
    result = score(scenario, actual)
    assert result.correct
    assert not result.false_positive


def test_wrong_tool_is_incorrect_not_false_positive():
    scenario = make_scenario(expected_tool="compute_stats")
    actual = ToolCall(tool_name="find_leaks", args={})
    result = score(scenario, actual)
    assert not result.correct
    assert not result.false_positive  # false_positive is reserved for negative scenarios


def test_right_tool_wrong_arg_value_is_incorrect():
    scenario = make_scenario(expected_args={"position": "BTN"})
    actual = ToolCall(tool_name="query_hands", args={"position": "SB"})
    result = score(scenario, actual)
    assert not result.correct


def test_missing_expected_arg_is_incorrect():
    scenario = make_scenario(expected_args={"position": "BTN"})
    actual = ToolCall(tool_name="query_hands", args={"limit": 10})
    result = score(scenario, actual)
    assert not result.correct


def test_unlisted_actual_args_are_ignored():
    """Only keys present in expected_args are checked - extra real args (limit, offset,
    player_hash) shouldn't fail a scenario that doesn't care about them."""
    scenario = make_scenario(expected_args={"position": "BTN"})
    actual = ToolCall(tool_name="query_hands", args={"position": "BTN", "limit": 50, "offset": 0, "player_hash": "abc123"})
    result = score(scenario, actual)
    assert result.correct


def test_negative_scenario_no_call_is_correct():
    scenario = make_scenario(category="negative", expected_tool=None)
    actual = ToolCall(tool_name=None, args={})
    result = score(scenario, actual)
    assert result.correct
    assert not result.false_positive


def test_negative_scenario_with_call_is_false_positive():
    scenario = make_scenario(category="negative", expected_tool=None)
    actual = ToolCall(tool_name="query_hands", args={})
    result = score(scenario, actual)
    assert not result.correct
    assert result.false_positive


def test_unexpected_args_absent_violation_is_incorrect():
    scenario = make_scenario(expected_tool="compute_stats", unexpected_args_absent=("player_hash",))
    actual = ToolCall(tool_name="compute_stats", args={"player_hash": "should-not-be-here"})
    result = score(scenario, actual)
    assert not result.correct


def test_unexpected_args_absent_satisfied_is_correct():
    scenario = make_scenario(expected_tool="compute_stats", unexpected_args_absent=("player_hash",))
    actual = ToolCall(tool_name="compute_stats", args={"position": "BTN"})
    result = score(scenario, actual)
    assert result.correct
