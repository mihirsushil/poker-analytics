"""Unit tests for the scenario loader, plus a sanity check on the real scenario set."""
from pathlib import Path

import pytest

from poker_analytics.eval.loader import load_scenarios
from poker_analytics.eval.tool_specs import TOOL_SPECS

FIXTURES = Path(__file__).parent / "fixtures"
REAL_SCENARIOS_DIR = Path(__file__).parent.parent / "eval" / "scenarios"
VALID_TOOL_NAMES = {t["name"] for t in TOOL_SPECS}


def test_loads_and_merges_multiple_files_sorted_by_id():
    scenarios = load_scenarios(FIXTURES / "eval_scenarios_ok")
    ids = [s.id for s in scenarios]
    assert ids == sorted(ids)
    assert {"fixture_a1", "fixture_a2", "fixture_b1"} == set(ids)


def test_defaults_are_applied_when_omitted():
    scenarios = load_scenarios(FIXTURES / "eval_scenarios_ok")
    by_id = {s.id: s for s in scenarios}
    assert by_id["fixture_a2"].expected_args == {}
    assert by_id["fixture_a2"].unexpected_args_absent == ()
    assert by_id["fixture_a2"].notes == ""


def test_duplicate_id_across_files_raises():
    with pytest.raises(ValueError, match="duplicate scenario id"):
        load_scenarios(FIXTURES / "eval_scenarios_dup")


def test_real_scenario_set_loads_cleanly():
    scenarios = load_scenarios(REAL_SCENARIOS_DIR)
    assert 40 <= len(scenarios) <= 60, f"expected 40-60 scenarios per the Phase 3 plan, got {len(scenarios)}"


def test_real_scenario_set_only_references_real_tools():
    scenarios = load_scenarios(REAL_SCENARIOS_DIR)
    for s in scenarios:
        if s.expected_tool is not None:
            assert s.expected_tool in VALID_TOOL_NAMES, f"{s.id} references unknown tool {s.expected_tool!r}"


def test_real_scenario_set_has_negative_scenarios():
    scenarios = load_scenarios(REAL_SCENARIOS_DIR)
    negatives = [s for s in scenarios if s.expected_tool is None]
    assert len(negatives) >= 5, "need enough negative scenarios for a stable false-positive rate"


def test_real_scenario_set_covers_every_tool():
    scenarios = load_scenarios(REAL_SCENARIOS_DIR)
    covered = {s.expected_tool for s in scenarios if s.expected_tool is not None}
    assert covered == VALID_TOOL_NAMES, f"missing coverage for {VALID_TOOL_NAMES - covered}"
