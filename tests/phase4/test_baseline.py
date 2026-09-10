import pytest

from poker_analytics.phase4.baseline import MajorityClassBaseline
from poker_analytics.phase4.extract import DecisionExample


def _example(action_type: str) -> DecisionExample:
    return DecisionExample(
        hand_pk=1, site="PokerStars", source="phh", street="preflop", position="BTN",
        hole_cards="Ah,Kh", board_so_far="", pot_before=1.0, stack_before=99.0,
        action_history="(first to act)", action_type=action_type, amount=0.0, to_amount=None,
        action_label=action_type,
    )


def test_picks_the_most_common_class():
    train = [_example("check")] * 5 + [_example("fold")] * 2
    baseline = MajorityClassBaseline(train)
    assert baseline.majority_class == "check"
    assert baseline.majority_count == 5


def test_predict_ignores_the_input_state():
    train = [_example("call")] * 3 + [_example("fold")] * 1
    baseline = MajorityClassBaseline(train)
    assert baseline.predict(_example("raise")) == "call"
    assert baseline.predict(_example("bet")) == "call"


def test_empty_training_set_raises():
    with pytest.raises(ValueError):
        MajorityClassBaseline([])
