"""Extraction sanity checks: does the state recorded at each decision point
(pot, stack, board, action history) actually match the real action sequence
that produced it, and are seats with unknown hole cards correctly excluded
from the label set while still contributing to pot/history for later
decisions in the same hand? Same "reconstruct the real sequence and check
the arithmetic" spirit as the Phase 1 money-conservation tests.
"""
from datetime import datetime

from poker_analytics.db import connect, insert_hand
from poker_analytics.models import Action, ActionType, Hand, Seat, Street
from poker_analytics.phase4.extract import extract_decision_points


def _heads_up_hand() -> Hand:
    """SB (hero, known cards) calls, BB (unknown cards) checks; flop: SB bets,
    BB calls. Only SB's two actions should produce decision examples."""
    sb = Seat(seat_no=1, player_hash="sb_hash", starting_stack=100.0, position="SB", hole_cards=["Ah", "Kh"])
    bb = Seat(seat_no=2, player_hash="bb_hash", starting_stack=100.0, position="BB", hole_cards=None)

    actions = [
        Action(Street.PREFLOP, 0, "sb_hash", ActionType.POST_SB, 1.0),
        Action(Street.PREFLOP, 1, "bb_hash", ActionType.POST_BB, 2.0),
        Action(Street.PREFLOP, 2, "sb_hash", ActionType.CALL, 1.0),
        Action(Street.PREFLOP, 3, "bb_hash", ActionType.CHECK, 0.0),
        Action(Street.FLOP, 4, "sb_hash", ActionType.BET, 5.0),
        Action(Street.FLOP, 5, "bb_hash", ActionType.CALL, 5.0),
    ]

    return Hand(
        hand_id="test-1",
        site="PokerStars",
        source="raw_text",
        source_file="test.txt",
        game="Hold'em",
        limit_type="No Limit",
        is_tournament=False,
        tournament_id=None,
        currency="$",
        table_name="Test Table",
        max_seats=2,
        button_seat=1,
        played_at=datetime(2026, 1, 1),
        seats=[sb, bb],
        actions=actions,
        board=["2c", "7d", "9h", "Ks", "3d"],
        pots=[],
        total_pot=14.0,
        rake=0.0,
    )


def _extract_from_hand(hand: Hand):
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.unlink(path)  # connect() creates it fresh; a pre-existing empty file confuses sqlite
    try:
        conn = connect(path)
        insert_hand(conn, hand)
        conn.close()
        return extract_decision_points(path)
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_only_seats_with_known_hole_cards_produce_examples():
    examples = _extract_from_hand(_heads_up_hand())
    assert len(examples) == 2
    assert all(ex.hole_cards == "Ah,Kh" for ex in examples)


def test_preflop_decision_state_matches_action_sequence():
    examples = _extract_from_hand(_heads_up_hand())
    preflop = examples[0]
    assert preflop.street == "preflop"
    assert preflop.action_type == "call"
    assert preflop.action_label == "call"
    assert preflop.pot_before == 3.0  # SB's 1 + BB's 2, before SB's call lands
    assert preflop.stack_before == 99.0  # 100 - the 1 already posted as SB
    assert preflop.board_so_far == ""  # preflop: no community cards yet
    assert preflop.action_history == "(first to act)"  # blinds aren't decisions


def test_flop_decision_state_reflects_preceding_actions():
    examples = _extract_from_hand(_heads_up_hand())
    flop = examples[1]
    assert flop.street == "flop"
    assert flop.action_type == "bet"
    assert flop.action_label == "bet 5"
    assert flop.pot_before == 4.0  # 1 + 2 + 1 (SB's preflop call)
    assert flop.stack_before == 98.0  # 100 - 1 (SB post) - 1 (SB call)
    assert flop.board_so_far == "2c,7d,9h"  # only the flop's 3 cards, not turn/river
    assert flop.action_history == "SB: call | BB: check"


def test_raise_label_uses_to_amount_not_amount():
    hero = Seat(seat_no=1, player_hash="h", starting_stack=200.0, position="BTN", hole_cards=["Qs", "Qd"])
    villain = Seat(seat_no=2, player_hash="v", starting_stack=200.0, position="BB", hole_cards=None)
    actions = [
        Action(Street.PREFLOP, 0, "v", ActionType.POST_BB, 2.0),
        Action(Street.PREFLOP, 1, "h", ActionType.RAISE, 8.0, to_amount=10.0),
    ]
    hand = Hand(
        hand_id="test-2", site="PokerStars", source="phh", source_file="t",
        game="Hold'em", limit_type="No Limit", is_tournament=False, tournament_id=None,
        currency="$", table_name="T", max_seats=2, button_seat=1, played_at=datetime(2026, 1, 1),
        seats=[hero, villain], actions=actions, board=[], pots=[], total_pot=10.0, rake=0.0,
    )
    examples = _extract_from_hand(hand)
    assert len(examples) == 1
    assert examples[0].action_label == "raise to 10"
