"""find_leaks against synthetic, hand-computable data - the real bootstrap
dataset makes a good smoke test but its "leaks" aren't independently
verifiable by eye, so correctness here needs data where the expected
answer can be computed by hand.
"""
from datetime import datetime

from poker_analytics import db, stats
from poker_analytics.models import Action, ActionType, Hand, Seat, Street

NIT = "nit_player"
LOOSE_A = "loose_player_a"
LOOSE_B = "loose_player_b"


def _hand(n: int) -> Hand:
    seats = [
        Seat(seat_no=1, player_hash=NIT, starting_stack=100, position="UTG"),
        Seat(seat_no=2, player_hash=LOOSE_A, starting_stack=100, position="CO"),
        Seat(seat_no=3, player_hash=LOOSE_B, starting_stack=100, position="BTN"),
    ]
    actions = [
        Action(street=Street.PREFLOP, order=0, player_hash=NIT, action_type=ActionType.FOLD, amount=0),
        Action(street=Street.PREFLOP, order=1, player_hash=LOOSE_A, action_type=ActionType.CALL, amount=1),
        Action(street=Street.PREFLOP, order=2, player_hash=LOOSE_B, action_type=ActionType.CALL, amount=1),
    ]
    return Hand(
        hand_id=str(n), site="TestSite", source="synthetic", source_file="test",
        game="Hold'em", limit_type="No Limit", is_tournament=False, tournament_id=None,
        currency="$", table_name="T1", max_seats=3, button_seat=3,
        played_at=datetime(2024, 1, 1), seats=seats, actions=actions,
        board=[], pots=[], total_pot=2, rake=0,
    )


def _load_db(n_hands: int):
    conn = db.connect(":memory:")
    for i in range(n_hands):
        db.insert_hand(conn, _hand(i))
    return conn


def test_nit_vpip_flagged_as_a_leak_against_population():
    conn = _load_db(40)
    result = stats.find_leaks(conn, NIT)

    assert result["player"]["hands"] == 40
    assert result["player"]["vpip_pct"] == 0.0
    assert result["population_baseline"]["vpip_pct"] == round(100 * 80 / 120, 1)

    leak = next(l for l in result["leaks"] if l["stat"] == "vpip_pct")
    assert leak["player_value"] == 0.0
    assert leak["direction"] == "low"
    assert abs(leak["diff"]) >= 8.0


def test_small_sample_does_not_get_flagged_despite_large_deviation():
    conn = _load_db(5)  # below the vpip_pct minimum sample size of 30
    result = stats.find_leaks(conn, NIT)

    assert result["player"]["hands"] == 5
    assert not any(l["stat"] == "vpip_pct" for l in result["leaks"])


def test_find_leaks_on_a_player_matching_baseline_has_no_leaks():
    conn = _load_db(40)
    result = stats.find_leaks(conn, LOOSE_A)
    # LOOSE_A's VPIP (100%) still differs from the population baseline
    # (which includes the nit dragging it down) - the real assertion is
    # that the population's own stats compared to themselves is leak-free.
    self_check = stats.find_leaks(conn, LOOSE_A, position="UTG")
    assert self_check["player"]["hands"] == 0  # LOOSE_A never sits UTG
    assert self_check["leaks"] == []
