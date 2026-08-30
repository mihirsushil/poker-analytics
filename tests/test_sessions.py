"""Gap-based session detection over synthetic, hand-computable timestamps."""
from datetime import datetime

from poker_analytics import db, sessions
from poker_analytics.models import Hand, Seat

PLAYER = "hero"


def _hand(n: int, played_at: datetime, table_name: str, net_result: float | None) -> Hand:
    seats = [Seat(seat_no=1, player_hash=PLAYER, starting_stack=100, net_result=net_result)]
    return Hand(
        hand_id=str(n), site="TestSite", source="synthetic", source_file="test",
        game="Hold'em", limit_type="No Limit", is_tournament=False, tournament_id=None,
        currency="$", table_name=table_name, max_seats=1, button_seat=1,
        played_at=played_at, seats=seats, actions=[], board=[], pots=[], total_pot=0, rake=0,
    )


def _load_db(hands: list[Hand]):
    conn = db.connect(":memory:")
    for h in hands:
        db.insert_hand(conn, h)
    return conn


def test_hands_within_the_gap_merge_into_one_session():
    hands = [
        _hand(1, datetime(2024, 1, 1, 10, 0), "T1", 5),
        _hand(2, datetime(2024, 1, 1, 10, 10), "T1", -2),
        _hand(3, datetime(2024, 1, 1, 10, 25), "T1", 3),
    ]
    conn = _load_db(hands)
    result = sessions.session_summary(conn, PLAYER, gap_minutes=30)
    assert len(result) == 1
    assert result[0]["hand_count"] == 3
    assert result[0]["net_result"] == 6
    assert result[0]["duration_minutes"] == 25.0


def test_a_gap_over_the_threshold_starts_a_new_session():
    hands = [
        _hand(1, datetime(2024, 1, 1, 10, 0), "T1", 5),
        _hand(2, datetime(2024, 1, 1, 10, 45), "T1", -2),  # 45 min later
    ]
    conn = _load_db(hands)
    result = sessions.session_summary(conn, PLAYER, gap_minutes=30)
    assert len(result) == 2
    assert [s["hand_count"] for s in result] == [1, 1]


def test_multi_tabling_is_one_session_not_one_per_table():
    hands = [
        _hand(1, datetime(2024, 1, 1, 10, 0, 0), "T1", 5),
        _hand(2, datetime(2024, 1, 1, 10, 0, 30), "T2", -2),
        _hand(3, datetime(2024, 1, 1, 10, 1, 0), "T1", 1),
    ]
    conn = _load_db(hands)
    result = sessions.session_summary(conn, PLAYER, gap_minutes=30)
    assert len(result) == 1
    assert result[0]["tables"] == ["T1", "T2"]
    assert result[0]["hand_count"] == 3


def test_no_hands_returns_empty_list():
    conn = db.connect(":memory:")
    assert sessions.session_summary(conn, "nobody") == []
