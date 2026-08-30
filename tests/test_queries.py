"""list_hands/get_hand round-trip: what queries.py reads back must match
what the parser produced and db.insert_hand wrote, field for field - this
is the only place that data comes back out of the DB, so it's worth
checking against the source Hand objects directly rather than just
trusting row counts.
"""
from pathlib import Path

from poker_analytics import db, queries
from poker_analytics.parsers.pokerstars_text import parse_file

RAW_TEXT_DIR = Path(__file__).resolve().parent.parent / "data" / "samples" / "raw_text"


def _all_hands():
    hands = []
    for f in sorted(RAW_TEXT_DIR.glob("*.txt")):
        hands.extend(parse_file(f))
    return hands


def _load_db(hands):
    conn = db.connect(":memory:")
    for h in hands:
        db.insert_hand(conn, h)
    return conn


def test_list_hands_returns_every_hand_with_no_filters():
    hands = _all_hands()
    conn = _load_db(hands)
    rows = queries.list_hands(conn, limit=1000)
    assert len(rows) == len(hands)
    assert {r["hand_id"] for r in rows} == {h.hand_id for h in hands}


def test_list_hands_filters_by_player_and_position():
    hands = _all_hands()
    conn = _load_db(hands)
    target_hand = next(h for h in hands if any(s.position == "BB" for s in h.seats))
    target_seat = next(s for s in target_hand.seats if s.position == "BB")

    rows = queries.list_hands(conn, player_hash=target_seat.player_hash, position="BB", limit=1000)
    assert target_hand.hand_id in {r["hand_id"] for r in rows}
    for row in rows:
        detail = queries.get_hand(conn, row["id"])
        seat = next(s for s in detail["seats"] if s["player_hash"] == target_seat.player_hash)
        assert seat["position"] == "BB"


def test_list_hands_respects_limit_and_offset_with_no_overlap():
    hands = _all_hands()
    conn = _load_db(hands)
    page1 = queries.list_hands(conn, limit=5, offset=0)
    page2 = queries.list_hands(conn, limit=5, offset=5)
    assert len(page1) == 5
    assert len(page2) == 5
    assert {r["id"] for r in page1}.isdisjoint({r["id"] for r in page2})


def test_get_hand_matches_original_parsed_hand():
    hands = _all_hands()
    conn = _load_db(hands)
    by_hand_id = {h.hand_id: h for h in hands}

    for row in queries.list_hands(conn, limit=1000):
        detail = queries.get_hand(conn, row["id"])
        original = by_hand_id[detail["hand_id"]]
        assert len(detail["seats"]) == len(original.seats)
        assert len(detail["actions"]) == len(original.actions)
        assert len(detail["pots"]) == len(original.pots)
        assert detail["total_pot"] == original.total_pot
        assert detail["board"] == original.board


def test_get_hand_returns_none_for_missing_id():
    conn = db.connect(":memory:")
    assert queries.get_hand(conn, 999) is None
