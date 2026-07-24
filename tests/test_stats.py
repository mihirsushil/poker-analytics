"""VPIP/PFR are the two stats the project's milestone requires
hand-validating. This cross-checks stats.vpip_pfr() (SQL-based) against an
independently-written, brute-force recomputation directly over parsed Hand
objects - two different code paths agreeing is the actual validation, not
just eyeballing a handful of hands.
"""
from pathlib import Path

from poker_analytics import db, stats
from poker_analytics.models import ActionType, Street
from poker_analytics.parsers.phh import parse_phhs_file
from poker_analytics.parsers.pokerstars_text import parse_file

PHH_FIXTURE = Path(__file__).resolve().parent.parent / "data" / "samples" / "phh" / "fixtures" / "sample_50_hands.phhs"
RAW_TEXT_DIR = Path(__file__).resolve().parent.parent / "data" / "samples" / "raw_text"


def _brute_force_vpip_pfr(hands, player_hash: str | None = None) -> dict:
    vpip_flags, pfr_flags = [], []
    for hand in hands:
        for seat in hand.seats:
            if player_hash is not None and seat.player_hash != player_hash:
                continue
            preflop_actions = [a for a in hand.actions if a.street == Street.PREFLOP and a.player_hash == seat.player_hash]
            vpip_flags.append(any(a.action_type in (ActionType.CALL, ActionType.BET, ActionType.RAISE) for a in preflop_actions))
            pfr_flags.append(any(a.action_type in (ActionType.BET, ActionType.RAISE) for a in preflop_actions))
    n = len(vpip_flags)
    return {
        "hands": n,
        "vpip_pct": round(100 * sum(vpip_flags) / n, 1) if n else None,
        "pfr_pct": round(100 * sum(pfr_flags) / n, 1) if n else None,
    }


def _load_db(hands):
    conn = db.connect(":memory:")
    for h in hands:
        db.insert_hand(conn, h)
    return conn


def test_vpip_pfr_matches_brute_force_on_phh_fixture():
    hands = parse_phhs_file(PHH_FIXTURE)
    assert len(hands) >= 46
    conn = _load_db(hands)

    expected = _brute_force_vpip_pfr(hands)
    actual = stats.vpip_pfr(conn)
    assert actual["hands"] == expected["hands"]
    assert actual["vpip_pct"] == expected["vpip_pct"]
    assert actual["pfr_pct"] == expected["pfr_pct"]


def test_vpip_pfr_matches_brute_force_per_player():
    hands = parse_phhs_file(PHH_FIXTURE)
    conn = _load_db(hands)
    some_player = hands[0].seats[0].player_hash

    expected = _brute_force_vpip_pfr(hands, player_hash=some_player)
    actual = stats.vpip_pfr(conn, player_hash=some_player)
    assert actual == expected


def test_vpip_pfr_matches_brute_force_on_raw_text_fixtures():
    hands = []
    for f in sorted(RAW_TEXT_DIR.glob("*.txt")):
        hands.extend(parse_file(f))
    conn = _load_db(hands)

    expected = _brute_force_vpip_pfr(hands)
    actual = stats.vpip_pfr(conn)
    assert actual == expected


def test_milestone_100_hands_with_stats():
    """The project's Phase 1 milestone: 100+ hands parsed with a correct,
    computable VPIP/PFR. Loads every vendored/fetched bootstrap hand
    (raw-text fixtures + any fetched phh bulk volume) into one DB.
    """
    hands = []
    for f in sorted(RAW_TEXT_DIR.glob("*.txt")):
        hands.extend(parse_file(f))
    bulk_dir = Path(__file__).resolve().parent.parent / "data" / "samples" / "phh" / "bulk"
    if bulk_dir.exists():
        for f in sorted(bulk_dir.glob("*.phhs")):
            hands.extend(parse_phhs_file(f))
    else:
        hands.extend(parse_phhs_file(PHH_FIXTURE))

    conn = db.connect(":memory:")
    inserted = 0
    for h in hands:
        try:
            db.insert_hand(conn, h)
            inserted += 1
        except Exception:
            pass  # duplicate (site, hand_id) across overlapping fetch runs

    assert inserted >= 100
    report = stats.vpip_pfr(conn)
    assert report["vpip_pct"] is not None
    assert report["pfr_pct"] is not None
    assert 0 <= report["vpip_pct"] <= 100
    assert 0 <= report["pfr_pct"] <= 100
