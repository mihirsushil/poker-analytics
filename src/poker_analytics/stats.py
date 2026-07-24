"""VPIP/PFR/3-bet/etc. stats engine, sliceable by player, position, and date.

`player_hash=None` computes population-wide stats across every seat in the
filtered set (useful for the bootstrap dataset, which has no single
"hero" - and later, as a baseline to flag deviations against, e.g. for a
find_leaks-style tool). Passing a specific `player_hash` narrows to one
player's own tendencies, which is the self-play use case.

Definitions here are the standard HUD ones, deliberately simplified where
a full-precision version would need more game-state bookkeeping than this
project's scope calls for (documented per-function). VPIP and PFR are the
two stats the project's own milestone requires hand-validating; see
tests/test_stats.py for that cross-check against a brute-force
recomputation over the same hands.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


def _filter_clause(player_hash: str | None, position: str | None, start_date: str | None, end_date: str | None):
    clauses = []
    params: list = []
    if player_hash is not None:
        clauses.append("hs.player_hash = ?")
        params.append(player_hash)
    if position is not None:
        clauses.append("hs.position = ?")
        params.append(position)
    if start_date is not None:
        clauses.append("h.played_at >= ?")
        params.append(start_date)
    if end_date is not None:
        clauses.append("h.played_at <= ?")
        params.append(end_date)
    where = (" AND " + " AND ".join(clauses)) if clauses else ""
    return where, params


def vpip_pfr(conn: sqlite3.Connection, player_hash: str | None = None, position: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:
    where, params = _filter_clause(player_hash, position, start_date, end_date)
    rows = conn.execute(
        f"""
        SELECT
            MAX(CASE WHEN a.street = 'preflop' AND a.action_type IN ('call', 'bet', 'raise') THEN 1 ELSE 0 END) AS vpip,
            MAX(CASE WHEN a.street = 'preflop' AND a.action_type IN ('bet', 'raise') THEN 1 ELSE 0 END) AS pfr
        FROM hand_seats hs
        JOIN hands h ON h.id = hs.hand_id
        LEFT JOIN actions a ON a.hand_id = hs.hand_id AND a.player_hash = hs.player_hash
        WHERE 1=1{where}
        GROUP BY hs.id
        """,
        params,
    ).fetchall()
    n = len(rows)
    if n == 0:
        return {"hands": 0, "vpip_pct": None, "pfr_pct": None}
    return {
        "hands": n,
        "vpip_pct": round(100 * sum(r[0] for r in rows) / n, 1),
        "pfr_pct": round(100 * sum(r[1] for r in rows) / n, 1),
    }


@dataclass
class _PreflopRaise:
    order: int
    player_hash: str


def three_bet_and_fold_to_three_bet(conn: sqlite3.Connection, player_hash: str | None = None, position: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:
    """3-bet%% = (times made the first re-raise preflop) / (times had the
    chance to, i.e. acted preflop facing exactly one prior raise and were
    not that raiser). fold_to_3bet%% = (times folded facing a 3-bet after
    having opened) / (times faced a 3-bet after opening).
    """
    where, params = _filter_clause(player_hash, position, start_date, end_date)
    hand_ids = [r[0] for r in conn.execute(
        f"SELECT DISTINCT hs.hand_id FROM hand_seats hs JOIN hands h ON h.id = hs.hand_id WHERE 1=1{where}", params
    ).fetchall()]
    if not hand_ids:
        return {"three_bet_opportunities": 0, "three_bet_pct": None, "faced_3bet_after_opening": 0, "fold_to_3bet_pct": None}

    three_bet_opps = 0
    three_bets_made = 0
    opened_then_faced_3bet = 0
    folded_to_3bet = 0

    q_marks = ",".join("?" * len(hand_ids))
    action_rows = conn.execute(
        f"""
        SELECT hand_id, action_order, player_hash, action_type
        FROM actions
        WHERE street = 'preflop' AND hand_id IN ({q_marks})
        ORDER BY hand_id, action_order
        """,
        hand_ids,
    ).fetchall()

    by_hand: dict[int, list[tuple]] = {}
    for row in action_rows:
        by_hand.setdefault(row[0], []).append(row)

    target_players = {player_hash} if player_hash else None

    for hid, rows in by_hand.items():
        raises = [(order, ph) for (_, order, ph, atype) in rows if atype in ("bet", "raise")]
        if len(raises) < 1:
            continue
        opener = raises[0][1]
        seat_players = [r[2] for r in rows]
        distinct_players_in_order = list(dict.fromkeys(seat_players))

        if len(raises) >= 1:
            # Anyone acting after the opener, before a second raise happens, had a 3-bet opportunity.
            first_raise_order = raises[0][0]
            second_raise_order = raises[1][0] if len(raises) >= 2 else None
            for order, ph, atype in [(o, p, a) for (_, o, p, a) in rows]:
                if order <= first_raise_order or ph == opener:
                    continue
                if second_raise_order is not None and order > second_raise_order:
                    continue
                if target_players and ph not in target_players:
                    continue
                three_bet_opps += 1
                if atype in ("bet", "raise") and order == second_raise_order:
                    three_bets_made += 1

        if len(raises) >= 2 and (not target_players or opener in target_players):
            opened_then_faced_3bet += 1
            reraiser = raises[1][1]
            opener_action_after = next((atype for (_, o, ph, atype) in rows if ph == opener and o > raises[1][0]), None)
            if opener_action_after == "fold":
                folded_to_3bet += 1

    return {
        "three_bet_opportunities": three_bet_opps,
        "three_bet_pct": round(100 * three_bets_made / three_bet_opps, 1) if three_bet_opps else None,
        "faced_3bet_after_opening": opened_then_faced_3bet,
        "fold_to_3bet_pct": round(100 * folded_to_3bet / opened_then_faced_3bet, 1) if opened_then_faced_3bet else None,
    }


def cbet_pct(conn: sqlite3.Connection, player_hash: str | None = None, position: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:
    """c-bet%% = (times bet the flop after being the last preflop raiser and
    reaching the flop) / (times were the last preflop raiser and reached
    the flop).
    """
    where, params = _filter_clause(player_hash, position, start_date, end_date)
    hand_ids = [r[0] for r in conn.execute(
        f"SELECT DISTINCT hs.hand_id FROM hand_seats hs JOIN hands h ON h.id = hs.hand_id WHERE 1=1{where}", params
    ).fetchall()]
    if not hand_ids:
        return {"opportunities": 0, "cbet_pct": None}

    q_marks = ",".join("?" * len(hand_ids))
    rows = conn.execute(
        f"""
        SELECT hand_id, street, action_order, player_hash, action_type
        FROM actions
        WHERE hand_id IN ({q_marks}) AND street IN ('preflop', 'flop')
        ORDER BY hand_id, action_order
        """,
        hand_ids,
    ).fetchall()
    by_hand: dict[int, list[tuple]] = {}
    for row in rows:
        by_hand.setdefault(row[0], []).append(row)

    target_players = {player_hash} if player_hash else None
    opportunities = 0
    cbets = 0
    for hid, hand_rows in by_hand.items():
        preflop_raises = [r for r in hand_rows if r[1] == "preflop" and r[4] in ("bet", "raise")]
        flop_rows = [r for r in hand_rows if r[1] == "flop"]
        if not preflop_raises or not flop_rows:
            continue
        last_raiser = preflop_raises[-1][3]
        if target_players and last_raiser not in target_players:
            continue
        opportunities += 1
        if flop_rows[0][3] == last_raiser and flop_rows[0][4] in ("bet", "raise"):
            cbets += 1

    return {"opportunities": opportunities, "cbet_pct": round(100 * cbets / opportunities, 1) if opportunities else None}


def wtsd_and_wsd(conn: sqlite3.Connection, player_hash: str | None = None, position: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:
    """WTSD%% = went to showdown, given saw the flop. W$SD%% = won money,
    given went to showdown.

    "Went to showdown" is NOT simply "this seat's hole_cards are known" -
    a seat that mucks without showing at showdown never gets hole_cards
    populated, which would make W$SD look inflated (biased toward winners,
    who show far more often than losers do). Instead: the hand reached a
    real showdown (at least one seat's hole_cards were revealed) AND this
    seat never folded.
    """
    where, params = _filter_clause(player_hash, position, start_date, end_date)
    rows = conn.execute(
        f"""
        SELECT hs.hand_id, hs.player_hash, hs.net_result,
               EXISTS(SELECT 1 FROM actions a WHERE a.hand_id = hs.hand_id AND a.player_hash = hs.player_hash AND a.street = 'flop') AS saw_flop,
               EXISTS(SELECT 1 FROM actions a WHERE a.hand_id = hs.hand_id AND a.player_hash = hs.player_hash AND a.action_type = 'fold') AS folded,
               EXISTS(SELECT 1 FROM hand_seats hs2 WHERE hs2.hand_id = hs.hand_id AND hs2.hole_cards IS NOT NULL) AS hand_had_showdown
        FROM hand_seats hs
        JOIN hands h ON h.id = hs.hand_id
        WHERE 1=1{where}
        """,
        params,
    ).fetchall()
    saw_flop_rows = [r for r in rows if r[3]]
    if not saw_flop_rows:
        return {"saw_flop": 0, "wtsd_pct": None, "showdowns": 0, "wsd_pct": None}
    showdown_rows = [r for r in saw_flop_rows if r[5] and not r[4]]
    wtsd_pct = round(100 * len(showdown_rows) / len(saw_flop_rows), 1)
    resolved_showdowns = [r for r in showdown_rows if r[2] is not None]
    wsd_pct = round(100 * sum(1 for r in resolved_showdowns if r[2] > 0) / len(resolved_showdowns), 1) if resolved_showdowns else None
    return {"saw_flop": len(saw_flop_rows), "wtsd_pct": wtsd_pct, "showdowns": len(showdown_rows), "wsd_pct": wsd_pct}


def aggression_factor(conn: sqlite3.Connection, player_hash: str | None = None, position: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:
    """(bets + raises) / calls, postflop only (the conventional scope for AF)."""
    where, params = _filter_clause(player_hash, position, start_date, end_date)
    row = conn.execute(
        f"""
        SELECT
            SUM(CASE WHEN a.action_type IN ('bet', 'raise') THEN 1 ELSE 0 END) AS aggressive,
            SUM(CASE WHEN a.action_type = 'call' THEN 1 ELSE 0 END) AS calls
        FROM hand_seats hs
        JOIN hands h ON h.id = hs.hand_id
        JOIN actions a ON a.hand_id = hs.hand_id AND a.player_hash = hs.player_hash
        WHERE a.street != 'preflop'{where}
        """,
        params,
    ).fetchone()
    aggressive, calls = row[0] or 0, row[1] or 0
    return {"aggressive_actions": aggressive, "calls": calls, "aggression_factor": round(aggressive / calls, 2) if calls else None}


def full_report(conn: sqlite3.Connection, player_hash: str | None = None, position: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict:
    report = {"filters": {"player_hash": player_hash, "position": position, "start_date": start_date, "end_date": end_date}}
    report.update(vpip_pfr(conn, player_hash, position, start_date, end_date))
    report.update(three_bet_and_fold_to_three_bet(conn, player_hash, position, start_date, end_date))
    report.update(cbet_pct(conn, player_hash, position, start_date, end_date))
    report.update(wtsd_and_wsd(conn, player_hash, position, start_date, end_date))
    report.update(aggression_factor(conn, player_hash, position, start_date, end_date))
    return report


def by_position(conn: sqlite3.Connection, player_hash: str | None = None, start_date: str | None = None, end_date: str | None = None) -> dict[str, dict]:
    positions = [r[0] for r in conn.execute("SELECT DISTINCT position FROM hand_seats WHERE position IS NOT NULL").fetchall()]
    return {pos: full_report(conn, player_hash, pos, start_date, end_date) for pos in positions}
