"""Read-side queries over the hands database.

`db.py` only ever writes (`connect`, `insert_hand`); this module is where
every read comes from, for the MCP server and anything else that needs to
look hands back up rather than just insert them.
"""
from __future__ import annotations

import sqlite3


def list_hands(
    conn: sqlite3.Connection,
    player_hash: str | None = None,
    position: str | None = None,
    site: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Summary rows for hands matching the given filters, newest first.

    `player_hash`/`position` filter to hands where some seat matches both
    (a seat's position only means something in the context of one seat, so
    filtering by position alone - with no player - matches any hand that
    had *a* seat in that position).
    """
    clauses = []
    params: list = []
    joins = ""
    if player_hash is not None or position is not None:
        joins = "JOIN hand_seats hs ON hs.hand_id = h.id"
        if player_hash is not None:
            clauses.append("hs.player_hash = ?")
            params.append(player_hash)
        if position is not None:
            clauses.append("hs.position = ?")
            params.append(position)
    if site is not None:
        clauses.append("h.site = ?")
        params.append(site)
    if start_date is not None:
        clauses.append("h.played_at >= ?")
        params.append(start_date)
    if end_date is not None:
        clauses.append("h.played_at <= ?")
        params.append(end_date)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = conn.execute(
        f"""
        SELECT DISTINCT h.id, h.hand_id, h.site, h.game, h.limit_type,
               h.is_tournament, h.table_name, h.played_at, h.board,
               h.total_pot, h.rake
        FROM hands h
        {joins}
        {where}
        ORDER BY h.played_at DESC
        LIMIT ? OFFSET ?
        """,
        [*params, limit, offset],
    ).fetchall()

    return [
        {
            "id": r[0],
            "hand_id": r[1],
            "site": r[2],
            "game": r[3],
            "limit_type": r[4],
            "is_tournament": bool(r[5]),
            "table_name": r[6],
            "played_at": r[7],
            "board": r[8].split(",") if r[8] else [],
            "total_pot": r[9],
            "rake": r[10],
        }
        for r in rows
    ]


def get_hand(conn: sqlite3.Connection, id: int) -> dict | None:
    """Full detail for one hand (by internal `hands.id`, as returned by
    `list_hands`): the hand row plus every seat/action/pot that belongs to
    it, reassembled from the normalized schema.
    """
    hand_row = conn.execute(
        """
        SELECT id, hand_id, site, source, source_file, game, limit_type,
               is_tournament, tournament_id, currency, table_name,
               max_seats, button_seat, played_at, board, total_pot, rake
        FROM hands WHERE id = ?
        """,
        (id,),
    ).fetchone()
    if hand_row is None:
        return None

    seats = conn.execute(
        """
        SELECT seat_no, player_hash, starting_stack, is_button, position,
               sitting_out, hole_cards, net_result
        FROM hand_seats WHERE hand_id = ? ORDER BY seat_no
        """,
        (id,),
    ).fetchall()

    actions = conn.execute(
        """
        SELECT street, action_order, player_hash, action_type, amount,
               to_amount, is_all_in
        FROM actions WHERE hand_id = ? ORDER BY action_order
        """,
        (id,),
    ).fetchall()

    pots = conn.execute(
        "SELECT pot_no, amount, winners FROM pots WHERE hand_id = ? ORDER BY pot_no",
        (id,),
    ).fetchall()

    return {
        "id": hand_row[0],
        "hand_id": hand_row[1],
        "site": hand_row[2],
        "source": hand_row[3],
        "source_file": hand_row[4],
        "game": hand_row[5],
        "limit_type": hand_row[6],
        "is_tournament": bool(hand_row[7]),
        "tournament_id": hand_row[8],
        "currency": hand_row[9],
        "table_name": hand_row[10],
        "max_seats": hand_row[11],
        "button_seat": hand_row[12],
        "played_at": hand_row[13],
        "board": hand_row[14].split(",") if hand_row[14] else [],
        "total_pot": hand_row[15],
        "rake": hand_row[16],
        "seats": [
            {
                "seat_no": s[0],
                "player_hash": s[1],
                "starting_stack": s[2],
                "is_button": bool(s[3]),
                "position": s[4],
                "sitting_out": bool(s[5]),
                "hole_cards": s[6].split(",") if s[6] else None,
                "net_result": s[7],
            }
            for s in seats
        ],
        "actions": [
            {
                "street": a[0],
                "order": a[1],
                "player_hash": a[2],
                "action_type": a[3],
                "amount": a[4],
                "to_amount": a[5],
                "is_all_in": bool(a[6]),
            }
            for a in actions
        ],
        "pots": [
            {"pot_no": p[0], "amount": p[1], "winners": p[2].split(",") if p[2] else []}
            for p in pots
        ],
    }
