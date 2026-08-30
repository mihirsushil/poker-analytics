"""Session detection for one player's hand history.

A "session" only means something from a single player's point of view (the
same timestamp has a different player at every seat), so unlike the stats
module this always takes a required `player_hash` - there's no
population-wide equivalent. Sessions are detected the way HUD tools (PT4,
HM3) do it: a session is a maximal run of that player's hands, ordered by
time and regardless of table (multi-tabling is one session, not several),
where no gap between consecutive hands exceeds `gap_minutes`.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime


def session_summary(conn: sqlite3.Connection, player_hash: str, gap_minutes: int = 30) -> list[dict]:
    rows = conn.execute(
        """
        SELECT h.played_at, h.table_name, hs.net_result
        FROM hand_seats hs
        JOIN hands h ON h.id = hs.hand_id
        WHERE hs.player_hash = ?
        ORDER BY h.played_at ASC
        """,
        (player_hash,),
    ).fetchall()
    if not rows:
        return []

    gap = gap_minutes * 60
    sessions: list[list[tuple]] = [[rows[0]]]
    prev_time = datetime.fromisoformat(rows[0][0])
    for row in rows[1:]:
        played_at = datetime.fromisoformat(row[0])
        if (played_at - prev_time).total_seconds() > gap:
            sessions.append([])
        sessions[-1].append(row)
        prev_time = played_at

    summaries = []
    for session_rows in sessions:
        start = datetime.fromisoformat(session_rows[0][0])
        end = datetime.fromisoformat(session_rows[-1][0])
        net_results = [r[2] for r in session_rows if r[2] is not None]
        summaries.append({
            "session_start": start.isoformat(),
            "session_end": end.isoformat(),
            "duration_minutes": round((end - start).total_seconds() / 60, 1),
            "hand_count": len(session_rows),
            "net_result": round(sum(net_results), 2) if net_results else None,
            "tables": sorted({r[1] for r in session_rows}),
        })
    return summaries
