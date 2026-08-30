"""MCP server exposing the hand database as tools for an LLM client:
listing/inspecting hands, computing stats, comparing positions, flagging
leaks against the population baseline, and summarizing sessions.

Every tool opens its own short-lived connection - call volume from an MCP
client is low enough that pooling would be premature.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from . import queries, sessions, stats
from .db import connect

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "poker.db"

mcp = MCPServer("poker-analytics")


def _conn() -> sqlite3.Connection:
    return connect(DB_PATH)


@mcp.tool()
def query_hands(
    player_hash: str | None = None,
    position: str | None = None,
    site: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """List hands matching the given filters, newest first. Returns hand
    summaries (id, hand_id, site, table, date, board, pot) - pass a
    returned `id` to get_hand for full detail."""
    conn = _conn()
    try:
        return queries.list_hands(conn, player_hash, position, site, start_date, end_date, limit, offset)
    finally:
        conn.close()


@mcp.tool()
def get_hand(id: int) -> dict | None:
    """Full detail for one hand by its internal id (from query_hands):
    every seat, action, and pot."""
    conn = _conn()
    try:
        return queries.get_hand(conn, id)
    finally:
        conn.close()


@mcp.tool()
def compute_stats(
    player_hash: str | None = None,
    position: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """VPIP/PFR/3-bet/fold-to-3-bet/c-bet/WTSD/W$SD/aggression-factor for
    the given player (or the whole population if player_hash is omitted),
    optionally sliced by position and date range."""
    conn = _conn()
    try:
        return stats.full_report(conn, player_hash, position, start_date, end_date)
    finally:
        conn.close()


@mcp.tool()
def compare_positions(
    player_hash: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict[str, dict]:
    """Full stat report broken out per position (BTN/SB/BB/UTG/.../CO)."""
    conn = _conn()
    try:
        return stats.by_position(conn, player_hash, start_date, end_date)
    finally:
        conn.close()


@mcp.tool()
def find_leaks(
    player_hash: str,
    position: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Flag stats where a player deviates from the population baseline by
    more than a per-stat threshold, above the minimum sample size needed
    to trust the number."""
    conn = _conn()
    try:
        return stats.find_leaks(conn, player_hash, position, start_date, end_date)
    finally:
        conn.close()


@mcp.tool()
def session_summary(player_hash: str, gap_minutes: int = 30) -> list[dict]:
    """Split a player's hands into sessions (a new session starts after
    `gap_minutes` with no hands, regardless of table) and summarize each
    one's duration, hand count, net result, and tables played."""
    conn = _conn()
    try:
        return sessions.session_summary(conn, player_hash, gap_minutes)
    finally:
        conn.close()


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
