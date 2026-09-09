"""JSON-schema tool specs for the 6 MCP tools, for handing to a model's native tool-use
API during the eval. Mirrors src/poker_analytics/mcp_server.py's signatures and docstrings
exactly - if that file's tools change, update this list to match.
"""
from __future__ import annotations

TOOL_SPECS: list[dict] = [
    {
        "name": "query_hands",
        "description": (
            "List hands matching the given filters, newest first. Returns hand summaries "
            "(id, hand_id, site, table, date, board, pot) - pass a returned `id` to get_hand "
            "for full detail."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_hash": {"type": "string"},
                "position": {"type": "string", "enum": ["UTG", "UTG+1", "MP", "CO", "BTN", "SB", "BB"]},
                "site": {"type": "string"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                "limit": {"type": "integer", "default": 50},
                "offset": {"type": "integer", "default": 0},
            },
        },
    },
    {
        "name": "get_hand",
        "description": (
            "Full detail for one hand by its internal id (from query_hands): every seat, "
            "action, and pot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
        },
    },
    {
        "name": "compute_stats",
        "description": (
            "VPIP/PFR/3-bet/fold-to-3-bet/c-bet/WTSD/W$SD/aggression-factor for the given "
            "player (or the whole population if player_hash is omitted), optionally sliced "
            "by position and date range."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_hash": {"type": "string"},
                "position": {"type": "string", "enum": ["UTG", "UTG+1", "MP", "CO", "BTN", "SB", "BB"]},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
        },
    },
    {
        "name": "compare_positions",
        "description": "Full stat report broken out per position (BTN/SB/BB/UTG/.../CO).",
        "input_schema": {
            "type": "object",
            "properties": {
                "player_hash": {"type": "string"},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
        },
    },
    {
        "name": "find_leaks",
        "description": (
            "Flag stats where a player deviates from the population baseline by more than a "
            "per-stat threshold, above the minimum sample size needed to trust the number."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_hash": {"type": "string"},
                "position": {"type": "string", "enum": ["UTG", "UTG+1", "MP", "CO", "BTN", "SB", "BB"]},
                "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                "end_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["player_hash"],
        },
    },
    {
        "name": "session_summary",
        "description": (
            "Split a player's hands into sessions (a new session starts after gap_minutes "
            "with no hands, regardless of table) and summarize each one's duration, hand "
            "count, net result, and tables played."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "player_hash": {"type": "string"},
                "gap_minutes": {"type": "integer", "default": 30},
            },
            "required": ["player_hash"],
        },
    },
]
