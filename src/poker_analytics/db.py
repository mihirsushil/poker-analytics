"""SQLite schema and hand persistence.

One `Hand` (see models.py) maps to one row in `hands`, N rows in
`hand_seats`, N rows in `actions`, and N rows in `pots`. `source` on
`hands` tracks data provenance (raw_text bootstrap fixture / phh bootstrap
volume / self_play) so bootstrap and self-played data can always be told
apart, per the project's data-honesty rules.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import Hand

SCHEMA = """
CREATE TABLE IF NOT EXISTS hands (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id TEXT NOT NULL,
    site TEXT NOT NULL,
    source TEXT NOT NULL,
    source_file TEXT NOT NULL,
    game TEXT NOT NULL,
    limit_type TEXT NOT NULL,
    is_tournament INTEGER NOT NULL,
    tournament_id TEXT,
    currency TEXT NOT NULL,
    table_name TEXT NOT NULL,
    max_seats INTEGER NOT NULL,
    button_seat INTEGER NOT NULL,
    played_at TEXT NOT NULL,
    board TEXT NOT NULL,       -- comma-separated cards, e.g. "Ac,Kd,7h,2s,9c"
    total_pot REAL NOT NULL,
    rake REAL NOT NULL,
    UNIQUE(site, hand_id)
);

CREATE TABLE IF NOT EXISTS hand_seats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    seat_no INTEGER NOT NULL,
    player_hash TEXT NOT NULL,
    starting_stack REAL NOT NULL,
    is_button INTEGER NOT NULL,
    position TEXT,
    sitting_out INTEGER NOT NULL,
    hole_cards TEXT,            -- comma-separated, NULL if never revealed
    net_result REAL
);

CREATE INDEX IF NOT EXISTS idx_hand_seats_player ON hand_seats(player_hash);
CREATE INDEX IF NOT EXISTS idx_hand_seats_hand ON hand_seats(hand_id);

CREATE TABLE IF NOT EXISTS actions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    street TEXT NOT NULL,
    action_order INTEGER NOT NULL,
    player_hash TEXT NOT NULL,
    action_type TEXT NOT NULL,
    amount REAL NOT NULL,
    to_amount REAL,
    is_all_in INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_actions_hand ON actions(hand_id);
CREATE INDEX IF NOT EXISTS idx_actions_player ON actions(player_hash);

CREATE TABLE IF NOT EXISTS pots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    hand_id INTEGER NOT NULL REFERENCES hands(id) ON DELETE CASCADE,
    pot_no INTEGER NOT NULL,
    amount REAL NOT NULL,
    winners TEXT NOT NULL       -- comma-separated player_hashes
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def insert_hand(conn: sqlite3.Connection, hand: Hand) -> int:
    """Insert a fully-parsed Hand. Raises sqlite3.IntegrityError on (site, hand_id) duplicates."""
    cur = conn.execute(
        """
        INSERT INTO hands (
            hand_id, site, source, source_file, game, limit_type, is_tournament,
            tournament_id, currency, table_name, max_seats, button_seat,
            played_at, board, total_pot, rake
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            hand.hand_id, hand.site, hand.source, hand.source_file, hand.game,
            hand.limit_type, int(hand.is_tournament), hand.tournament_id,
            hand.currency, hand.table_name, hand.max_seats, hand.button_seat,
            hand.played_at.isoformat(), ",".join(hand.board), hand.total_pot, hand.rake,
        ),
    )
    hand_row_id = cur.lastrowid

    for seat in hand.seats:
        conn.execute(
            """
            INSERT INTO hand_seats (
                hand_id, seat_no, player_hash, starting_stack, is_button,
                position, sitting_out, hole_cards, net_result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hand_row_id, seat.seat_no, seat.player_hash, seat.starting_stack,
                int(seat.is_button), seat.position, int(seat.sitting_out),
                ",".join(seat.hole_cards) if seat.hole_cards else None,
                seat.net_result,
            ),
        )

    for action in hand.actions:
        conn.execute(
            """
            INSERT INTO actions (
                hand_id, street, action_order, player_hash, action_type,
                amount, to_amount, is_all_in
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                hand_row_id, action.street.value, action.order, action.player_hash,
                action.action_type.value, action.amount, action.to_amount,
                int(action.is_all_in),
            ),
        )

    for pot in hand.pots:
        conn.execute(
            "INSERT INTO pots (hand_id, pot_no, amount, winners) VALUES (?, ?, ?, ?)",
            (hand_row_id, pot.pot_no, pot.amount, ",".join(pot.winners)),
        )

    conn.commit()
    return hand_row_id
