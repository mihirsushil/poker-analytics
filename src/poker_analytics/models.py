"""In-memory representation of a parsed hand, shared by every input parser
(raw PokerStars text, PHH, and eventually other sites/formats). Every
parser's job is to produce a Hand; everything downstream (DB, stats, MCP
tools) only ever sees this shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Street(str, Enum):
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"


class ActionType(str, Enum):
    POST_ANTE = "post_ante"
    POST_SB = "post_sb"
    POST_BB = "post_bb"
    FOLD = "fold"
    CHECK = "check"
    CALL = "call"
    BET = "bet"
    RAISE = "raise"


@dataclass
class Action:
    street: Street
    order: int  # 0-indexed order within the whole hand, across streets
    player_hash: str
    action_type: ActionType
    amount: float  # chips/currency put in with *this* action
    to_amount: float | None = None  # for raises: "raises X to Y" -> Y
    is_all_in: bool = False


@dataclass
class Seat:
    seat_no: int
    player_hash: str
    starting_stack: float
    is_button: bool = False
    position: str | None = None  # BTN/SB/BB/UTG/UTG+1/.../HJ/CO, filled in post-parse
    sitting_out: bool = False
    hole_cards: list[str] | None = None  # only known for hero, or shown at showdown
    net_result: float | None = None  # winnings - contributed, filled from summary


@dataclass
class Pot:
    pot_no: int  # 0 = main pot, 1+ = side pots, in creation order
    amount: float
    winners: list[str] = field(default_factory=list)  # player_hashes


@dataclass
class Hand:
    hand_id: str  # site's own hand ID, e.g. "159582406766"
    site: str  # "PokerStars"
    source: str  # "raw_text" | "phh" | "self_play" - data provenance
    source_file: str

    game: str  # "Hold'em"
    limit_type: str  # "No Limit" | "Pot Limit" | "Limit"
    is_tournament: bool
    tournament_id: str | None
    currency: str  # "$" | "" for tournament chips

    table_name: str
    max_seats: int
    button_seat: int
    played_at: datetime

    seats: list[Seat]
    actions: list[Action]
    board: list[str]  # up to 5 cards, in reveal order
    pots: list[Pot]
    total_pot: float
    rake: float

    def seat_for(self, player_hash: str) -> Seat | None:
        return next((s for s in self.seats if s.player_hash == player_hash), None)

    def hero_seat(self) -> Seat | None:
        """The seat whose hole cards were dealt-and-shown to us (the 'Dealt to X' line).

        For bootstrap (non-self-played) data this isn't really "the user" -
        it's just whichever seat the original hand history was exported
        from - but it's the only seat we reliably have hole cards for
        pre-showdown, so it's the natural default subject for stats.
        """
        return next((s for s in self.seats if s.hole_cards is not None), None)
