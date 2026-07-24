"""Parser for raw PokerStars hand-history .txt exports (Hold'em only).

Handles the edge cases the project plan calls out explicitly: antes,
all-ins (including short-stack partial blind posts), uncalled bets, side
pots, mucked/shown hole cards, sitting-out players, and both the modern
("PokerStars Hand #...") and legacy ("PokerStars Game #...") header formats.
Names are hashed before they ever reach a Hand object - nothing downstream
of this module ever sees a real screen name.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from ..hashing import hash_player_name
from ..models import Action, ActionType, Hand, Pot, Seat, Street
from ..positions import assign_positions

_HAND_ID_RE = re.compile(r"(?:Hand|Game) #(?P<hand_id>\d+)")
_TOURNEY_ID_RE = re.compile(r"Tournament #(?P<tourney_id>\d+)")
_CASH_STAKES_RE = re.compile(r"\((\$)?[\d.]+/(\$)?[\d.]+(?:\s*\w+)?\)")
_TIMESTAMP_RE = re.compile(r"(?P<timestamp>\d{4}/\d{2}/\d{2} \d{1,2}:\d{2}:\d{2}) \w+")
_TABLE_RE = re.compile(r"^Table '(?P<table>[^']*)' (?P<max_seats>\d+)-max(?: Seat #(?P<button>\d+) is the button)?")
_SEAT_RE = re.compile(r"^Seat (?P<seat_no>\d+): (?P<name>.+) \(\$?(?P<stack>[\d.,]+) in chips\)( is sitting out)?\s*$")
_DEALT_RE = re.compile(r"^Dealt to (?P<name>.+?) \[(?P<cards>.+)\]$")
_STREET_RE = re.compile(r"^\*\*\* (FLOP|TURN|RIVER) \*\*\* (?:\[.+?\] )?\[(?P<cards>.+)\]$")
_UNCALLED_RE = re.compile(r"^Uncalled bet \(\$?(?P<amount>[\d.,]+)\) returned to (?P<name>.+)$")
_COLLECTED_RE = re.compile(r"^(?P<name>.+?) collected \$?(?P<amount>[\d.,]+) from (?:the )?(?P<pot>main pot|side pot|pot)$")
_TOTAL_POT_SIMPLE_RE = re.compile(r"^Total pot \$?(?P<total>[\d.,]+) \| Rake \$?(?P<rake>[\d.,]+)")
_TOTAL_POT_SPLIT_RE = re.compile(
    r"^Total pot \$?(?P<total>[\d.,]+) Main pot \$?(?P<main>[\d.,]+)\. "
    r"(?:Side pot \$?(?P<side1>[\d.,]+)\.\s*)?(?:Side pot-2 \$?(?P<side2>[\d.,]+)\.\s*)?"
    r"\| Rake \$?(?P<rake>[\d.,]+)"
)
_BOARD_RE = re.compile(r"^Board \[(?P<cards>.+)\]$")

_POST_ANTE_RE = re.compile(r"^posts the ante \$?(?P<amount>[\d.,]+)$")
_POST_SB_RE = re.compile(r"^posts small blind \$?(?P<amount>[\d.,]+)( and is all-in)?$")
_POST_BB_RE = re.compile(r"^posts big blind \$?(?P<amount>[\d.,]+)( and is all-in)?$")
_FOLD_RE = re.compile(r"^folds\s*$")
_CHECK_RE = re.compile(r"^checks\s*$")
_CALL_RE = re.compile(r"^calls \$?(?P<amount>[\d.,]+)( and is all-in)?$")
_BET_RE = re.compile(r"^bets \$?(?P<amount>[\d.,]+)( and is all-in)?$")
_RAISE_RE = re.compile(r"^raises \$?(?P<amount>[\d.,]+) to \$?(?P<to>[\d.,]+)( and is all-in)?$")
_SHOWS_RE = re.compile(r"^shows \[(?P<cards>.+)\]")


class ParseError(ValueError):
    pass


def _num(s: str) -> float:
    return float(s.replace(",", ""))


def parse_file(path: str | Path) -> list[Hand]:
    """Split a hand-history export on blank lines and parse each hand."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    blocks = [b.strip("\n") for b in re.split(r"\n\s*\n", text) if b.strip()]
    return [parse_hand_text(block, str(path)) for block in blocks]


def parse_hand_text(text: str, source_file: str) -> Hand:
    lines = [line for line in text.splitlines() if line.strip() != ""]
    if not lines:
        raise ParseError("empty hand text")

    if "Hold'em" not in lines[0]:
        raise ParseError(f"unsupported game (Hold'em only): {lines[0]!r}")

    hand_id_match = _HAND_ID_RE.search(lines[0])
    timestamp_match = _TIMESTAMP_RE.search(lines[0])
    if hand_id_match is None or timestamp_match is None:
        raise ParseError(f"unrecognized header line: {lines[0]!r}")

    tourney_match = _TOURNEY_ID_RE.search(lines[0])
    is_tournament = tourney_match is not None
    tournament_id = tourney_match.group("tourney_id") if tourney_match else None

    currency = ""
    if not is_tournament:
        stakes_match = _CASH_STAKES_RE.search(lines[0])
        if stakes_match and (stakes_match.group(1) or stakes_match.group(2)):
            currency = "$"

    limit_type = next((lt for lt in ("No Limit", "Pot Limit", "Limit") if lt in lines[0]), "No Limit")
    played_at = datetime.strptime(timestamp_match.group("timestamp"), "%Y/%m/%d %H:%M:%S")

    table_match = _TABLE_RE.match(lines[1])
    if table_match is None:
        raise ParseError(f"unrecognized table line: {lines[1]!r}")
    button_seat = int(table_match.group("button")) if table_match.group("button") else None

    idx = 2
    seats: list[Seat] = []
    name_to_hash: dict[str, str] = {}
    while idx < len(lines) and (m := _SEAT_RE.match(lines[idx])):
        name = m.group("name")
        player_hash = name_to_hash.setdefault(name, hash_player_name(name))
        seats.append(
            Seat(
                seat_no=int(m.group("seat_no")),
                player_hash=player_hash,
                starting_stack=_num(m.group("stack")),
                is_button=(button_seat == int(m.group("seat_no"))),
                sitting_out=m.group(0).rstrip().endswith("is sitting out"),
            )
        )
        idx += 1

    def resolve(name: str) -> str:
        return name_to_hash.setdefault(name, hash_player_name(name))

    actions: list[Action] = []
    board: list[str] = []
    pots: list[Pot] = []
    hero_hole_cards: tuple[str, list[str]] | None = None
    shown_cards: dict[str, list[str]] = {}
    sb_hash: str | None = None
    bb_hash: str | None = None
    total_contributed: dict[str, float] = {}
    total_collected: dict[str, float] = {}
    street_contrib: dict[str, float] = {}
    street = Street.PREFLOP
    order = 0
    total_pot = 0.0
    rake = 0.0
    pot_collections: dict[str, list[tuple[str, float]]] = {"pot": [], "main pot": [], "side pot": []}
    in_summary = False

    def add_action(player_hash: str, action_type: ActionType, amount: float, to_amount: float | None = None, all_in: bool = False):
        nonlocal order
        actions.append(Action(street=street, order=order, player_hash=player_hash, action_type=action_type, amount=amount, to_amount=to_amount, is_all_in=all_in))
        order += 1

    for line in lines[idx:]:
        line = line.rstrip()
        if line == "*** HOLE CARDS ***":
            continue
        if line in ("*** SHOW DOWN ***", "*** SUMMARY ***"):
            if line == "*** SUMMARY ***":
                in_summary = True
            continue
        if in_summary:
            if (m := _TOTAL_POT_SPLIT_RE.match(line)):
                total_pot = _num(m.group("total"))
                rake = _num(m.group("rake"))
                if m.group("main"):
                    pots.append(Pot(pot_no=0, amount=_num(m.group("main")), winners=[]))
                if m.group("side1"):
                    pots.append(Pot(pot_no=1, amount=_num(m.group("side1")), winners=[]))
                if m.group("side2"):
                    pots.append(Pot(pot_no=2, amount=_num(m.group("side2")), winners=[]))
                continue
            if (m := _TOTAL_POT_SIMPLE_RE.match(line)):
                total_pot = _num(m.group("total"))
                rake = _num(m.group("rake"))
                pots.append(Pot(pot_no=0, amount=total_pot, winners=[]))
                continue
            if (m := _BOARD_RE.match(line)):
                board = m.group("cards").split()
                continue
            continue  # per-seat summary lines: not needed, money already reconciled from action stream

        if (m := _STREET_RE.match(line)):
            street = Street[m.group(1)]
            street_contrib = {}
            board = m.group("cards").split()
            continue

        if (m := _DEALT_RE.match(line)):
            h = resolve(m.group("name"))
            cards = m.group("cards").split()
            hero_hole_cards = (h, cards)
            continue

        if (m := _UNCALLED_RE.match(line)):
            h = resolve(m.group("name"))
            total_contributed[h] = total_contributed.get(h, 0.0) - _num(m.group("amount"))
            continue

        if (m := _COLLECTED_RE.match(line)):
            h = resolve(m.group("name"))
            amount = _num(m.group("amount"))
            total_collected[h] = total_collected.get(h, 0.0) + amount
            pot_collections[m.group("pot")].append((h, amount))
            continue

        if (m := _SHOWS_RE.match(line)) and ":" in line:
            name, _ = line.split(":", 1)
            shown_cards[resolve(name.strip())] = m.group("cards").split()
            continue

        if ": " not in line:
            continue  # e.g. "X has returned", "X finished the tournament in Nth place"
        name, rest = line.split(": ", 1)
        rest = rest.strip()
        h = resolve(name)

        if (m := _POST_ANTE_RE.match(rest)):
            amt = _num(m.group("amount"))
            total_contributed[h] = total_contributed.get(h, 0.0) + amt
            add_action(h, ActionType.POST_ANTE, amt)
        elif (m := _POST_SB_RE.match(rest)):
            amt = _num(m.group("amount"))
            sb_hash = h
            total_contributed[h] = total_contributed.get(h, 0.0) + amt
            street_contrib[h] = street_contrib.get(h, 0.0) + amt
            add_action(h, ActionType.POST_SB, amt, all_in=bool(m.group(1)))
        elif (m := _POST_BB_RE.match(rest)):
            amt = _num(m.group("amount"))
            bb_hash = h
            total_contributed[h] = total_contributed.get(h, 0.0) + amt
            street_contrib[h] = street_contrib.get(h, 0.0) + amt
            add_action(h, ActionType.POST_BB, amt, all_in=bool(m.group(1)))
        elif _FOLD_RE.match(rest):
            add_action(h, ActionType.FOLD, 0.0)
        elif _CHECK_RE.match(rest):
            add_action(h, ActionType.CHECK, 0.0)
        elif (m := _CALL_RE.match(rest)):
            amt = _num(m.group("amount"))
            total_contributed[h] = total_contributed.get(h, 0.0) + amt
            street_contrib[h] = street_contrib.get(h, 0.0) + amt
            add_action(h, ActionType.CALL, amt, all_in=bool(m.group(1)))
        elif (m := _BET_RE.match(rest)):
            amt = _num(m.group("amount"))
            total_contributed[h] = total_contributed.get(h, 0.0) + amt
            street_contrib[h] = street_contrib.get(h, 0.0) + amt
            add_action(h, ActionType.BET, amt, all_in=bool(m.group(1)))
        elif (m := _RAISE_RE.match(rest)):
            amt = _num(m.group("amount"))
            to_amt = _num(m.group("to"))
            increment = to_amt - street_contrib.get(h, 0.0)
            total_contributed[h] = total_contributed.get(h, 0.0) + increment
            street_contrib[h] = to_amt
            add_action(h, ActionType.RAISE, amt, to_amount=to_amt, all_in=bool(m.group(1)))
        # else: ignore lines we don't model yet (e.g. "X has returned", "X said ...")

    if hero_hole_cards:
        h, cards = hero_hole_cards
        seat = next((s for s in seats if s.player_hash == h), None)
        if seat:
            seat.hole_cards = cards
    for h, cards in shown_cards.items():
        seat = next((s for s in seats if s.player_hash == h), None)
        if seat and seat.hole_cards is None:
            seat.hole_cards = cards

    for seat in seats:
        seat.net_result = round(total_collected.get(seat.player_hash, 0.0) - total_contributed.get(seat.player_hash, 0.0), 2)

    if not pots:
        pots = [Pot(pot_no=0, amount=total_pot, winners=[])]
    pot_name_order = ["main pot", "pot", "side pot"]
    remaining_collections = {k: list(v) for k, v in pot_collections.items()}
    for pot in sorted(pots, key=lambda p: p.pot_no):
        for key in pot_name_order:
            if remaining_collections[key]:
                h, _ = remaining_collections[key].pop(0)
                pot.winners.append(h)

    hand_id = hand_id_match.group("hand_id")

    hand = Hand(
        hand_id=hand_id,
        site="PokerStars",
        source="raw_text",
        source_file=source_file,
        game="Hold'em",
        limit_type=limit_type,
        is_tournament=is_tournament,
        tournament_id=tournament_id,
        currency=currency,
        table_name=table_match.group("table"),
        max_seats=int(table_match.group("max_seats")),
        button_seat=button_seat if button_seat is not None else (seats[0].seat_no if seats else 0),
        played_at=played_at,
        seats=seats,
        actions=actions,
        board=board,
        pots=pots,
        total_pot=total_pot,
        rake=rake,
    )
    assign_positions(hand, sb_hash, bb_hash)
    return hand
