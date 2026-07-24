"""Loader for the public phh-dataset (uoftcprg/phh-dataset, MIT license) -
real, pre-anonymized 2009 online-poker hands in the PHH format. This is the
bulk-volume bootstrap source (see data/samples/README.md); the raw-text
PokerStars parser (pokerstars_text.py) is still the one built for real
self-played exports.

PHH represents a whole multi-hand file as one TOML document, with each
hand as a top-level table keyed by its index ("[1]", "[2]", ...). Players
are indexed p1..pN in action order (p1 = SB, p2 = BB, ..., pN = button);
`seats` maps each pN to its physical seat number. Hole cards are already
redacted ("????") for every player except at showdown reveals - there is
no privileged "hero" seat in this data, unlike a real self-played export.

Only the No Limit Hold'em variant ('NT') is supported, matching the rest
of this project's scope.
"""
from __future__ import annotations

import re
import tomllib
from datetime import datetime
from pathlib import Path

from ..hashing import hash_player_name
from ..models import Action, ActionType, Hand, Pot, Seat, Street

_ACTION_RE = re.compile(
    r"^(?:"
    r"d dh p(?P<dh_idx>\d+) (?P<dh_cards>\S+)"
    r"|d db (?P<db_cards>\S+)"
    r"|p(?P<f_idx>\d+) f"
    r"|p(?P<cc_idx>\d+) cc"
    r"|p(?P<cbr_idx>\d+) cbr (?P<cbr_amount>[\d.]+)"
    r"|p(?P<sm_idx>\d+) sm (?P<sm_cards>\S+)"
    r")$"
)


def _split_cards(card_str: str) -> list[str]:
    return [card_str[i : i + 2] for i in range(0, len(card_str), 2)]


def parse_phhs_file(path: str | Path, source: str = "phh") -> list[Hand]:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    hands = []
    for key in sorted(data, key=int):
        hand = _parse_one_hand(data[key], source_file=str(path), source=source)
        if hand is not None:
            hands.append(hand)
    return hands


def _parse_one_hand(fields: dict, source_file: str, source: str) -> Hand | None:
    if fields.get("variant") != "NT":
        return None  # only No Limit Hold'em is in scope for this project

    players: list[str] = fields["players"]
    seat_nos: list[int] = fields["seats"]
    starting_stacks: list[float] = fields["starting_stacks"]
    blinds = fields.get("blinds_or_straddles", [0] * len(players))
    antes = fields.get("antes", [0] * len(players))
    n = len(players)

    player_hashes = [hash_player_name(p) for p in players]
    seats = [
        Seat(
            seat_no=seat_nos[i],
            player_hash=player_hashes[i],
            starting_stack=float(starting_stacks[i]),
            is_button=(i == n - 1),  # pN (last action-order index) is always the button
        )
        for i in range(n)
    ]

    total_contributed = [float(antes[i]) for i in range(n)]
    street_contrib = [0.0] * n
    street_top = 0.0
    street = Street.PREFLOP
    order = 0
    actions: list[Action] = []
    board: list[str] = []
    shown_cards: dict[int, list[str]] = {}
    folded = [False] * n

    # Blinds/straddles apply immediately, before the first explicit action.
    # A *negative* value is a dead blind (e.g. a returning player posting a
    # missed blind): it goes straight into the pot but doesn't establish a
    # street contribution for calling purposes, unlike a live blind.
    for i, amt in enumerate(blinds):
        amt = float(amt)
        if amt == 0:
            continue
        if amt < 0:
            total_contributed[i] += -amt
            continue
        total_contributed[i] += amt
        street_contrib[i] = amt
        action_type = ActionType.POST_SB if i == 0 else ActionType.POST_BB
        actions.append(Action(street=street, order=order, player_hash=player_hashes[i], action_type=action_type, amount=amt))
        order += 1
    street_top = max(street_contrib) if street_contrib else 0.0

    for raw in fields.get("actions", []):
        m = _ACTION_RE.match(raw)
        if m is None:
            continue  # unrecognized/unsupported micro-action (e.g. run-it-twice markers)

        if m.group("db_cards"):
            _settle_uncalled(street_contrib, total_contributed)
            board.extend(_split_cards(m.group("db_cards")))
            street = {Street.PREFLOP: Street.FLOP, Street.FLOP: Street.TURN, Street.TURN: Street.RIVER}.get(street, Street.RIVER)
            street_contrib = [0.0] * n
            street_top = 0.0
            continue

        if m.group("dh_idx"):
            continue  # hole-card deal marker; real cards only ever appear at "sm" showdown reveals in this obfuscated dataset

        if m.group("f_idx"):
            i = int(m.group("f_idx")) - 1
            folded[i] = True
            actions.append(Action(street=street, order=order, player_hash=player_hashes[i], action_type=ActionType.FOLD, amount=0.0))
            order += 1
            continue

        if m.group("cc_idx"):
            i = int(m.group("cc_idx")) - 1
            delta = street_top - street_contrib[i]
            action_type = ActionType.CHECK if delta <= 0 else ActionType.CALL
            total_contributed[i] += max(delta, 0.0)
            street_contrib[i] = street_top
            actions.append(Action(street=street, order=order, player_hash=player_hashes[i], action_type=action_type, amount=max(delta, 0.0)))
            order += 1
            continue

        if m.group("cbr_idx"):
            i = int(m.group("cbr_idx")) - 1
            to_amount = float(m.group("cbr_amount"))
            delta = to_amount - street_contrib[i]
            action_type = ActionType.BET if street_top == 0 else ActionType.RAISE
            total_contributed[i] += delta
            street_contrib[i] = to_amount
            street_top = to_amount
            actions.append(Action(street=street, order=order, player_hash=player_hashes[i], action_type=action_type, amount=delta, to_amount=to_amount))
            order += 1
            continue

        if m.group("sm_idx"):
            i = int(m.group("sm_idx")) - 1
            if m.group("sm_cards") != "????":
                shown_cards[i] = _split_cards(m.group("sm_cards"))
            continue

    _settle_uncalled(street_contrib, total_contributed)

    for i, cards in shown_cards.items():
        seats[i].hole_cards = cards

    winnings = fields.get("winnings")
    collected = None
    if winnings is not None and sum(winnings) > 0:
        collected = [float(w) for w in winnings]
    if collected is None:
        # Either no `winnings` field, or a degenerate all-zero one (a known
        # data-quality gap in this dataset despite there being a real pot) -
        # fall back to inferring the winner ourselves.
        collected = _infer_winnings(n, folded, shown_cards, total_contributed)

    for i, seat in enumerate(seats):
        seat.net_result = None if collected is None else round(collected[i] - total_contributed[i], 2)

    total_pot = round(sum(total_contributed), 2)
    winner_hashes = [player_hashes[i] for i in range(n) if collected and collected[i] > 0] if collected else []
    pots = [Pot(pot_no=0, amount=total_pot, winners=winner_hashes)]

    played_at = datetime(fields["year"], fields["month"], fields["day"])

    hand = Hand(
        hand_id=str(fields["hand"]),
        site=fields.get("venue", "Unknown"),
        source=source,
        source_file=source_file,
        game="Hold'em",
        limit_type="No Limit",
        is_tournament=False,
        tournament_id=None,
        currency=fields.get("currency_symbol", ""),
        table_name=str(fields.get("table", "")),
        max_seats=fields.get("seat_count", n),
        button_seat=seat_nos[-1],
        played_at=played_at,
        seats=seats,
        actions=actions,
        board=board,
        pots=pots,
        total_pot=total_pot,
        rake=0.0,  # not recorded per-hand in this dataset
    )
    from ..positions import assign_positions

    sb_hash = player_hashes[0] if blinds and float(blinds[0]) else None
    bb_hash = player_hashes[1] if len(blinds) > 1 and float(blinds[1]) else None
    assign_positions(hand, sb_hash, bb_hash)
    return hand


def _settle_uncalled(street_contrib: list[float], total_contributed: list[float]) -> None:
    """PHH action strings have no explicit "uncalled bet returned" marker
    (unlike the raw-text format) - a bet/raise nobody calls is just followed
    by folds. If the street's top contribution was never matched by a
    second player, the unmatched excess never really entered the pot and
    must be excluded, exactly as PokerStars' own "Uncalled bet returned"
    line does for the raw-text format.
    """
    if not street_contrib:
        return
    ranked = sorted(street_contrib, reverse=True)
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else 0.0
    if top > second:
        excess = top - second
        i = street_contrib.index(top)
        total_contributed[i] -= excess
        street_contrib[i] -= excess


def _infer_winnings(n: int, folded: list[bool], shown_cards: dict[int, list[str]], total_contributed: list[float]) -> list[float] | None:
    """Best-effort winner inference for the ~1 in 5 hands missing an explicit
    `winnings` field: an uncontested pot (everyone else folded), or a
    showdown where exactly one player revealed real cards (others mucked
    without showing, the standard tell that they lost). Anything more
    ambiguous than that (e.g. a genuine multi-way showdown with no
    `winnings` field) is left as an unknown result rather than guessed.
    """
    live = [i for i in range(n) if not folded[i]]
    if len(live) == 1:
        winner = live[0]
    else:
        real_shows = [i for i in live if i in shown_cards]
        if len(real_shows) == 1:
            winner = real_shows[0]
        else:
            return None
    pot = sum(total_contributed)
    return [pot if i == winner else 0.0 for i in range(n)]
