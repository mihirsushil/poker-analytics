"""Assign standard position labels (BTN/SB/BB/UTG/.../HJ/CO) to seats.

Blinds are identified directly from the hand's post-SB/post-BB actions
(not from seat-number arithmetic), because a sitting-out or missing player
can shift who actually posts - see data/samples/README.md's
player-sitting-out.txt case. Everything after the big blind is ordered by
seat distance from the button and labeled from the button backwards
(CO, HJ, LJ, then generic UTG/UTG+n for the rest).
"""
from __future__ import annotations

from .models import Hand, Seat


def assign_positions(hand: Hand, sb_player_hash: str | None, bb_player_hash: str | None) -> None:
    seats_by_no = {s.seat_no: s for s in hand.seats}
    ordered_seat_nos = sorted(seats_by_no)
    n = len(ordered_seat_nos)
    if n == 0:
        return

    button_idx = ordered_seat_nos.index(hand.button_seat) if hand.button_seat in seats_by_no else 0
    # Seats in action order starting immediately after the button, wrapping around.
    rotation = ordered_seat_nos[button_idx + 1:] + ordered_seat_nos[: button_idx + 1]

    seats_by_no[hand.button_seat].position = "BTN" if hand.button_seat in seats_by_no else None

    if n == 2:
        # Heads-up: the button also posts the small blind.
        other = next(s for s in ordered_seat_nos if s != hand.button_seat)
        seats_by_no[hand.button_seat].position = "BTN/SB"
        seats_by_no[other].position = "BB"
        return

    remaining = [no for no in rotation if no != hand.button_seat]
    if sb_player_hash is not None and (sb_seat := _seat_for_hash(hand, sb_player_hash)):
        sb_seat.position = "SB"
        remaining = [no for no in remaining if no != sb_seat.seat_no]
    if bb_player_hash is not None and (bb_seat := _seat_for_hash(hand, bb_player_hash)):
        bb_seat.position = "BB"
        remaining = [no for no in remaining if no != bb_seat.seat_no]

    count = len(remaining)
    late_labels = ["CO", "HJ", "LJ"]
    for i, seat_no in enumerate(remaining):
        from_button = count - i  # 1 = seat directly before the button (CO)
        if from_button <= len(late_labels):
            label = late_labels[from_button - 1]
        else:
            utg_index = count - from_button  # 0 for the first seat after BB
            label = "UTG" if utg_index == 0 else f"UTG+{utg_index}"
        seats_by_no[seat_no].position = label


def _seat_for_hash(hand: Hand, player_hash: str) -> Seat | None:
    return hand.seat_for(player_hash)
