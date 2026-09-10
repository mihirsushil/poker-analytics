"""Decision-point extraction.

Walks the Phase 1 schema (hands/hand_seats/actions/pots) and emits one
DecisionExample per action taken by a seat whose hole cards are known
(hero, or anyone whose cards were shown at showdown) - that's the only
population we can build a supervised (state, action) pair for, since a
folded/unshown hand's cards are never in the data by design (privacy - see
hashing.py). Blind/ante posts are not decisions and are excluded from the
label set, but they still count as actions in the running pot/stack math
and in the action-history text.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

NON_DECISION_ACTIONS = {"post_ante", "post_sb", "post_bb"}

STREET_BOARD_SIZE = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


@dataclass
class DecisionExample:
    hand_pk: int
    site: str
    source: str  # provenance passthrough from hands.source (raw_text | phh)
    street: str
    position: str | None
    hole_cards: str  # comma-separated, e.g. "Ac,Kd"
    board_so_far: str  # comma-separated, "" preflop
    pot_before: float
    stack_before: float
    action_history: str  # compact prior-actions-this-hand text
    action_type: str  # fold | check | call | bet | raise - the grading label
    amount: float
    to_amount: float | None
    action_label: str  # natural-language target, e.g. "raise to 75"


def _format_action(a_type: str, position: str | None) -> str:
    pos = position or "?"
    return f"{pos}: {a_type}"


def _normalize_label(action_type: str, amount: float, to_amount: float | None) -> str:
    if action_type == "bet":
        return f"bet {amount:.0f}"
    if action_type == "raise":
        sizing = to_amount if to_amount is not None else amount
        return f"raise to {sizing:.0f}"
    return action_type  # fold | check | call


def extract_decision_points(db_path: str | Path) -> list[DecisionExample]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    hands = conn.execute(
        "SELECT id, site, source, board FROM hands ORDER BY id"
    ).fetchall()

    examples: list[DecisionExample] = []

    for hand in hands:
        hand_pk = hand["id"]
        board_cards = [c for c in hand["board"].split(",") if c]

        seats = conn.execute(
            """SELECT player_hash, starting_stack, position, hole_cards
               FROM hand_seats WHERE hand_id = ?""",
            (hand_pk,),
        ).fetchall()
        seat_by_hash = {s["player_hash"]: s for s in seats}

        actions = conn.execute(
            """SELECT street, action_order, player_hash, action_type, amount, to_amount
               FROM actions WHERE hand_id = ? ORDER BY action_order""",
            (hand_pk,),
        ).fetchall()

        pot_running = 0.0
        contributed: dict[str, float] = {s["player_hash"]: 0.0 for s in seats}
        history_lines: list[str] = []

        for a in actions:
            player_hash = a["player_hash"]
            seat = seat_by_hash.get(player_hash)
            street = a["street"]
            action_type = a["action_type"]

            if (
                seat is not None
                and seat["hole_cards"]
                and action_type not in NON_DECISION_ACTIONS
            ):
                n_board = STREET_BOARD_SIZE.get(street, 0)
                stack_before = seat["starting_stack"] - contributed[player_hash]
                examples.append(
                    DecisionExample(
                        hand_pk=hand_pk,
                        site=hand["site"],
                        source=hand["source"],
                        street=street,
                        position=seat["position"],
                        hole_cards=seat["hole_cards"],
                        board_so_far=",".join(board_cards[:n_board]),
                        pot_before=round(pot_running, 2),
                        stack_before=round(stack_before, 2),
                        action_history=" | ".join(history_lines) if history_lines else "(first to act)",
                        action_type=action_type,
                        amount=a["amount"],
                        to_amount=a["to_amount"],
                        action_label=_normalize_label(action_type, a["amount"], a["to_amount"]),
                    )
                )

            # Update running state for every action, decision or not, so
            # later decisions in the same hand see accurate pot/stack/history.
            pot_running += a["amount"]
            contributed[player_hash] = contributed.get(player_hash, 0.0) + a["amount"]
            if action_type not in NON_DECISION_ACTIONS:
                history_lines.append(_format_action(action_type, seat["position"] if seat else None))

    conn.close()
    return examples


def write_jsonl(examples: list[DecisionExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(asdict(ex)) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract decision points from the hand DB.")
    parser.add_argument("--db", default="data/poker.db")
    parser.add_argument("--out", default="phase4/data/decisions.jsonl")
    args = parser.parse_args()

    examples = extract_decision_points(args.db)
    write_jsonl(examples, args.out)
    print(f"Extracted {len(examples)} decision points -> {args.out}")


if __name__ == "__main__":
    main()
