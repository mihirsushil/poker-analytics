"""Turn extracted DecisionExamples into instruction/response pairs for
causal-LM LoRA fine-tuning, and split them into train/val/test.

Known bias, documented rather than hidden: hole cards are only known for a
seat when they were revealed (hero's own cards, or a showdown reveal), so
this population skews toward hands that *didn't* end in an early fold for
that seat. The label distribution (see phase4/README.md) is check/call/bet/
raise-heavy with very few folds - a real property of the data, not a bug -
and the majority-class baseline in baseline.py reflects that skew honestly
rather than being corrected for it.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path

from .extract import DecisionExample

SYSTEM_PROMPT = (
    "You are a No-Limit Hold'em decision assistant. Given the state of a hand at the "
    "moment it is your turn to act, respond with exactly one action: fold, check, call, "
    "\"bet <amount>\", or \"raise to <amount>\". Respond with the action only, no explanation."
)


def format_prompt(ex: DecisionExample) -> str:
    board = ex.board_so_far if ex.board_so_far else "(preflop, no board yet)"
    return (
        f"Street: {ex.street}\n"
        f"Your position: {ex.position or 'unknown'}\n"
        f"Your hole cards: {ex.hole_cards}\n"
        f"Board: {board}\n"
        f"Pot before your action: {ex.pot_before}\n"
        f"Your stack before your action: {ex.stack_before}\n"
        f"Action so far this hand: {ex.action_history}\n"
        "What do you do?"
    )


def to_chat_example(ex: DecisionExample) -> dict:
    """Chat-template shape consumed by finetune.py and infer.py."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_prompt(ex)},
            {"role": "assistant", "content": ex.action_label},
        ],
        # Kept alongside for scoring - not part of the training text itself.
        "action_type": ex.action_type,
    }


def load_examples(path: str | Path) -> list[DecisionExample]:
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(DecisionExample(**json.loads(line)))
    return examples


def split(
    examples: list[DecisionExample], seed: int = 0, val_frac: float = 0.1, test_frac: float = 0.1
) -> tuple[list[DecisionExample], list[DecisionExample], list[DecisionExample]]:
    """Group-split by `hand_pk` so decisions from the same hand never leak across splits."""
    by_hand: dict[int, list[DecisionExample]] = {}
    for ex in examples:
        by_hand.setdefault(ex.hand_pk, []).append(ex)

    hand_ids = list(by_hand.keys())
    rng = random.Random(seed)
    rng.shuffle(hand_ids)

    n_hands = len(hand_ids)
    n_val = int(n_hands * val_frac)
    n_test = int(n_hands * test_frac)

    val_hands = set(hand_ids[:n_val])
    test_hands = set(hand_ids[n_val : n_val + n_test])

    train: list[DecisionExample] = []
    val: list[DecisionExample] = []
    test: list[DecisionExample] = []
    for hid in hand_ids:
        group = by_hand[hid]
        if hid in val_hands:
            val.extend(group)
        elif hid in test_hands:
            test.extend(group)
        else:
            train.extend(group)
    return train, val, test


def write_chat_jsonl(examples: list[DecisionExample], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(to_chat_example(ex)) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Split decision points into train/val/test chat JSONL.")
    parser.add_argument("--in-path", default="phase4/data/decisions.jsonl")
    parser.add_argument("--out-dir", default="phase4/data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    examples = load_examples(args.in_path)
    train, val, test = split(examples, seed=args.seed)

    out_dir = Path(args.out_dir)
    write_chat_jsonl(train, out_dir / "train.jsonl")
    write_chat_jsonl(val, out_dir / "val.jsonl")
    write_chat_jsonl(test, out_dir / "test.jsonl")

    print(f"train={len(train)} val={len(val)} test={len(test)} -> {out_dir}/")


if __name__ == "__main__":
    main()
