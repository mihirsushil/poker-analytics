"""Benchmark harness: decision-prediction accuracy for the majority-class
baseline, a fine-tuned LoRA adapter, and (only if a real API key is
present) a prompted frontier model - against the held-out test split.

Grading is on the *categorical* action bucket (fold/check/call/bet/raise),
not exact sizing - matching a bet/raise amount character-for-character
isn't the interesting question for a smoke-scale run, and neither the
majority baseline nor a lightly fine-tuned small model can reasonably be
graded on it yet. bucket_prediction() is the one place that logic lives.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .baseline import MajorityClassBaseline
from .dataset import load_examples, split
from .extract import DecisionExample

ACTION_BUCKETS = ("fold", "check", "call", "bet", "raise")


def bucket_prediction(text: str) -> str:
    """Map free-form model output to one of ACTION_BUCKETS, or 'unparseable'."""
    lowered = text.strip().lower()
    lowered = re.sub(r'^[\s"\'.]+|[\s"\'.]+$', "", lowered)
    for bucket in ACTION_BUCKETS:
        if lowered.startswith(bucket):
            return bucket
    return "unparseable"


@dataclass
class ArmResult:
    name: str
    n: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0


def score_arm(name: str, examples: list[DecisionExample], predict_fn) -> ArmResult:
    correct = 0
    for ex in examples:
        pred = bucket_prediction(predict_fn(ex))
        if pred == ex.action_type:
            correct += 1
    return ArmResult(name=name, n=len(examples), correct=correct)


def run_benchmark(
    decisions_path: str,
    adapter_dir: str | None,
    frontier_model: str | None,
    seed: int = 0,
    limit: int | None = None,
) -> dict:
    examples = load_examples(decisions_path)
    train, _val, test = split(examples, seed=seed)
    if limit:
        test = test[:limit]

    results: list[ArmResult] = []

    baseline = MajorityClassBaseline(train)
    results.append(score_arm("majority_class_baseline", test, baseline.predict))

    if adapter_dir and Path(adapter_dir).exists():
        from .infer import predict_action

        results.append(
            score_arm(
                f"lora_finetune ({adapter_dir})",
                test,
                lambda ex: predict_action(ex, adapter_dir),
            )
        )
    else:
        print(f"skipping lora_finetune arm - {adapter_dir!r} does not exist yet, run finetune.py first")

    if frontier_model:
        from .frontier import FrontierModelClient, is_configured

        if is_configured():
            client = FrontierModelClient(frontier_model)
            results.append(score_arm(f"frontier ({frontier_model})", test, client.predict))
        else:
            print("skipping frontier arm - ANTHROPIC_API_KEY not set (see phase4/README.md)")

    return {
        "n_test": len(test),
        "majority_class": baseline.majority_class,
        "class_distribution_train": baseline.distribution,
        "results": [
            {"arm": r.name, "n": r.n, "correct": r.correct, "accuracy": round(r.accuracy, 4)}
            for r in results
        ],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark baseline vs LoRA fine-tune vs frontier model.")
    parser.add_argument("--decisions", default="phase4/data/decisions.jsonl")
    parser.add_argument("--adapter-dir", default="phase4/adapters/smoke_test")
    parser.add_argument("--frontier-model", default="claude-haiku-4-5", help="set to '' to skip the frontier arm entirely")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="cap test set size, useful for a fast smoke run")
    parser.add_argument("--out", default="phase4/benchmark_results.json")
    args = parser.parse_args()

    report = run_benchmark(
        decisions_path=args.decisions,
        adapter_dir=args.adapter_dir,
        frontier_model=args.frontier_model or None,
        seed=args.seed,
        limit=args.limit,
    )
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
