"""Majority-class baseline: always predict whichever action_type is most
common in the training split. The floor any real model has to beat.
"""
from __future__ import annotations

import collections

from .extract import DecisionExample


class MajorityClassBaseline:
    def __init__(self, train_examples: list[DecisionExample]) -> None:
        counts = collections.Counter(ex.action_type for ex in train_examples)
        if not counts:
            raise ValueError("cannot build a majority-class baseline from zero examples")
        self.majority_class, self.majority_count = counts.most_common(1)[0]
        self.distribution = dict(counts)

    def predict(self, ex: DecisionExample) -> str:  # noqa: ARG002 - ignores state by design
        return self.majority_class
