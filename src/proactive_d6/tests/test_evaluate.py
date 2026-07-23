from __future__ import annotations

import unittest

from proactive_d1.core import LabelFreeChunk, LabeledChunk
from proactive_d6.evaluate import _expanded_metrics


def example(chunk_index: int, gold: int) -> LabeledChunk:
    return LabeledChunk(
        feature=LabelFreeChunk(
            input_index=0,
            video_path="x.mp4",
            domain="Chef",
            fold=0,
            chunk_index=chunk_index,
            total_chunks=4,
            interval=(float(chunk_index), float(chunk_index + 1)),
            raw_response="$silent$",
            values={},
        ),
        gold_interrupt=gold,
    )


class EvaluateTest(unittest.TestCase):
    def test_expanded_metrics_reports_both_classes_and_confusion(self) -> None:
        examples = [example(0, 1), example(1, 1), example(2, 0), example(3, 0)]
        decisions = {(0, 0): 1, (0, 1): 0, (0, 2): 0, (0, 3): 1}
        metrics = _expanded_metrics(examples, decisions)
        self.assertEqual(
            {name: metrics[name] for name in ("tp", "fp", "tn", "fn")},
            {"tp": 1, "fp": 1, "tn": 1, "fn": 1},
        )
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["gmean_f1"], 0.5)

