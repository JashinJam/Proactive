from __future__ import annotations

import unittest

import numpy as np

from proactive_d1.core import LabelFreeChunk, LabeledChunk
from proactive_d5.robust import (
    cross_validate_multiview_linear,
    drop_assistant_history,
    static_promotion_gate,
)


def _examples() -> list[LabeledChunk]:
    result: list[LabeledChunk] = []
    for input_index in range(12):
        for chunk_index in range(2):
            result.append(
                LabeledChunk(
                    feature=LabelFreeChunk(
                        input_index=input_index,
                        video_path=f"v{input_index}.mp4",
                        domain="test",
                        fold=input_index % 3,
                        chunk_index=chunk_index,
                        total_chunks=2,
                        interval=(float(chunk_index), float(chunk_index + 1)),
                        raw_response="$silent$",
                        values={},
                    ),
                    gold_interrupt=chunk_index,
                )
            )
    return result


class RobustInputTest(unittest.TestCase):
    def test_drop_assistant_history_is_answer_free_and_nonmutating(self) -> None:
        rows = [
            {
                "video_path": "v.mp4",
                "video_intervals": [[0, 1], [1, 2]],
                "dialog": [
                    [{"role": "user", "text": "help"}],
                    [
                        {"role": "user", "text": "help"},
                        {"role": "assistant", "text": "$interrupt$act"},
                    ],
                ],
            }
        ]
        transformed, audit = drop_assistant_history(rows)
        self.assertEqual(audit["assistant_turns_removed"], 1)
        self.assertEqual(len(transformed[0]["dialog"][1]), 1)  # type: ignore[index]
        self.assertEqual(len(rows[0]["dialog"][1]), 2)  # type: ignore[index]

    def test_drop_assistant_history_rejects_answers(self) -> None:
        with self.assertRaisesRegex(ValueError, "answer-free"):
            drop_assistant_history([{"answers": ["$silent$"]}])


class RobustOofTest(unittest.TestCase):
    def test_multiview_oof_covers_all_methods_and_views(self) -> None:
        examples = _examples()
        clean = np.asarray(
            [[example.gold_interrupt, example.feature.chunk_index] for example in examples],
            dtype=np.float64,
        )
        views = {
            "clean": clean,
            "history4": clean + np.asarray([0.1, 0.0]),
            "assistant_drop": clean + np.asarray([0.0, 0.1]),
            "frame_jitter": clean - np.asarray([0.1, 0.0]),
        }
        decisions, details, heads = cross_validate_multiview_linear(
            examples,
            views,
            ("signal", "chunk"),
            clean_view="clean",
            training_views=tuple(views),
            folds=3,
            calibration_fold_offset=1,
            seed=7,
            max_iterations=10,
            l2_weights=(0.01,),
            l2_reduction="sum",
        )
        expected = {example.key for example in examples}
        self.assertEqual(len(details), 3)
        self.assertEqual(set(heads["robust"]), {0, 1, 2})
        for method in ("standard", "robust"):
            for view in views:
                self.assertEqual(set(decisions[method][view]), expected)

    def test_static_gate_requires_each_perturbation(self) -> None:
        def result(macro: float, predicted: int = 5) -> dict[str, object]:
            return {
                "overall": {
                    "macro_f1": macro,
                    "tp": predicted,
                    "fp": 0,
                    "support": 10,
                }
            }

        metrics = {
            "standard": {
                "clean": result(0.70),
                "history4": result(0.50),
                "assistant_drop": result(0.40),
                "frame_jitter": result(0.60),
            },
            "robust": {
                "clean": result(0.699),
                "history4": result(0.52),
                "assistant_drop": result(0.41),
                "frame_jitter": result(0.609),
            },
        }
        audit = static_promotion_gate(
            metrics,
            clean_view="clean",
            perturbation_views=("history4", "assistant_drop", "frame_jitter"),
            maximum_clean_drop=0.002,
            minimum_perturbation_gain=0.01,
        )
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["worst_perturbation_delta"], 0.009)


if __name__ == "__main__":
    unittest.main()
