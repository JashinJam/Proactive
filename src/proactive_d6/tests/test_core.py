from __future__ import annotations

import copy
import unittest

import numpy as np

from proactive_d1.core import LabelFreeChunk, LabeledChunk
from proactive_d1.neural_core import cross_validate_neural_matrix
from proactive_d6.core import (
    D6_VARIANTS,
    LAST2_GROUPS,
    apply_stage_thresholds,
    build_structured_stages,
    calibrate_stage_thresholds,
    cross_validate_structured_calibration,
)


def _chunk(
    index: int,
    *,
    total: int = 7,
    fold: int = 0,
    input_index: int = 0,
) -> LabelFreeChunk:
    return LabelFreeChunk(
        input_index=input_index,
        video_path=f"video-{input_index}.mp4",
        domain="Tutorial",
        fold=fold,
        chunk_index=index,
        total_chunks=total,
        interval=(float(index), float(index + 1)),
        raw_response="",
        values={"is_first_chunk": float(index == 0)},
    )


def _stage_row() -> dict[str, object]:
    user = {"role": "user", "text": "help"}
    assistants = [
        {"role": "assistant", "text": f"$interrupt${letter}"}
        for letter in ("A", "B", "C")
    ]
    counts = [0, 1, 2, 2, 3, 3, 3]
    return {
        "video_path": "video-0.mp4",
        "video_intervals": [[index, index + 1] for index in range(len(counts))],
        "query": "help",
        "domain": "Tutorial",
        "dialog": [[user, *assistants[:count]] for count in counts],
    }


class StructuredStageTest(unittest.TestCase):
    def test_builds_mutually_exclusive_causal_stages(self) -> None:
        chunks = [_chunk(index) for index in range(7)]
        families, previous, audit = build_structured_stages([_stage_row()], chunks)
        np.testing.assert_array_equal(previous, [0, 1, 1, 0, 1, 0, 0])
        self.assertEqual(
            families["position"],
            ("first", "second", "2-4", "2-4", "2-4", "5-9", "5-9"),
        )
        self.assertEqual(
            families["last_action"],
            (
                "first",
                "previous_interrupt",
                "previous_interrupt",
                "previous_silent",
                "previous_interrupt",
                "previous_silent",
                "previous_silent",
            ),
        )
        self.assertEqual(
            families["last2"],
            ("first", "second", "ii", "is", "si", "is", "ss"),
        )
        self.assertEqual(sum(audit["counts"]["last2"].values()), 7)  # type: ignore[union-attr]
        self.assertFalse(audit["labels_read"])
        self.assertFalse(audit["predictions_read"])

    def test_prefix_is_invariant_and_answers_are_rejected(self) -> None:
        chunks = [_chunk(index) for index in range(7)]
        row = _stage_row()
        original, _, _ = build_structured_stages([row], chunks)
        changed = copy.deepcopy(row)
        extra = {"role": "assistant", "text": "$interrupt$future"}
        for index in range(5, 7):
            changed["dialog"][index] = [*changed["dialog"][index], extra]  # type: ignore[index]
        mutated, _, _ = build_structured_stages([changed], chunks)
        for family in original:
            self.assertEqual(original[family][:5], mutated[family][:5])

        row["answers"] = ["$silent$"] * 7
        with self.assertRaisesRegex(ValueError, "answer-stripped"):
            build_structured_stages([row], chunks)

    def test_rejects_decreasing_cumulative_dialog(self) -> None:
        row = _stage_row()
        row["dialog"][3] = row["dialog"][3][:1]  # type: ignore[index]
        with self.assertRaisesRegex(ValueError, "decreased"):
            build_structured_stages([row], [_chunk(index) for index in range(7)])


class ThresholdPolicyTest(unittest.TestCase):
    def test_pins_first_and_applies_frozen_shrinkage(self) -> None:
        logits = [-2.0, 2.0, 5.0, 6.0, 7.0, 8.0]
        labels = [0, 1, 0, 0, 1, 1]
        stages = ["first", "first", "ii", "ii", "ii", "ii"]
        thresholds, detail = calibrate_stage_thresholds(
            logits,
            labels,
            stages,
            LAST2_GROUPS,
            global_threshold=0.0,
            minimum_group_rows=2,
            shrinkage_pseudocount=4.0,
            shrunk=True,
            pin_first=True,
        )
        self.assertEqual(thresholds["first"], 0.0)
        self.assertEqual(
            detail["groups"]["first"]["fallback_reason"],  # type: ignore[index]
            "first_group_pinned_to_global",
        )
        self.assertAlmostEqual(
            detail["groups"]["ii"]["local_threshold"], 6.5  # type: ignore[index]
        )
        self.assertAlmostEqual(
            detail["groups"]["ii"]["shrinkage_weight"], 0.5  # type: ignore[index]
        )
        self.assertAlmostEqual(thresholds["ii"], 3.25)
        self.assertEqual(thresholds["ss"], 0.0)

    def test_unshrunk_and_missing_mapping_behavior(self) -> None:
        thresholds, _ = calibrate_stage_thresholds(
            [5.0, 6.0, 7.0, 8.0],
            [0, 0, 1, 1],
            ["ii"] * 4,
            LAST2_GROUPS,
            global_threshold=0.0,
            minimum_group_rows=2,
            shrinkage_pseudocount=256.0,
            shrunk=False,
            pin_first=True,
        )
        self.assertAlmostEqual(thresholds["ii"], 6.5)
        with self.assertRaisesRegex(ValueError, "cover every stage"):
            apply_stage_thresholds([0.0], ["ii"], {"first": 0.0})


class StructuredCrossValidationTest(unittest.TestCase):
    def test_global_policy_exactly_reuses_d4_training(self) -> None:
        examples: list[LabeledChunk] = []
        rows: list[list[float]] = []
        for index in range(100):
            fold = index % 5
            first = float((index % 11) - 5)
            second = float(((index * 3) % 7) - 3)
            label = int(1.2 * first - 0.4 * second + (index % 3) * 0.1 > 0)
            examples.append(
                LabeledChunk(
                    _chunk(1, total=2, fold=fold, input_index=index),
                    label,
                )
            )
            rows.append([first, second])
        values = np.asarray(rows, dtype=np.float32)
        names = ("first", "second")
        stages = {
            "position": tuple("second" for _ in examples),
            "last_action": tuple(
                "previous_interrupt" if index % 2 else "previous_silent"
                for index in range(len(examples))
            ),
            "last2": tuple("second" for _ in examples),
        }
        reference, reference_folds = cross_validate_neural_matrix(
            examples,
            values,
            names,
            folds=5,
            calibration_fold_offset=1,
            seed=11,
            max_iterations=80,
            l2_weights=[0.001, 0.01],
            l2_reduction="sum",
        )
        decisions, details, logits, thresholds = cross_validate_structured_calibration(
            examples,
            values,
            names,
            stages,
            folds=5,
            calibration_fold_offset=1,
            seed=11,
            max_iterations=80,
            l2_weights=[0.001, 0.01],
            l2_reduction="sum",
            minimum_group_rows=1,
            shrinkage_pseudocount=256.0,
            pin_first=True,
        )
        self.assertEqual(decisions["d4_global_replay"], reference)
        self.assertEqual(len(logits), len(examples))
        for fold_index, reference_fold in enumerate(reference_folds):
            for variant in D6_VARIANTS:
                detail = details[variant][fold_index]
                self.assertEqual(
                    detail["selected_l2_weight"],
                    reference_fold["selected_l2_weight"],
                )
                self.assertEqual(
                    detail["global_threshold_logit"],
                    reference_fold["threshold_logit"],
                )
                self.assertTrue(detail["d4_model_reused_across_all_variants"])
        self.assertEqual(
            set(thresholds["last2_shrunk"]),
            {example.key for example in examples},
        )


if __name__ == "__main__":
    unittest.main()
