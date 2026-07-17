from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from proactive_r0.artifacts import sha256_file
from proactive_r0.core import CausalInferenceConfig, StarterKitSymbols

from proactive_d1.core import (
    LinearDecisionHead,
    LinearModel,
    attach_gold_labels,
    binary_metrics,
    build_label_free_chunks,
    causal_scalar_values,
    cross_validate_linear,
    decision_answer,
    decisions_from_feature,
    feature_names,
    make_fold_manifest,
    metrics_for_subset,
    paired_session_bootstrap,
    load_decision_head,
    predict_feature_values,
    serialize_decision_head,
    select_threshold,
    strip_answers,
    validate_fold_manifest,
)
from proactive_d1.deploy import (
    process_session_with_fused_head,
    process_session_with_scalar_head,
)
from proactive_d1.analyze import decision_changes, grouped_metrics
from proactive_d1.analyze_neural import tag_margin_summary
from proactive_d1.audit_threshold import decisions_at_threshold, threshold_gate
from proactive_d1.internvl_features import (
    NeuralDecisionFeatures,
    tag_sequence_log_probability,
    validate_batched_tag_suffixes,
    validate_tag_suffix,
)
from proactive_d1.neural_core import (
    cross_validate_neural_matrix,
    load_aligned_neural_cache,
    neural_matrix,
)
from proactive_d1.merge_neural import validate_session_arrays


def source_rows(count: int = 10) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(count):
        rows.append(
            {
                "video_path": f"video-{index}.mp4",
                "video_intervals": [[0.0, 2.0], [2.0, 10.0]],
                "query": f"Help with task {index}",
                "task": f"Task {index}",
                "domain": "Chef" if index % 2 == 0 else "Tutorial",
                "dialog": [
                    [{"role": "user", "text": f"Help with task {index}"}],
                    [
                        {"role": "user", "text": f"Help with task {index}"},
                        {"role": "assistant", "text": "Opening guidance"},
                    ],
                ],
                "answers": ["$interrupt$Start", "$silent$"],
            }
        )
    return rows


def r0_records(count: int = 10) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(count):
        records.append(
            {
                "input_index": index,
                "video_path": f"video-{index}.mp4",
                "prediction": {
                    "video_path": f"video-{index}.mp4",
                    "answers": ["$silent$", "$silent$"],
                },
                "chunks": [
                    {
                        "chunk_index": 0,
                        "interval": [0.0, 2.0],
                        "model_input_frames": 16,
                        "raw_response": "Start with the first step.",
                    },
                    {
                        "chunk_index": 1,
                        "interval": [2.0, 10.0],
                        "model_input_frames": 32,
                        "raw_response": "$silent$",
                    },
                ],
            }
        )
    return records


class FoldManifestTest(unittest.TestCase):
    def test_assignment_is_label_independent_and_domain_balanced(self) -> None:
        rows = source_rows()
        changed = copy.deepcopy(rows)
        for row in changed:
            row["answers"] = ["$silent$", "$interrupt$changed"]
        first = make_fold_manifest(rows, folds=5, seed="fixed")
        second = make_fold_manifest(changed, folds=5, seed="fixed")
        self.assertEqual(first, second)
        fold_by_index = validate_fold_manifest(first, rows)
        self.assertEqual(set(fold_by_index.values()), set(range(5)))
        for domain in ("Chef", "Tutorial"):
            domain_folds = [
                fold_by_index[index]
                for index, row in enumerate(rows)
                if row["domain"] == domain
            ]
            self.assertEqual(sorted(domain_folds), list(range(5)))


class FeatureConstructionTest(unittest.TestCase):
    def test_label_free_features_reject_embedded_answers(self) -> None:
        rows = source_rows()
        manifest = make_fold_manifest(rows, folds=5, seed="fixed")
        folds = validate_fold_manifest(manifest, rows)
        with self.assertRaisesRegex(ValueError, "must not contain answers"):
            build_label_free_chunks(rows, r0_records(), folds, 4)

    def test_raw_response_and_temporal_features_align(self) -> None:
        rows = source_rows()
        manifest = make_fold_manifest(rows, folds=5, seed="fixed")
        folds = validate_fold_manifest(manifest, rows)
        features = build_label_free_chunks(
            strip_answers(rows), r0_records(), folds, max_history_turns=4
        )
        self.assertEqual(len(features), 20)
        self.assertEqual(features[0].values["is_first_chunk"], 1.0)
        self.assertEqual(features[0].values["raw_malformed_nonempty"], 1.0)
        self.assertEqual(features[0].values["r0_decision_interrupt"], 0.0)
        self.assertEqual(features[0].values["r0f_decision_interrupt"], 1.0)
        self.assertEqual(features[1].values["history_turn_fraction"], 0.25)
        self.assertNotIn("relative_position", features[0].values)
        self.assertNotIn("log1p_session_chunks", features[0].values)
        self.assertAlmostEqual(features[0].values["log1p_observed_end_sec"], 1.0986122886681098)
        direct = causal_scalar_values(
            row=strip_answers(rows)[0],
            chunk_index=0,
            interval=(0.0, 2.0),
            previous_end=None,
            model_input_frames=16,
            raw_response="Start with the first step.",
            r0_answer="$silent$",
            domains=["Chef", "Tutorial"],
            max_history_turns=4,
            max_model_frames=32,
        )
        self.assertEqual(features[0].values, direct)
        labeled = attach_gold_labels(features, rows)
        self.assertEqual([item.gold_interrupt for item in labeled[:2]], [1, 0])


class CalibrationTest(unittest.TestCase):
    def test_exact_threshold_prefers_separable_decisions(self) -> None:
        threshold, metrics = select_threshold(
            [-3.0, -2.0, 2.0, 3.0], [0, 0, 1, 1]
        )
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertGreater(threshold, -2.0)
        self.assertLess(threshold, 2.0)

    def test_five_fold_oof_covers_every_chunk(self) -> None:
        rows = source_rows()
        manifest = make_fold_manifest(rows, folds=5, seed="fixed")
        folds = validate_fold_manifest(manifest, rows)
        features = build_label_free_chunks(strip_answers(rows), r0_records(), folds, 4)
        labeled = attach_gold_labels(features, rows)
        names = feature_names("temporal", ["Chef", "Tutorial"])
        decisions, details = cross_validate_linear(
            labeled,
            names,
            folds=5,
            calibration_fold_offset=1,
            seed=7,
            max_iterations=60,
            l2_weight=0.001,
        )
        self.assertEqual(len(decisions), 20)
        self.assertEqual(len(details), 5)
        metrics = metrics_for_subset(labeled, decisions, include_first=True)
        self.assertGreater(metrics["macro_f1"], 0.99)
        for detail in details:
            self.assertNotIn(detail["test_fold"], detail["fit_folds"])
            self.assertNotIn(detail["calibration_fold"], detail["fit_folds"])

    def test_bootstrap_and_response_fallback(self) -> None:
        rows = source_rows()
        manifest = make_fold_manifest(rows, folds=5, seed="fixed")
        folds = validate_fold_manifest(manifest, rows)
        features = build_label_free_chunks(strip_answers(rows), r0_records(), folds, 4)
        labeled = attach_gold_labels(features, rows)
        candidate = {item.key: item.gold_interrupt for item in labeled}
        baseline = decisions_from_feature(labeled, "r0_decision_interrupt")
        bootstrap = paired_session_bootstrap(
            labeled, candidate, baseline, repetitions=100, seed=9
        )
        self.assertGreater(bootstrap["delta_macro_f1_p2_5"], 0.0)
        self.assertEqual(decision_answer("$silent$", 1), "$interrupt$Please continue with the next step.")
        self.assertEqual(decision_answer("Do this", 1), "$interrupt$Do this")
        self.assertEqual(decision_answer("Do this", 0), "$silent$")

    def test_binary_metrics_use_macro_of_both_classes(self) -> None:
        metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(metrics["interrupt_f1"], 0.5)
        self.assertAlmostEqual(metrics["silent_f1"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)

    def test_deployment_threshold_requires_complete_oof_logits(self) -> None:
        rows = source_rows()
        manifest = make_fold_manifest(rows, folds=5, seed="fixed")
        folds = validate_fold_manifest(manifest, rows)
        features = build_label_free_chunks(strip_answers(rows), r0_records(), folds, 4)
        labeled = attach_gold_labels(features, rows)
        logits = {
            example.key: 1.0 if example.gold_interrupt else -1.0
            for example in labeled
        }
        decisions = decisions_at_threshold(labeled, logits, 0.0)
        self.assertEqual(
            [decisions[example.key] for example in labeled],
            [example.gold_interrupt for example in labeled],
        )
        logits.pop(labeled[0].key)
        with self.assertRaisesRegex(ValueError, "complete OOF logits"):
            decisions_at_threshold(labeled, logits, 0.0)

    def test_threshold_gate_checks_global_and_local_stability(self) -> None:
        reference = {
            "macro_f1": 0.634,
            "interrupt_f1": 0.635,
            "silent_f1": 0.633,
        }
        unified = {
            "macro_f1": 0.632,
            "interrupt_f1": 0.631,
            "silent_f1": 0.633,
        }
        gate = {
            "max_overall_macro_f1_drop": 0.005,
            "max_worst_fold_macro_f1_drop": 0.02,
            "max_local_offset_macro_f1_drop": 0.005,
            "min_bootstrap_delta_p2_5": -0.01,
            "min_class_f1": 0.6,
        }
        result = threshold_gate(
            reference,
            unified,
            fold_deltas=[-0.01, 0.0],
            local_sweep_deltas=[-0.004, -0.002],
            bootstrap={"delta_macro_f1_p2_5": -0.008},
            gate=gate,
        )
        self.assertTrue(result["passed"])
        failed = threshold_gate(
            reference,
            unified,
            fold_deltas=[-0.03],
            local_sweep_deltas=[-0.004],
            bootstrap={"delta_macro_f1_p2_5": -0.008},
            gate=gate,
        )
        self.assertFalse(failed["passed"])

    def test_analysis_groups_and_change_categories(self) -> None:
        rows = [
            {"group": "a", "gold_interrupt": 1, "predicted_interrupt": 1, "r0f_interrupt": 0},
            {"group": "a", "gold_interrupt": 0, "predicted_interrupt": 0, "r0f_interrupt": 0},
            {"group": "b", "gold_interrupt": 0, "predicted_interrupt": 1, "r0f_interrupt": 0},
        ]
        grouped = grouped_metrics(rows, lambda row: str(row["group"]))
        self.assertEqual(grouped["a"]["candidate"]["macro_f1"], 1.0)
        changes = decision_changes(rows)
        self.assertEqual(changes["corrected_fn"], 1)
        self.assertEqual(changes["introduced_fp"], 1)
        self.assertEqual(changes["unchanged_correct"], 1)

    def test_serialized_decision_head_round_trip(self) -> None:
        head = LinearDecisionHead(
            feature_names=("a", "b"),
            model=LinearModel(
                mean=(1.0, 2.0),
                scale=(2.0, 4.0),
                weight=(1.0, -2.0),
                bias=0.25,
                train_loss=0.5,
            ),
            threshold_logit=-0.1,
        )
        payload = serialize_decision_head(head, {"purpose": "test"})
        loaded = load_decision_head(payload)
        self.assertEqual(loaded, head)
        decision, logit = predict_feature_values(loaded, {"a": 3.0, "b": 2.0})
        self.assertEqual(decision, 1)
        self.assertAlmostEqual(logit, 1.25)
        with self.assertRaisesRegex(ValueError, "missing"):
            predict_feature_values(loaded, {"a": 3.0})

    def test_forced_tag_scoring_helpers(self) -> None:
        input_ids = torch.tensor([[10, 11, 2, 3]])
        self.assertEqual(validate_tag_suffix(input_ids, [2, 3]), 2)
        with self.assertRaisesRegex(ValueError, "changed"):
            validate_tag_suffix(input_ids, [3, 2])
        logits = torch.tensor([[0.0, 2.0, -1.0], [3.0, 0.0, -2.0]])
        score = tag_sequence_log_probability(logits, [1, 0])
        expected = torch.log_softmax(logits, dim=-1)[[0, 1], [1, 0]].sum()
        self.assertAlmostEqual(score, float(expected), places=6)

    def test_batched_candidate_suffix_validation(self) -> None:
        input_ids = torch.tensor([[10, 11, 2, 3], [10, 11, 4, 5]])
        self.assertEqual(
            validate_batched_tag_suffixes(input_ids, [[2, 3], [4, 5]]), 2
        )
        with self.assertRaisesRegex(ValueError, "row 1"):
            validate_batched_tag_suffixes(input_ids, [[2, 3], [5, 4]])
        with self.assertRaisesRegex(ValueError, "align"):
            validate_batched_tag_suffixes(input_ids, [[2, 3]])

    def test_aligned_neural_cache_and_oof_matrix(self) -> None:
        rows = source_rows()
        manifest = make_fold_manifest(rows, folds=5, seed="fixed")
        folds = validate_fold_manifest(manifest, rows)
        features = build_label_free_chunks(strip_answers(rows), r0_records(), folds, 4)
        examples = attach_gold_labels(features, rows)
        keys = [example.key for example in examples]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            np.savez(
                path,
                hidden_state=np.asarray(
                    [[example.gold_interrupt, example.feature.chunk_index] for example in examples],
                    dtype=np.float32,
                ),
                tag_margin=np.asarray(
                    [2 * example.gold_interrupt - 1 for example in examples], dtype=np.float32
                ),
                silent_log_probability=np.full(len(examples), -2.0, dtype=np.float32),
                interrupt_log_probability=np.full(len(examples), -1.0, dtype=np.float32),
                prompt_tokens=np.full(len(examples), 10, dtype=np.int32),
                input_index=np.asarray([key[0] for key in keys], dtype=np.int32),
                chunk_index=np.asarray([key[1] for key in keys], dtype=np.int32),
            )
            cache = load_aligned_neural_cache(path, examples, hidden_size=2)
        scalar_names = feature_names("temporal", ["Chef", "Tutorial"])
        values, names = neural_matrix(examples, cache, scalar_names, "tag_only")
        decisions, details = cross_validate_neural_matrix(
            examples,
            values,
            names,
            folds=5,
            calibration_fold_offset=1,
            seed=11,
            max_iterations=50,
            l2_weights=[0.0001, 0.001],
            l2_reduction="sum",
        )
        self.assertEqual(len(decisions), len(examples))
        self.assertEqual(len(details), 5)
        self.assertGreater(
            metrics_for_subset(examples, decisions, include_first=True)["macro_f1"],
            0.99,
        )

    def test_online_scalar_deployment_uses_current_chunk_only(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.responses = iter(["Start now", "$silent$"])

            def generate(self, frames, messages, max_new_tokens):
                return next(self.responses)

        starter = StarterKitSymbols(
            system_prompt="system",
            normalize_dialog_turns=lambda turns: [
                {"role": "assistant", "content": str(turn["text"])} for turn in turns
            ],
            extract_frames=lambda *args, **kwargs: [object()] * 16,
        )
        head = LinearDecisionHead(
            feature_names=(
                "is_first_chunk",
                "domain=Chef",
                "domain=Tutorial",
            ),
            model=LinearModel(
                mean=(0.0, 0.0, 0.0),
                scale=(1.0, 1.0, 1.0),
                weight=(2.0, 0.0, 0.0),
                bias=-1.0,
                train_loss=0.0,
            ),
            threshold_logit=0.0,
        )
        result = process_session_with_scalar_head(
            row=strip_answers(source_rows(1))[0],
            input_index=0,
            video_folder=Path("/unused"),
            model=FakeModel(),
            starter=starter,
            config=CausalInferenceConfig(
                frames_per_interval=16,
                max_frames=32,
                max_history_turns=4,
                max_new_tokens=8,
            ),
            head=head,
        )
        self.assertEqual(
            result["prediction"]["answers"],  # type: ignore[index]
            ["$interrupt$Start now", "$silent$"],
        )

    def test_online_fused_deployment_combines_scalar_tag_and_hidden(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.responses = iter(["Start now", "Keep going"])
                self.features = iter(
                    [
                        NeuralDecisionFeatures(
                            hidden_state=np.asarray([1.0, 0.0], dtype=np.float32),
                            silent_log_probability=-2.0,
                            interrupt_log_probability=-1.0,
                            tag_margin=1.0,
                            prompt_tokens=20,
                            hidden_max_abs_difference=0.0,
                            hidden_cosine_similarity=1.0,
                        ),
                        NeuralDecisionFeatures(
                            hidden_state=np.asarray([-1.0, 0.0], dtype=np.float32),
                            silent_log_probability=-1.0,
                            interrupt_log_probability=-2.0,
                            tag_margin=-1.0,
                            prompt_tokens=24,
                            hidden_max_abs_difference=0.0,
                            hidden_cosine_similarity=1.0,
                        ),
                    ]
                )

            def generate(self, frames, messages, max_new_tokens):
                return next(self.responses)

            def extract_decision_features(self, frames, messages):
                return next(self.features)

        starter = StarterKitSymbols(
            system_prompt="system",
            normalize_dialog_turns=lambda turns: [
                {"role": "assistant", "content": str(turn["text"])} for turn in turns
            ],
            extract_frames=lambda *args, **kwargs: [object()] * 16,
        )
        head = LinearDecisionHead(
            feature_names=(
                "is_first_chunk",
                "domain=Chef",
                "domain=Tutorial",
                "tag_margin",
                "hidden_0000",
                "hidden_0001",
            ),
            model=LinearModel(
                mean=(0.0,) * 6,
                scale=(1.0,) * 6,
                weight=(0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
                bias=0.0,
                train_loss=0.0,
            ),
            threshold_logit=0.0,
        )
        result = process_session_with_fused_head(
            row=strip_answers(source_rows(1))[0],
            input_index=0,
            video_folder=Path("/unused"),
            model=FakeModel(),
            starter=starter,
            config=CausalInferenceConfig(
                frames_per_interval=16,
                max_frames=32,
                max_history_turns=4,
                max_new_tokens=8,
            ),
            head=head,
        )
        self.assertEqual(
            result["prediction"]["answers"],  # type: ignore[index]
            ["$interrupt$Start now", "$silent$"],
        )
        chunks = result["chunks"]
        self.assertEqual([chunk["tag_margin"] for chunk in chunks], [1.0, -1.0])
        self.assertEqual([chunk["decision_interrupt"] for chunk in chunks], [1, 0])
        self.assertEqual(
            [chunk["decision_feature_mode"] for chunk in chunks],
            ["sequential", "sequential"],
        )
        self.assertEqual(
            [chunk["candidate_forward_passes"] for chunk in chunks], [2, 2]
        )

    def test_merge_cache_validation_and_tag_auc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "session.npz"
            np.savez(
                path,
                hidden_state=np.ones((2, 3), dtype=np.float32),
                tag_margin=np.asarray([-1.0, 2.0], dtype=np.float32),
                silent_log_probability=np.asarray([-1.0, -3.0], dtype=np.float32),
                interrupt_log_probability=np.asarray([-2.0, -1.0], dtype=np.float32),
                prompt_tokens=np.asarray([4, 5], dtype=np.int32),
                input_index=np.asarray(7, dtype=np.int32),
                chunk_index=np.asarray([0, 1], dtype=np.int32),
            )
            record = {
                "feature_path": str(path),
                "feature_sha256": sha256_file(path),
                "input_index": 7,
                "extracted_chunks": 2,
                "source_chunks": 2,
                "complete_session": True,
            }
            arrays = validate_session_arrays(record, hidden_size=3)
            self.assertEqual(arrays["hidden_state"].shape, (2, 3))
            broken = dict(record)
            broken["complete_session"] = False
            with self.assertRaisesRegex(ValueError, "partial"):
                validate_session_arrays(broken, hidden_size=3)
        margin = tag_margin_summary(
            [
                {"gold_interrupt": 0, "tag_margin": -2.0},
                {"gold_interrupt": 0, "tag_margin": -1.0},
                {"gold_interrupt": 1, "tag_margin": 1.0},
                {"gold_interrupt": 1, "tag_margin": 2.0},
            ]
        )
        self.assertEqual(margin["roc_auc"], 1.0)
        self.assertEqual(margin["zero_threshold_metrics"]["macro_f1"], 1.0)  # type: ignore[index]


if __name__ == "__main__":
    unittest.main()
