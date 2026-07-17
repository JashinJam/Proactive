from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from proactive_d2.run_final_mlp_oof import (
    DEFAULT_CONFIG,
    TEST_LABEL_SENTINEL,
    _assert_restricted_fold_labels,
    _atomic_output_directory,
    _fit_head_for_fold,
    main,
    _promotion_checks,
    _restricted_fold_labels,
    _validate_protocol_config,
    _verified_d1_macro_f1,
)


class FinalMLPOOFProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.fit = np.asarray([0, 1], dtype=np.int64)
        self.calibration = np.asarray([2, 3], dtype=np.int64)
        self.test = np.asarray([4, 5], dtype=np.int64)
        self.labels = np.asarray([0, 1, 0, 1, 1, 0], dtype=np.int64)

    def test_restricted_fold_labels_mask_only_test_rows(self) -> None:
        restricted = _restricted_fold_labels(
            self.labels, self.fit, self.calibration, self.test
        )
        np.testing.assert_array_equal(restricted[self.fit], [0, 1])
        np.testing.assert_array_equal(restricted[self.calibration], [0, 1])
        np.testing.assert_array_equal(
            restricted[self.test], [TEST_LABEL_SENTINEL, TEST_LABEL_SENTINEL]
        )
        np.testing.assert_array_equal(self.labels, [0, 1, 0, 1, 1, 0])

    def test_fold_label_assertions_reject_visible_test_label_and_overlap(self) -> None:
        restricted = _restricted_fold_labels(
            self.labels, self.fit, self.calibration, self.test
        )
        exposed = restricted.copy()
        exposed[self.test[0]] = 1
        with self.assertRaisesRegex(ValueError, "sentinel"):
            _assert_restricted_fold_labels(
                exposed, self.fit, self.calibration, self.test
            )
        with self.assertRaisesRegex(ValueError, "disjoint"):
            _assert_restricted_fold_labels(
                restricted,
                np.asarray([0, 4]),
                self.calibration,
                self.test,
            )

    def test_head_accepts_only_sentinel_masked_test_labels(self) -> None:
        restricted = _restricted_fold_labels(
            self.labels, self.fit, self.calibration, self.test
        )
        values = np.asarray([[-2.0], [2.0], [-1.0], [1.0], [3.0], [-3.0]])
        predictions, details = _fit_head_for_fold(
            values,
            ("value",),
            restricted,
            self.fit,
            self.calibration,
            self.test,
            l2_weights=[0.001],
            seed=17,
            max_iterations=20,
            test_fold=0,
        )
        self.assertEqual(predictions.shape, (2,))
        self.assertEqual(details["selected_l2_weight"], 0.001)

        exposed = restricted.copy()
        exposed[self.test] = self.labels[self.test]
        with self.assertRaisesRegex(ValueError, "sentinel"):
            _fit_head_for_fold(
                values,
                ("value",),
                exposed,
                self.fit,
                self.calibration,
                self.test,
                l2_weights=[0.001],
                seed=17,
                max_iterations=20,
                test_fold=0,
            )

    def test_verified_d1_macro_f1_is_bound_to_metrics(self) -> None:
        metrics = {"overall": {"macro_f1": 0.6341}}
        self.assertEqual(_verified_d1_macro_f1(metrics, 0.6341), 0.6341)
        with self.assertRaisesRegex(ValueError, "differs"):
            _verified_d1_macro_f1(metrics, 0.6)

    def test_promotion_optional_requirements_follow_boolean_flags(self) -> None:
        config: dict[str, object] = {
            "min_delta_macro_f1_vs_d1": 0.005,
            "min_positive_folds": 4,
            "require_positive_session_bootstrap_lower_bound": False,
            "require_non_first_chunk_gain": False,
            "require_both_class_f1_nonzero": False,
        }
        checks = _promotion_checks(
            candidate_macro_f1=0.65,
            d1_macro_f1=0.6341,
            bootstrap_lower_bound=-0.1,
            positive_folds=4,
            non_first_macro_f1=0.2,
            d1_non_first_macro_f1=0.3,
            interrupt_f1=0.0,
            silent_f1=0.6,
            config=config,
        )
        self.assertTrue(all(checks.values()))

        enabled = dict(config)
        enabled.update(
            {
                "require_positive_session_bootstrap_lower_bound": True,
                "require_non_first_chunk_gain": True,
                "require_both_class_f1_nonzero": True,
            }
        )
        enabled_checks = _promotion_checks(
            candidate_macro_f1=0.65,
            d1_macro_f1=0.6341,
            bootstrap_lower_bound=-0.1,
            positive_folds=4,
            non_first_macro_f1=0.2,
            d1_non_first_macro_f1=0.3,
            interrupt_f1=0.0,
            silent_f1=0.6,
            config=enabled,
        )
        self.assertFalse(enabled_checks["positive_bootstrap_lower_bound"])
        self.assertFalse(enabled_checks["non_first_chunk_gain"])
        self.assertFalse(enabled_checks["both_classes_nonzero"])

    def test_frozen_protocol_fields_are_validated(self) -> None:
        config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
        _validate_protocol_config(config)

        changed = copy.deepcopy(config)
        changed["training"]["optimizer"] = "sgd"
        with self.assertRaisesRegex(ValueError, "training.optimizer"):
            _validate_protocol_config(changed)

        changed = copy.deepcopy(config)
        changed["evaluation"]["promotion"][
            "require_non_first_chunk_gain"
        ] = "true"
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            _validate_protocol_config(changed)

    def test_atomic_output_publishes_only_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "audit-output"
            expected_staging = final.with_name(f"{final.name}.incomplete-{os.getpid()}")
            with _atomic_output_directory(final) as staging:
                self.assertEqual(staging, expected_staging)
                self.assertFalse(final.exists())
                (staging / "runtime.json").write_text("{}\n", encoding="utf-8")
            self.assertTrue(final.is_dir())
            self.assertFalse(expected_staging.exists())
            self.assertTrue((final / "runtime.json").is_file())

    def test_atomic_output_preserves_unique_failure_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "formal-output"
            staging = final.with_name(f"{final.name}.incomplete-{os.getpid()}")
            with self.assertRaisesRegex(RuntimeError, "deliberate"):
                with _atomic_output_directory(final) as active:
                    self.assertEqual(active, staging)
                    raise RuntimeError("deliberate failure")
            self.assertFalse(final.exists())
            self.assertTrue(staging.is_dir())
            failure = json.loads(
                (staging / "failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["status"], "incomplete")
            self.assertEqual(failure["exception_type"], "RuntimeError")
            self.assertEqual(failure["pid"], os.getpid())

    def test_audit_only_main_uses_the_atomic_publication_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            final = Path(temporary) / "audit-only"

            def fake_run(
                argv: list[str], *, output_dir: Path, final_output_dir: Path
            ) -> None:
                self.assertIn("--audit-only", argv)
                self.assertEqual(final_output_dir, final.resolve())
                self.assertFalse(final_output_dir.exists())
                (output_dir / "runtime.json").write_text(
                    '{"status":"audit-only"}\n', encoding="utf-8"
                )

            with patch(
                "proactive_d2.run_final_mlp_oof._run_main", side_effect=fake_run
            ):
                main(
                    [
                        "--config",
                        str(DEFAULT_CONFIG),
                        "--output-dir",
                        str(final),
                        "--audit-only",
                    ]
                )
            self.assertTrue(final.is_dir())
            self.assertEqual(
                json.loads((final / "runtime.json").read_text(encoding="utf-8"))[
                    "status"
                ],
                "audit-only",
            )


if __name__ == "__main__":
    unittest.main()
