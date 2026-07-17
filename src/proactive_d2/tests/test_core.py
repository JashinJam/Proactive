from __future__ import annotations

import unittest

import numpy as np

from proactive_d1.core import LinearModel, binary_metrics, select_threshold
from proactive_d2.core import (
    ResidualDecisionHead,
    ResidualMLPModel,
    fit_residual_mlp,
    load_residual_head,
    predict_residual_logits,
    residual_parameter_count,
    serialize_residual_head,
)


class ResidualMLPTest(unittest.TestCase):
    def test_parameter_count_and_serialization_round_trip(self) -> None:
        base = LinearModel(
            mean=(0.0, 0.0),
            scale=(1.0, 1.0),
            weight=(1.0, -1.0),
            bias=0.25,
            train_loss=0.5,
        )
        model = ResidualMLPModel(
            base=base,
            hidden_weight=((1.0, 0.0), (0.0, 1.0)),
            hidden_bias=(0.0, 0.0),
            output_weight=(0.2, -0.1),
            output_bias=0.05,
            fit_loss=0.4,
            calibration_loss=0.45,
            best_epoch=3,
            epochs_run=7,
        )
        head = ResidualDecisionHead(
            feature_names=("a", "b"), model=model, threshold_logit=0.1
        )
        payload = serialize_residual_head(head, {"test": True})
        loaded = load_residual_head(payload)
        values = [[1.0, 2.0], [-1.0, 0.5]]
        self.assertTrue(
            np.allclose(
                predict_residual_logits(model, values),
                predict_residual_logits(loaded.model, values),
                rtol=0.0,
                atol=0.0,
            )
        )
        self.assertEqual(residual_parameter_count(2, 2), 9)

    def test_residual_learns_nonlinear_quadrant_signal(self) -> None:
        generator = np.random.default_rng(7)
        values = generator.normal(size=(600, 2)).astype(np.float32)
        labels = ((values[:, 0] * values[:, 1]) > 0).astype(np.int64)
        base = LinearModel(
            mean=(0.0, 0.0),
            scale=(1.0, 1.0),
            weight=(0.0, 0.0),
            bias=0.0,
            train_loss=1.0,
        )
        model, _ = fit_residual_mlp(
            values[:400],
            labels[:400],
            values[400:],
            labels[400:],
            base,
            hidden_width=8,
            learning_rate=0.01,
            weight_decay=0.001,
            batch_size=64,
            max_epochs=100,
            patience=20,
            min_delta=0.0001,
            gradient_clip_norm=1.0,
            seed=9,
        )
        logits = predict_residual_logits(model, values[400:])
        threshold, _ = select_threshold(logits, labels[400:].tolist())
        predictions = [int(value >= threshold) for value in logits]
        metrics = binary_metrics(labels[400:].tolist(), predictions)
        self.assertGreater(metrics["macro_f1"], 0.85)
        self.assertGreater(model.best_epoch, 0)


if __name__ == "__main__":
    unittest.main()
