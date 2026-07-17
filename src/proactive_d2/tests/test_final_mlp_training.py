from __future__ import annotations

import unittest
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from proactive_d2.final_mlp_cache import STATE_NAMES, bfloat16_tensor_to_uint16
from proactive_d2.final_mlp_training import (
    FinalMLPCacheArrays,
    export_adapter_features,
    train_adapter_fold,
)


class _ToyMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.base = nn.Linear(width, width, bias=False)
        self.adapter = nn.Linear(width, width, bias=False)
        nn.init.eye_(self.base.weight)
        nn.init.zeros_(self.adapter.weight)
        self.base.weight.requires_grad_(False)
        self.adapter_enabled = True

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.float()
        batch_offset = 0.125 * (value.shape[0] - 1)
        result = self.base(value) + batch_offset
        return result + self.adapter(value) if self.adapter_enabled else result


class _ToyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mlp = _ToyMLP(width)


class _ToyPeft(nn.Module):
    def __init__(self, layer: _ToyLayer) -> None:
        super().__init__()
        self.layer = layer

    @contextmanager
    def disable_adapter(self):
        previous = self.layer.mlp.adapter_enabled
        self.layer.mlp.adapter_enabled = False
        try:
            yield
        finally:
            self.layer.mlp.adapter_enabled = previous


class _BatchShapeNorm(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 0.25 * (value.shape[0] - 1)


class _BatchShapeHead(nn.Linear):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        logits = super().forward(value)
        offset = torch.tensor(
            [0.125, -0.25], dtype=logits.dtype, device=logits.device
        )
        return logits + offset * (value.shape[0] - 1)


def _toy_cache(rows: int = 12, width: int = 4) -> tuple[FinalMLPCacheArrays, np.ndarray]:
    x = torch.zeros(rows, 1, width, dtype=torch.bfloat16)
    x[:, 0, 0] = torch.linspace(-2, 2, rows).to(torch.bfloat16)
    residual = torch.zeros_like(x)
    reference_mlp = x.clone()
    local_base_mlp = x.clone()
    reference_hidden = x.clone()
    local_base_hidden = x.clone()
    values = {
        "residual": residual,
        "normalized": x,
        "reference_mlp_output": reference_mlp,
        "local_base_mlp_output": local_base_mlp,
        "reference_final_hidden": reference_hidden,
        "local_base_final_hidden": local_base_hidden,
    }
    state_bits: dict[str, np.ndarray] = {}
    for candidate in ("silent", "interrupt"):
        for name in STATE_NAMES:
            state_bits[f"{candidate}_{name}_bits"] = bfloat16_tensor_to_uint16(
                values[name]
            )
    labels = (x[:, 0, 0].float().numpy() > 0).astype(np.int64)
    return (
        FinalMLPCacheArrays(
            state_bits=state_bits,
            base_hidden_state=x[:, 0].float().numpy(),
            base_tag_margin=np.zeros(rows, dtype=np.float32),
            prompt_tokens=np.ones(rows, dtype=np.int32),
            input_index=np.arange(rows, dtype=np.int32),
            chunk_index=np.zeros(rows, dtype=np.int32),
        ),
        labels,
    )


class FinalMLPTrainingTest(unittest.TestCase):
    def test_toy_adapter_trains_and_exports_fixed_shape_features(self) -> None:
        torch.manual_seed(23)
        cache, labels = _toy_cache()
        layer = _ToyLayer(width=4)
        peft = _ToyPeft(layer)
        lm_head = nn.Linear(4, 2, bias=False).to(dtype=torch.bfloat16)
        with torch.no_grad():
            lm_head.weight.zero_()
            lm_head.weight[1, 0] = 1.0
            lm_head.weight[0, 0] = -1.0
        lm_head.weight.requires_grad_(False)
        initial = {
            name: parameter.detach().clone()
            for name, parameter in peft.named_parameters()
            if parameter.requires_grad
        }
        result = train_adapter_fold(
            cache,
            labels,
            np.asarray([0, 1, 2, 3, 8, 9, 10, 11], dtype=np.int64),
            np.asarray([4, 5, 6, 7], dtype=np.int64),
            peft_model=peft,
            initial_state=initial,
            decoder_layer=layer,
            final_norm=_BatchShapeNorm(),
            lm_head=lm_head,
            silent_token_ids=[0],
            interrupt_token_ids=[1],
            device=torch.device("cpu"),
            learning_rate=0.05,
            weight_decay=0.0,
            batch_size=4,
            max_epochs=4,
            patience=2,
            min_delta=1e-6,
            gradient_clip_norm=1.0,
            seed=31,
        )
        self.assertTrue(np.isfinite(result.best_calibration_loss))
        self.assertGreater(result.best_epoch, 0)
        self.assertGreaterEqual(len(result.history), 2)
        self.assertGreater(
            max(float(epoch["max_gradient_norm"] or 0.0) for epoch in result.history),
            0.0,
        )
        margins, hidden, candidate_difference = export_adapter_features(
            cache,
            np.arange(cache.rows, dtype=np.int64),
            batch_size=4,
            peft_model=peft,
            decoder_layer=layer,
            final_norm=_BatchShapeNorm(),
            lm_head=lm_head,
            silent_token_ids=[0],
            interrupt_token_ids=[1],
            device=torch.device("cpu"),
        )
        self.assertEqual(margins.shape, (12,))
        self.assertEqual(hidden.shape, (12, 4))
        self.assertEqual(candidate_difference, 0.0)
        self.assertTrue(np.isfinite(margins).all())
        self.assertTrue(np.isfinite(hidden).all())

    def test_zero_adapter_replay_cancels_batch_shape_offsets(self) -> None:
        cache, _ = _toy_cache(rows=65, width=4)
        layer = _ToyLayer(width=4)
        peft = _ToyPeft(layer)
        lm_head = _BatchShapeHead(4, 2, bias=False).to(dtype=torch.bfloat16)
        lm_head.weight.requires_grad_(False)
        margins, hidden, candidate_difference = export_adapter_features(
            cache,
            np.arange(cache.rows, dtype=np.int64),
            batch_size=64,
            peft_model=peft,
            decoder_layer=layer,
            final_norm=_BatchShapeNorm(),
            lm_head=lm_head,
            silent_token_ids=[0],
            interrupt_token_ids=[1],
            device=torch.device("cpu"),
        )
        np.testing.assert_array_equal(hidden, cache.base_hidden_state)
        np.testing.assert_array_equal(margins, cache.base_tag_margin)
        self.assertEqual(candidate_difference, 0.0)


if __name__ == "__main__":
    unittest.main()
