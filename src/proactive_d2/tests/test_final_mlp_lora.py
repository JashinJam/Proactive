from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from proactive_d2.final_mlp_lora import (
    FinalMLPStateCapture,
    decision_margin_from_logits,
    final_mlp_cache_bytes_per_chunk,
    final_mlp_lora_parameter_count,
    final_mlp_target_regex,
    reconstruct_final_hidden,
    tag_sequence_log_probability_tensor,
)
from proactive_d2.final_mlp_cache import (
    bfloat16_tensor_to_uint16,
    state_from_bit_arrays,
    state_to_bit_arrays,
    uint16_to_bfloat16_tensor,
)
from proactive_d2.final_mlp_training import fixed_shape_batches


class _FakeDecoderLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.post_attention_layernorm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 2 * width, bias=False),
            nn.GELU(),
            nn.Linear(2 * width, width, bias=False),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden + 0.25
        normalized = self.post_attention_layernorm(residual)
        return residual + self.mlp(normalized)


class FinalMLPLoRATest(unittest.TestCase):
    def test_parameter_and_cache_accounting(self) -> None:
        self.assertEqual(final_mlp_lora_parameter_count(1024, 3072, 8), 98_304)
        self.assertEqual(
            final_mlp_cache_bytes_per_chunk(
                candidates=2,
                tag_length=3,
                hidden_size=1024,
                bytes_per_value=2,
                stored_tensors_per_candidate=6,
            ),
            73_728,
        )
        self.assertEqual(
            final_mlp_target_regex(27),
            r"model\.language_model\.layers\.27\.mlp\.(gate_proj|up_proj|down_proj)",
        )

    def test_capture_reconstructs_exact_final_hidden(self) -> None:
        torch.manual_seed(7)
        layer = _FakeDecoderLayer(width=6).eval()
        final_norm = nn.LayerNorm(6).eval()
        inputs = torch.randn(2, 8, 6)
        with FinalMLPStateCapture(layer, tag_length=3) as capture:
            full = final_norm(layer(inputs))
        reconstructed = reconstruct_final_hidden(layer, final_norm, capture.state())
        expected = full[:, -4:-1, :]
        torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)

    def test_reference_cache_removes_local_base_offset(self) -> None:
        torch.manual_seed(11)
        layer = _FakeDecoderLayer(width=4).eval()
        final_norm = nn.LayerNorm(4).eval()
        inputs = torch.randn(1, 7, 4)
        with FinalMLPStateCapture(layer, tag_length=2) as capture:
            full = final_norm(layer(inputs))
        raw = capture.state()
        local_base = layer.mlp(raw.normalized)
        reference = capture.mlp_output() + 0.125
        corrected = type(raw)(
            residual=raw.residual,
            normalized=raw.normalized,
            reference_mlp_output=reference,
            local_base_mlp_output=local_base,
        )
        reconstructed = reconstruct_final_hidden(layer, final_norm, corrected)
        expected = final_norm(raw.residual + reference)
        torch.testing.assert_close(reconstructed, expected, rtol=0, atol=0)
        self.assertFalse(torch.equal(reconstructed, full[:, -3:-1, :]))

    def test_final_hidden_cache_removes_local_norm_offset(self) -> None:
        torch.manual_seed(13)
        layer = _FakeDecoderLayer(width=4).eval()
        final_norm = nn.LayerNorm(4).eval()
        residual = torch.randn(1, 2, 4)
        normalized = torch.randn(1, 2, 4)
        local_base_mlp = layer.mlp(normalized).detach()
        reference_mlp = local_base_mlp.clone()
        local_base_hidden = final_norm(residual + reference_mlp).detach()
        reference_hidden = local_base_hidden + 0.25
        from proactive_d2.final_mlp_lora import FinalMLPScoringState

        state = FinalMLPScoringState(
            residual=residual,
            normalized=normalized,
            reference_mlp_output=reference_mlp,
            local_base_mlp_output=local_base_mlp,
            reference_final_hidden=reference_hidden,
            local_base_final_hidden=local_base_hidden,
        )
        reconstructed = reconstruct_final_hidden(layer, final_norm, state)
        torch.testing.assert_close(reconstructed, reference_hidden, rtol=0, atol=0)

    def test_tag_margin_is_differentiable_and_correct(self) -> None:
        silent = torch.tensor(
            [[[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]]], requires_grad=True
        )
        interrupt = torch.tensor(
            [[[0.0, 2.0, -1.0], [2.0, 1.0, 0.0]]], requires_grad=True
        )
        silent_ids = [0, 2]
        interrupt_ids = [1, 0]
        margin = decision_margin_from_logits(
            silent, interrupt, silent_ids, interrupt_ids
        )
        expected = tag_sequence_log_probability_tensor(
            interrupt, interrupt_ids
        ) - tag_sequence_log_probability_tensor(silent, silent_ids)
        torch.testing.assert_close(margin, expected)
        margin.sum().backward()
        self.assertGreater(float(silent.grad.abs().sum()), 0.0)
        self.assertGreater(float(interrupt.grad.abs().sum()), 0.0)

    def test_invalid_dimensions_fail(self) -> None:
        with self.assertRaises(ValueError):
            final_mlp_lora_parameter_count(0, 3072, 8)
        with self.assertRaises(ValueError):
            final_mlp_cache_bytes_per_chunk(
                candidates=2, tag_length=0, hidden_size=1024, bytes_per_value=2
            )

    def test_bfloat16_bit_round_trip_is_exact(self) -> None:
        values = torch.tensor(
            [-3.5, -0.0, 0.0, 0.125, 1.0, 17.25], dtype=torch.bfloat16
        )
        bits = bfloat16_tensor_to_uint16(values)
        self.assertEqual(bits.dtype.name, "uint16")
        restored = uint16_to_bfloat16_tensor(bits)
        self.assertTrue(torch.equal(values, restored))
        self.assertTrue(torch.equal(values.view(torch.uint16), restored.view(torch.uint16)))

    def test_corrected_state_bit_round_trip_is_exact(self) -> None:
        torch.manual_seed(19)
        tensors = [torch.randn(1, 3, 5, dtype=torch.bfloat16) for _ in range(6)]
        state = type(self)._state(*tensors)
        arrays = state_to_bit_arrays(state, remove_batch_dimension=False)
        restored = state_from_bit_arrays(arrays)
        for original, candidate in zip(
            tensors,
            (
                restored.residual,
                restored.normalized,
                restored.reference_mlp_output,
                restored.local_base_mlp_output,
                restored.reference_final_hidden,
                restored.local_base_final_hidden,
            ),
        ):
            self.assertTrue(torch.equal(original, candidate))

    def test_fixed_shape_batches_pad_without_changing_real_order(self) -> None:
        batches = fixed_shape_batches([5, 2, 9, 1, 4], batch_size=3)
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0][1], 3)
        self.assertEqual(batches[0][0].tolist(), [5, 2, 9])
        self.assertEqual(batches[1][1], 2)
        self.assertEqual(batches[1][0].tolist(), [1, 4, 4])

    @staticmethod
    def _state(
        residual: torch.Tensor,
        normalized: torch.Tensor,
        reference: torch.Tensor,
        local_base: torch.Tensor,
        reference_hidden: torch.Tensor,
        local_base_hidden: torch.Tensor,
    ):
        from proactive_d2.final_mlp_lora import FinalMLPScoringState

        return FinalMLPScoringState(
            residual=residual,
            normalized=normalized,
            reference_mlp_output=reference,
            local_base_mlp_output=local_base,
            reference_final_hidden=reference_hidden,
            local_base_final_hidden=local_base_hidden,
        )


if __name__ == "__main__":
    unittest.main()
