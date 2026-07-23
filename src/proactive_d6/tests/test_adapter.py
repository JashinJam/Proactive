from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from proactive_d6.adapter import (
    CausalVisualMemory,
    LoRALinear,
    MEMORY_PARAMETERS,
    differentiable_tag_log_probability,
    memory_parameter_count,
)


class AdapterTest(unittest.TestCase):
    def test_differentiable_tag_score_matches_d4_reduction(self) -> None:
        logits = torch.randn(1, 3, 17, dtype=torch.bfloat16, requires_grad=True)
        token_ids = [2, 8, 4]
        actual = differentiable_tag_log_probability(logits, token_ids)
        values = logits.float()[0]
        ids = torch.tensor(token_ids)
        expected = (
            values.gather(1, ids[:, None]).squeeze(1)
            - torch.logsumexp(values, dim=-1)
        ).sum()
        self.assertTrue(torch.equal(actual, expected))
        actual.backward()
        self.assertIsNotNone(logits.grad)

    def test_memory_parameter_count_and_zero_injection(self) -> None:
        torch.manual_seed(1)
        memory = CausalVisualMemory()
        self.assertEqual(memory_parameter_count(memory), MEMORY_PARAMETERS)
        previous = memory.initial_state(torch.device("cpu"))
        update = memory(torch.randn(1, 1024), torch.randn(17, 1024), previous)
        self.assertEqual(tuple(update.state.shape), (1, 128))
        self.assertTrue(torch.equal(update.residual, torch.zeros_like(update.residual)))
        self.assertTrue(torch.isfinite(update.attention_entropy))
        self.assertGreaterEqual(float(update.normalized_attention_entropy), 0.0)
        self.assertLessEqual(float(update.normalized_attention_entropy), 1.0 + 1e-6)

    def test_memory_reset_and_chunk_detach(self) -> None:
        torch.manual_seed(2)
        memory = CausalVisualMemory(input_size=1024, memory_size=128, heads=4)
        query = torch.randn(1, 1024)
        visual = torch.randn(9, 1024)
        zero = memory.initial_state(torch.device("cpu"))
        first = memory(query, visual, zero)
        repeated = memory(query, visual, memory.initial_state(torch.device("cpu")))
        torch.testing.assert_close(first.state, repeated.state, rtol=0, atol=0)
        detached = first.state.detach()
        self.assertFalse(detached.requires_grad)
        second = memory(query, visual, detached)
        self.assertFalse(torch.equal(first.state, second.state))

    def test_future_visual_mutation_does_not_change_past_update(self) -> None:
        torch.manual_seed(3)
        memory = CausalVisualMemory()
        state = memory.initial_state(torch.device("cpu"))
        query = torch.randn(1, 1024)
        current = torch.randn(11, 1024)
        past = memory(query, current, state)
        future_a = torch.randn(7, 1024)
        future_b = future_a + 100.0
        historical_a = memory(query, current, state)
        historical_b = memory(query, current, state)
        memory(query, future_a, past.state.detach())
        memory(query, future_b, past.state.detach())
        torch.testing.assert_close(historical_a.state, historical_b.state, rtol=0, atol=0)

    def test_lora_zero_b_is_exact_identity_and_parameter_count(self) -> None:
        torch.manual_seed(4)
        base = nn.Linear(7, 11, bias=False)
        wrapper = LoRALinear(base, rank=3, alpha=6)
        value = torch.randn(2, 5, 7)
        expected = base(value)
        actual = wrapper(value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(wrapper.lora_a.numel() + wrapper.lora_b.numel(), 54)
        wrapper.lora_b.data.fill_(0.25)
        self.assertFalse(torch.equal(wrapper(value), expected))
        wrapper.enabled = False
        torch.testing.assert_close(wrapper(value), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
