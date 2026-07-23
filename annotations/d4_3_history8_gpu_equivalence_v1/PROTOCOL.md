# D4.3 history8 GPU equivalence protocol

Status: frozen before GPU inference on 2026-07-21.

## Question

Does the online shared-vision deployment path with `max_history_turns=8`
reproduce the completed D4.2 history8 generation cache, neural features,
full-development head logits, decisions, and answers?

## Frozen inputs

- InternVL3.5-1B revision `9191dbccf312b537016f041b25d61c72e7c5c9f3`.
- BF16, SDPA, greedy decode, 32 cumulative frames, 16 frames per interval,
  eight dialog turns, and 64 new tokens.
- D4.2 history8 head SHA256
  `dab9eaf100ea301055ab4d68856d406fb5927864bc96c71f2038688067b904c5`.
- Source indices `143,356,472,609`, in source order. They cover all four domains,
  102 chunks, and long official assistant histories.
- Reference generation records SHA256
  `af52d7454c36051c11302b5201c38df364e8797e82d5742c67c53c7216d0bc39`.
- Reference neural cache SHA256
  `a9780f9aef5b3c0cf66aafd873b5b32484b6330cd524ad3a06d1a6de8be9727e`.
- Reference final records SHA256
  `daad7670a3ce9aaeaab3909966d97250529f9e0040aa454427667952c1facded`.

## Gates

All 102 chunks must match raw response, prompt-token count, tag margin, hidden
state, dialog features, decision, and answer. Numeric tag-margin/hidden/logit
maximum absolute differences must be no greater than `1e-6`. Total parameters
must remain below 2B, peak allocated memory must stay below the resource
admission estimate, and every measured session must stay below 300 seconds.

This smoke is an engineering equivalence check, not a new performance estimate.
