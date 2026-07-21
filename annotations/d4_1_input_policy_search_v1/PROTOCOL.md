# D4.1 Input-Policy Search Protocol

Status: frozen before D4.1 model inference.

This user-authorized branch audits four inference-input limits on public validation. It does not change the D3 scientific baseline or the frozen D4 submission package.

## Frozen System

- Backbone: `OpenGVLab/InternVL3_5-1B-HF` revision `9191dbccf312b537016f041b25d61c72e7c5c9f3`.
- D4 head: exact 1,052-parameter `dialog_stage_fused` head, SHA256 `531431710a01a71bdd02ffd7a9758428fe282323cc41fae2c1d6859e45408b13`.
- Threshold: `0.1263874797442615`.
- BF16, SDPA, shared vision, greedy decoding, official starter-kit system prompt, official chunk-aligned dialog.
- Inference strips `answers` before sampling, prompting, model calls, and decision-feature extraction.

Only `max_frames`, `frames_per_interval`, `max_history_turns`, and `max_new_tokens` may vary.

## Frozen Samples

Seed `20260720` assigns within-domain session-length quartiles without reading labels. Search and confirmation each contain 80 non-overlapping sessions: 20 per domain and 5 per domain/quartile. Full evaluation uses all 700 public-validation sessions.

## Variants And Selection

The baseline is `(32,16,4,64)`. The predefined search contains the 3 x 3 visual grid, five history windows, and four generation limits, deduplicated to 16 configurations. Search independently chooses the best visual, history, and generation setting and composes a joint configuration. An existing tuple is recorded as an alias; otherwise it becomes the seventeenth unique configuration.

All unique configurations run on confirmation. Confirmation Macro F1 selects two non-baseline candidates. Full evaluation runs only baseline plus those two candidates.

Ranking uses official Macro F1, then official G-mean F1, then lower total model inference GPU time, then stable variant ID. A configuration with any session above 300 seconds of measured model inference cannot be named deployable best.

## Reporting Boundary

The final claim is limited to "D4.1 public-validation best input policy." Public-validation selection is val-supervised and is not hidden-test or independent generalization evidence. Promotion into `configs/d4_*`, `submission/d4_small/manifest.json`, or the frozen head requires separate authorization and a GPU equivalence smoke.

The live official rules page could not be fetched during implementation on 2026-07-20. The local starter-kit snapshot remains the executable source for the 300-second inference-only limit and official scorer. Rules must be checked again before any submission.
