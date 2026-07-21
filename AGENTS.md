# Agent Instructions

Read and follow [Agent.md](Agent.md) before inspecting, editing, training, evaluating, or delegating work in this project. `Agent.md` is the canonical instruction file; this file exists for tools that auto-discover `AGENTS.md`.

## Current D4.2 Experiment Status

- D4.2 completed on 2026-07-21 as a post-D4.1, val-supervised input-policy adaptation audit over all 700 public-validation sessions / 9,935 chunks. It is not hidden-test or independent-generalization evidence.
- The frozen model-side configuration is InternVL3.5-1B, official prompt/dialog, BF16 greedy shared-vision inference, and the 1,051-feature D1-fused-plus-dialog-stage schema. Four policies were evaluated: baseline `(max_frames=32, frames_per_interval=16, max_history_turns=4, max_new_tokens=64)`, `history8=(32,16,8,64)`, `frames16=(16,16,4,64)`, and `tokens16=(32,16,4,16)`.
- Each policy receives a fresh five-fold session OOF 1,052-parameter standardized class-balanced float64 linear head: three folds fit, one calibrates, one tests, with L2 grid `{1e-5,1e-4,1e-3,1e-2}` and exact calibration-fold Macro-F1 threshold selection. Experiment config SHA256 is `71b88e99482a9d80bfd401f34604c7df5ab34b0aea723919c33e6fbf8caee453`.
- OOF Macro/G-mean results are `history8 0.6988/0.6988`, `frames16 0.6854/0.6854`, baseline `0.6846/0.6846`, and `tokens16 0.6844/0.6843`. `history8` improves Macro by `+0.0142`; its 5,000-repetition paired-session interval is `[+0.008166,+0.020363]`, with 5/5 positive folds and 4/4 positive domains.
- The all-development `history8` train-fit uses L2 `0.01` and median-fold threshold `0.12101525136349107`. Its Macro `0.7469` is training sanity only. Head SHA256 is `dab9eaf100ea301055ab4d68856d406fb5927864bc96c71f2038688067b904c5`.
- The original D4 config, head, and submission remain frozen and unchanged. Do not promote D4.2, overwrite `submission/d4_small`, or claim deployment/hidden-test improvement without separate user authorization and an exact GPU equivalence smoke. See [CURRENT_ROUTE.md](CURRENT_ROUTE.md) and [Agent.md](Agent.md) for the full interpretation and active gates.
