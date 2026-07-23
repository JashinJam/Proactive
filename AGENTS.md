# Agent Instructions

Read and follow [Agent.md](Agent.md) before inspecting, editing, training, evaluating, or delegating work in this project. `Agent.md` is the canonical instruction file; this file exists for tools that auto-discover `AGENTS.md`.

## Current D4.2/D4.3 Experiment Status

- D4.2 completed on 2026-07-21 as a post-D4.1, val-supervised input-policy adaptation audit over all 700 public-validation sessions / 9,935 chunks. It is not hidden-test or independent-generalization evidence.
- The frozen model-side configuration is InternVL3.5-1B, official prompt/dialog, BF16 greedy shared-vision inference, and the 1,051-feature D1-fused-plus-dialog-stage schema. Four policies were evaluated: baseline `(max_frames=32, frames_per_interval=16, max_history_turns=4, max_new_tokens=64)`, `history8=(32,16,8,64)`, `frames16=(16,16,4,64)`, and `tokens16=(32,16,4,16)`.
- Each policy receives a fresh five-fold session OOF 1,052-parameter standardized class-balanced float64 linear head: three folds fit, one calibrates, one tests, with L2 grid `{1e-5,1e-4,1e-3,1e-2}` and exact calibration-fold Macro-F1 threshold selection. Experiment config SHA256 is `71b88e99482a9d80bfd401f34604c7df5ab34b0aea723919c33e6fbf8caee453`.
- OOF Macro/G-mean results are `history8 0.6988/0.6988`, `frames16 0.6854/0.6854`, baseline `0.6846/0.6846`, and `tokens16 0.6844/0.6843`. `history8` improves Macro by `+0.0142`; its 5,000-repetition paired-session interval is `[+0.008166,+0.020363]`, with 5/5 positive folds and 4/4 positive domains.
- The all-development `history8` train-fit uses L2 `0.01` and median-fold threshold `0.12101525136349107`. Its Macro `0.7469` is training sanity only. Head SHA256 is `dab9eaf100ea301055ab4d68856d406fb5927864bc96c71f2038688067b904c5`.
- D4.3 completed the authorized four-domain, 102-chunk GPU equivalence smoke: all discrete fields match, hidden/tag differences are zero, maximum logit difference is `1.22e-7`, peak allocated memory is 3.48 GB, and maximum session wall time is 112.70 seconds. `submission/d4_2_history8_small` is now the independent active leaderboard-engineering candidate. The original D4 config, head, and `submission/d4_small` remain frozen and unchanged; do not claim hidden-test improvement or upload externally without separate authorization. See [CURRENT_ROUTE.md](CURRENT_ROUTE.md) and [Agent.md](Agent.md) for the full interpretation and active gates.

## Current D5 Experiment Status

- On 2026-07-22 the exact-query-grouped protocol was withdrawn. All active D5 evaluation now uses the exact D4.2 five-fold session manifest with algorithm `domain_stratified_sha256_round_robin`, seed `d1-session-oof-v1`, three fit folds, one calibration fold, and one test fold. Do not regenerate, regroup, or substitute this manifest for D5 comparisons.
- The D5 `history8` baseline exactly reproduces frozen D4.2 predictions and metrics at Macro/G-mean `0.6988/0.6988`; predictions SHA256 is `d154789b8f41583558878e93b9bb618643a5f64d1ad5b397d84cfd592e31c121`.
- Re-run D5 results are: causal multiscale `0.6988` (`+0.0000`), selected dual-view `0.6846` (`-0.0142`), visual-temporal residual `0.6983` (`-0.0005`), and robust multiview clean `0.6918` (`-0.0070`). Every candidate fails its frozen gate; self-fed robustness inference was not run.
- End these D5 families without post-hoc variants on the same folds. Old grouped-fold outputs may remain as historical experiment artifacts but are not an active protocol, baseline, or promotion source. D4.2 `history8` remains the active leaderboard-engineering candidate; do not claim hidden-test improvement or upload externally without separate authorization.
