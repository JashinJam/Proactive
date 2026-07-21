# D4.2 Adapted Input-Policy OOF Protocol

Date frozen: 2026-07-21

D4.2 tests whether a small set of mechanism-motivated inference policies benefits
after the complete D4 dialog-stage decision head is retrained for that policy. It
does not modify the InternVL3.5-1B backbone, official prompt/dialog semantics,
greedy decoding, BF16 precision, feature family, fold manifest, or scorer.

The four frozen policies are `(max_frames, frames_per_interval,
max_history_turns, max_new_tokens)`:

- `baseline`: `(32, 16, 4, 64)`; exact D4 OOF replay control.
- `history8`: `(32, 16, 8, 64)`; test whether extra assistant history becomes
  useful after the scalar/neural head is adapted to its changed distribution.
- `frames16`: `(16, 16, 4, 64)`; test compact cumulative visual context as a
  regularizer and lower-compute policy.
- `tokens16`: `(32, 16, 4, 16)`; test whether a short decision-oriented decode
  reduces response noise and latency.

`history16` is excluded because it duplicates the history mechanism and was no
better than `history8` in the completed D4.1 full audit. No joint configuration,
additional window, frame rate, token cap, feature, L2 value, or threshold may be
added after D4.2 results are visible.

For every policy, cover all 700 public-development sessions with answers removed
before model-facing inference and record source-ordered raw responses,
tag-sequence scores, 1,024-dimensional causal hidden states, timing, and exact
input parameters. To minimize redundant GPU forwards, baseline and `history8`
reuse their hash-pinned D4.1 full-run raw generations while recomputing the
decision feature forward; every recomputed prompt-token count and tag score must
exactly match the D4.1 reference. `tokens16` runs the 16-token generation but
reuses the D4.2 baseline hidden/tag features because its frames and messages are
identical and `max_new_tokens` is not an input to feature extraction. It must
match baseline interval, frame, prompt, hidden, and tag identity exactly. Timing
combines the actually run component with the same-session cached component, and
experimental wall time is reported separately. `frames16` runs both forwards.
The baseline OOF must exactly reproduce the pinned D1/D3-D predictions, official
metrics, and hashes before candidates are ranked.

Use the frozen five-fold domain-stratified session manifest. For each policy and
test fold, use three folds for class-balanced float64 logistic-regression fitting,
one distinct fold for L2/threshold calibration, and one fold for testing. Search
only `L2 in {1e-5, 1e-4, 1e-3, 1e-2}` with sum reduction. Select L2 by calibration
Macro F1 and select the exact calibration Macro-F1 threshold. Merge the five test
folds in original session/chunk order and score with the pinned official scorer.

Rank policies by official OOF Macro F1, then G-mean F1, lower total model
inference seconds, and stable policy ID. Report both class metrics, confusion
counts, predicted interrupt rate, fold/domain/chunk-position results, decision
changes, and 5,000 paired-session bootstrap intervals against baseline.

After OOF ranking, fit exactly one all-700 head for the winning policy. Use the
median of its five selected L2 values and the median of its five calibration
thresholds; do not reselect either value from full-fit predictions. Serialize the
head and report its official full-fit score only as training sanity. Do not mutate
the frozen D4 head, D4 configs, or submission package.

This candidate set was chosen after inspecting D4.1 public-validation results.
Consequently, D4.2 is a post-selection, val-supervised mechanism audit: its fold
tests are held out from each head fit, but the overall comparison is not independent
of public-validation candidate selection and is not hidden-test evidence.
