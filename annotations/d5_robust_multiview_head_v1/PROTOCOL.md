# D5 robust multiview linear-head protocol

Status: re-frozen before the user-directed D4-fold rerun on 2026-07-22. The views,
equal weights, learner, and gates are unchanged from the 2026-07-21 protocol.

## Question

Can one fixed 1,052-parameter standardized linear head retain the clean
history8 score on the D4.2 session folds while becoming materially less sensitive to
three causal input perturbations? This is a post-selection public-validation
robustness audit, not hidden-test or independent-generalization evidence.

## Frozen views

All views use the same pinned InternVL3.5-1B model, official system prompt, BF16
greedy shared-vision inference, `max_frames=32`, `frames_per_interval=16`,
`max_new_tokens=64`, and 1,051-feature D1-fused-plus-dialog-stage schema.

1. `clean_history8`: official answer-stripped dialog, eight retained history
   turns, and `uniform_cumulative_v1` sampling.
2. `history4`: official answer-stripped dialog, four retained history turns, and
   `uniform_cumulative_v1` sampling. The completed hash-pinned D4.2 cache is
   reused without modification.
3. `assistant_drop`: delete every assistant turn from every causal dialog prefix
   before inference, retain the initial user query, use history8 and uniform
   cumulative sampling. The transformation must reject source rows containing
   gold answers.
4. `frame_jitter`: retain official answer-stripped dialog and history8, but when
   more than 32 frames have been observed select the midpoint of each of the 32
   uniform stride bins: `int((index + 0.5) * observed_frames / 32)`. Prefixes of
   at most 32 frames remain unchanged. The policy is deterministic and receives
   no future timestamp, label, or prediction.

Every newly extracted feature shard is answer-free. Complete coverage, source
order, chunk order, frame policy, hidden width, finite values, and hashes are
validated before labels are attached.

## Frozen OOF training

Use the exact D4.2 `domain_stratified_sha256_round_robin` manifest with five outer
folds, seed `d1-session-oof-v1`, and calibration offset one. The standard comparator fits on `clean_history8`
only. For each test rotation it chooses L2 from
`{1e-5,1e-4,1e-3,1e-2}` and an exact clean calibration-fold Macro-F1 threshold,
then applies that unchanged model and threshold to every view.

The sole robust candidate is
`equal_mix_clean_history4_assistant_drop_frame_jitter_v1`. For each rotation,
concatenate the four views for the three fit folds, repeating each chunk label
exactly once per view. Fit one standardized class-balanced float64 logistic head
with the same LBFGS implementation, `max_iter=120`, sum-reduced L2 grid, and base
seed 20260714. Comparator fits use
`seed + test_fold * 100 + grid_index`; robust fits use
`seed + 10000 + test_fold * 100 + grid_index`. Choose L2 and one threshold by
exact Macro-F1 on the equally
concatenated four-view calibration data. Apply the unchanged head and threshold
to every test view. There is no view weighting, feature change, extra parameter,
threshold per view, or hyperparameter search.

The comparator's clean predictions must exactly reproduce the frozen D4.2
history8 prediction SHA256. Failure invalidates the experiment.

## Static gate and self-fed early stop

The candidate is eligible for self-fed inference only if:

- clean Macro-F1 changes by at least `-0.002` versus the clean-only comparator;
- candidate minus comparator Macro-F1 is at least `+0.010` separately on
  `history4`, `assistant_drop`, and `frame_jitter`;
- neither candidate nor comparator collapses to one class on any view.

If any static check fails, stop before self-fed GPU inference. Since final
robustness uses the minimum improvement across all perturbations, an already
failed static view cannot be rescued by self-fed performance.

If eligible, self-fed evaluation removes official assistant turns and processes
each held-out session sequentially with the corresponding frozen OOF head. A
predicted `$silent$` adds no assistant turn; a predicted `$interrupt$` appends
that chunk's causal generated utterance for later chunks. The query, video
prefix, model, sampling, and decoding remain frozen. Self-fed promotion requires
at least `+0.010` Macro-F1 over the clean-trained comparator under the identical
self-fed protocol, in addition to every static gate.

Passing all gates permits a separate all-development robust-head refit and exact
GPU equivalence smoke; it does not alter either existing D4 submission bundle or
authorize external upload. Failure ends this robustness head family without
post-hoc view weights, thresholds, L2 values, perturbation strengths, or model
variants on these folds.
