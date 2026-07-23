# D6 Low-Dimensional Structured Calibration v1 Preregistration

Frozen on 2026-07-21 before any D6 model fit, threshold result, or metric inspection.

## Objective

Test whether the organizer-visible recent action pattern can improve the exact D4
interrupt/silent decision through a small, deployable threshold policy rather than
another high-dimensional feature fusion.

D6 is a CPU-only, public-validation-supervised leaderboard-engineering study. It
does not change the InternVL backbone, D4 feature matrix, D4 linear-head training,
frame policy, dialog window, or utterance generation. It is not hidden-test evidence.

## Motivation and D5 Boundary

D5 found a positive but split-sensitive action-history signal. Its primary combined
2,100 input columns and contained three exact duplicate-column pairs, so part of the
gain could reflect changed effective regularization rather than new information.

D6 freezes D5 v1 and does not delete columns from or rerun it. D6 instead uses only:

1. the scalar logit from an exactly reproduced D4 fold head;
2. one mutually exclusive causal stage name derived from chunk position and at most
   the two most recent organizer-visible prior actions.

No D3 hidden delta or D5 continuous/window feature enters a D6 calibration policy.

## Fixed Sources and Information Order

Use the same frozen 700-session public-development input, R0 label-free records,
D1 session folds, neural cache, D4 matrix definition, and official scorer as D5.

The required order is:

```text
source rows
  -> strip_answers()
  -> build frozen D4 label-free matrix and causal stage names
  -> attach public labels
  -> fit D4 on fit folds
  -> select D4 L2 and global threshold on the calibration fold
  -> derive structured thresholds on that same calibration fold
  -> evaluate once on the untouched test fold
```

Feature and stage construction must reject source rows containing `answers`. It may
not read frozen predictions, future dialog, future chunks, model-generated history,
human ratings, or external data.

## Exact D4 Reference

For every outer test fold, reproduce D4 exactly:

- three fit folds, one calibration fold, one test fold;
- the frozen D4 1,051-column matrix;
- the frozen L2 grid, optimizer, seed policy, and exact global-threshold selection;
- no change to standardization or class balancing.

`d4_global_replay` must match the frozen D4 prediction and metric hashes before any
D6 candidate is interpreted. All structured variants inherit the D4-selected model
and L2 for that fold; they may change only the threshold applied to its logits.

## Frozen Stage Families

### Position control

`first`, `second`, `2-4`, `5-9`, `10+` from the causal zero-based chunk index.

### Last-action control

`first`, `previous_interrupt`, `previous_silent`. The previous action is recovered
only from the assistant-count increment already visible in the current official
dialog prefix.

### Last-two primary family

- `first` for chunk 0;
- `second` for chunk 1, regardless of its single prior action;
- `ii`, `is`, `si`, or `ss` thereafter, from the two most recent visible prior
  actions in chronological order.

These groups are mutually exclusive. The `first` group always uses the exact global
D4 threshold and is never locally calibrated.

## Frozen Threshold Rule

For each eligible non-first group on the calibration fold:

1. require at least 64 rows and both gold classes;
2. select the local exact Macro-F1 threshold using the same deterministic tie break
   as D4;
3. for a shrunk policy, compute

```text
effective_n = 2 * min(group_interrupt_count, group_silent_count)
weight = effective_n / (effective_n + 256)
group_threshold = global_threshold
                  + weight * (local_threshold - global_threshold)
```

An ineligible group falls back to the global threshold. The constants 64 and 256
are fixed here and must not be searched. Test-fold labels never affect group
eligibility, thresholds, weights, or fallback.

## Frozen Variants

Run in this order:

1. `d4_global_replay`;
2. `position_shrunk`;
3. `last_action_shrunk`;
4. `last2_unshrunk`;
5. `last2_shrunk`.

`last2_shrunk` is the sole preregistered primary. Controls explain whether any gain
comes from coarse position, only the immediately previous action, or unregularized
group thresholds. A control cannot replace the primary after results are known.

## Stability Splits

After the frozen-fold run, rerun only `d4_global_replay` and `last2_shrunk` under
three new label-independent, domain-stratified session assignments:

```text
d6-stability-20260721-a
d6-stability-20260721-b
d6-stability-20260721-c
```

Each split independently refits D4, selects its global threshold, and derives D6
group thresholds using the same fit/calibration/test rotation. These are robustness
checks on the same public data, not independent test sets.

## Primary Promotion Gate

The primary becomes eligible for a deployment-threshold transport audit only if all
conditions hold:

1. official OOF Macro F1 gain over exact D4 is at least `0.005`;
2. paired session-bootstrap 95% lower bound is strictly positive;
3. at least four of five fixed test folds have positive Macro gain;
4. at least three of four domains have positive Macro gain;
5. non-first-chunk Macro improves;
6. interrupt and silent F1 are each at least `0.67`;
7. all three stability-split primary-minus-D4 gains are positive.

If any condition fails, freeze D6 v1 without deployment integration. Do not search
new stage bins, minimum counts, shrinkage constants, thresholds, L2 values, or model
families on the same protocol.

## Required Audits

- exact D4 OOF prediction and metric hashes;
- answer-stripped stage construction and prefix invariance to future dialog changes;
- exact offline/online last-two stage replay;
- mutually exclusive stage coverage for all 9,935 chunks;
- first-stage threshold equals the D4 global threshold in every fold;
- structured variants reuse the exact D4-selected model and L2;
- complete prediction order/schema validation and official scoring;
- fold/domain/position diagnostics and paired session bootstrap;
- three new stability splits;
- no GPU, human evaluation, external data, leaderboard upload, or Docker upload.

## Post-Gate Action

A passing primary does not immediately replace D4. It permits one separately audited
transport step: add the median of the five frozen per-stage offsets relative to each
fold's global threshold onto the existing final D4 threshold, then verify full-data
offline/online decisions, adapter integration, CPU preflight, and one GPU smoke.
Any full-data score is training sanity only.
