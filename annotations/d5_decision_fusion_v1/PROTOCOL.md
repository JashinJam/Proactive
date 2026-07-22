# D5 Decision Fusion v1 Preregistration

Frozen on 2026-07-21 before any D5 model fit or metric inspection.

## Objective

Maximize the official C1 Small interrupt/silent Macro F1 with no additional
human evaluation. The experiment tests one preregistered primary: whether the
union of frozen D4 dialog-stage signal, frozen D3 causal dynamics, and a compact
causal prior-action history improves on the exact D4 OOF reference.

This is public-validation-supervised leaderboard engineering. It is not hidden
test evidence and does not retroactively change the scientific status of D3 or
the diagnostic status of D3-D/D4.

## Fixed Sources

- the 700-session official public-development JSONL;
- the frozen label-free R0 session records;
- the frozen label-independent D1 five-fold session manifest;
- the frozen 9,935-row D1 neural cache;
- the frozen D3 and D4 OOF predictions and metrics;
- the unmodified official scorer.

No external data, new model inference, human rating, state annotation, future
video, future dialog, current/future answers, or model-generated history is
allowed. Source `answers` are removed before every feature builder and are
attached only after the matrices and folds are frozen.

## Frozen Feature Blocks

### D4 base

Use the exact 1,051-column D1-fused-plus-eight-dialog-stage matrix from D3-D.
The `d4_replay` variant must reproduce the frozen D4 decisions, prediction hash,
and metric hash before any new variant is accepted.

### D3 dynamics

Recompute the frozen D3 dynamics from the D1 cache. When appended to D4, omit
only `has_previous_chunk` from the D3 scalar block because the identical column
already exists in D4. The remaining block contains seven dynamics scalars and,
for the full variant, the 1,024-dimensional current-minus-previous hidden delta.

### Causal prior-action history

At chunk `i`, compare the assistant-turn count in `dialog[i]` with the previous
visible prefix. This reveals only the organizer-provided action after chunk
`i-1`. Update the history with that previous action before deriving features for
chunk `i`. Never append the current model prediction.

The 18 frozen columns are:

```text
action_lag2_interrupt
action_lag2_available
action_lag3_interrupt
action_lag3_available
action_lag4_interrupt
action_lag4_available
action_interrupt_rate_last2
action_interrupt_rate_last4
action_interrupt_rate_last8
action_log1p_consecutive_interrupts
action_log1p_consecutive_silents
action_log1p_chunks_since_silent
action_last2_ii
action_last2_is
action_last2_si
action_last2_ss
action_transition_rate_last4
action_transition_rate_last8
```

Rates use all available actions up to the named window and are zero when no
action is available. Last-two indicators are all zero until two prior actions
exist. `chunks_since_silent` is the number of visible prior actions since the
last silent action, or the full visible history length when no silent action has
occurred. Run lengths and chunks-since values use `log1p`.

## Frozen Variants

Run in this order:

1. `d4_replay`;
2. `d4_plus_dynamic_scalar`;
3. `d4_plus_full_dynamics`;
4. `d4_plus_action_history`;
5. `d4_plus_full_dynamics_history`.

The fifth variant is the sole preregistered primary. The other four are controls.
A positive control may motivate a separately frozen follow-up but cannot replace
the primary post hoc.

## Fixed Training

- five session-level folds from the frozen label-independent manifest;
- for each rotation: three fit folds, the next fold for L2/threshold calibration,
  and one untouched test fold;
- class-balanced float64 linear logistic regression with LBFGS;
- L2 grid `[1e-5, 1e-4, 1e-3, 1e-2]`, sum reduction;
- exact calibration-fold Macro F1 threshold selection;
- seed `20260714`, maximum 120 iterations;
- no new L2, threshold, feature, window, or model-capacity search after results.

## Stability Check

After the frozen-fold run, rerun only `d4_replay` and the primary under three
label-independent, domain-stratified five-fold session assignments:

```text
d5-stability-20260721-a
d5-stability-20260721-b
d5-stability-20260721-c
```

Each assignment retrains both models with the same fit/calibration/test rotation.
It is a robustness diagnostic on the same 700 sessions, not independent data.

## Primary Promotion Gate

The primary replaces D4 only if all conditions hold on the frozen split:

1. official OOF Macro F1 gain over exact D4 is at least `0.005`;
2. paired session-bootstrap 95% lower bound is positive;
3. at least four of five test folds have positive Macro gain;
4. at least three of four domains have positive Macro gain;
5. non-first-chunk Macro improves;
6. interrupt and silent F1 are each at least `0.67`;
7. all three stability splits have positive primary-minus-D4 Macro gain.

If the frozen gain is positive but below `0.005`, retain D5 as a secondary
candidate only. If it is non-positive or unstable, stop this feature family.

## Required Automatic Audits

- reject target-bearing feature inputs;
- prefix invariance to future dialog rows;
- exact offline/online action-history replay;
- exact action-history lag-1 agreement with D4's assistant-addition feature;
- exact D4 OOF replay before new variants;
- complete prediction order/schema validation and official scoring;
- fold/domain/position diagnostics and session bootstrap;
- no GPU use in OOF/stability runs;
- no leaderboard, Docker, or registry upload without user authorization.

## Post-Gate Action

Only a passing primary is eligible for a single all-development refit using the
median frozen-fold L2 and threshold policy, followed by complete offline/online
replay, submission-adapter integration, CPU preflight, and one-session GPU smoke.
The full-fit score is training sanity only.
