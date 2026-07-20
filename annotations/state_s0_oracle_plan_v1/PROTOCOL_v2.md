# S0 Oracle-Plan Structured State Decoding Protocol v2

## Revision Reason

The single-state v1 engineering smoke was target-isolated, but all three tasks
showed a strong monotonic preference for lower option digits. Equal token length
does not remove candidate-token priors. No formal run or target evaluation was
performed under v1.

V2 freezes content-free contextual calibration before formal predictions. For
each session and target, score the same option mapping using only the query and
query-only oracle plan, with no video and no prior dialog. Every observed state
uses:

```text
calibrated_score(option) = observed_logp(option) - content_free_logp(option)
```

The calibration is computed once per session, reads no dynamic state target,
and is shared by both dialog views. Raw, content-free, and calibrated scores are
all recorded. No temperature, scale, alternative mapping, permutation, prompt,
or post-prediction calibration search is allowed.

## Question

Can the frozen Small backbone decode the current procedural state from the
causal video/query/dialog prefix when it is given a query-only four-step oracle
plan?

This is an oracle-plan, predicted-dynamic-state feasibility study. It is not a
deployable planner, a hidden-test claim, or a decision-head promotion study.

## Evaluation Isolation

- The 20-session / 80-state U1 formal oracle remains evaluation-only.
- The prediction runner reads only `inputs.jsonl`, containing the static plan
  but no current step, progress, evidence, recovery, confidence, or answer.
- `targets.jsonl` is read only after both prediction views are complete and
  hashed.
- Existing U1 ratings, review keys, current outputs, R0/D1 errors, and gold
  interrupt/silent answers are never read.
- The experimenter has inspected the oracle schema/examples/aggregate target
  distribution. This is state-label-aware protocol design frozen before formal
  predictions, not a never-seen-label benchmark.

## Causal Inputs And Views

At each state, use only task/query, the query-only four-step static plan,
explicit `video_intervals_so_far`, and dialog available before the chunk.
Frames follow R0 (`16/interval`, cumulative cap `32`).

1. `official_dialog`: complete inference-visible dialog prefix.
2. `no_assistant_history`: remove prior assistant turns while retaining query
   and any other turns.

The content-free calibration view contains only query + static plan.

## Fixed Candidate Scoring

```text
step:     1=s1, 2=s2, 3=s3, 4=s4
progress: 1=not_started, 2=ongoing, 3=complete,
          4=deviated, 5=recovered
error:    1=absent, 2=present
```

Candidates must be distinct equal-length round-tripping token sequences. The
visual tower runs once per observed state; all language candidates remain
batch-one. Ties use configured option order. There are 11 observed language
passes per state and 11 content-free passes per session.

## Metrics And Frozen Bands

Report step accuracy/Macro F1/ordinal MAE, progress accuracy/Macro F1,
error-present accuracy/Macro F1, joint step-progress accuracy, mean task Macro
F1, domain/position composite correctness, confidence/entropy, and a paired
10,000-repetition session bootstrap between dialog views (seed `20260717`).

```text
strong zero-shot signal       mean task Macro F1 >= 0.45
weak but usable signal        mean task Macro F1 >= 0.35 and < 0.45
insufficient zero-shot signal mean task Macro F1 < 0.35
```

S0 only selects the initialization strategy for S1 and cannot promote a
submission model. Poor S0 does not disprove state utility.

