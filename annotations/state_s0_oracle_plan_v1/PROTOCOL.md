# S0 Oracle-Plan Structured State Decoding Protocol

## Question

Can the frozen Small backbone decode the current procedural state from the
causal video/query/dialog prefix when it is given a query-only four-step oracle
plan?

This is an oracle-plan, predicted-dynamic-state feasibility study. It is not a
deployable planner, a hidden-test claim, or a decision-head promotion study.

## Evaluation Isolation

- The 20-session / 80-state U1 formal oracle remains evaluation-only.
- The prediction runner reads only `inputs.jsonl`, which contains the query-only
  static plan but no current step, progress, evidence, recovery, confidence, or
  current/future answer.
- `targets.jsonl` is read only by the evaluator after prediction artifacts are
  complete and hashed.
- Existing U1 ratings, review keys, current outputs, R0/D1 errors, and gold
  interrupt/silent answers are never read.
- The experimenter has previously inspected the oracle schema, examples, and
  aggregate target distribution. Therefore this is frozen-before-prediction,
  state-label-aware protocol design, not a never-seen-label benchmark.

## Causal Inputs

At each selected chunk, the model may use only:

- task and query;
- the four-step static plan written from task/query before video inspection;
- official dialog available before the current chunk;
- frames from the explicit `video_intervals_so_far`, capped by the frozen R0
  `max_frames=32` policy.

Two views are frozen:

1. `official_dialog`: the complete inference-visible dialog prefix.
2. `no_assistant_history`: remove prior assistant turns while retaining the
   query and any non-assistant turns; this diagnoses dialog-policy dependence.

## Fixed Candidate Scoring

No free-form state JSON is generated. The frozen model teacher-forces option
digits and scores their complete autoregressive log probability:

```text
step:     1=s1, 2=s2, 3=s3, 4=s4
progress: 1=not_started, 2=ongoing, 3=complete,
          4=deviated, 5=recovered
error:    1=absent, 2=present
```

Every candidate group must tokenize into distinct, equal-length sequences and
round-trip exactly. The visual tower is evaluated once per state; candidate
language passes remain batch-one. Ties are resolved by the configured option
order. No threshold, candidate wording, prompt, decoding, or smoothing search
is allowed after predictions are visible.

## Metrics

Report for each view:

- step accuracy, Macro F1, and ordinal MAE;
- progress accuracy and Macro F1;
- error-present accuracy and Macro F1;
- joint step-plus-progress accuracy;
- mean of the three task Macro F1 values;
- per-domain and per-position composite correctness;
- confidence/entropy diagnostics from the candidate softmax.

Compare the two views with a paired 10,000-repetition session bootstrap over
per-state composite correctness, seed `20260717`.

## Frozen Interpretation Bands

The mean of step/progress/error Macro F1 is classified as:

```text
strong zero-shot signal       >= 0.45
weak but usable signal        >= 0.35 and < 0.45
insufficient zero-shot signal < 0.35
```

These bands decide only how S1 should be initialized. S0 cannot promote a
submission model. A poor S0 result does not disprove state utility because it
may reflect a zero-shot interface failure.

