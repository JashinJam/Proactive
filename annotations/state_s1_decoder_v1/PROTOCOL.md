# S1 Lightweight Procedural-State Decoder Protocol

## Objective

Train a small decoder on frozen causal D3 features to predict explicit state:

```text
current_step: s1 | s2 | s3 | s4
progress: not_started | ongoing | complete | deviated | recovered
error_present: false | true
```

S0 showed insufficient zero-shot decoding and must not be prompt-tuned further.
S1 tests whether explicit supervision can recover state from representations that
already support D3 causal dynamics.

## New Data Split

Select 32 sessions without reading answers, R0/D1/D3 outputs, errors, ratings, or
state labels. Exclude the 20 U1 formal sessions and four historical R1 sessions.

Per domain:

- split remaining sessions into short/middle/long thirds by chunk count;
- select three short, three middle, and two long sessions by frozen hash seed;
- assign two held-out sessions per domain using a frozen rotated band schedule,
  yielding three short, three middle, and two long held-out sessions overall;
- assign the remaining six sessions to state training.

This yields 24 training and 8 held-out sessions, all domain-balanced. Every
chunk is annotated in sequence; the held-out annotations are not visible during
state-probe model/L2 selection. The old U1 formal 80 states are secondary
state-label-aware cross-check only.

The first label-free preparation attempt assigned all held-out sessions to
short/middle bands. It was rejected before any annotation and retained as an
engineering manifest. The formal preparation uses the rotated schedule above.

## Annotation Causality

1. Write exactly four macro steps from task/query before video inspection.
2. Process chunks in increasing order.
3. At chunk `i`, inspect only `video_intervals_so_far` and `prior_dialog` supplied
   for that chunk; never inspect interval gaps or future chunks first.
4. Never read current/future answers, model predictions, decision errors,
   ratings, or existing oracle state files.
5. `current_step_id` points to the active, just-completed-not-advanced, or
   recovery-needed macro step.
6. `confidence` is annotator confidence. It is reported diagnostically and is
   not a regression target.

The four-step plan must be completed and hashed for every selected session
before any S1 video frame or dialog beyond the query is inspected. Static plan
steps are deliberately coarse: each step must describe a distinct procedural
phase, cover the whole requested task, and avoid encoding timestamps, observed
objects, model behavior, or likely intervention points.

Use the following frozen dynamic-state semantics:

- `not_started`: no task action toward the current macro step is yet visible;
- `ongoing`: the current step has started but its completion cues are not yet
  satisfied;
- `complete`: the current step has just met its completion cues and no action
  from the next step is yet visible;
- `deviated`: the visible action is off-plan, incorrect, blocked, or abandons
  the current task step;
- `recovered`: the current chunk visibly resumes the plan after the immediately
  preceding annotated state was `deviated`.

Advance `current_step_id` only when the next macro step visibly begins. A
dialog instruction alone does not prove that the user executed it. A time gap
between candidate intervals does not authorize inspecting the omitted video;
the first later visible state may nevertheless be farther ahead if the visible
evidence establishes it. `next_step_id` is the following static step, or JSON
`null` for `s4`.

`incompletion_or_error_evidence` records concrete visible evidence that the
current step is incomplete, incorrect, blocked, or off-plan. It is not a C1
interrupt label. The frozen binary `error_present` target is derived as whether
this list is non-empty. Therefore `complete` requires an empty list, while
`not_started`, `ongoing`, and `deviated` require at least one item. `recovered`
may remain incomplete and normally retains an item. Every state must contain
at least one completion or incompletion/error evidence item and a confidence in
`[0, 1]`.

The annotation schema retains evidence/recovery text for auditing and later
utterance studies, but the first S1 model predicts only step, progress, and a
binary non-empty incomplete/error flag.

## Frozen Decoder Variants

All heads are standardized linear models with class-balanced cross entropy.
L2 is selected only by five-fold training-session CV from:

```text
[1e-4, 1e-3, 1e-2, 1e-1, 1.0]
```

For each feature variant, one shared L2 value is selected for all three heads.
For every candidate L2, compute step, progress, and error Macro F1 from pooled
out-of-fold predictions across the five training-session folds, average the
three values, and select the largest mean. Ties within `1e-12` choose the
stronger regularization (larger L2). Standardization statistics, class weights,
and model parameters are fit only on the four training folds. After selection,
refit the three heads on all 24 training sessions and evaluate the eight
held-out sessions exactly once.

Variants:

1. `temporal_only`: seven D1 causal temporal features; diagnostic control.
2. `current_d1`: D1 1,043-dimensional current-chunk fused representation;
   diagnostic control.
3. `d3_dynamics`: D3 2,075-dimensional current plus causal dynamics; only
   promotion-eligible state decoder.

No nonlinear layer, recurrence, PCA, feature search, class merge, transition
smoothing, or threshold search is allowed in S1 v1.

## Frozen Annotation Quality Control

Before any decoder training, validate exact session/chunk coverage, source
identity, four-step IDs, timestamps, provenance, state values, evidence rules,
and absence of target markers. Freeze separate SHA256 manifests for train and
held-out annotations.

Review all eight held-out sessions in a second sequential pass. Also re-annotate
four training sessions selected label-independently before the first pass, one
per domain. Report agreement before adjudication:

```text
selection seed: 20260718-state-s1-qc-v1
selection key:  SHA256(seed + NUL + domain + NUL + video_path), smallest per domain

Arts and Crafts: input 33  / 0b4dd0cf02c47a4c.mp4
Chef:            input 95  / 2590c2cacdf4a6c2.mp4
Handyman:        input 20  / 0573fc312656e95a.mp4
Tutorial:        input 592 / d9c5c309f8bc0a0d.mp4
```

```text
step exact agreement                  >= 0.75
step within-one agreement             >= 0.95
progress exact agreement              >= 0.65
error-present Cohen kappa              >= 0.50
```

If any threshold fails, adjudicate the affected sessions, document the changed
rows, and repeat the full validator. This quality-control gate concerns label
reliability only and must not inspect decoder predictions or held-out metrics.
The held-out labels remain inaccessible to decoder/CV code until the selected
train-only models and their artifact hashes are frozen.

## Metrics And Gate

Evaluate once on the eight new held-out sessions:

- step/progress/error accuracy and Macro F1;
- step ordinal MAE;
- joint step-progress accuracy;
- per-state composite correctness;
- transition diagnostics and per-domain results;
- paired session bootstrap against `temporal_only`, 10,000 repetitions, seed
  `20260717`.

The `d3_dynamics` decoder passes S1 only if:

```text
mean task Macro F1                           >= 0.50
composite correctness delta vs temporal     >= 0.05
paired-session bootstrap delta CI95 low     > 0
positive composite delta domains            >= 3/4
step Macro F1                               >= 0.40
progress Macro F1                           >= 0.35
```

Run the frozen decoder on the old formal 80 only after the new held-out result.
This secondary check cannot rescue a failed primary gate.

## Downstream Rule

Only a passing S1 decoder may enter `D3 + predicted-state posterior` decision
OOF or predicted-state utterance generation. S1 itself cannot promote a
submission model.
