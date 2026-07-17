# R1 Oracle Compact-State Annotation Protocol v1

## Purpose

This pilot tests whether compact procedural state can help the frozen R0 model. It is an evaluation-only oracle upper bound, not a deployable state generator and not a held-out result.

The first pilot contains four sessions, one per public domain, and 50 chunks. The sessions are selected by a frozen domain-stratified SHA256 rule that does not inspect labels or R0 errors. The sample is intentionally too small for a population-level claim; a useful signal must be confirmed on a larger pre-registered subset.

## Information Boundary

Static plans may use only `task` and `query`. Do not inspect the video, dialog, or answers while authoring the plan. Generic procedural knowledge is allowed.

For chunk `i`, dynamic state may use only:

- `task` and `query`;
- `dialog[i]`, which is already inference-visible;
- video frames at timestamps no later than `video_intervals[i][1]`.

The annotator must not inspect:

- `answers`, including the current answer;
- `dialog[j]` for any `j > i`;
- video frames later than the current interval end;
- R0 predictions or error categories.

Annotate chunks in chronological order. A state may be carried forward by retaining the previous `last_update_chunk`; if any dynamic field changes, set `last_update_chunk` to the current chunk.

## State Semantics

- `current_step_id`: the step the user is currently expected to execute.
- `progress`: `not_started`, `ongoing`, `complete`, `deviated`, or `recovered`.
- `completion_evidence`: concrete visual evidence already observed by the interval end.
- `incompletion_or_error_evidence`: concrete missing, incorrect, or off-plan evidence already observed.
- `next_step_id`: next planned step, or `null` when the procedure is complete.
- `confidence`: confidence in the state interpretation, from 0 to 1.
- `last_update_chunk`: most recent chunk at which any dynamic state field changed.

Static `completion_cues` and `incompletion_cues` describe what would distinguish the step states. They must not encode whether the assistant should speak.

## Forbidden Target Leakage

Annotations must never contain `$interrupt$`, `$silent$`, a gold label, a gold utterance, or an instruction such as "should interrupt". The schema validator rejects these markers and any causal timestamp beyond the interval end.

## Controlled Variants

All variants use the frozen R0 model, frames, history, decoding, canonicalization, and scorer.

| Variant | State fields exposed |
|---|---|
| `r0_frozen` | None; predictions are extracted from the frozen full R0 artifact |
| `null` | Identical state wrapper with `status: unavailable` |
| `step` | Current step text only |
| `cues` | Current step plus static completion/incompletion cues |
| `full` | Cues plus goal, progress, observed evidence, next step, confidence, and update index |

`null` is the prompt-scaffold control. Plan-state effects are primarily measured against `null`; `r0_frozen` reveals whether the wrapper itself changes behavior.

## Decision Gate

The pilot can validate the protocol and reveal gross effects, but cannot pass the R1 scientific gate alone. Before building a state predictor, repeat any promising result on a larger pre-registered, session-level subset and report bootstrap uncertainty, class metrics, interrupt rate, malformed count, first-chunk recall, and per-domain behavior.

