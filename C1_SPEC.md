# C1 EgoProactive Task Specification

> Local snapshot: 2026-07-13  
> Scope: stable task, data, metric, and submission facts only  
> Live rules: <https://wearable-ai-workshop.github.io/challenge_rules.html>

Competition rules can change. Re-check the live page before a submission; do not update this file from memory.

## 1. Task

Given a first-person procedural video stream, one user query, and the dialog available so far, produce one decision for each candidate interval:

```text
$silent$
```

or:

```text
$interrupt$<utterance>
```

The ranked C1 decision metric evaluates whether to interrupt, not the semantic quality of the utterance.

## 2. Public Validation Data

Local canonical input:

```text
data/egoproactive/wearable_ai_2026_egoproactive_val_700.jsonl
```

The public set contains:

- 700 session rows;
- 9,935 candidate intervals in total;
- four high-level domains;
- one video per session row.

Top-level fields:

```text
video_path
duration_in_sec
video_intervals
query
domain
task
answers
dialog
```

Alignment constraints:

```text
len(video_intervals) == len(answers) == len(dialog)
```

`video_intervals[i]` contains absolute timestamps `[start, end]`. Most intervals are approximately 8 seconds, but the first/last interval can be shorter and gaps can exist. Never replace absolute time with cumulative interval duration.

`dialog[i]` is the history available before decision `i`. Consecutive silent chunks may have identical dialog history.

The public schema does **not** provide plan state, current step, completion cues, OOP labels, or recovery-plan targets.

## 3. Causal Constraint

For interval `i`, the model may use:

- the user query;
- dialog available before `i`;
- past intervals;
- frames from the current interval up to its end timestamp;
- internal state derived causally from the above.

It may not use:

- frames after interval `i`;
- later dialog or gold decisions;
- a full-session caption or feature computed from future frames;
- future step boundaries or future plan updates;
- labels from another interval to construct the current input.

Any preprocessing shared across time must be proven causal at the feature level, not only at the final model call.

## 4. Prediction Schema

One output row per input session, in exactly the input order:

```json
{
  "video_path": "example.mp4",
  "answers": [
    "$interrupt$Start with the first step.",
    "$silent$"
  ]
}
```

Requirements:

- exactly 700 rows for the full public validation set;
- `video_path` matches the corresponding input row;
- `len(prediction.answers) == len(input.video_intervals)`;
- each answer starts with exactly `$interrupt$` or is exactly `$silent$`;
- rows and answers remain in source order.

The official parser classifies a response as interrupt only when its left-stripped text starts with `$interrupt$`; all other responses are scored as silent. This fallback is not permission to emit malformed output.

## 5. Metrics

Let interrupt be the positive class:

```text
Interrupt F1 = harmonic_mean(interrupt precision, interrupt recall)
Silent F1    = harmonic_mean(silent precision, silent recall)

Macro F1  = (Interrupt F1 + Silent F1) / 2
G-mean F1 = sqrt(Interrupt F1 * Silent F1)
```

Macro F1 is the primary local/leaderboard metric. G-mean is diagnostic and penalizes class imbalance more strongly.

Every result report must include both class F1s, TP/FP/TN/FN, and predicted interrupt rate. A single Macro F1 value is insufficient to diagnose collapse.

## 6. Official Evaluation

Canonical local implementation:

```text
data/starter_kit/run_evaluation.py
```

From `data/starter_kit/`, evaluate an existing prediction file with the official entry point and the appropriate prediction-path arguments:

```bash
python run_evaluation.py --task proactive --eval-only
```

Check `python run_evaluation.py --help` and the current starter-kit README for exact path flags. Do not maintain a second metric implementation as the source of reported scores.

Before reporting a full result, verify:

- the scorer processed all 700 sessions and 9,935 decisions;
- no skipped or truncated rows;
- no shard was mistaken for a full prediction file;
- source order was preserved;
- the metrics JSON and prediction SHA256 were saved.

## 7. Small Division and Eligibility

The current rules define Small as at most 2B total parameters. Treat all inference-time components as part of the submitted system and sum their total parameters unless the live rules explicitly state otherwise. Count all MoE parameters, not only active parameters.

Prize eligibility requires open weights under acceptable terms. Closed APIs can be useful as offline research teachers only when competition rules and licenses permit; they are not a deployable Small submission.

## 8. Local Evaluation Policy

The public validation labels are visible. Therefore:

- tuning on all 700 rows measures public-set fit, not held-out generalization;
- any model, prompt, threshold, plan, or cue derived from these labels must be marked `val-supervised`;
- use session-level splits for internal development;
- never split individual intervals from one session across train and validation;
- report both split metrics and full public-set leaderboard metrics when applicable.

## 9. Authoritative Local References

- [Starter-kit README](data/starter_kit/README.md)
- [Official scorer](data/starter_kit/run_evaluation.py)
- [Official proactive tests](data/starter_kit/tests/test_run_evaluation_proactive.py)
- [Leaderboard mirror](wearable-ai-leaderboard/README.md)
- [PWR audit](literature/papers/challenge1_proactive/PWR_audit.md)

