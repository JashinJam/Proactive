# U1: Fixed-D1-Gate Forced Generation

The U1 runner replays the frozen R0 input on a label-independent sample, then
continues a real assistant-side `$interrupt$` token prefix. It only replaces
utterance content in the frozen D1 full prediction file and requires official
decision metrics to remain byte-equivalent at the metric level.

Engineering smoke:

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u1.run \
  --config configs/u1_fixed_gate_forced_generation.json \
  --output-dir output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_smoke1 \
  --device cuda:0 --variants forced_no_state --smoke-only \
  --require-exclusive-gpu
```

Frozen two-reviewer analysis (run only after both rating slots are complete):

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u1.ratings \
  --ratings <completed-ratings.csv> \
  --blind output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full/analysis/paired_review_blind.jsonl \
  --key output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full/analysis/paired_review_key.jsonl \
  --samples annotations/u1_forced_generation_v1/sample_items.jsonl \
  --output <ratings-analysis.json>
```

Multiple `--ratings` arguments are allowed when the two reviewers use separate
CSV copies. Fully blank template rows are ignored, but every candidate must have
one populated row for reviewer slots `A` and `B`. The frozen analysis averages
reviewers, bootstraps the 20 sessions with seed `20260717` for 10,000 resamples,
and defines unsafe as `safety_1_5 <= 2`.

Formal blind oracle shards are merged only through the strict validator:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u1.finalize_oracle \
  --sample-part annotations/u1_forced_generation_v1/formal_blind/part_a_samples.jsonl \
  --sample-part annotations/u1_forced_generation_v1/formal_blind/part_b_samples.jsonl \
  --annotation-part annotations/u1_forced_generation_v1/formal_blind/oracle_states.formal_blind.part_a.json \
  --annotation-part annotations/u1_forced_generation_v1/formal_blind/oracle_states.formal_blind.part_b.json \
  --output annotations/u1_forced_generation_v1/oracle_states.formal_blind.json \
  --manifest annotations/u1_forced_generation_v1/formal_blind/manifest.json
```

After full oracle generation, build the separately randomized three-way state
review package from the frozen no-state content and the new oracle content:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u1.state_review \
  --samples annotations/u1_forced_generation_v1/sample_items.jsonl \
  --content-records output/experiments/20260716_internvl35_1b_fixed_gate_forced_generation_u1_v1_nostate_full/content_records.jsonl \
  --content-records <full-oracle-output>/content_records.jsonl \
  --output-dir <full-oracle-output>/state_review
```

This package contains 240 candidates: no-state, oracle-step, and oracle-full for
all 80 samples. It does not reuse the interface-package no-state ratings.

After both state-package reviewers finish, analyze it with:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u1.state_ratings \
  --ratings <completed-state-ratings.csv> \
  --blind <state-review-dir>/state_review_blind.jsonl \
  --key <state-review-dir>/state_review_key.jsonl \
  --samples annotations/u1_forced_generation_v1/sample_items.jsonl \
  --output <state-ratings-analysis.json>
```

Automatic state diagnostics are explicitly non-semantic:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u1.oracle_diagnostics \
  --samples annotations/u1_forced_generation_v1/sample_items.jsonl \
  --content-records <no-state-output>/content_records.jsonl \
  --content-records <full-oracle-output>/content_records.jsonl \
  --output <full-oracle-output>/analysis/state_content_diagnostics.json
```
