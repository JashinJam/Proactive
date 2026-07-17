# C1 Small R0: InternVL3.5-1B No-Plan Baseline

This component freezes the first active-route experiment: a causal, zero-shot,
no-plan baseline on the public EgoProactive validation set.

## Fixed Interface

- The model receives the official system prompt, query, and at most four dialog
  turns available before the current decision.
- Frames are extracted from absolute source intervals with the official starter
  kit function. The current call contains only intervals at or before the
  current chunk.
- Cumulative frames use the starter kit's uniform stride and a 32-frame cap.
- The frames are passed to InternVL as one ordered video, one visual token block
  per sampled frame. Video frames are explicitly resized to 448x448 to match the
  checkpoint vision tower; the bundled 384x384 video processor config produces
  an invalid 27x27 patch grid with the checkpoint's 0.5 pixel shuffle.
- Generation is greedy. No plan, threshold, class prior, or validation label is
  used during generation.
- GPU occupancy is checked through NVML before model loading. Existing compute
  processes produce a warning; pass `--require-exclusive-gpu` when a run must
  hard-fail instead of sharing a lightly occupied device.
- Malformed model responses follow the official scorer's silent fallback, then
  are canonicalized to the submission schema. Raw responses are retained in
  `session_records.jsonl`.

## Model Qualification

The pinned `OpenGVLab/InternVL3_5-1B-HF` snapshot has 1,060,897,792 unique
parameters across all stored BF16 tensors. Its Apache-2.0 model card and exact
weight hash are recorded in `models/internvl35_1b_hf.json` and the experiment
config. No auxiliary learned component is used.

## Commands

Run static eligibility and data checks without loading the model:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_r0.run \
  --config configs/r0_internvl35_1b_no_plan.json \
  --output-dir /tmp/r0_audit \
  --audit-only
```

Run a one-session smoke test only after the selected GPU is fully idle:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_r0.run \
  --config configs/r0_internvl35_1b_no_plan.json \
  --output-dir output/experiments/20260713_internvl35_1b_no_plan_smoke \
  --device cuda:0 \
  --max-sessions 1
```

Run the full 700-session baseline by omitting `--max-sessions` and
`--output-dir`. Add `--resume` after an interrupted run. The runner stores one
durable record after every completed session and materializes the official
prediction file only from that ordered prefix.

For session-level data parallelism, run contiguous shards with matching
`--num-shards N --shard-index K`, then invoke `python -m proactive_r0.merge`
with every completed shard directory. The merger rejects missing, duplicate, or
out-of-order global input indices before calling the official full scorer.

## Tests

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python -m unittest discover \
  -s src/proactive_r0/tests -v
```
