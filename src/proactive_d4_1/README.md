# D4.1: Frozen D4 Inference-Input Policy Search

D4.1 is a val-supervised public-validation audit of four causal inference limits:

```text
max_frames
frames_per_interval
max_history_turns
max_new_tokens
```

The InternVL3.5-1B backbone, 1,052-parameter D4 head, threshold `0.1263874797442615`, BF16, shared-vision feature extraction, greedy decoding, official system prompt, and official dialog remain frozen. The runner strips `answers` before inference and calls `proactive_d4.deploy.process_session_with_dialog_stage_head` for every session.

## Run

From an existing tmux window:

```bash
bash scripts/run_d4_1_input_policy_search.sh
```

The default launcher uses `/home/quewenjun/miniconda3/envs/wearable_ai/bin/python`, requires four GPUs with no existing compute processes, and writes to:

```text
output/experiments/20260720_internvl35_1b_d4_1_input_policy_search_v1/
```

Common overrides:

```bash
GPU_IDS=1,2,3,4 NUM_GPUS=4 bash scripts/run_d4_1_input_policy_search.sh
DRY_RUN=1 bash scripts/run_d4_1_input_policy_search.sh
NUM_SHARDS=4 MAX_TASK_ATTEMPTS=3 bash scripts/run_d4_1_input_policy_search.sh
SMOKE_ONLY=1 NUM_GPUS=1 GPU_IDS=4 bash scripts/run_d4_1_input_policy_search.sh
```

Keep `NUM_SHARDS` unchanged when resuming an existing experiment. `GPU_IDS` may change between invocations, provided every requested GPU is idle.

`SMOKE_ONLY=1` loads the frozen model once per task and runs the baseline plus one non-baseline configuration on the same label-blind short session. It writes `gpu_smoke.json`, verifies legal outputs and complete timing, and rejects GPUs with preexisting compute processes. It does not select or promote a policy.

## Resume Contract

Every stage/variant/shard owns a task hash, source-ordered session index list, `status.json`, `run.log`, environment/code snapshots, fsynced `session_records.jsonl`, atomic `predictions.jsonl`, and runtime summary. Completed tasks are skipped. A valid prefix resumes at the next session. Failed or terminated tasks retry without overwriting completed records. Any changed effective config, sample manifest, stage plan, task hash, frozen artifact hash, or shard order is rejected.

## Comparison

Rebuild all completed-stage artifacts independently with:

```bash
PYTHONPATH=src python -m proactive_d4_1.compare --experiment-dir <path>
```

`compare` loads the pinned official starter-kit scorer for Macro/G-mean and class metrics, then adds decision changes, domain/task/position/length-quartile strata, 5,000 paired session bootstraps, timing distributions, GPU-seconds, peak memory, and the 300-second model-inference limit check.

The final claim is restricted to "D4.1 public-validation best input policy." `best_inference.json` does not modify D4 configs, the submission manifest, or the bundled head. Promotion requires separate authorization and an equivalence GPU smoke.

The estimated formal workload is 80--90 GPU-hours, or roughly 20--30 hours of wall time with four GPUs. Actual time depends on session lengths and hardware.
