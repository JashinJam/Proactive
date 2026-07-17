# C1 Small D1: Session-Held-Out Decision Calibration

D1 separates the ranked interrupt/silent decision from free-form generation. The
first stage is a no-GPU control using only causal temporal metadata and frozen R0
response properties. The completed neural stage adds forced-tag likelihood
margins and frozen multimodal hidden states. The promoted fused OOF reference is
Macro F1 `0.6341`; this is public-validation development evidence.

The split is deterministic, domain-stratified, and session-level. For outer test
fold `f`, fold `(f + 1) % 5` calibrates the decision threshold and the remaining
three folds fit the linear model. Every session receives exactly one out-of-fold
prediction. No chunks from one session cross folds.

All trained or threshold-tuned D1 results use public validation labels and are
therefore `val-supervised`, not held-out test evidence.

## Commands

Audit source fingerprints, frozen references, fold assignment, and label-free
feature construction:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.run_scalar \
  --config configs/d1_internvl35_1b_scalar_oof.json \
  --output-dir /tmp/d1_scalar_audit \
  --audit-only
```

Run the complete CPU out-of-fold control:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.run_scalar \
  --config configs/d1_internvl35_1b_scalar_oof.json \
  --output-dir output/experiments/20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2
```

After the OOF feature set and threshold policy are frozen, fit and serialize one
full-development deployment head:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.finalize_scalar \
  --config configs/d1_internvl35_1b_scalar_final.json
```

The resulting full-fit metric is only a training sanity check. The OOF metric
remains the generalization estimate within public development data.

`proactive_d1.deploy.process_session_with_scalar_head` is the online serving
path. It applies the serialized head inside the per-session loop where domain,
current interval, prior interval end, visible dialog, and current frame count
are available. The unmodified starter `model.generate(frames, messages)`
signature alone does not expose enough metadata for this policy.

Audit or smoke the complete online runner with
`proactive_d1.run_deploy`. It removes `answers` before generation, loads the
frozen R0 model and serialized head, applies the head inside the causal session
loop, and emits the unchanged official prediction schema.

Extract frozen causal hidden states and forced-tag score margins. Start with a
bounded smoke run before launching a full cache:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.extract_neural \
  --config configs/d1_internvl35_1b_neural_features.json \
  --device cuda:0 --max-sessions 1 --max-chunks-per-session 2 \
  --output-dir output/features/20260714_internvl35_1b_d1_neural_feature_cache_v1_smoke1
```

Feature extraction removes source answers before building messages. Saved
hidden states are taken at the final causal prompt token before either forced
tag, and the two candidate-prefix states are checked for equality.

After all four complete shards finish, validate and consolidate them:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.merge_neural
```

The merge rejects partial, duplicated, missing, non-finite, misordered, or
hash-mismatched session caches and does not read or store source answers.

After pinning the merged `features.npz` SHA256 in the neural OOF config, run the
tag-only, scalar+tag, hidden-linear, and fused-linear controls:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.run_neural \
  --config configs/d1_internvl35_1b_neural_oof.json
```

Promotion is measured against the frozen `response_temporal` OOF control, not
against R0 or R0-F.

Build domain/fold/position, error-change, tag-margin AUC, and regularization
diagnostics from frozen neural OOF predictions:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.analyze_neural \
  --experiment-dir output/experiments/20260714_internvl35_1b_neural_decision_head_d1_oof_v1
```

Fit the single deployable fused head with L2 and threshold policies frozen from
the clean five-fold OOF run:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.finalize_neural \
  --config configs/d1_internvl35_1b_neural_final.json
```

The sequential online fused path remains the correctness oracle. The promoted
`shared_vision` path reuses one projected video representation across the two
original batch-one language passes; it is byte-equivalent on 127 chunks and
reduces measured wall time by 9.15%. Run the deployed mode with:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.run_deploy \
  --config configs/d1_internvl35_1b_neural_deploy_shared_vision.json \
  --output-dir output/experiments/d1_shared_vision_smoke \
  --device cuda:0 --max-sessions 1
```

Verify online raw responses, tag margins, logits, decisions, and answers against
the frozen offline cache and final-head records:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.verify_deploy \
  --deploy-dir output/experiments/d1_shared_vision_smoke \
  --final-dir output/experiments/20260715_internvl35_1b_neural_decision_head_d1_final_v1 \
  --r0-dir output/experiments/20260713_internvl35_1b_no_plan_r0 \
  --cache output/features/20260714_internvl35_1b_d1_neural_feature_cache_v1/features.npz
```

For a bounded equivalence audit, add `--record-hidden-state` to `run_deploy`
and `--require-hidden-state` to `verify_deploy`. Do not record hidden vectors in
a full submission run. Compare latency, memory, predictions, and official
metrics against a sequential run with:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.compare_deploy_modes \
  --reference-dir output/experiments/d1_sequential_benchmark \
  --candidate-dir output/experiments/d1_shared_vision_benchmark
```

`batched` and `prefix_cache` remain implemented only as rejected, auditable
controls. They must not replace `shared_vision` without a new equivalence and
performance result.

Audit transport of the final head's single median threshold across the five
frozen OOF models:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d1.audit_threshold \
  --config configs/d1_internvl35_1b_threshold_robustness.json
```

The completed audit first reproduces the original OOF predictions byte for
byte. The single threshold obtains official Macro F1 `0.6330` versus the
fold-specific `0.6341` and passes all frozen deployment-robustness checks. This
is a public-validation deployment diagnostic; its post-hoc threshold sweep must
not be used to replace the serialized final threshold.

## Tests

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python -m unittest discover \
  -s src/proactive_d1/tests -v
```
