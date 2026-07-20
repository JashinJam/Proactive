# D3: Frozen-Cache Causal Dynamics

D3 tests whether label-free cross-chunk changes add stable session-held-out
decision signal beyond D1 `fused_linear=0.6341`. It does not rerun the backbone
or consume U1 ratings.

The frozen dynamic scalars are:

```text
has_previous_chunk
tag_margin_delta_previous
tag_margin_abs_delta_previous
tag_margin_delta_history_mean
hidden_cosine_previous
hidden_delta_rms_previous
hidden_cosine_history_mean
hidden_delta_rms_history_mean
```

The high-dimensional feature is `current_hidden - previous_hidden`. First-chunk
dynamics are all zero. Historical means exclude the current chunk. No future
row, label, full-session length, or later dialog is read.

Only `dynamics_fused` is promotion-eligible. The D1 replay, scalar-only, and
hidden-delta-only variants are diagnostic controls and cannot replace the
primary after results are visible.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d3.run \
  --config configs/d3_internvl35_1b_causal_dynamics_oof.json \
  --output-dir <output-dir> --audit-only
```

Remove `--audit-only` for the single formal CPU OOF run.

After promotion, fit one serialized all-development head with the frozen median
OOF L2 and threshold:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d3.finalize \
  --config configs/d3_internvl35_1b_causal_dynamics_final.json
```

Its full-fit score is training sanity only; it does not replace the OOF result.

Verify that a session-local online state reproduces every offline dynamic
feature, final-head logit, and decision over all cached chunks:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d3.verify \
  --final-dir output/experiments/20260717_internvl35_1b_causal_dynamics_d3_final_v1 \
  --cache output/features/20260714_internvl35_1b_d1_neural_feature_cache_v1/features.npz
```

Freeze the post-hoc interpretation audit without retraining or tuning:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d3.analyze \
  --oof-dir output/experiments/20260717_internvl35_1b_causal_dynamics_d3_oof_v1
```

The promoted head can run through the existing D1 deployment CLI with
`configs/d3_internvl35_1b_causal_dynamics_deploy_shared_vision.json`. A GPU
smoke recorded with `--record-hidden-state` can be checked against every frozen
source feature and final-head output:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d3.verify_deploy \
  --deployment-dir <deployment-dir> \
  --final-dir output/experiments/20260717_internvl35_1b_causal_dynamics_d3_final_v1 \
  --cache output/features/20260714_internvl35_1b_d1_neural_feature_cache_v1/features.npz
```
