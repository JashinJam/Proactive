# C1 Small D2: Decision Representation Experiments

## Rejected Residual MLP Control

D2 keeps the InternVL3.5-1B backbone, frozen causal neural cache, feature order,
session folds, response content, and official scorer from D1. The only new
capacity is a width-8 GELU residual MLP added to the exact per-fold D1 linear
logit. Its output layer starts at zero, so epoch zero reproduces the D1 base.

The rotating calibration fold controls class-balanced BCE early stopping and
then selects the final decision threshold. The complete test fold remains
unused until predictions are frozen. This is public-validation-supervised OOF
evidence, not a hidden-test result.

The completed preregistered run obtains official Macro F1 `0.6351`, only
`+0.0010` over D1 `0.6341`. Its paired-session bootstrap interval crosses zero,
and only three of five folds improve strictly, so it does not pass promotion.
D1 remains the scientific and deployment baseline; do not refit or deploy this
MLP as the current submission model.

Run the preregistered CPU experiment:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d2.run_mlp \
  --config configs/d2_internvl35_1b_residual_mlp_oof.json
```

Run tests:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python -m unittest discover \
  -s src/proactive_d2/tests -v
```

## Rejected Final-Language-MLP LoRA OOF

The completed frozen experiment targets only layer 27 language-MLP
`gate_proj`, `up_proj`, and `down_proj` with rank-8 LoRA. This adds 98,304
trainable parameters. Utterance generation keeps the adapter disabled; the
adapter is used only by the binary decision path.

The two bounded feasibility smokes remain historical failures. A later
four-state full-cache attempt also exposed final-RMSNorm batch-shape drift at
input 11/chunk 4. The formal cache therefore stores reference/local MLP output
and reference/local final hidden in addition to residual/normalized input.
Training computes adapter-enabled minus adapter-disabled deltas under the same
fixed batch-64 shape for the MLP, final norm, and LM-head margin:

```text
reference_mlp + same_batch_adapted_mlp - same_batch_disabled_mlp
reference_hidden + same_batch_adapted_norm - same_batch_base_norm
cached_margin + same_batch_adapted_margin - same_batch_base_margin
```

The six-state cache completed over 700 sessions / 9,935 chunks and matches the
fixed D1 hidden/margin exactly with exact prompt/key order. The single frozen
five-fold OOF then obtained official Macro F1 `0.6357`, only `+0.0016` over D1
`0.6341`; its session-bootstrap interval crosses zero and only 2/5 folds
improve. It is rejected. Do not full-refit or post-hoc tune another LoRA
variant on these folds.

Run the exact fixed-shape replay smoke on an extracted session:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
/home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d2.verify_final_mlp_replay \
  --session-cache /path/to/session_0011.npz \
  --device cuda:0
```

See `reports/20260715_internvl35_1b_final_mlp_lora_feasibility.md` for the
historical engineering audit and
`reports/20260716_internvl35_1b_final_mlp_lora_oof.md` for the formal cache,
OOF metrics, promotion decision, and route consequence.
