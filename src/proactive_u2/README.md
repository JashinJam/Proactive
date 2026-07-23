# U2: Fixed-D4 Early Utterance Grounding

U2 keeps the D4 interrupt decisions fixed and compares six paired content views
on the complete review-informed early-chunk intersection frozen by the protocol.
It does not train a model, tune the gate, or search frame/history settings.

Prepare the sanitized sample:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u2.prepare \
  --config configs/u2_internvl35_1b_d4_early_grounding_prepare_v1.json
```

Run all paired views with one model load:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u2.run \
  --config configs/u2_internvl35_1b_d4_early_grounding_v1.json \
  --output-dir output/experiments/20260720_internvl35_1b_d4_early_grounding_u2_v1 \
  --device cuda:0 --require-exclusive-gpu
```

Automatic output differences are mechanism diagnostics only. Grounding and
hallucination comparisons remain pending until the generated blind package is
rated under the explicit current-visual-support rubric.

The completed automatic run and its non-promotion decision are documented in
[`reports/20260720_internvl35_1b_d4_early_grounding_u2.md`](../../reports/20260720_internvl35_1b_d4_early_grounding_u2.md).
