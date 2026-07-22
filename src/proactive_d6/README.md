# D6 Structured Calibration

D6 keeps the exact D4 fold model and changes only the threshold applied to its
logit. The preregistered primary uses a six-state summary of causal prior actions
with calibration-fold thresholds shrunk toward the global D4 threshold.

Run tests:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m unittest discover -s src/proactive_d6/tests -v
```

Run the frozen CPU-only OOF and stability study:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d6.run \
  --config configs/d6_internvl35_1b_structured_calibration_oof_v1.json
```

Protocol:
[`annotations/d6_structured_calibration_v1/PROTOCOL.md`](../../annotations/d6_structured_calibration_v1/PROTOCOL.md).

Result:
[`reports/20260721_internvl35_1b_structured_calibration_d6.md`](../../reports/20260721_internvl35_1b_structured_calibration_d6.md).
