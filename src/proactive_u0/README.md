# U0: Frozen D1 Utterance Audit

This component performs a deterministic, read-only audit of the promoted D1
OOF answers and prepares a separate blind human-review package. It does not run
the model, train parameters, change D1 decisions, or recompute a replacement
competition metric.

Run:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u0.run \
  --config configs/u0_d1_utterance_audit.json \
  --output-dir output/experiments/20260716_internvl35_1b_d1_utterance_u0_v1
```

Test:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python -m unittest discover \
  -s src/proactive_u0/tests -v
```
