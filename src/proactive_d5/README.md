# D5 Decision Fusion

D5 is a CPU-only, public-validation-supervised decision experiment. It exactly
replays D4, then tests the preregistered union of D4 dialog stage, non-duplicate
D3 dynamics, and answer-blind causal prior-action history. It does not run model
inference or use human ratings.

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_d5.run \
  --config configs/d5_internvl35_1b_decision_fusion_oof_v1.json
```

Run unit tests:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m unittest discover -s src/proactive_d5/tests -v
```

The frozen protocol is
[`annotations/d5_decision_fusion_v1/PROTOCOL.md`](../../annotations/d5_decision_fusion_v1/PROTOCOL.md).
