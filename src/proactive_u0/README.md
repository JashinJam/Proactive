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

## Two-Reviewer Ratings Analysis

The U0 A/B aggregation has a dedicated validator because silent U0 items leave
all content fields blank and are not U1 variant pairs. Run the frozen analysis:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src \
  /home/lanjinxin/miniconda3/envs/wearable_ai/bin/python \
  -m proactive_u0.analyze_ratings \
  --config configs/u0_dual_reviewer_analysis_v1.json \
  --output-dir output/experiments/20260720_u0_dual_reviewer_analysis_v1
```

The analysis preserves both original reviewer rows, reports session-bootstrap
intervals and agreement, and writes a separate adjudication list. It does not
replace ratings with reconciled values or estimate full-validation prevalence.
