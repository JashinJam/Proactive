# D4.2 History8 Small Candidate

This is the independent, GPU-verified history8 candidate. It does not overwrite
the frozen D4 bundle and is not an external submission or hidden-test result.

Use the existing adapter with all candidate paths explicit:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python -m proactive_d4.submission \
  --config configs/d4_2_internvl35_1b_history8_deploy_shared_vision_v1.json \
  --manifest submission/d4_2_history8_small/manifest.json \
  --head-path submission/d4_2_history8_small/decision_head.json \
  --input-jsonl /input/egoproactive_test.jsonl \
  --video-dir /input/videos \
  --model-dir /opt/models/InternVL3_5-1B-HF \
  --starter-kit-dir starter_kit \
  --output-jsonl /output/predictions.jsonl \
  --work-dir /tmp/d42_history8_runtime \
  --device cuda:0
```

The input must omit `answers` and provide chunk-aligned cumulative official
`dialog`. The model uses only causal video through the current interval. The
source-code license and official Docker template remain unresolved external
gates; no upload is authorized by this bundle.
