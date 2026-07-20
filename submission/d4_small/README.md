# D4 Small Submission Adapter

This directory defines the model-facing contract that will be placed behind the
official test Docker template after it is released. It does not guess the final
container base image, mount paths, command, time limit, or network policy.

The adapter accepts an organizer input JSONL with chunk-aligned `dialog`, reads
videos only through the supplied candidate intervals, runs the frozen D4 model,
and atomically writes the official two-field prediction JSONL.

```bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=/opt/wearable_ai/src

python -m proactive_d4.submission \
  --input-jsonl /input/egoproactive_test.jsonl \
  --video-dir /input/videos \
  --model-dir /opt/models/InternVL3_5-1B-HF \
  --head-path /opt/wearable_ai/submission/d4_small/decision_head.json \
  --starter-kit-dir /opt/wearable_ai/starter_kit \
  --output-jsonl /output/predictions.jsonl \
  --work-dir /tmp/d4_runtime \
  --device cuda:0
```

The default hidden-test contract rejects any input row containing `answers`.
`--allow-input-answers-for-local-audit` exists only for a public-validation
preflight or smoke; the shared runner removes that field before inference.

## Local Collaboration Check

The bundled head is the only learned D4 artifact not provided by the InternVL
snapshot. Verify it after pulling:

```bash
sha256sum submission/d4_small/decision_head.json
# 531431710a01a71bdd02ffd7a9758428fe282323cc41fae2c1d6859e45408b13
```

With local model, starter-kit, and public data paths already available, run a
CPU-only contract check without passing `--head-path`; the adapter will use the
bundled head by default:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python -m proactive_d4.submission \
  --input-jsonl /path/to/wearable_ai_2026_egoproactive_val_700.jsonl \
  --video-dir /path/to/egoproactive/val \
  --model-dir /path/to/InternVL3_5-1B-HF \
  --starter-kit-dir /path/to/starter_kit \
  --work-dir /tmp/d4_handoff_preflight \
  --preflight-only \
  --allow-input-answers-for-local-audit \
  --max-sessions 1
```

The local-audit flag is required only because the public validation JSONL
contains `answers`. Omit it for organizer hidden input.

## Bundle Inputs

- project `src/proactive_r0`, `src/proactive_d1`, `src/proactive_d3`, and
  `src/proactive_d4`;
- official `starter_kit/model.py` and `run_generate_proactive.py` at the hashes
  pinned by the frozen D4 config;
- the pinned InternVL3.5-1B-HF snapshot;
- bundled D4 `decision_head.json` from this directory, with SHA256
  `531431710a01a71bdd02ffd7a9758428fe282323cc41fae2c1d6859e45408b13`;
- `manifest.json` and `requirements-tested.txt` from this directory.

## Remaining Template Work

After the organizer releases the official template, map its input/output paths
to this CLI, apply its resource and health-check requirements, build without
network access at runtime, and rerun the frozen one-session equivalence smoke.

Prize-source eligibility is not yet marked complete: the backbone is
Apache-2.0, but this project currently has no top-level source-code license.
Selecting that license is an owner decision and is intentionally not made by
the adapter.
