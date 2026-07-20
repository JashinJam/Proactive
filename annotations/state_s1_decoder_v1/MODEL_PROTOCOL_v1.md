# S1 v1 Frozen Linear-Decoder Implementation

This addendum freezes implementation details that were not needed for the
query-only plan freeze. It is frozen before any decoder is trained and before
held-out labels exist.

## Train-Only Session CV

Assign the 24 training sessions to five folds independently of state labels:

```text
seed: 20260718-state-s1-cv-v1
key: SHA256(seed + NUL + domain + NUL + video_path)
algorithm: sort within each domain by key, then assign rank modulo five
```

The generated `cv_split_manifest.json` is the source of truth. No state label,
model output, D1/D3 error, answer, U1 rating, or held-out annotation is used to
create it.

## Features

- `temporal_only`: the seven frozen D1 temporal scalars in their registered
  order;
- `current_d1`: 18 response-temporal scalars, the frozen tag margin, and the
  1,024-dimensional current hidden state (1,043 total);
- `d3_dynamics`: `current_d1` plus eight causal dynamics scalars and the
  1,024-dimensional current-minus-previous hidden delta (2,075 total).

The feature cache and dynamics construction must exactly reuse the frozen D1
and D3 code and SHA256-pinned cache. No target is read during feature creation.

## Classifiers And L2

Fit one standardized `sklearn.linear_model.LogisticRegression` head per target
with `solver=lbfgs`, `class_weight=balanced`, `fit_intercept=true`,
`max_iter=2000`, `tol=1e-8`, and `C=1/L2`. Use scikit-learn `1.7.2`, NumPy
`1.24.4`, and SciPy `1.15.3` from the existing `proassist` environment.

For a fold, standardization mean and scale are computed from its training
sessions only; constant scales are replaced by one. A head may only emit
classes observed in that fold. Macro F1 is nevertheless computed over the full
frozen target vocabulary, so an unlearned class receives zero rather than being
silently removed from the metric.

For each feature variant and L2 candidate, pool all five out-of-fold
predictions, compute full-vocabulary Macro F1 for step/progress/error, and
average the three. Select the largest mean; ties within `1e-12` select larger
L2. Refit all three heads using that shared L2 on all 24 train sessions.

Training accepts a train-annotation path only. Held-out evaluation is a separate
command that requires the frozen model package and its SHA256 before accepting
a held-out-annotation path.
