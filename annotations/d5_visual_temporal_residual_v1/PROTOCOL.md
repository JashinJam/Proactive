# D5 visual temporal residual protocol

Status: re-frozen before the user-directed D4-fold rerun on 2026-07-22; the
architecture, optimizer, and gates are unchanged.

This experiment adds one strictly causal visual temporal candidate. For every
chunk, the frozen InternVL vision tower processes the frames selected by
`causal_multiscale_16_8_8_v1`. The projected 1,024-dimensional pooler outputs of
the current interval's selected frames are averaged and L2-normalized. No future
interval, generated answer, or label is available to this feature extractor.

Within each D4.2 session-level OOF rotation, the frozen D5 history8 feature
schema first fits its class-balanced linear base on the three fit folds and
selects L2 from `{1e-5,1e-4,1e-3,1e-2}` on the calibration fold. A single learned
`Linear(1024,32)` projection and one-layer `GRU(input=32,hidden=32)` consume each
session in chronological chunk order. A `Linear(32,1)` output is added to the
detached base logit. The temporal module has exactly 39,073 parameters; no bidir-
ectional state, attention, width, layer, or pooling alternative is permitted.

Temporal optimization is fixed to AdamW, learning rate `3e-4`, weight decay
`1e-2`, class-balanced BCE, at most 100 epochs, calibration-loss patience 10,
gradient norm clipping 1.0, and seed 20260721. The best calibration-loss state is
restored, then the exact calibration-fold Macro-F1 threshold is selected once.
The calibration reuse is explicitly val-supervised and is not independent model
selection evidence.

Promotion requires at least +0.005 Macro-F1 over the D4-fold history8 baseline,
paired-session bootstrap 95% lower bound above zero, positive changes in at least
four of five folds and three of four domains, non-declining previous-interrupt and
previous-silent strata, and predictions in both classes. Failure ends this model
family. Total parameters must remain below 2B. Evidence remains post-selection
public-validation evidence, not hidden-test or independent-generalization proof.
