# D5 causal multiscale frame sampling protocol

Status: re-frozen before the user-directed D4-fold rerun on 2026-07-22; the
sampler, learner, and gates are unchanged from the 2026-07-21 protocol.

This experiment changes only the causal selection of at most 32 already observed
frames. For each chunk, `causal_multiscale_16_8_8_v1` keeps up to 16 frames from
the current interval, eight uniformly spaced frames including the available tail
of the immediately previous interval, and eight nearest frames at uniform
absolute-time anchors over older observed intervals. When a tier underfills, the
most recent unselected observed frames fill the remaining capacity. Selection is
deduplicated and returned in chronological source order. The official extractor
does not sample the exact interval endpoint; "tail" therefore means its last
available extracted frame.

The backbone, official prompt/dialog normalization, greedy decoding, history8
text context, 64-token generation cap, shared-vision decision features, 1,051
feature schema, D4.2 session folds, learner, L2 grid, and calibration rule remain
frozen. A fresh policy-matched linear head is fitted in each OOF rotation. The sole
primary comparison is against the exactly reproduced D4.2 history8 baseline
Macro-F1 0.6988. No sampler variant or budget may be selected after labels are
evaluated.

Promotion requires Macro improvement of at least 0.005, paired-session bootstrap
95% lower bound above zero, positive changes in at least four of five folds and
three of four domains, non-declining previous-interrupt and silent strata, and no
class collapse. Failure stops this sampling family but does not block separately
pre-registered fusion experiments. All evidence remains post-selection and
public-validation supervised, not independent or hidden-test evidence.
