# D5 uniform/multiscale dual-view fusion protocol

Status: re-frozen before the user-directed D4-fold rerun on 2026-07-22; the
candidate definitions, selection rule, and gates are unchanged.

This experiment combines the frozen history8 uniform-cumulative view with the
frozen causal-multiscale 16/8/8 view. The uniform view supplies the complete
1,051-feature D1-fused-plus-dialog-stage vector. The second view contributes only
the policy-matched difference in tag margin and the 1,024-dimensional causal
language hidden state. No voting or generated-answer selection is used.

Two heads are pre-registered. `shared_delta` appends one 1,025-dimensional view
difference to the uniform vector. `dialog_gated_delta` appends two copies of that
difference, multiplied respectively by `1-assistant_added_since_previous` and
`assistant_added_since_previous`. The gate is an existing strictly causal dialog
feature. Both candidates use the same D4.2 session-level outer folds,
class-balanced float64 linear learner, four-value L2 grid, and exact calibration-
fold Macro-F1 threshold selection as the D4-fold history8 baseline. The gated candidate
is selected only if its Macro-F1 exceeds `shared_delta` by at least 0.002;
otherwise `shared_delta` is the family candidate. Added head parameters must stay
below 10,000 and total parameters below 2B.

The selected family candidate is promoted only if it improves the D4-fold
history8 baseline by at least 0.005, has paired-session bootstrap 95% lower bound
above zero, improves at least four of five folds and three of four domains, does
not decline in either previous-interrupt or previous-silent strata, and predicts
both classes. No fusion widths, gates, interactions, L2 values, or thresholds may
be introduced after labels are evaluated. This remains post-selection public-
validation evidence, not independent or hidden-test evidence.
