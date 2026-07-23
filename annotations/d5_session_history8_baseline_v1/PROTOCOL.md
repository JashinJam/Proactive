# D5 D4-fold history8 baseline protocol

Status: frozen before the user-directed D4-fold rerun on 2026-07-22.

The experiment must reproduce the frozen D4.2 history8 OOF result on the exact
`domain_stratified_sha256_round_robin` session manifest with seed
`d1-session-oof-v1`. Three folds fit, one calibrates, and one tests. The generated
predictions and official metrics must match the D4.2 artifacts byte-for-byte.

The D4.2 history8 generation records and neural cache are reused byte-for-byte.
The feature schema, class-balanced float64 linear learner, four-value L2 grid,
calibration rule, prompt, decoding, and backbone are unchanged. The result is a
post-selection public-validation replay, not independent or hidden-test evidence.
No alternative fold assignment, seed, L2 value, or threshold may be selected
after labels are evaluated.
