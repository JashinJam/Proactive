# Proactive R1

This component runs the C1 Small oracle compact-state pilot defined in `CURRENT_ROUTE.md`. It imports the frozen R0 model adapter and response canonicalization without modifying R0 or the official starter kit.

The controlled variants are `null`, `step`, `cues`, and `full`. The frozen R0 predictions for the same pilot sessions are extracted as a fifth reference. See `annotations/r1_oracle_pilot_v1/PROTOCOL.md` for the causal annotation boundary and interpretation limits.

No R1 result is submission-ready: the state is manually annotated from the evaluation video prefix and is explicitly non-deployable.
