# Current Route: C1 Small, PWR-Inspired

> Updated: 2026-07-20
> Status: D3 at official OOF Macro F1 0.6690 remains the formally promoted scientific baseline; D4 remains the frozen leaderboard-engineering candidate and now has a bundled 1,052-parameter head, hidden-input submission adapter, strict two-field publisher, parameter/license/dependency manifests, a CPU preflight, 48 passing R0--D4 regression tests, and an exact 10-chunk adapter GPU smoke; next merge the teammate-owned frame/history policy, adapt the official Docker template when released, and otherwise return to early-chunk utterance/grounding while remaining ratings are pending; leaderboard/container upload still requires user authorization
> Objective: maximize official C1 Macro F1 in the Small division without sacrificing causality, reproducibility, or prize eligibility

## 1. Decisions Already Made

1. **Primary target is C1 Small.** ProAssist-8B and LiveStar-8B are historical baselines, not deployable Small backbones.
2. **PWR changes the problem formulation.** The decision should be conditioned on procedural state and visual completion/incompletion evidence, not treated as an isolated gate.
3. **Official PWR is not reproducible today.** No public training code, checkpoints, plan/cue targets, or Pro2Bench training annotations were found.
4. **Plan-state scaling is paused, and RL remains ineligible.** U1-V does not localize the main failure to state, while D3-D shows dialog policy explains most decision gain. Resume state work only on the frozen evidence gates; do not start GRPO.
5. **Granularity is a testable hypothesis.** It becomes a modeled component only if oracle granularity changes the metric reliably.
6. **STRIDE data is optional boundary supervision.** It is not direct C1 interrupt supervision and cannot silently become a prize-critical dependency.
7. **R0 backbone is InternVL3.5-1B-HF.** The pinned Apache-2.0 checkpoint at revision `9191dbccf312b537016f041b25d61c72e7c5c9f3` contains 1,060,897,792 unique parameters and is within the 2B Small limit.
8. **The four-session R1 result is a protocol pilot, not a no-effect proof.** It found no benefit for the current zero-shot text serialization, but 50 chunks are insufficient to rule out a real state effect.
9. **Decision calibration is complete before further state scaling.** D1 proves a frozen causal fused head reaches `0.6341`; both the residual MLP and final-MLP LoRA add only unstable `+0.0010/+0.0016`. The calibrated interface is now strong enough to revisit oracle state without treating generic head capacity as the main confound.
10. **The deployed D1 threshold is sufficiently stable.** Replacing per-fold calibration thresholds with the serialized median threshold changes OOF Macro `0.6341 -> 0.6330`; the full-precision drop is only `0.00113`, the worst-fold drop is `0.00424`, and the predeclared robustness gate passes.
11. **A generic nonlinear head is not the missing capacity.** The single preregistered width-8 residual MLP reaches `0.6351`, only `+0.0010` over D1, with paired-session 95% interval `[-0.0011,+0.0031]`, 3/5 strictly positive folds, and two domains slightly worse. It is not promoted and must not be post-hoc tuned on the same folds.
12. **Final-MLP adaptation learns tag-margin signal but does not provide a stable fused gain.** Naive three-position BF16 replay and the two historical feasibility smokes retain their failed status. A six-state, same-batch-corrected cache later reproduced all 9,935 D1 rows exactly and enabled the one frozen OOF run. `adapted_fused_linear` reaches `0.6357`, only `+0.0016` over D1, with bootstrap `[-0.00425,+0.00756]` and 2/5 positive folds. It is rejected; do not search more ranks, layers, or learning rates on the same split.
13. **Decision and content quality are separate objectives.** D1 remains frozen for leaderboard development while utterance quality is audited independently. Content diagnostics must not be folded into or substituted for the official C1 Macro F1.
14. **U0/U1 diagnostics now block the larger oracle-state replication.** Reviewer A and U1-V identify a strong language-history interface effect but no repeatable step/progress residual. Keep the frozen state package gate, but do not expand annotation while it remains unresolved.
15. **D3 establishes cross-chunk dynamics as a stable decision signal.** The preregistered primary reaches official OOF Macro `0.6690`, `+0.0349` over exact D1 replay, with positive session-bootstrap lower bound, 5/5 positive folds, 4/4 positive domains, and positive non-first gain. It is promoted for decision development, but the gain includes official dialog-history policy signal and must not be described as purely visual procedural understanding.
16. **U1-V identifies assistant history as the forced-generation bottleneck.** Removing assistant history makes all 80/80 samples fall back; removing the current interval lowers fallback `30.0% -> 26.25%` and does not trigger the preregistered current-visual gate. Masking all pixels changes wording at a threshold boundary but does not change aggregate fallback. Treat vision as an unstable content modifier, not a reliable state decoder.
17. **D3-D reconstructs the decision gain from official dialog policy, and D4 is the frozen leaderboard candidate.** Eight answer-stripped causal dialog-stage scalars alone reach `0.6618`; D1 fused plus those scalars reaches diagnostic OOF Macro `0.6846`, with 5/5 positive folds, 4/4 positive domains, and session-bootstrap interval `[+0.0418,+0.0591]` versus D1. D4 does not retroactively promote this result: it full-refits exactly that one feature set, serializes a 1,052-parameter head, reproduces all cached online decisions, and passes an exact GPU smoke. Do not search related features.
18. **D4 model-facing submission packaging is complete before the official template.** The adapter accepts arbitrary hidden JSONL/video mount paths, requires chunk-aligned official `dialog`, rejects `answers` by default, rewrites only runtime paths/hashes, invokes the frozen D4 runner without a scorer, and atomically publishes exact `video_path/answers` rows. The exact head is bundled in the handoff package. CPU preflight, 48 regressions, and a physical-GPU one-session smoke pass; the adapter prediction is byte-identical to the frozen D4 smoke. The official Docker base/interface remains pending until its announced release, and the project top-level source license still requires an owner decision.

## 2. Target System Shape

The intended Small system separates internal state from outward speech:

```text
causal video chunks + query + prior dialog
                  |
                  v
       compact state updater
       current step / progress / cue evidence / confidence
                  |
                  v
       interrupt-silent decision
          |                 |
       silent           interrupt + concise utterance
```

Internal state should be allowed to update on every chunk. The outward gate controls whether the assistant speaks; it must not freeze all procedural tracking after a false negative.

Candidate compact state:

```text
goal
current_step
progress: not_started | ongoing | complete | deviated | recovered
completion_evidence
incompletion_or_error_evidence
next_step
confidence
last_update_chunk
```

This is our proposed representation, not an official PWR schema.

## 3. Experiment Ladder

All stages must use the same Small backbone, causal input, split, scorer, and decoding policy unless the experiment explicitly studies one of them.

### R0: Freeze a Reproducible No-Plan Baseline

Deliverables:

- an open-weight Small backbone with verified total parameters;
- full 700-session predictions in original order;
- official Macro/interrupt/silent/G-mean metrics;
- deterministic config and data fingerprint;
- explicit declaration of whether public validation labels were used for tuning.

Backbone decision (2026-07-13): use `OpenGVLab/InternVL3_5-1B-HF` at revision `9191dbccf312b537016f041b25d61c72e7c5c9f3`. The exact BF16 checkpoint has 1,060,897,792 stored unique parameters: 304,012,288 vision, 751,632,384 language, and 5,253,120 projector parameters. The weight SHA256 is `11effd1da2fc0929957d56de4129d1e7d2aed044ea878d40bf83e95b63d38b39`; model metadata is recorded in `models/internvl35_1b_hf.json`.

R0 completed on 2026-07-14 over all 700 public-validation sessions and 9,935 chunks. The official scorer reports Macro F1 `0.4630`, G-mean F1 `0.4541`, interrupt P/R/F1 `0.5286/0.2879/0.3728`, and silent P/R/F1 `0.4571/0.7002/0.5531` (TP/FP/TN/FN `1541/1374/3209/3811`). The model predicted interrupt on `29.34%` of chunks, versus `53.87%` interrupt support in the public labels, so the dominant R0 error is missed interventions rather than over-interruption. The frozen predictions SHA256 is `312d0375dd67be2fb244d622a9f302734082f0f251ef0b8dd190b00880879820`; the [formal report](reports/20260713_internvl35_1b_no_plan_r0.md) and [complete artifact](output/experiments/20260713_internvl35_1b_no_plan_r0/) are the reference for all R1 comparisons.

R0 also exposed an orthogonal format-adherence issue: 633/9,935 raw generations did not start with a valid tag and were scored as silent under the official fallback. R1 must keep the same canonicalization and report this count separately, so a plan-state gain is not confused with a format-compliance change. GPU occupancy is checked before loading; light sharing is permitted with a warning, while `--require-exclusive-gpu` remains available for strict runs.

### R0-F: Response-Intent Format Repair

R0-F completed on 2026-07-14 by re-canonicalizing the frozen R0 raw generations without rerunning the model. Explicit tags are preserved; empty raw text remains silent; other non-empty text is prefixed with `$interrupt$`. The repair function reads no per-response label or threshold, but the rule family was selected after public-validation error analysis, so the experiment is explicitly `val-supervised`.

The official full-set result is Macro F1 `0.5362`, G-mean `0.5340`, interrupt P/R/F1 `0.6119/0.4056/0.4879`, and silent P/R/F1 `0.5020/0.6995/0.5845` (TP/FP/TN/FN `2171/1377/3206/3181`). Relative to R0, 633 decisions change: 630 become TP and 3 become FP. The frozen prediction SHA256 is `cfda7d147ac3203ff5750a5b65fbac54af5f2bcf4aef4d4fa16db700b25c0e37`.

R0-F was the first public-validation leaderboard candidate and is now a historical control superseded by D1. Its dominant gain is the public convention that 699/700 first chunks are interrupt; hidden-test transfer must not be assumed. Two label-independent grammar attempts were rejected at one-session smoke because they predicted interrupt on 13--14/14 chunks and collapsed silent F1 to zero. See the [R0-F report](reports/20260714_internvl35_1b_r0f_format_ablation.md).

### R1: Oracle Compact Plan Upper Bound

Create a larger, auditable, preregistered oracle-state replication on sessions independent of the four-session pilot. Compare:

- no plan;
- current step only;
- current step + completion/incompletion cues;
- compact full state.

Gate: proceed only if state information produces a clear, repeatable benefit or a diagnostically useful class tradeoff. If the oracle plan does not help, do not build a planner.

Active R1 target after D2: determine whether oracle state can recover interrupt recall without erasing silent precision on the stabilized D1 decision interface. Freeze causal inputs, frame/history policy, split construction, response canonicalization, and scorer before evaluation. Any prompt-format intervention is a separate ablation, not part of the plan-state comparison.

R1 protocol pilot completed on 2026-07-14 over a label-independent, domain-stratified four-session subset (50 chunks). Frozen R0 scored subset Macro F1 `0.5398`; null wrapper `0.5169`; step `0.4949`; cues `0.3592`; full compact state `0.5169`. Full state increased interrupt recall from null `0.3667` to `0.5000`, but reduced silent recall from `0.7500` to `0.5500`, producing no Macro gain. It changed the null confusion matrix by `+4 TP/+4 FP/-4 TN/-4 FN`. The R1 scientific gate was not passed, and the proposed 16-session expansion of the same zero-shot text serialization is cancelled rather than resumed.

The pilot also exposed a separate format confound: all four full-state first chunks generated instruction-like text or a malformed tag and were scored silent. Two subsequent grammar-controlled smokes collapsed toward all-interrupt and were stopped before the complete factorial. Posthoc response-intent repair gives frozen R0 `0.5994` and full state `0.5895` on the same four sessions, so state still does not beat the repaired no-plan reference. See the [R1 pilot report](reports/20260714_internvl35_1b_oracle_state_r1_pilot_v1.md).

This is a **no-positive-signal pilot**, not evidence that procedural state has zero value. Four sessions cannot establish a population effect or a stable null result. A larger preregistered session-level oracle-state replication remains a possible future route, but U1-V and D3-D now pause it: neither localizes the active residual to step/progress, while dialog policy explains the stronger decision gains. If the frozen state gate later reactivates replication, it must use an independent evaluation set and a decision-level interface; it must not simply scale the cancelled zero-shot serialization.

### D1: Frozen Multimodal Decision-Head Calibration

Decouple the ranked interrupt/silent decision from free-form tag generation. Keep the R0 InternVL backbone frozen and test whether causal features already contain a linearly recoverable decision signal.

Protocol:

- freeze a deterministic five-fold split at the session level, stratified by domain without inspecting labels;
- never place chunks from one session in different folds;
- use three folds for fitting, one fold for threshold calibration, and one fold for testing, rotating until every session has one out-of-fold prediction;
- begin with no-GPU temporal/raw-response baselines, then add tag-sequence score margin and the final causal multimodal hidden state;
- train only a linear head first; add a small MLP or LoRA only after a positive held-out signal;
- use R0-F only as the response-text repair/output baseline, not as hidden-test evidence;
- run the official scorer on the merged out-of-fold prediction file and label every trained/tuned result `val-supervised`.

Required controls:

| Variant | Purpose |
|---|---|
| temporal/raw-response | Measure first-chunk, causal chunk-index/elapsed-time, visible-dialog, and format priors without new visual features |
| tag-margin | Test whether the rejected grammar had ranking signal but a bad zero threshold |
| hidden-linear | Test linear separability of the frozen 1,024-dimensional causal multimodal state |
| fused-linear | Test hidden state plus causal temporal/tag features without nonlinear capacity |

Gate for a promoted candidate:

1. out-of-fold Macro F1 improves over R0-F by at least `0.015`;
2. the session-bootstrap confidence interval for the paired improvement has a positive lower bound;
3. both class F1 values remain non-collapsed;
4. the gain is not confined to first chunks and improves a meaningful portion of mid-session false negatives;
5. per-domain results and predicted interrupt rate are reported.

After the scalar control passed this original gate, the neural increment was judged against the stronger `response_temporal` OOF reference: require at least `+0.005` Macro, a positive paired session-bootstrap lower bound, non-collapsed class F1, and a non-first-chunk gain.

If the scalar controls already explain the gain, treat it as annotation-policy calibration rather than visual understanding. If tag margins or hidden features show no held-out separability, do not launch a full hidden-head training run by default.

Scalar-control result (2026-07-14): completed over all 700 sessions / 9,935 chunks with the frozen five-fold session split. The effective v2 uses no full-session length, final video duration, relative complete-session position, future dialog, or future frames. `temporal`, `temporal_domain`, and `response_temporal` obtain official OOF Macro F1 `0.6081`, `0.6104`, and `0.6119`, respectively, versus R0-F `0.5362`. The best paired session-bootstrap delta is `+0.0756`, with 95% interval `[+0.0660, +0.0853]`; all five folds and all four domains improve. Non-first-chunk Macro rises from `0.4915` to `0.5843`, so the effect is not confined to the first-chunk convention.

The scalar control therefore passes the D1 promotion gate. Its interpretation is deliberately narrow: timing features alone reach `0.6081`, while domain and frozen-response properties add only `0.0038`, so most of the gain is public annotation-policy calibration rather than demonstrated visual understanding. The OOF file combines five different heads and is not itself a deployable test-time model. Its next-stage requirement was to serialize one full-development head and measure tag-margin/causal-hidden/fused increments; both are now complete below. A first implementation that used total-session-derived relative-position features was invalidated immediately, archived outside the active project, and must not be cited. See the [scalar D1 report](reports/20260714_internvl35_1b_causal_scalar_decision_head_d1_oof_v2.md).

Scalar deployment update: the selected `response_temporal` policy has been refit on all 700 public-development sessions and serialized as one 19-parameter head under `20260714_internvl35_1b_causal_scalar_decision_head_d1_final_v1`. Its threshold is the median of the five already frozen OOF calibration thresholds, not a threshold reselected on full-fit predictions. The official train-fit sanity Macro is `0.6136`; the head, predictions, and metrics reproduce byte-for-byte on an independent refit. This does not replace the OOF `0.6119` estimate.

Neural increment result (2026-07-15): the four-shard label-free cache completed over all 700 sessions / 9,935 chunks with hidden shape `[9935, 1024]`, zero candidate-prefix hidden difference, and a peak of about 3.14 GB per extraction GPU. Under the same frozen split, `tag_only`, `scalar_tag`, `hidden_linear`, and `fused_linear` obtain official OOF Macro F1 `0.5313`, `0.6172`, `0.6031`, and **`0.6341`**. The fused head improves the scalar reference by `+0.0222`; paired session bootstrap gives median `+0.0223` and 95% interval `[+0.0123, +0.0322]`. All five folds and four domains improve, and non-first Macro rises `0.5843 -> 0.6045`. Tag margin or hidden alone does not pass, so the positive result is complementary fusion rather than a stand-alone neural gate.

An explicitly post-hoc expanded L2 grid produced fused Macro `0.6336`, below the clean first run, so it was rejected. The clean `fused_linear` selection has been refit on all public-development sessions as one 1,044-parameter head under `20260715_internvl35_1b_neural_decision_head_d1_final_v1`; its full-fit Macro `0.6719` is training sanity only. An independent refit is byte-identical. A 10-chunk online GPU smoke reproduced the frozen R0 raw responses and tag margins exactly, matched all decisions/answers, and differed in logit by at most `9.03e-8`; runtime was 43.258 seconds with 3.47 GB peak memory. See the [complete neural D1 report](reports/20260715_internvl35_1b_neural_decision_head_d1.md).

Inference optimization update (2026-07-15): simple batch-of-two scoring was byte-equivalent but 18.12% slower and used 19.59% more peak memory; cropped prefix-cache scoring changed tag margin by as much as `0.1134` and was also slower, so both were rejected. The promoted `shared_vision` path computes the projected video representation once while preserving the two original batch-one language passes. On an eight-session, four-domain, short/long 127-chunk benchmark, all hidden states and tag margins are exactly equal to the sequential cache, predictions and official metrics are byte-identical, wall time improves `500.892 -> 455.056s` (`9.15%`), and peak memory is unchanged. Use `configs/d1_internvl35_1b_neural_deploy_shared_vision.json` for deployment and retain the sequential config as the correctness oracle. See the [inference optimization report](reports/20260715_internvl35_1b_d1_inference_optimization.md).

Threshold robustness update (2026-07-15): the five frozen linear rotations and original prediction SHA256 were reproduced exactly before applying the final head's single median threshold. Official Macro changes `0.6341 -> 0.6330`; paired session delta interval is `[-0.0048,+0.0024]`, the worst fold changes by `-0.00424`, and all five preregistered deployment checks pass. Keep threshold `0.1256053793821626`; the descriptive OOF sweep must not be used for a post-hoc replacement. See the [threshold audit](reports/20260715_internvl35_1b_d1_threshold_robustness.md).

### D2: Lightweight State-Aware Supervision

D2 residual-MLP control completed on 2026-07-15. It freezes the D1 1,043-dimensional input and exact per-fold linear logit, then adds a zero-initialized `1043 -> 8 -> 1` GELU residual with 8,361 new parameters. Official OOF Macro is `0.6351` versus D1 `0.6341`; paired-session delta median is `+0.0010`, 95% interval `[-0.0011,+0.0031]`, only 3/5 folds are strictly positive, and Arts/Chef decline slightly. It fails the predeclared `+0.005`, positive-bootstrap, and 4/5-fold gates. Do not full-refit, submit, or post-hoc tune this MLP. See the [D2 report](reports/20260715_internvl35_1b_residual_mlp_d2.md).

Final-language-MLP feasibility audit completed on 2026-07-15. InternVL has 28 Qwen3 language layers; the audited adapter targets only layer 27 MLP `gate_proj/up_proj/down_proj`, rank 8, alpha 16, dropout 0. It adds 98,304 trainable parameters, for `1,060,996,096` base-plus-adapter parameters or `1,060,997,140` including the 1,044-parameter fused head. Those are the hypothetical submission numbers **if the rejected LoRA system were deployed**; it is not the current candidate. The current D1 base-plus-head system is `1,060,898,836` total/active parameters, approximately `1.060898836B`.

The historical feasibility v1 stores only residual/normalized states and fails BF16 local replay with base hidden/logit/margin differences `0.1875/0.125/0.0030565`. Feasibility v2 adds reference/local MLP outputs; its two fixed chunks are zero-adapter exact, but its preregistered non-deployed local/full adapted-margin diagnostic is `0.0105085 > 0.01`, so both smoke artifacts remain failed.

The first four-state full-cache attempt exposed a second BF16 tail at `(input 11, chunk 4)`: final RMSNorm hidden differed by `0.03125`. It was stopped and preserved. The replacement formal cache stores six tensors per candidate: residual, normalized input, reference/local MLP output, and reference/local final hidden. Formal replay computes adapter-enabled minus adapter-disabled deltas under the same fixed batch-64 shape, including a same-shape LM-head margin correction. A 16-chunk regression and the full 700-session / 9,935-chunk merge both obtain exact zero differences against D1 hidden/margin and exact prompt/key matches.

The six-state cache uses `73,728 bytes/chunk`, or `732,487,680 bytes` (`698.554688 MiB`) uncompressed; the compressed file is `604,798,620 bytes`. Four-shard extraction consumed `6.8764 GPU-hours`, `1.8278 h` wall time, and at most `2.906 GiB` per GPU. The merged feature SHA256 is `2c4d7d4d69e54e7156404f747a3ff65cd6c6652c4623dd4d50aad9f538dd455e`.

The single frozen five-fold OOF completed on 2026-07-16. `adapted_tag_only`, `adapted_hidden_linear`, and `adapted_fused_linear` obtain official Macro F1 `0.5879/0.6064/0.6357`. The primary fused result is `+0.0016` over D1; paired-session bootstrap is `[-0.00425,+0.00756]`, only folds 2 and 4 improve, and only the non-first and non-collapse gates pass. Non-first Macro does improve `0.60454 -> 0.61017`, concentrated at chunks 5+, but Chef declines `-0.0086` and only 2/4 domains improve. No adapter is promoted or full-refit. See the [formal OOF report](reports/20260716_internvl35_1b_final_mlp_lora_oof.md) and retain the [feasibility audit](reports/20260715_internvl35_1b_final_mlp_lora_feasibility.md) as engineering history.

Do not try more ranks, layers, MLP widths, L2 ranges, or post-hoc learning rates. The remaining U1 ratings retain their frozen authority for future utterance/state conclusions, but they no longer block D4 leaderboard decision engineering. Any later larger oracle-state replication must not reuse the four-session pilot as its evaluation set and must be sized for session-level uncertainty. Proceed to granularity only if state information then gives a repeatable benefit.

### D3: Frozen-Cache Causal Dynamics

D3 is preregistered on 2026-07-17 while U1 human ratings are pending. It is independent of utterance ratings and uses no new backbone inference. The hypothesis is that strictly causal cross-chunk changes contain procedural-progress signal missing from D1's current-chunk representation.

The frozen label-free dynamics are eight scalars (previous/history-mean tag-margin differences plus previous/history-mean hidden cosine and RMS change) and the 1,024-dimensional `current_hidden - previous_hidden`. First-chunk dynamics are zero; all historical means exclude the current chunk. The split, class-balanced linear learner, four-value clean D1 L2 grid, calibration policy, threshold policy, and official scorer remain unchanged.

Only `dynamics_fused = D1 fused + 8 dynamics scalars + hidden delta` is promotion-eligible. `d1_fused_replay`, scalar-only dynamics, and hidden-delta-only dynamics are frozen diagnostics and cannot be selected after results. Promotion requires at least `+0.005` official Macro over D1 `0.6341`, positive session-bootstrap lower bound, at least 4/5 positive folds and 3/4 positive domains, positive non-first gain, and both class F1 values at least `0.60`. No post-hoc feature, L2, rolling-window, or threshold search is allowed on the same folds.

D3 completed on 2026-07-17 over all 700 sessions / 9,935 chunks. Exact D1 replay reproduces Macro `0.6341` and the frozen prediction SHA256. Diagnostic scalar and hidden-delta variants reach `0.6551/0.6594`; the single promotion-eligible `dynamics_fused` reaches official Macro **`0.6690`**, interrupt/silent F1 `0.6845/0.6535`, and predicted-interrupt rate `50.82%`. Relative to D1, paired-session bootstrap gives median `+0.03506` and 95% interval `[+0.02654,+0.04372]`; all five folds and all four domains improve, while non-first Macro rises `0.60454 -> 0.64460`. All preregistered gates pass. Do not post-hoc search new dynamics, windows, L2 values, or thresholds on these folds.

The interpretation is narrower than “visual progress solved.” For all 9,235 non-first public-validation chunks, whether `dialog[i]` adds an assistant turn over `dialog[i-1]` exactly equals the previous gold interrupt decision. This is a legal causal input under the official prefix, but creates an annotation/dialog-policy signal. D3 gains `+0.04061` in the no-new-assistant-turn group and only `+0.00145` in the new-turn group. Treat D3 as a valid official-protocol decision improvement, not proof of purely visual temporal understanding; self-fed dialog robustness remains open.

A 2,076-parameter all-development head is serialized using frozen median L2 `0.01` and threshold `0.14439966662436324`, for 1,060,899,868 total/active system parameters. Independent OOF and final refits are byte-identical. The session-local online state reproduces every cached dynamic feature exactly and all 9,935 decisions, with maximum logit difference `2.95e-7`. D3 is integrated into the existing shared-vision deployment CLI; a physical-GPU 10-chunk smoke matches frozen raw response, prompt tokens, margin, hidden, dynamics, decision, and answer exactly, with maximum logit difference `7.22e-8`, 42.989s wall time, and 3.466 GB peak memory. Full submission/container packaging remains separate and requires user authorization. See the [formal D3 report](reports/20260717_internvl35_1b_causal_dynamics_d3.md).

### D3-D and D4: Dialog-Policy Control to Leaderboard Candidate

D3-D completed on 2026-07-19 as a preregistered, CPU-only, non-promotable
mechanism control. It derives eight scalars only from the answer-stripped official
dialog prefix. The smallest two-feature previous-assistant-increment control
reaches Macro `0.6475`; eight dialog-stage scalars without neural features reach
`0.6618`, reconstructing `79.4%` of the D3 gain. D1 fused plus the two increment
features reaches `0.6749`; D1 fused plus all eight reaches **`0.6846`**, interrupt/
silent F1 `0.6893/0.6799`, and non-first Macro `0.6596`.

For the `0.6846` diagnostic, all five folds and four domains improve over D1;
the paired-session interval is `[+0.0418,+0.0591]`. The most stable top dialog
coefficient in every fold is time since the last assistant addition. After all
OOF predictions were frozen, `assistant_added_since_previous` again matches the
previous gold action on 9,235/9,235 non-first chunks. This establishes that
official-dialog intervention rhythm, not demonstrated procedural-state decoding,
explains most of the measured gain. It remains a causal and benchmark-visible
leaderboard signal, with a self-fed-dialog distribution-shift caveat.

Do not retroactively promote `0.6846`: the D3-D protocol explicitly made all five
variants diagnostic. D4 completed on 2026-07-19 using exactly the frozen eight
features and the existing finalization policy: median OOF L2 `0.01`, median OOF
threshold `0.1263874797442615`, and one full-development fit. The serialized head
has 1,052 parameters (1,060,898,844 total system parameters); its full-fit Macro
`0.7393` is training sanity only. Online dialog features match offline values with
zero difference, all 9,935 cached decisions match, and maximum logit difference is
`2.55e-7`. A physical-GPU shared-vision smoke matches raw response, prompt tokens,
tag margin, hidden, eight dialog features, decision, and answer on 10/10 chunks;
maximum logit difference is `6.32e-8` and peak memory is 3.466 GB.

D4 is now the frozen **leaderboard-engineering candidate**, while D3 remains the
formally promoted scientific baseline. No feature, L2-grid, threshold-policy, or
history-window search is allowed on this branch. On 2026-07-20 the model-facing
submission adapter completed: it accepts organizer JSONL/video paths, requires
chunk-aligned official dialog, rejects answers by default, writes only the official
two prediction fields atomically, and records parameter/input/output hashes. CPU
preflight and 48 regressions pass. A physical-GPU one-session adapter smoke takes
44.206s, peaks at 3,466,037,248 bytes, sees no preexisting process on the selected
GPU, and produces a byte-identical prediction file plus 10/10 exact raw response,
prompt, margin, dialog features, logits, decisions, and answers versus the frozen
D4 smoke. The hidden-test form values are model license `Apache-2.0` and total/active
parameters `1.060898844B/1.060898844B`.

The official Docker template is still scheduled for release to top validation
participants on 2026-08-08, so final CMD/mount/resource adaptation remains pending.
The teammate-owned frame/history search must produce a synchronized config/manifest
and rerun the adapter smoke before replacing the current defaults. The project also
lacks a top-level source-code license; this owner decision must be resolved before
claiming prize-source eligibility. Do not upload to the leaderboard or container registry without user authorization.
See the [submission audit](reports/20260720_internvl35_1b_d4_submission_entrypoint_audit.md),
[submission contract](submission/d4_small/README.md),
[combined U1-V/D3-D report](reports/20260719_u1_visual_reliance_and_d3_dialog_policy_control.md)
and [D4 report](reports/20260719_internvl35_1b_d4_dialog_stage_candidate.md).

### U0: Frozen-Gate Utterance Audit

Audit the existing D1 fused OOF answers without running or training a model. Produce reproducible full-set statistics by domain, task, chunk position, confusion outcome, fallback status, and session repetition. Freeze a 200-item blind human-review sample with separate review and answer-key files; the review file must not expose gold decisions, gold utterances, D1 confidence, or source-system labels.

U0 is complete only when the source hashes, sampling seed, exact stratum counts, rubric, rating template, and generated artifact hashes are recorded. Human ratings may remain pending, but automatic statistics and the review package must be reproducible byte-for-byte.

U0 automatic audit completed on 2026-07-16 over all 700 sessions / 9,935 chunks. D1 predicts 4,613 interrupts, of which 2,586 (`56.06%`) use the hard-coded fallback; 1,647/3,165 binary TP are fallback. Fallback binary precision is `63.69%` versus `74.89%` for non-fallback text, but these are decision-label precisions and do not establish semantic correctness. The second chunk is the sharpest interface failure: 423/426 predicted interrupts are fallback. A deterministic 200-item, five-stratum, four-domain-balanced blind-review package is frozen; automatic artifacts reproduce byte-for-byte with manifest SHA256 `92ba38ec6f600086464eb4098d5a9242fcfcf0350fc3ed213aecdb153fd07291`. See the [U0 report](reports/20260716_d1_utterance_u0_audit.md).

### U1: Fixed-D1-Gate Forced Generation

Use the exact frozen D1 OOF interrupt/silent decisions. On a label-independent sample from chunks where D1 predicts interrupt and raw R0 chose silent, compare:

- the current hard-coded fallback;
- forced interrupt generation without plan/state;
- forced generation with answer-blind oracle current/next step;
- forced generation with answer-blind oracle step, progress, visible evidence, and recovery action.

All variants must have identical decisions and sample order. Oracle annotations may use only the query, official prior dialog, and video evidence through the current interval; current/future gold answers, future dialog, and future video are prohibited. Evaluate content separately from official Macro. If forced no-state generation succeeds, prioritize repairing the gate-to-language interface; if only oracle state succeeds, proceed with the larger state replication; if both fail, prioritize fit-fold-only utterance supervision before a deployable state updater.

U1 progress through 2026-07-17: a label-independent sample is frozen from D1-fallback/R0-explicit-silent chunks, excluding the old four R1 sessions. It contains 20 sessions / 80 chunks, exactly 20 per domain and 20 per second/2--4/5--9/10+ position. A provenance audit found that the original 16-chunk smoke annotator had already inspected the corresponding generation outputs, so that oracle file is now explicitly nonblind and engineering-only.

All 20 sessions / 80 states were therefore re-annotated by two isolated-context agents from sanitized inputs. Static plans used only query/task; dynamic states used prior dialog and only the explicit causal video intervals, excluding interval gaps, future information, gold answers, and model outputs. The merged formal oracle SHA256 is `e8f1e0736398d46193009ddb3966599ccc2f8629cfaecdd55f270b5ec6018850`; strict coverage, timestamp, target-marker, step-reference, confidence, and provenance checks pass.

The full formal run reproduces 80/80 frozen R0 raw responses and preserves all 9,935 D1 decisions plus official Macro `0.6341`. No-state, oracle-step, and oracle-full yield 56/80, 56/80, and 57/80 non-empty continuations. Step changes 43 texts but never changes empty/nonempty status; full changes 53 and recovers one additional second-chunk continuation. These are non-semantic diagnostics, not evidence of content gain. The frozen state package rerates no-state/step/full together as 240 blind candidates rather than reusing interface-package no-state scores. See the [formal U1 report](reports/20260717_u1_formal_blind_oracle_generation.md). Do not promote/reject state until both reviewers finish and `state_ratings.py` applies the preregistered gates.

U1-V completed on 2026-07-19 over the same frozen 80 chunks without reading
reviewer scores. The reused full view has 30% fallback. Removing assistant history
makes 80/80 fall back, a `+70pp` change; removing only the current interval lowers
fallback to `26.25%`, with mean text similarity `0.7734`, so the preregistered
current-visual gate does not trigger. Full masking keeps fallback at 30% but lowers
mean similarity to `0.6479`, narrowly crossing the frozen `0.65` any-visual
threshold. Qualitative discordant cases show that vision sometimes changes the
specific step, but history supplies the generation skeleton and can produce
plausible stale instructions without current visual evidence. Prioritize early
language cold start and grounding only after the D4 decision candidate; do not
resume S1 from this result.

### R2: Granularity Sensitivity

For the same sessions, construct coarse, medium, and fine oracle step/cue variants. Hold all other inputs fixed.

Gate: activate this stage only after D1/D2 provide a decision interface that can demonstrably use state. Build a dedicated granularity predictor only if granularity materially changes Macro F1 or consistently explains FP/FN patterns. Otherwise treat it as an annotation/calibration detail.

### R3: Predicted Compact State

Replace oracle state with a deployable causal state updater. Measure:

- state accuracy separately from decision accuracy;
- oracle-to-predicted performance gap;
- false-negative state staleness;
- parameter and latency cost.

S0 preregistration update (2026-07-17): while U1 human ratings remain pending,
run an independent oracle-plan/predicted-dynamic-state feasibility study on the
existing 20-session / 80-state formal set. The prediction runner receives the
query-only four-step static plan, current causal frames, and either the official
dialog prefix or a diagnostic prefix with assistant history removed. It scores
fixed equal-length candidates for step (`s1--s4`), progress (five protocol
classes), and error-present, with one shared vision pass and no free-form JSON.
The formal oracle remains evaluation-only and is stored separately from runner
inputs. Report step/progress/error Macro F1, joint accuracy, domain/position
breakdowns, and paired session bootstrap between dialog views. Mean task Macro
F1 `>=0.45` is a strong zero-shot signal, `0.35--0.45` is weak-but-usable, and
`<0.35` is insufficient zero-shot signal. This classification is frozen before
state predictions but was designed after the experimenter had inspected the
oracle schema/examples/aggregate distribution; it is not a never-seen-label
benchmark and cannot promote a submission model.

S0 engineering revision before formal inference: the target-isolated one-state
v1 smoke exposed a strong monotonic option-digit prior despite equal token
length. V1 is retained as an unevaluated engineering failure and must not be
cited as state efficacy. V2 freezes one content-free calibration per
session/target using query + oracle plan but no video/dialog, then predicts from
`observed_logp - content_free_logp`. Raw and calibrated scores are retained;
no temperature, prompt, mapping, or permutation search is allowed after formal
predictions.

S0 result (2026-07-17): both target-isolated 80-state views completed with the
frozen v2 calibration and no ratings/answers read. `official_dialog` obtains
step/progress/error Macro F1 `0.2226/0.1348/0.5098`, mean task Macro `0.2891`,
joint step-progress accuracy `0.05`, and composite correctness `0.4083`.
`no_assistant_history` obtains `0.2601/0.0167/0.2024`, mean task Macro `0.1597`,
joint accuracy `0`, and composite `0.2333`. Both fail the frozen weak threshold
`0.35`, and neither has a fully correct three-field state. Official dialog does
improve composite by `+0.1750`, paired-session 95% `[+0.1125,+0.2375]`, but the
predictions collapse toward `s4/complete/error-present`; without assistant
history they collapse toward `s1/not-started/error-absent`. This is dialog-stage
signal, not reliable visual state decoding. Do not tune S0 further.

S1 preparation is frozen: 32 new answer/model-output/rating-blind sessions,
excluding U1 formal and old R1, contain 444 contiguous states. The primary split
is 24 train sessions / 318 states and 8 held-out sessions / 126 states, balanced
by domain and covering short/middle/long. All 32 query-only four-step plans were
frozen before new S1 video inspection (plan SHA256 `eefc6a0ab4c4da6ee66182a39e69da2e9dc175ee53901f998a6ae722e251ba71`).
The plan author had previously seen official dialog for train inputs 20 and 25
during engineering inspection; this is recorded rather than claimed as isolated
plan authorship. Dynamic annotation is append-only and the renderer only exposes
the next unrecorded explicit interval. As of 2026-07-18, inputs 20 and 25 are
complete: 2/32 sessions and 23/444 states, record SHA256
`f81a2b00e2fb676a9a6d1ef22a4c81e517158550aeec5f7325b363095683e4bb`.

The S1 model path is also frozen before label completion: a label-independent
five-fold train-session split, shared-L2 selection over pooled train OOF, exact
`7/1043/2075` feature variants, strict annotation validation, and physically
separate train-only and one-shot held-out commands. Nine unit tests and the
full 9,935-row label-free feature audit pass. No decoder has been trained and no
held-out annotation has been read.

S1 pause decision (2026-07-18): reviewer A completed U0 and the 160-candidate
U1 interface package. Forced no-state improves content composite by `+1.1725`,
session-bootstrap 95% `[+0.8875,+1.45]`, and all four domains are positive, but
hallucination increases `+2.5pp`, just above the frozen `+2pp` cap. The effect is
negative at second chunks (`-0.35`, 80% generation fallback) and large at 5--9
and 10+ (`+2.06/+1.93`). Actual model history is capped at four assistant turns;
one-turn contexts have 81.5% fallback, while two-to-four-turn contexts rarely
fall back. This strongly implicates the language/history interface but does not
prove that vision is unused. Reviewer B was not read, and the separate
240-candidate no-state/step/full package is not present in the supplied A CSV.

Pause the remaining 421 S1 states without deleting any asset. U1-V and D3-D are
now complete: U1-V identifies assistant history rather than current visual state
as the dominant generation dependency, and D3-D reconstructs most or more than
all of D3's OOF gain from official dialog-stage features. Neither result satisfies
the state-resumption gate. Resume S1 only if the separate state-package ratings
pass or a new, independently frozen residual audit localizes repeatable errors to
step/progress transitions. Reviewer B and the state package may be incorporated
later under their frozen gates, but they no longer block D4 decision engineering.
See the [A-only diagnostic](reports/20260718_u0_u1_reviewer_a_diagnostic.md),
[combined U1-V/D3-D report](reports/20260719_u1_visual_reliance_and_d3_dialog_policy_control.md),
and [S0 report](reports/20260717_internvl35_1b_oracle_plan_state_s0.md).

### R4: Noisy-Plan Robustness

Train or evaluate with realistic state errors:

- one-step lag;
- skipped update;
- incorrect completion;
- wrong step index;
- stale cue;
- recovery transition error.

This addresses PWR's gold-plan teacher-forcing gap before any policy optimization.

### R5: Training Strategy Decision

Only after R0--R4 decide among:

- supervised state/decision training;
- distillation from an offline planner;
- lightweight decision calibration;
- sequence or boundary pretraining;
- GRPO or another policy objective.

RL must target an observed residual error, not serve as the default response to weak supervision.

## 4. Required Ablations

At minimum, preserve these comparisons:

| Axis | Variants |
|---|---|
| State | none / oracle / predicted / noisy |
| State content | step only / cues / full compact state |
| Update policy | outward-interrupt only / internal every chunk |
| Granularity | coarse / medium / fine, if R2 is active |
| Decision calibration | fixed decoding / threshold or class-weight change |
| Data policy | held-out / `val-supervised` |

Do not compare results produced by different session ordering, partial subsets, or non-official metric implementations.

## 5. Explicit Non-Goals for the Current Stage

- faithfully reproducing the unpublished PWR system;
- reviving ProAssist-8B or LiveStar-8B as the Small submission;
- treating action boundaries as interrupt labels;
- launching full-corpus STRIDE training before its data schema and license audit;
- starting GRPO before a stable SFT/no-plan baseline and oracle-state study;
- optimizing content quality at the expense of the decision metric, while content is not ranked in C1 validation;
- building a large online planner whose parameters violate the Small budget.

## 6. Open Decisions

The following require evidence or user direction and must not be silently assumed:

- source and cost of oracle plan/cue annotation;
- train/dev/eval protocol given the public validation set;
- whether state generation is offline, distilled, or fully online;
- whether STRIDE-derived boundary pretraining survives license and semantic-gap review;
- leaderboard target and acceptable latency envelope.
- which approved top-level source-code license the project owner selects before prize submission;
- whether a Validation Phase upload uses the transparent five-head OOF artifact or another explicitly identified prediction source.

Record a resolved decision here, with date and supporting report, before downstream agents depend on it.

## 7. Primary Evidence

- [PWR audit](literature/papers/challenge1_proactive/PWR_audit.md)
- [C1 task specification](C1_SPEC.md)
- [Official starter kit](data/starter_kit/README.md)
- [Active literature index](literature/README.md)
- [2026-07-13 archive manifest](../deprecated/wearable_ai_challenge/2026-07-13_pre_pwr_reset/MANIFEST.md)
