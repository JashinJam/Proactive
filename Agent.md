# Wearable AI Challenge Agent Guide

> Scope: `/home/quewenjun/workspace/proactive_vlm/Proactive`
> Updated: 2026-07-22
> Status: normative instructions for every human or AI agent working in this project

## 1. Project Objective

The project targets **ECCV Wearable AI Challenge 1: EgoProactive**, with the following priorities:

1. Compete in the **Small** division and improve the official leaderboard metric.
2. Preserve causal streaming behavior: no future video, dialog, labels, or derived future features.
3. Optimize the official decision metric, Macro F1 over `$interrupt$` and `$silent$`.
4. Keep every claimed result reproducible and every data dependency legally traceable.
5. Prefer a compact, PWR-inspired procedural-state design over copying PWR's two large models.

Unless the user explicitly changes scope, work on C1 Small only. Challenge 2, Challenge 3, ProAssist-8B, and LiveStar-8B are historical references, not active implementation targets.

## 2. Required Reading Order

Before changing code or launching an experiment, read in this order:

1. This file.
2. [CURRENT_ROUTE.md](CURRENT_ROUTE.md), which is the living technical direction.
3. [C1_SPEC.md](C1_SPEC.md), which contains stable task and evaluation facts.
4. The [official challenge rules](https://wearable-ai-workshop.github.io/challenge_rules.html) and local [starter-kit README](starter_kit/README.md).
5. [PWR full audit](literature/papers/challenge1_proactive/PWR_audit.md).
6. The README and tests of the component being changed.
7. [daily_work_log.md](../daily_work_log.md) only when historical context is needed.

The daily log, old literature rankings, archived plans, and old experiment reports describe history. They do not define the current route.

## 3. Source of Truth

When sources disagree, use this precedence:

1. Current official challenge rules and official scorer behavior.
2. The user's latest explicit decision.
3. `C1_SPEC.md` for task facts and `CURRENT_ROUTE.md` for route decisions.
4. This file for engineering process.
5. Audits based on primary papers or source code.
6. Raw experiment artifacts and metrics recomputed with the official scorer.
7. Daily logs, old guides, presentations, and literature summaries.

Never silently choose the number or interpretation that makes a result look better. Record the conflict, the selected interpretation, and its evidence.

## 4. Current Technical Position

- Official PWR training code, weights, Pro2Bench training annotations, plan targets, and cue targets are not publicly obtainable as of 2026-07-13.
- Any implementation in this project is therefore **PWR-inspired** or a **paper-spec reimplementation**, not an official PWR reproduction.
- R0, R0-F, the four-session R1 protocol pilot, D1, D2, D3, U1-V, D3-D, D4, D4.1, D4.2, and D4.3 are complete. The active formally promoted scientific baseline remains D3 `dynamics_fused`, official OOF Macro F1 `0.6690`. D4 remains an unchanged frozen reference bundle for the non-promotable `0.6846` D3-D diagnostic. The independently packaged D4.2 `history8` system is now the active leaderboard-engineering baseline after D4.3 reproduced all online fields on 102/102 chunks across four domains; maximum logit difference is `1.22e-7`, peak allocated memory is 3.48 GB, and maximum session wall time is 112.70 seconds. Adapt the future official Docker template on release; external submission still requires explicit user authorization.
- The fused head combines 18 strictly causal scalar features, one fixed-tag margin, and a 1,024-dimensional final causal hidden state. Tag-only and hidden-only do not beat the scalar control; do not report `0.6341` as a stand-alone hidden-state result.
- One 1,044-parameter fused head has been refit on all public-development sessions and serialized. Its `0.6719` score is train-fit sanity only. The promoted `shared_vision` runner matches 127/127 frozen hidden states, margins, decisions, and answers, reduces benchmark wall time by `9.15%`, and does not increase peak memory.
- The final D1 median threshold has a formal OOF transport audit: a single threshold gives official Macro `0.6330` versus fold-specific `0.6341`, a full-precision drop of `0.00113`, and passes all preregistered robustness checks. Keep the serialized threshold; do not adopt a post-hoc threshold from the audit sweep.
- The preregistered D2 width-8 residual MLP is complete at official OOF Macro `0.6351`, only `+0.0010` over D1 with a bootstrap interval crossing zero and 3/5 positive folds. It is rejected. Do not call it the active baseline, full-refit it, or search more MLP variants on the same folds.
- The final-language-MLP LoRA engineering audit is complete. Both historical two-chunk smokes remain marked failed: v1 exposes BF16 MLP batch-shape drift, while v2 makes those two zero-adapter chunks exact but misses a non-deployed local/full margin diagnostic (`0.0105085 > 0.01`). A later four-state full-cache attempt also failed at `(input 11, chunk 4)` because final RMSNorm differed by `0.03125`. Never relabel these artifacts as passed or efficacy evidence.
- The replacement six-state label-free cache is complete over 700 sessions / 9,935 chunks. It uses same-batch adapter-enabled minus disabled corrections for the final MLP, final norm, and LM-head margin; the merged cache matches the fixed D1 hidden/margin exactly and prompt/key order exactly. Feature SHA256 is `2c4d7d4d69e54e7156404f747a3ff65cd6c6652c4623dd4d50aad9f538dd455e`.
- The single frozen rank-8 final-MLP OOF is complete. `adapted_fused_linear` obtains official Macro `0.6357`, only `+0.0016` over D1; its session-bootstrap interval `[-0.00425,+0.00756]` crosses zero and only 2/5 folds improve. It is rejected. Do not full-refit it or search rank, layer targets, learning rate, batch size, L2 range, or other LoRA variants on the same folds.
- Online serving must use the custom causal session runner because the unmodified starter `generate(frames, messages)` interface omits interval/domain metadata. Use `configs/d1_internvl35_1b_neural_deploy_shared_vision.json`; retain the sequential config as the correctness oracle. Batched and cropped-prefix-cache modes are rejected controls, not deployment candidates.
- D3 adds eight strictly causal previous/history dynamics scalars and a 1,024-dimensional current-minus-previous hidden delta to D1. Its primary result has a positive session-bootstrap interval, 5/5 positive folds, 4/4 positive domains, and positive non-first gain; do not search new dynamics, history windows, L2 values, or thresholds on the same folds. The serialized 2,076-parameter final head and online state reproduce 9,935/9,935 cached decisions. The D3 shared-vision GPU CLI also passes a 10-chunk exact smoke against frozen R0/cache/final artifacts; full submission/container packaging remains pending and requires user authorization.
- D3-D confirms that D3's gain is primarily dialog-policy explainable. Eight dialog-only scalars reach `0.6618`; D1 fused plus the frozen eight reaches diagnostic Macro `0.6846`, with 5/5 positive folds, 4/4 positive domains, and bootstrap `[+0.0418,+0.0591]` versus D1. The official assistant-addition signal equals previous gold interrupt on 9,235/9,235 non-first chunks. This signal is causal and benchmark-visible but can shift under self-fed dialog. D3-D variants remain explicitly non-promotable; do not retroactively adopt `0.6846` as a scientific baseline or search related features.
- D4 uses exactly the frozen D1-fused-plus-eight-dialog-stage matrix. Its full-development head uses OOF-median L2 `0.01` and threshold `0.1263874797442615`, has SHA256 `531431710a01a71bdd02ffd7a9758428fe282323cc41fae2c1d6859e45408b13`, and brings total parameters to 1,060,898,844. Full-fit Macro `0.7393` is training sanity only. Online replay matches 9,935/9,935 decisions with maximum logit difference `2.55e-7`; the 10-chunk GPU smoke matches all frozen intermediate/output checks with maximum logit difference `6.32e-8`. Use `configs/d4_internvl35_1b_dialog_stage_deploy_shared_vision_v1.json`; do not alter features or calibration.
- D4.1 is a completed, separate user-authorized public-validation input-policy audit; it does not relax the frozen D4 feature/calibration rule. With the D4 head, threshold, prompt, decoding, and backbone frozen, it searches only `max_frames`, `frames_per_interval`, `max_history_turns`, and `max_new_tokens` through non-overlapping stratified 80-session search/confirmation subsets and a three-policy 700-session full stage. The full result keeps default `(32,16,4,64)` best at Macro `0.7393`; `history8=(32,16,8,64)` scores `0.7378`, and `history16=(32,16,16,64)` scores `0.7372`. D4.1 config SHA256 is `c906abff796954d1039e207710f7c98b61b05685add19bcda2fa9d6f50409fab`. These are val-supervised full-fit-head results, not hidden-test evidence.
- D4.2 is the completed policy-matched five-fold OOF audit on 700 sessions / 9,935 chunks. It evaluates baseline `(32,16,4,64)`, `history8=(32,16,8,64)`, `frames16=(16,16,4,64)`, and `tokens16=(32,16,4,16)`, fitting a fresh standardized class-balanced float64 1,052-parameter linear head per policy with three fit folds, one calibration fold, one test fold, and L2 grid `{1e-5,1e-4,1e-3,1e-2}`. `history8` reaches official OOF Macro/G-mean `0.6988/0.6988` versus exact baseline `0.6846/0.6846`; its paired-session interval is `[+0.008166,+0.020363]`, with 5/5 positive folds and 4/4 positive domains. `frames16` scores `0.6854` and cuts recorded model time by about 33% but has an interval crossing zero; `tokens16` scores `0.6844` and is rejected. The final all-development `history8` fit uses L2 `0.01` and median-fold threshold `0.12101525136349107`; Macro `0.7469` is train-fit sanity only. Its head SHA256 is `dab9eaf100ea301055ab4d68856d406fb5927864bc96c71f2038688067b904c5`. D4.3 passes a four-domain 102-chunk exact GPU smoke and packages `submission/d4_2_history8_small`; the frozen D4 config, head, and submission remain unchanged. This promotion is engineering-only and does not convert the val-supervised result into hidden-test evidence.
- D5 has been re-run on the exact D4.2 session folds after the exact-query-grouped protocol was withdrawn. The D5 `history8` baseline exactly reproduces D4.2 at `0.6988`; causal multiscale reaches `0.6988`, dual-view at most `0.6846`, and the visual-temporal residual `0.6983`, so all fail their frozen gates. The final equal-mix clean/history4/assistant-drop/frame-jitter head reaches clean `0.6918`; its per-view deltas are `-0.0070/+0.0012/+0.2215/-0.0024`, so the static robustness gate fails and self-fed is not run. End these D5 families without post-hoc sampler, fusion, temporal, view-weight, perturbation, L2, or threshold variants on the same folds. The old grouped-fold outputs are historical artifacts only; D4.2 `history8` remains the active leaderboard-engineering candidate.
- D6 query-conditioned causal visual memory plus late-attention LoRA is preregistered against the exact D4.2 `history8` folds. Its 102-chunk zero-init/causality smoke passes exactly, while the rotation-0 trainability/resource smoke is still in progress and no efficacy result exists. Do not modify `src/proactive_d6`, its query-memory config/protocol, or active experiment artifacts while that run is live.
- A separate collaborator-owned historical branch uses the artifact names D5 decision fusion and D6 structured calibration but predates D4.2. Against the older D4 `0.6846` baseline, its action-history fusion reaches `0.6912` but fails the bootstrap and alternate-split gates; its structured-calibration primary reaches `0.6747` and fails decisively. Preserve those protocols, code, and reports as closed evidence. Do not confuse them with the active D4.2-session-fold D5 families or the query-memory LoRA D6, and do not promote, refit, or deploy them.
- D4 submission uses `src/proactive_d4/submission.py` and `submission/d4_small/manifest.json`; the exact 1,052-parameter head is bundled at `submission/d4_small/decision_head.json`. The current adapter requires chunk-aligned `dialog` and rejects `answers`; the only exception flag is for explicit local public-val audit, after which `strip_answers()` still runs before inference. The public schema and starter runner support that dialog contract, but hidden-test field availability and organizer-provided versus self-fed assistant-history semantics remain unconfirmed until organizer clarification or the official template. Published rows are projected to exact `video_path/answers`, strictly validated, and atomically written. Current form values are model license `Apache-2.0`, total parameters `1.060898844B`, and active parameters `1.060898844B`. Do not claim prize-source eligibility until the owner selects a top-level project source license and the official template's treatment of the organizer starter kit is confirmed.
- S0 oracle-plan zero-shot state decoding is complete and insufficient: official-dialog step/progress/error Macro F1 is `0.2226/0.1348/0.5098`, mean task Macro `0.2891 < 0.35`; no-assistant mean Macro is `0.1597`. Dialog history raises composite correctness by `+0.1750` with positive session-bootstrap lower bound, but predictions collapse to endpoint states. Do not tune S0 prompts/options/calibration. S1 assets remain frozen at 2/32 sessions and 23/444 states, including all query-only plans, the train-only CV split, validator, append-only annotation path, and separate train/evaluation commands. U1-V and D3-D do not satisfy the state-resumption gates. Resume S1 only if the separate oracle-state ratings pass or a newly frozen independent residual audit localizes repeatable errors to step/progress transitions.
- Reviewers A and B have both completed the 200-item U0 hard-stratum package, and the dedicated aggregation validates all 400 rows. Pair-average spoken-content composite is `2.3063`; fallback/nonfallback are `1.5413/3.0713`, and second-chunk composite is `1.7857`. Groundedness is not stable across reviewers: A/B means are `2.4375/4.0125` and quadratic kappa is `0.0508`, so new content promotion requires an explicit fact-support rubric rather than silent averaging. Reviewer A has completed the U1 interface package (160 candidates), while U1 reviewer B and the separate 240-candidate U1 state package remain unread/unrated. The provisional A-only forced-no-state gain is `+1.1725`, session-bootstrap 95% `[+0.8875,+1.45]`, but its `+2.5pp` hallucination increase misses the frozen `+2pp` cap. Treat this as route-diagnostic evidence, not two-reviewer promotion.
- U1-V is complete over the frozen 80 chunks. Removing assistant history raises fallback `30% -> 100%`; removing current-interval video lowers it to `26.25%` and fails the current-visual gate; masking all video preserves 30% fallback while mean text similarity `0.6479` narrowly triggers the any-visual sensitivity threshold. Vision changes some step wording but does not supply a reliable state/grounding signal. Prioritize D4 decision engineering; leave early-chunk utterance cold-start/grounding as the later content route.
- U2 is complete automatically on the frozen 21-item fixed-D4 early pool. Removing assistant history makes 21/21 candidates fall back; removing current video does not worsen coverage. A current-interval predicted-fact block rescues 3/21 query-only cases (`+14.29pp` nonempty), below the frozen `+20pp` gate, and does not change full-history coverage. The fact strings are often imperative rather than pure observations, so no grounding or content promotion is claimed and no post-hoc prompt/frame/history tuning is allowed.
- U0 utterance auditing and its A/B aggregation are complete. U1 now has fresh isolated-context formal-blind annotations for all 20 sessions / 80 sampled states; the old four-session oracle smoke is explicitly nonblind and engineering-only. Full oracle-step/full generation reproduces 80/80 R0 raw outputs, preserves every D1 decision and official Macro `0.6341`, and yields 56/80 and 57/80 non-empty continuations versus no-state 56/80. A deterministic 240-candidate no-state/step/full blind-review package and frozen session-bootstrap analysis are ready. Reviewer A's interface ratings are complete; reviewer B and state-package conclusions remain pending, so neither generation interface nor state is formally promoted/rejected. The bounded D5 decision-engineering ladder is complete; the active order is now `retain D4.2 history8 -> complete the frozen D6 gate -> adapt the official Docker template on release -> external submission only with user authorization`. Oracle replication, granularity, predicted state, and RL remain gated.
- A dedicated granularity model is justified only if coarse/medium/fine oracle plans produce a clear and repeatable metric difference.
- GRPO is not the next default step. It becomes eligible only after plan-state benefit, a measured fused-head residual, and a stable supervised baseline are established.
- STRIDE code is retained as a schema and boundary-modeling reference. Its action intervals or step boundaries are not C1 interrupt labels.
- ProAssist and LiveStar are archived 8B baselines. Do not restore them as the main route without a new user decision.

See [CURRENT_ROUTE.md](CURRENT_ROUTE.md) for gates and open decisions.

## 5. Workspace Boundaries

This checkout is a Git repository. Preserve its current branch and dirty state; do not assume historical nested repositories are present.

| Path | Role | Write Policy |
|---|---|---|
| `src/` | New Small/PWR-inspired implementation | Primary active code area |
| `configs/` | Versioned experiment and model configs | Text configs only |
| `models/` | Small wrappers or lightweight metadata | Never store downloaded weights here |
| `output/` | Generated experiment artifacts | Never treat as source code |
| `logs/` | Runtime logs | Generated only |
| `reports/` | Reproducible experiment conclusions | One report per completed experiment/analysis |
| `literature/` | Active primary evidence and method notes | No route decisions except explicit audits |
| `STRIDE/` | Optional independent reference repo; absent in this checkout | Reference-first when present; assign one write owner before edits |
| `wearable-ai-leaderboard/` | Optional official mirror; absent in this checkout | Read-only when present unless explicitly updating from upstream |
| `/data1/wearable_ai_challenge_data` | External official data | Read-only; no project-local `data` symlink is assumed |
| `starter_kit/` | Project-local official starter-kit snapshot | Read-only except an explicitly authorized upstream update |
| `../deprecated/` | Historical snapshots | Read-only; restore explicitly, never edit in place |

Do not move, rename, delete, or recursively scan large data trees merely to understand the project. In particular:

- `/data1/wearable_ai_challenge_data` is external project data and is not exposed through a required project-local symlink.
- `/data2/download` is a shared multi-terabyte download/extraction area.
- Model caches and `/data1` datasets are not part of workspace cleanup.

## 6. Multi-Agent Ownership

Every multi-agent task must declare:

- objective;
- read scope;
- exact writable paths;
- dependencies on other agents;
- expected artifact;
- validation command or acceptance check.

Rules:

1. One write owner per file and one write owner per nested Git repo at a time.
2. Unassigned paths are read-only.
3. Parallelize independent audits, tests, and components; serialize shared interfaces and migrations.
4. Do not switch branches, rebase, commit, clean, reset, or stash while another agent uses the same repo.
5. Do not revert, overwrite, or "clean up" changes whose author is unknown.
6. Before editing a nested repo, record `git status --short --branch` and `git rev-parse HEAD`.
7. A handoff must state changed paths, interface decisions, commands run, results, and remaining risks.

Agents may inspect and implement normal scoped work autonomously. Multi-GPU training, downloads over 10 GB, external submissions, and operations that consume shared cluster reservations require explicit user authorization.

## 7. Git and Dependency Rules

- Do not assume the project root has Git history.
- Do not initialize a new umbrella Git repo or convert nested repos to submodules without user approval.
- Do not commit or push unless explicitly requested.
- Preserve dirty worktrees. Never use destructive Git commands to make a repo look clean.
- Keep upstream mirrors separable from project code. Do not patch the official scorer to improve a score.
- Pin model, processor, framework, and dataset revisions in each experiment.
- Never commit videos, model weights, checkpoints, Arrow data, predictions, caches, or credentials.

## 8. Data and License Rules

For every training source, record:

- source name and URL;
- dataset/file revision or hash;
- extraction/conversion version;
- top-level license;
- underlying video/content license;
- allowed use in a prize-eligible model;
- exact supervision derived from it.

An outer CC-BY-4.0 dataset license does not override a source video's NC, platform, or unclear terms. Data with NC or unresolved underlying terms must not become a critical dependency of a prize model.

The downloaded STRIDE/interlive corpus may be investigated in an isolated optional experiment for action intervals or step boundaries. It must not be described or consumed as direct C1 `$interrupt$` supervision. A conversion must state the semantic gap explicitly.

The public C1 validation labels may be used for local analysis, but any run trained or tuned on them must be labeled `val-supervised`. Do not present it as held-out generalization.

## 9. Causal and Modeling Constraints

- At chunk `i`, use only the query, dialog available before `i`, and video evidence whose timestamp is not later than the end of chunk `i`.
- Respect the absolute `[start, end]` values in `video_intervals`; intervals can be short or contain gaps.
- Do not index video features by cumulative interval duration when absolute timestamps are available.
- Do not use future step boundaries, future captions, future plan updates, or full-session summaries at inference.
- Internal state may update every chunk even when the outward action is `$silent$`; clearly distinguish internal state update from user-facing interruption.
- Count all inference-time components toward the Small parameter budget. For MoE models, count total parameters, not active parameters.

## 10. Experiment Contract

Use experiment IDs of the form:

```text
YYYYMMDD_<model>_<hypothesis>_<variant>
```

Store each run under `output/experiments/<experiment_id>/`. A leaderboard-relevant run must contain or reference:

```text
README.md              hypothesis, result, conclusion
config.json            all effective parameters
command.sh             exact command, without secrets
environment.txt        Python/CUDA/package versions
code_state.txt         repo HEADs and dirty status
data_manifest.json     source revisions, hashes, license status
metrics.json           official scorer output
predictions.jsonl      complete predictions in source order
run.log                execution log
```

Also record seed, parameter count, precision, hardware, wall time, peak memory, validation split policy, and whether plan/cues are oracle, predicted, or generated from gold information.

One experiment should answer one primary question. Compare against the same frozen baseline, data split, scorer, and decoding policy. Do not change the backbone, labels, prompt, sampling, and threshold in one ablation and attribute the result to a single factor.

## 11. Evaluation and Submission

Use the official scorer in `starter_kit/run_evaluation.py`. Do not copy its metric into a divergent evaluator.

Required reporting:

- Macro F1;
- interrupt precision/recall/F1;
- silent precision/recall/F1;
- G-mean F1 as a diagnostic;
- TP/FP/TN/FN;
- predicted interrupt rate;
- per-domain/task breakdown when sample counts permit.

Prediction requirements:

- preserve the original session order; never sort by `video_path` before submission;
- emit exactly one row per input session;
- match `len(answers)` to `len(video_intervals)`;
- emit exactly `$silent$` or `$interrupt$<utterance>`;
- validate the full file, not a 50-row threshold shard;
- record the prediction artifact SHA256.

Agents do not submit to an external leaderboard or upload models without explicit user authorization. Before a real submission, re-check the live official rules because dates, limits, and packaging requirements can change.

## 12. Documentation Rules

- `CURRENT_ROUTE.md` is the only active route document.
- `C1_SPEC.md` contains task facts, not method rankings.
- `Agent.md` contains process rules plus a concise synchronized current-position summary; detailed and authoritative experiment scores belong in `CURRENT_ROUTE.md` and `reports/`.
- `reports/` contains evidence-backed results and must separate facts from next-step proposals.
- `reports/` 下所有面向汇报的 Markdown 报告必须使用中文撰写；命令、路径、代码标识符、模型名和指标名按原始技术形式保留。
- `daily_work_log.md` is append-only history; do not rewrite old entries to fit the current view.
- Archived documents may be cited as historical evidence but cannot be silently restored as current instructions.

When a route decision changes, update `CURRENT_ROUTE.md` first, then update affected links. Do not create competing `plan_final_v2_new.md` files.

## 13. Archive Policy

Archive to:

```text
/home/lanjinxin/workspace/deprecated/wearable_ai_challenge/<date>_<reason>/
```

Each archive must preserve relative paths and include a `MANIFEST.md` with original path, reason, replacement, dependencies, Git state, and restoration procedure. Archive by moving, not deleting. Do not edit archived code in place; restore it or create a new active copy.

The 2026-07-13 route-reset snapshot is documented at [MANIFEST.md](../deprecated/wearable_ai_challenge/2026-07-13_pre_pwr_reset/MANIFEST.md).

## 14. Definition of Done

A task is complete only when:

- changes stay inside assigned ownership;
- interfaces and assumptions are documented;
- targeted tests pass;
- the official scorer is used for metric claims;
- no future leakage is introduced;
- Small parameter accounting is stated for deployable systems;
- data revisions and licenses are recorded;
- complete artifacts, not partial shards, support reported numbers;
- active documentation and experiment metadata are updated;
- user changes and archived history remain intact;
- unresolved risks are reported plainly.
