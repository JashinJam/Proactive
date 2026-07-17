# Wearable AI Challenge Agent Guide

> Scope: `/home/lanjinxin/workspace/wearable_ai_challenge`  
> Updated: 2026-07-16  
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
4. The [official challenge rules](https://wearable-ai-workshop.github.io/challenge_rules.html) and local [starter-kit README](data/starter_kit/README.md).
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
- R0, R0-F, the four-session R1 protocol pilot, the D1 scalar control, and the D1 neural increment are complete. The active scientific baseline is the clean `fused_linear` session-level OOF result, Macro F1 `0.6341`; it is public-validation-supervised development evidence, not a hidden-test claim.
- The fused head combines 18 strictly causal scalar features, one fixed-tag margin, and a 1,024-dimensional final causal hidden state. Tag-only and hidden-only do not beat the scalar control; do not report `0.6341` as a stand-alone hidden-state result.
- One 1,044-parameter fused head has been refit on all public-development sessions and serialized. Its `0.6719` score is train-fit sanity only. The promoted `shared_vision` runner matches 127/127 frozen hidden states, margins, decisions, and answers, reduces benchmark wall time by `9.15%`, and does not increase peak memory.
- The final D1 median threshold has a formal OOF transport audit: a single threshold gives official Macro `0.6330` versus fold-specific `0.6341`, a full-precision drop of `0.00113`, and passes all preregistered robustness checks. Keep the serialized threshold; do not adopt a post-hoc threshold from the audit sweep.
- The preregistered D2 width-8 residual MLP is complete at official OOF Macro `0.6351`, only `+0.0010` over D1 with a bootstrap interval crossing zero and 3/5 positive folds. It is rejected. Do not call it the active baseline, full-refit it, or search more MLP variants on the same folds.
- The final-language-MLP LoRA engineering audit is complete. Both historical two-chunk smokes remain marked failed: v1 exposes BF16 MLP batch-shape drift, while v2 makes those two zero-adapter chunks exact but misses a non-deployed local/full margin diagnostic (`0.0105085 > 0.01`). A later four-state full-cache attempt also failed at `(input 11, chunk 4)` because final RMSNorm differed by `0.03125`. Never relabel these artifacts as passed or efficacy evidence.
- The replacement six-state label-free cache is complete over 700 sessions / 9,935 chunks. It uses same-batch adapter-enabled minus disabled corrections for the final MLP, final norm, and LM-head margin; the merged cache matches the fixed D1 hidden/margin exactly and prompt/key order exactly. Feature SHA256 is `2c4d7d4d69e54e7156404f747a3ff65cd6c6652c4623dd4d50aad9f538dd455e`.
- The single frozen rank-8 final-MLP OOF is complete. `adapted_fused_linear` obtains official Macro `0.6357`, only `+0.0016` over D1; its session-bootstrap interval `[-0.00425,+0.00756]` crosses zero and only 2/5 folds improve. It is rejected. Do not full-refit it or search rank, layer targets, learning rate, batch size, L2 range, or other LoRA variants on the same folds.
- Online serving must use the custom causal session runner because the unmodified starter `generate(frames, messages)` interface omits interval/domain metadata. Use `configs/d1_internvl35_1b_neural_deploy_shared_vision.json`; retain the sequential config as the correctness oracle. Batched and cropped-prefix-cache modes are rejected controls, not deployment candidates.
- U0 utterance auditing is complete. U1 fixed-D1-gate generation has completed a 16-chunk no-state/step/full oracle engineering smoke and a full 80-chunk no-state generation: all R0 replays and D1 decisions are exact, official Macro remains `0.6341`, but no-state produces only 56/80 non-empty continuations and visibly unreliable content. Paired human ratings and the remaining full oracle annotations are pending, so neither interface-only generation nor state has been promoted or rejected. The immediate ladder remains `complete U1 paired/oracle evaluation -> larger pre-registered oracle-state replication only if state is implicated -> granularity after repeatable state gain -> predicted/noisy state -> training-strategy decision`.
- A dedicated granularity model is justified only if coarse/medium/fine oracle plans produce a clear and repeatable metric difference.
- GRPO is not the next default step. It becomes eligible only after plan-state benefit, a measured fused-head residual, and a stable supervised baseline are established.
- STRIDE code is retained as a schema and boundary-modeling reference. Its action intervals or step boundaries are not C1 interrupt labels.
- ProAssist and LiveStar are archived 8B baselines. Do not restore them as the main route without a new user decision.

See [CURRENT_ROUTE.md](CURRENT_ROUTE.md) for gates and open decisions.

## 5. Workspace Boundaries

The umbrella directory is **not** a Git repository. Active nested repositories are independent.

| Path | Role | Write Policy |
|---|---|---|
| `src/` | New Small/PWR-inspired implementation | Primary active code area |
| `configs/` | Versioned experiment and model configs | Text configs only |
| `models/` | Small wrappers or lightweight metadata | Never store downloaded weights here |
| `output/` | Generated experiment artifacts | Never treat as source code |
| `logs/` | Runtime logs | Generated only |
| `reports/` | Reproducible experiment conclusions | One report per completed experiment/analysis |
| `literature/` | Active primary evidence and method notes | No route decisions except explicit audits |
| `STRIDE/` | Independent Git repo, branch `ljx` | Reference-first; assign one write owner before edits |
| `wearable-ai-leaderboard/` | Official leaderboard mirror, branch `main` | Read-only unless explicitly updating from upstream |
| `data` | Symlink to `/data1/wearable_ai_challenge_data` | Treat source data and starter kit as read-only |
| `../deprecated/` | Historical snapshots | Read-only; restore explicitly, never edit in place |

Do not move, rename, delete, or recursively scan large data trees merely to understand the project. In particular:

- `data -> /data1/wearable_ai_challenge_data` is external project data.
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

Use the official scorer in `data/starter_kit/run_evaluation.py`. Do not copy its metric into a divergent evaluator.

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
