# Wearable AI Challenge: C1 Small

This workspace targets the ECCV Wearable AI Challenge proactive track, with the Small division and official Macro F1 leaderboard as the current priority.

## Start Here

- [Agent.md](Agent.md): mandatory engineering and multi-agent rules.
- [CURRENT_ROUTE.md](CURRENT_ROUTE.md): the only active technical route.
- [C1_SPEC.md](C1_SPEC.md): task, data, causality, metric, and submission facts.
- [PWR audit](literature/papers/challenge1_proactive/PWR_audit.md): primary method and reproducibility audit.
- [Active literature index](literature/README.md): retained C1 evidence.

## Active Layout

```text
src/                         new Small/PWR-inspired implementation
configs/                     experiment and model configs
starter_kit/                 local official starter-kit snapshot
/data1/...                   read-only external official data and model weights
output/                      generated run artifacts
reports/                     evidence-backed reports
literature/                  active C1 evidence
```

This checkout is a Git repository. Historical `STRIDE/` and `wearable-ai-leaderboard/` nested repositories are not present in this checkout; follow [Agent.md](Agent.md) if either is restored.

## Route Reset

The pre-PWR plans, completed 8B baselines, old ProAssist outputs, and non-C1 literature were moved without deletion to:

```text
/home/lanjinxin/workspace/deprecated/wearable_ai_challenge/
  2026-07-13_pre_pwr_reset/
```

See its [archive manifest](../deprecated/wearable_ai_challenge/2026-07-13_pre_pwr_reset/MANIFEST.md) for reasons, preserved Git state, artifact caveats, and restore instructions.
