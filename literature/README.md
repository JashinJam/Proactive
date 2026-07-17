# Active Literature Index

This directory contains evidence for C1 Small. It is not an execution-plan directory.

## Primary Audit

- [PWR_audit.md](papers/challenge1_proactive/PWR_audit.md): current primary audit of PWR v1, appendices, reproducibility gaps, and public artifact availability.

## Retained Method Notes

| File | Active Use |
|---|---|
| [STRIDE.md](papers/challenge1_proactive/STRIDE.md) | Boundary/sequence-denoising schema reference; not direct C1 labels |
| [ProAssist.md](papers/challenge1_proactive/ProAssist.md) | Historical baseline and data-pipeline evidence |
| [MMDuet2.md](papers/challenge1_proactive/MMDuet2.md) | Later-stage RL/timing reference, not current route |
| [R3-Streaming.md](papers/challenge1_proactive/R3-Streaming.md) | Routing and policy-collapse reference, not current route |
| [StreamPro.md](papers/challenge1_proactive/StreamPro.md) | Streaming training/reward reference, not current route |
| `StreamBridge_NeurIPS2025.pdf` | Decoupled activation architecture reference |
| `Proact-VL_ICML2026.pdf` | Compact decision-head and calibration reference |

[literature_review.md](literature_review.md) remains a historical C1 survey. Its top-level route-correction warning is binding: old method rankings and implementation priorities are not current decisions.

## Archived Literature

Track 2, Track 3, low-priority C1 PDFs, and the old bulk-download script were moved without deletion to:

```text
/home/lanjinxin/workspace/deprecated/wearable_ai_challenge/
  2026-07-13_pre_pwr_reset/project/literature/
```

See the archive `MANIFEST.md` before restoring anything. Restore only the paper needed for a concrete task; do not repopulate all tracks into the active tree.

## Adding a Source

For each new paper or repository:

1. record the primary URL, version, and access date;
2. separate paper claims from verified public artifacts;
3. state model size, training data, license, causal assumptions, and C1 label compatibility;
4. state whether it changes `CURRENT_ROUTE.md` or is only supporting evidence;
5. avoid assigning a priority before checking PWR and official-task compatibility.

