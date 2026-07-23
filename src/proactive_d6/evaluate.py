"""Merge five frozen D6 OOF folds, run the official scorer, and apply gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

from proactive_d1.core import (
    LabeledChunk,
    attach_gold_labels,
    build_label_free_chunks,
    prediction_rows,
)
from proactive_d4_1.compare import paired_session_bootstrap
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer

from .contract import load_experiment


DEFAULT_CONFIG = Path("configs/d6_internvl35_1b_query_memory_lora_oof_v1.json")


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _interrupt(answer: object) -> int:
    return int(str(answer).lstrip().startswith("$interrupt$"))


def _expanded_metrics(
    examples: Sequence[LabeledChunk], decisions: Mapping[tuple[int, int], int]
) -> dict[str, float | int]:
    if not examples:
        raise ValueError("D6 stratified metric subset is empty")
    labels = [example.gold_interrupt for example in examples]
    predictions = [int(decisions[example.key]) for example in examples]
    tp = sum(gold == 1 and pred == 1 for gold, pred in zip(labels, predictions))
    fp = sum(gold == 0 and pred == 1 for gold, pred in zip(labels, predictions))
    tn = sum(gold == 0 and pred == 0 for gold, pred in zip(labels, predictions))
    fn = sum(gold == 1 and pred == 0 for gold, pred in zip(labels, predictions))

    def divide(numerator: int, denominator: int) -> float:
        return 0.0 if denominator == 0 else numerator / denominator

    interrupt_precision = divide(tp, tp + fp)
    interrupt_recall = divide(tp, tp + fn)
    silent_precision = divide(tn, tn + fn)
    silent_recall = divide(tn, tn + fp)
    interrupt_f1 = divide(2 * tp, 2 * tp + fp + fn)
    silent_f1 = divide(2 * tn, 2 * tn + fp + fn)
    return {
        "macro_f1": (interrupt_f1 + silent_f1) / 2,
        "gmean_f1": math.sqrt(interrupt_f1 * silent_f1),
        "interrupt_precision": interrupt_precision,
        "interrupt_recall": interrupt_recall,
        "interrupt_f1": interrupt_f1,
        "silent_precision": silent_precision,
        "silent_recall": silent_recall,
        "silent_f1": silent_f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "predicted_interrupt_rate": (tp + fp) / len(examples),
        "support": len(examples),
    }


def _decisions_from_predictions(
    predictions: Sequence[Mapping[str, object]],
) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for input_index, prediction in enumerate(predictions):
        answers = prediction.get("answers")
        if not isinstance(answers, list):
            raise ValueError("D6 baseline prediction row is malformed")
        for chunk_index, answer in enumerate(answers):
            result[(input_index, chunk_index)] = _interrupt(answer)
    return result


def _frozen_fold_decisions(
    experiment_dir: Path,
    config_sha256: str,
    filename: str,
) -> tuple[dict[tuple[int, int], int], list[dict[str, object]]]:
    decisions: dict[tuple[int, int], int] = {}
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        fold_dir = experiment_dir / "folds" / f"fold_{fold}"
        summary = _load_object(fold_dir / "fold_summary.json")
        if summary.get("kind") != "d6_formal_oof_fold" or summary.get("status") != "complete":
            raise ValueError(f"D6 fold {fold} is incomplete")
        if summary.get("fold") != fold or summary.get("config_sha256") != config_sha256:
            raise ValueError(f"D6 fold {fold} identity changed")
        summaries.append(summary)
        for row in load_jsonl(fold_dir / filename):
            if row.get("gold_interrupt") != "SENTINEL_UNSEALED":
                raise ValueError("D6 frozen fold prediction contains unsealed gold")
            key = (int(row["input_index"]), int(row["chunk_index"]))
            if key in decisions:
                raise ValueError(f"D6 duplicate OOF prediction: {key}")
            decisions[key] = int(row["predicted_interrupt"])
    return decisions, summaries


def _stratified(
    examples: Sequence[LabeledChunk],
    candidate: Mapping[tuple[int, int], int],
    baseline: Mapping[tuple[int, int], int],
    source_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    def groups(
        key: Callable[[LabeledChunk], str | None]
    ) -> dict[str, dict[str, object]]:
        by_key: dict[str, list[LabeledChunk]] = {}
        for example in examples:
            value = key(example)
            if value is not None:
                by_key.setdefault(value, []).append(example)
        return {
            value: {
                "candidate": _expanded_metrics(selected, candidate),
                "baseline": _expanded_metrics(selected, baseline),
                "delta_macro_f1": float(
                    _expanded_metrics(selected, candidate)["macro_f1"]
                )
                - float(_expanded_metrics(selected, baseline)["macro_f1"]),
            }
            for value, selected in sorted(by_key.items())
        }

    def previous(example: LabeledChunk) -> str | None:
        chunk_index = example.feature.chunk_index
        if chunk_index == 0:
            return None
        answers = source_rows[example.feature.input_index]["answers"]
        return "previous_interrupt" if _interrupt(answers[chunk_index - 1]) else "previous_silent"  # type: ignore[index]

    return {
        "fold": groups(lambda example: str(example.feature.fold)),
        "domain": groups(lambda example: example.feature.domain),
        "previous_response": groups(previous),
        "chunk_position": groups(
            lambda example: "first" if example.feature.chunk_index == 0 else "non_first"
        ),
    }


def _bootstrap_rows(
    examples: Sequence[LabeledChunk], decisions: Mapping[tuple[int, int], int]
) -> list[dict[str, object]]:
    return [
        {
            "input_index": example.feature.input_index,
            "chunk_index": example.feature.chunk_index,
            "gold_interrupt": example.gold_interrupt,
            "predicted_interrupt": int(decisions[example.key]),
        }
        for example in examples
    ]


def _decision_changes(
    examples: Sequence[LabeledChunk],
    candidate: Mapping[tuple[int, int], int],
    baseline: Mapping[tuple[int, int], int],
) -> dict[str, int]:
    result = {
        "unchanged_correct": 0,
        "unchanged_incorrect": 0,
        "corrected_errors": 0,
        "new_errors": 0,
        "changed_decisions": 0,
        "interrupt_to_silent": 0,
        "silent_to_interrupt": 0,
    }
    for example in examples:
        gold = example.gold_interrupt
        candidate_correct = int(candidate[example.key]) == gold
        baseline_correct = int(baseline[example.key]) == gold
        if candidate_correct and baseline_correct:
            result["unchanged_correct"] += 1
        elif not candidate_correct and not baseline_correct:
            result["unchanged_incorrect"] += 1
        elif candidate_correct:
            result["corrected_errors"] += 1
        else:
            result["new_errors"] += 1
        if int(candidate[example.key]) != int(baseline[example.key]):
            result["changed_decisions"] += 1
            if int(baseline[example.key]) == 1:
                result["interrupt_to_silent"] += 1
            else:
                result["silent_to_interrupt"] += 1
    return result


def _score(
    *,
    starter_dir: Path,
    input_path: Path,
    predictions_path: Path,
    metrics_path: Path,
) -> dict[str, object]:
    log_path = metrics_path.with_suffix(".log")
    _run_official_scorer(
        starter_dir,
        input_path,
        predictions_path,
        metrics_path,
        log_path,
    )
    return _load_object(metrics_path)


def _diagnostic_summary(
    *,
    name: str,
    decisions: Mapping[tuple[int, int], int],
    examples: Sequence[LabeledChunk],
    output_dir: Path,
    inputs: object,
) -> dict[str, object]:
    predictions = prediction_rows(examples, dict(decisions))
    validate_prediction_rows(inputs.source_rows, predictions)  # type: ignore[attr-defined]
    predictions_path = output_dir / f"{name}_oof_predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    metrics = _score(
        starter_dir=inputs.starter_dir,  # type: ignore[attr-defined]
        input_path=inputs.input_path,  # type: ignore[attr-defined]
        predictions_path=predictions_path,
        metrics_path=output_dir / f"{name}_oof_metrics.json",
    )
    return {
        "selection_eligible": False,
        "head_refit": False,
        "overall": metrics["overall"],
        "predictions_sha256": sha256_file(predictions_path),
    }


def _write_report(path: Path, summary: Mapping[str, object]) -> None:
    overall = summary["overall"]
    gates = summary["promotion_gates"]
    lines = [
        "# D6 主干条件历史注入五折 OOF 报告",
        "",
        "> 本报告是公开 validation 上的 post-selection、val-supervised session OOF；不是 hidden-test 或独立泛化证据。",
        "",
        "## 结论",
        "",
        f"- D6 Macro-F1: `{overall['macro_f1']:.4f}`，D4.2 history8: `{summary['baseline_overall']['macro_f1']:.4f}`，差值 `{summary['delta_macro_f1']:+.4f}`。",
        f"- G-mean F1: `{overall['gmean_f1']:.4f}`。",
        f"- 晋升判定: `{'PASS' if summary['promotion_passed'] else 'FAIL'}`。",
        "- 外部上传仍未授权，本实验未产生任何外部提交。",
        "",
        "## 硬门",
        "",
    ]
    lines.extend(f"- `{name}`: `{'PASS' if passed else 'FAIL'}`" for name, passed in gates.items())
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "LoRA-disabled 与 memory-disabled 是冻结 test-only 诊断，复用主 head 且不参与候选选择。若任一晋升门失败，本结构族按预注册规则终止，不在相同 folds 上继续搜索 rank、层数、宽度或学习率。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(experiment_dir: Path, config_path: Path) -> dict[str, object]:
    inputs = load_experiment(config_path)
    experiment_dir = experiment_dir.resolve()
    output_dir = experiment_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    zero_init = _load_object(experiment_dir / "smokes" / "zero_init" / "summary.json")
    trainability = _load_object(
        experiment_dir / "smokes" / "rotation_0_trainability" / "summary.json"
    )
    for smoke, kind in (
        (zero_init, "d6_zero_init_causality_smoke"),
        (trainability, "d6_rotation0_trainability_smoke"),
    ):
        if smoke.get("kind") != kind or smoke.get("config_sha256") != inputs.config_sha256:
            raise ValueError(f"D6 smoke identity changed: {kind}")
        smoke_gates = smoke.get("gates")
        if not isinstance(smoke_gates, Mapping) or not all(smoke_gates.values()):
            raise ValueError(f"D6 smoke gate did not pass: {kind}")

    candidate, fold_summaries = _frozen_fold_decisions(
        experiment_dir, inputs.config_sha256, "primary_test_predictions_frozen.jsonl"
    )
    expected_keys = {
        (input_index, chunk_index)
        for input_index, row in enumerate(inputs.answer_free_rows)
        for chunk_index in range(len(row["video_intervals"]))
    }
    if set(candidate) != expected_keys or len(candidate) != 9935:
        raise ValueError("D6 OOF decisions do not cover 700 sessions / 9,935 chunks")

    records = [inputs.references[index] for index in range(len(inputs.references))]
    label_free = build_label_free_chunks(
        inputs.answer_free_rows,
        records,
        inputs.fold_by_index,
        max_history_turns=8,
        max_model_frames=32,
    )
    examples = attach_gold_labels(label_free, inputs.source_rows)
    predictions = prediction_rows(examples, candidate)
    validate_prediction_rows(inputs.source_rows, predictions)
    predictions_path = output_dir / "oof_predictions.jsonl"
    write_jsonl(predictions_path, predictions)
    official = _score(
        starter_dir=inputs.starter_dir,
        input_path=inputs.input_path,
        predictions_path=predictions_path,
        metrics_path=output_dir / "oof_metrics.json",
    )

    reference = inputs.config["d4_2_reference"]
    baseline_path = Path(str(reference["history8_oof_predictions"])).resolve()  # type: ignore[index]
    if sha256_file(baseline_path) != str(reference["history8_oof_predictions_sha256"]):  # type: ignore[index]
        raise ValueError("D6 baseline OOF prediction SHA256 changed")
    baseline_predictions = load_jsonl(baseline_path)
    validate_prediction_rows(inputs.source_rows, baseline_predictions)
    baseline = _decisions_from_predictions(baseline_predictions)
    if set(baseline) != expected_keys:
        raise ValueError("D6 baseline decisions do not cover every chunk")
    baseline_metrics = _expanded_metrics(examples, baseline)
    if round(float(baseline_metrics["macro_f1"]), 4) != float(reference["history8_macro_f1"]):  # type: ignore[index]
        raise ValueError("D6 did not reproduce the frozen D4.2 history8 baseline")

    internal = _expanded_metrics(examples, candidate)
    overall = official["overall"]
    if round(float(internal["macro_f1"]), 4) != float(overall["macro_f1"]):  # type: ignore[index]
        raise ValueError("D6 internal and official Macro-F1 differ")
    bootstrap = paired_session_bootstrap(
        _bootstrap_rows(examples, candidate),
        _bootstrap_rows(examples, baseline),
        repetitions=int(inputs.config["evaluation"]["bootstrap_repetitions"]),  # type: ignore[index]
        seed=int(inputs.config["evaluation"]["bootstrap_seed"]),  # type: ignore[index]
    )
    stratified = _stratified(examples, candidate, baseline, inputs.source_rows)
    positive_folds = sum(
        float(value["delta_macro_f1"]) > 0
        for value in stratified["fold"].values()  # type: ignore[union-attr]
    )
    positive_domains = sum(
        float(value["delta_macro_f1"]) > 0
        for value in stratified["domain"].values()  # type: ignore[union-attr]
    )
    evaluation = inputs.config["evaluation"]
    delta = float(overall["macro_f1"]) - float(reference["history8_macro_f1"])  # type: ignore[index]
    resource_passed = all(
        all(summary["resource_gates"].values())  # type: ignore[union-attr]
        for summary in fold_summaries
    )
    previous = stratified["previous_response"]
    positions = stratified["chunk_position"]
    gates = {
        "macro_gain_at_least_0_005": delta >= float(evaluation["minimum_macro_gain"]),  # type: ignore[index]
        "bootstrap_lower_bound_strictly_positive": float(
            bootstrap["delta_macro_f1_p2_5"]
        )
        > 0,
        "at_least_4_of_5_positive_folds": positive_folds
        >= int(evaluation["minimum_positive_folds"]),  # type: ignore[index]
        "at_least_3_of_4_positive_domains": positive_domains
        >= int(evaluation["minimum_positive_domains"]),  # type: ignore[index]
        "previous_interrupt_non_decrease": float(
            previous["previous_interrupt"]["delta_macro_f1"]  # type: ignore[index]
        )
        >= 0,
        "previous_silent_non_decrease": float(
            previous["previous_silent"]["delta_macro_f1"]  # type: ignore[index]
        )
        >= 0,
        "non_first_chunk_non_decrease": float(
            positions["non_first"]["delta_macro_f1"]  # type: ignore[index]
        )
        >= 0,
        "both_classes_non_degenerate": float(overall["interrupt_f1"]) > 0  # type: ignore[index]
        and float(overall["silent_f1"]) > 0,  # type: ignore[index]
        "all_fold_resource_gates_pass": resource_passed,
        "zero_init_and_causality_smoke_pass": all(zero_init["gates"].values()),  # type: ignore[union-attr]
        "trainability_and_48_hour_estimate_pass": all(
            trainability["gates"].values()  # type: ignore[union-attr]
        ),
        "parameter_limit_pass": int(inputs.config["parameter_accounting"]["total"])  # type: ignore[index]
        <= int(inputs.config["parameter_accounting"]["small_limit"]),  # type: ignore[index]
    }

    diagnostics: dict[str, object] = {}
    for name in ("lora_disabled", "memory_disabled"):
        decisions, _ = _frozen_fold_decisions(
            experiment_dir,
            inputs.config_sha256,
            f"{name}_test_predictions_frozen.jsonl",
        )
        if set(decisions) != expected_keys:
            raise ValueError(f"D6 {name} diagnostic coverage changed")
        diagnostics[name] = _diagnostic_summary(
            name=name,
            decisions=decisions,
            examples=examples,
            output_dir=output_dir,
            inputs=inputs,
        )

    summary = {
        "schema_version": 1,
        "kind": "d6_five_fold_oof_evaluation",
        "status": "complete",
        "classification": "post-selection val-supervised session OOF; not hidden-test evidence",
        "config_sha256": inputs.config_sha256,
        "sessions": 700,
        "chunks": 9935,
        "overall": overall,
        "internal_unrounded": internal,
        "baseline_overall": baseline_metrics,
        "delta_macro_f1": delta,
        "paired_session_bootstrap": bootstrap,
        "decision_changes_vs_baseline": _decision_changes(
            examples, candidate, baseline
        ),
        "positive_folds": positive_folds,
        "positive_domains": positive_domains,
        "stratified": stratified,
        "fixed_diagnostics": diagnostics,
        "fold_summaries": fold_summaries,
        "smokes": {
            "zero_init_causality": zero_init,
            "rotation_0_trainability": trainability,
        },
        "promotion_gates": gates,
        "promotion_passed": all(gates.values()),
        "family_action": "full_development_refit_required"
        if all(gates.values())
        else "terminate_structure_family_without_post_hoc_variants",
        "external_upload_authorized": False,
        "artifacts": {
            "oof_predictions_sha256": sha256_file(predictions_path),
            "oof_metrics_sha256": sha256_file(output_dir / "oof_metrics.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)
    _write_report(output_dir / "report_zh.md", summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args(argv)
    result = evaluate(Path(args.experiment_dir), Path(args.config))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
