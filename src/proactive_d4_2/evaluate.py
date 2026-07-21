"""Merge D4.2 features, run five-fold OOF, and refit the winning policy."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from proactive_d1.core import (
    LinearDecisionHead,
    LabeledChunk,
    attach_gold_labels,
    binary_metrics,
    build_label_free_chunks,
    feature_names,
    fit_linear_logistic,
    load_decision_head,
    predict_logits,
    prediction_rows,
    serialize_decision_head,
    strip_answers,
    validate_fold_manifest,
)
from proactive_d1.neural_core import (
    NeuralFeatureCache,
    cross_validate_neural_matrix,
    load_aligned_neural_cache,
)
from proactive_d3.dialog_control_core import (
    DIALOG_POLICY_NAMES,
    build_dialog_policy_features,
    dialog_control_matrix,
)
from proactive_d4_1.compare import (
    decision_change_statistics,
    flatten_decisions,
    load_official_scorer,
    paired_session_bootstrap,
    stratified_statistics,
    timing_statistics,
)
from proactive_d4_1.core import canonical_json, object_sha256
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer

from .core import (
    PolicyParameters,
    load_candidates,
    partition_indices,
    rank_summaries,
    validate_feature_records,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def length_quartiles(rows: Sequence[Mapping[str, object]]) -> dict[int, int]:
    """Assign within-domain length quartiles without labels."""
    if any("answers" in row for row in rows):
        raise ValueError("D4.2 length strata require answer-stripped rows")
    by_domain: dict[str, list[int]] = {}
    for input_index, row in enumerate(rows):
        by_domain.setdefault(str(row.get("domain")), []).append(input_index)
    result: dict[int, int] = {}
    for domain in sorted(by_domain):
        indices = sorted(
            by_domain[domain],
            key=lambda index: (
                len(rows[index]["video_intervals"]),  # type: ignore[arg-type,index]
                str(rows[index].get("video_path")),
            ),
        )
        for rank, input_index in enumerate(indices):
            result[input_index] = min(3, rank * 4 // len(indices)) + 1
    if set(result) != set(range(len(rows))):
        raise ValueError("D4.2 length strata do not cover every session")
    return result


def merge_candidate_records(
    *,
    experiment_dir: Path,
    candidate: Mapping[str, object],
    manifest: Mapping[str, object],
    rows: Sequence[dict[str, object]],
    num_shards: int,
    hidden_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shards = partition_indices(manifest, num_shards)
    records: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    candidate_id = str(candidate["candidate_id"])
    for shard_id, indices in enumerate(shards):
        shard_dir = (
            experiment_dir / "features" / candidate_id / f"shard_{shard_id:03d}"
        )
        status = _load_object(shard_dir / "status.json")
        if status.get("status") != "complete":
            raise ValueError(f"D4.2 feature shard is incomplete: {shard_dir}")
        shard_records = load_jsonl(shard_dir / "session_records.jsonl")
        validate_feature_records(
            shard_records,
            indices,
            rows,
            hidden_size=hidden_size,
            require_complete=True,
        )
        records.extend(shard_records)
        runtimes.append(_load_object(shard_dir / "runtime.json"))
    records.sort(key=lambda record: int(record["input_index"]))
    validate_feature_records(
        records,
        list(range(len(rows))),
        rows,
        hidden_size=hidden_size,
        require_complete=True,
    )
    return records, runtimes


def records_for_scalar_features(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Project D4 records to the R0 record schema expected by D1 feature code."""
    projected: list[dict[str, object]] = []
    for record in records:
        chunks = record.get("chunks")
        if not isinstance(chunks, list):
            raise ValueError("D4.2 feature record has no chunks")
        r0_answers: list[str] = []
        projected_chunks: list[dict[str, object]] = []
        for chunk in chunks:
            if not isinstance(chunk, Mapping):
                raise ValueError("D4.2 feature chunk is malformed")
            r0_answers.append(str(chunk["r0_answer"]))
            projected_chunks.append(
                {
                    "chunk_index": int(chunk["chunk_index"]),
                    "interval": list(chunk["interval"]),  # type: ignore[arg-type]
                    "model_input_frames": int(chunk["model_input_frames"]),
                    "raw_response": str(chunk["raw_response"]),
                }
            )
        projected.append(
            {
                "input_index": int(record["input_index"]),
                "video_path": str(record["video_path"]),
                "prediction": {
                    "video_path": str(record["video_path"]),
                    "answers": r0_answers,
                },
                "chunks": projected_chunks,
            }
        )
    return projected


def feature_arrays(
    records: Sequence[Mapping[str, object]], hidden_size: int
) -> dict[str, np.ndarray]:
    hidden: list[list[float]] = []
    tag_margin: list[float] = []
    silent: list[float] = []
    interrupt: list[float] = []
    prompt_tokens: list[int] = []
    input_indices: list[int] = []
    chunk_indices: list[int] = []
    for record in records:
        input_index = int(record["input_index"])
        for chunk in record["chunks"]:  # type: ignore[index]
            if not isinstance(chunk, Mapping):
                raise ValueError("D4.2 feature chunk is malformed")
            vector = [float(value) for value in chunk["hidden_state"]]  # type: ignore[index]
            if len(vector) != hidden_size:
                raise ValueError("D4.2 hidden width changed while merging")
            hidden.append(vector)
            tag_margin.append(float(chunk["tag_margin"]))
            silent.append(float(chunk["silent_log_probability"]))
            interrupt.append(float(chunk["interrupt_log_probability"]))
            prompt_tokens.append(int(chunk["prompt_tokens"]))
            input_indices.append(input_index)
            chunk_indices.append(int(chunk["chunk_index"]))
    arrays = {
        "hidden_state": np.asarray(hidden, dtype=np.float32),
        "tag_margin": np.asarray(tag_margin, dtype=np.float32),
        "silent_log_probability": np.asarray(silent, dtype=np.float32),
        "interrupt_log_probability": np.asarray(interrupt, dtype=np.float32),
        "prompt_tokens": np.asarray(prompt_tokens, dtype=np.int32),
        "input_index": np.asarray(input_indices, dtype=np.int32),
        "chunk_index": np.asarray(chunk_indices, dtype=np.int32),
    }
    if arrays["hidden_state"].shape[1:] != (hidden_size,):
        raise ValueError("D4.2 merged hidden matrix has the wrong shape")
    return arrays


def write_feature_cache(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    temporary.replace(path)


def build_candidate_matrix(
    *,
    source_rows: Sequence[dict[str, object]],
    answer_free_rows: Sequence[dict[str, object]],
    records: Sequence[Mapping[str, object]],
    fold_by_index: dict[int, int],
    candidate: Mapping[str, object],
    cache_path: Path,
    config: Mapping[str, object],
) -> tuple[list[LabeledChunk], np.ndarray, tuple[str, ...], dict[str, object]]:
    parameters = PolicyParameters.from_mapping(
        candidate["parameters"]  # type: ignore[arg-type]
    )
    label_free = build_label_free_chunks(
        answer_free_rows,
        records_for_scalar_features(records),
        fold_by_index,
        max_history_turns=parameters.max_history_turns,
        max_model_frames=parameters.max_frames,
    )
    dialog_values, dialog_audit = build_dialog_policy_features(
        answer_free_rows, label_free
    )
    examples = attach_gold_labels(label_free, source_rows)
    hidden_size = int(config["features"]["hidden_size"])  # type: ignore[index]
    cache = load_aligned_neural_cache(cache_path, examples, hidden_size)
    scalar_names = feature_names(
        str(config["features"]["scalar_variant"]),  # type: ignore[index]
        sorted({feature.domain for feature in label_free}),
    )
    values, names = dialog_control_matrix(
        examples,
        cache,
        scalar_names,
        dialog_values,
        str(config["features"]["dialog_variant"]),  # type: ignore[index]
    )
    if len(names) != int(config["features"]["feature_count"]):  # type: ignore[index]
        raise ValueError("D4.2 feature count differs from the frozen protocol")
    return examples, values, names, dialog_audit


def _run_scorer(
    *,
    starter_dir: Path,
    input_path: Path,
    predictions_path: Path,
    metrics_path: Path,
) -> dict[str, object]:
    _run_official_scorer(
        starter_dir,
        input_path,
        predictions_path,
        metrics_path,
        metrics_path.with_name("scorer.log"),
    )
    return _load_object(metrics_path)


def _write_comparison_csv(path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    fields = [
        "rank",
        "name",
        "candidate_id",
        "max_frames",
        "frames_per_interval",
        "max_history_turns",
        "max_new_tokens",
        "macro_f1",
        "gmean_f1",
        "interrupt_precision",
        "interrupt_recall",
        "interrupt_f1",
        "silent_precision",
        "silent_recall",
        "silent_f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "predicted_interrupt_rate",
        "delta_macro_f1_vs_baseline",
        "total_model_inference_seconds",
        "deployable",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, summary in enumerate(rank_summaries(summaries), start=1):
            overall = summary["overall"]
            timing = summary["timing"]
            writer.writerow(
                {
                    "rank": rank,
                    "name": summary["name"],
                    "candidate_id": summary["candidate_id"],
                    **summary["parameters"],
                    **{name: overall[name] for name in fields if name in overall},
                    "delta_macro_f1_vs_baseline": summary[
                        "delta_macro_f1_vs_baseline"
                    ],
                    "total_model_inference_seconds": timing[
                        "total_model_inference_seconds"
                    ],
                    "deployable": summary["deployable"],
                }
            )
    temporary.replace(path)


def _write_report(
    path: Path,
    ranking: Sequence[Mapping[str, object]],
    best: Mapping[str, object],
    final: Mapping[str, object],
) -> None:
    lines = [
        "# D4.2 输入策略适配五折 OOF 报告",
        "",
        "本实验在每个输入策略各自生成的 causal 特征上重新训练并校准完整 D4 线性头。候选由 D4.1 public-validation 结果后验选定，因此属于 val-supervised 机制诊断，不是 hidden-test 或独立泛化证据。",
        "",
        "## 五折结果",
        "",
        "| Rank | 配置 | 参数 `(frames, per_interval, history, tokens)` | Macro F1 | G-mean F1 | 相对基线 | Model seconds |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, summary in enumerate(ranking, start=1):
        parameters = summary["parameters"]
        policy = (
            parameters["max_frames"],
            parameters["frames_per_interval"],
            parameters["max_history_turns"],
            parameters["max_new_tokens"],
        )
        lines.append(
            f"| {rank} | `{summary['name']}` | `{policy}` | "
            f"{summary['overall']['macro_f1']:.4f} | "
            f"{summary['overall']['gmean_f1']:.4f} | "
            f"{summary['delta_macro_f1_vs_baseline']:+.4f} | "
            f"{summary['timing']['total_model_inference_seconds']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 最佳配置与全量拟合",
            "",
            f"OOF 最佳配置为 `{best['name']}`，参数 `{canonical_json(best['parameters'])}`，Macro F1 `{best['official_metrics']['macro_f1']:.4f}`。",
            f"在全部 700 sessions 上拟合后的训练闭环 Macro F1 为 `{final['train_fit_official']['macro_f1']:.4f}`；该数值只用于 train-fit sanity，不估计泛化性能。",
            "",
            "冻结的 D4 head、D4 配置和 submission 均未修改。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_experiment(experiment_dir: Path) -> dict[str, object]:
    experiment_dir = experiment_dir.expanduser().resolve()
    config = _load_object(experiment_dir / "config.json")
    config_sha256 = object_sha256(config)
    identity = _load_object(experiment_dir / "experiment_identity.json")
    if identity.get("experiment_config_sha256") != config_sha256:
        raise ValueError("D4.2 evaluation found a changed experiment config")
    plan = _load_object(experiment_dir / "feature_plan.json")
    if plan.get("experiment_config_sha256") != config_sha256:
        raise ValueError("D4.2 feature plan uses a different config")

    data = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    training = dict(config["training"])  # type: ignore[arg-type]
    evaluation = dict(config["evaluation"])  # type: ignore[arg-type]
    input_path = _resolve(data["input"])
    starter_dir = _resolve(starter_config["path"])
    source_rows = load_jsonl(input_path)
    answer_free_rows = strip_answers(source_rows)
    manifest = _load_object(experiment_dir / "source_manifest.json")
    fold_manifest = _load_object(experiment_dir / "fold_manifest.json")
    fold_by_index = validate_fold_manifest(fold_manifest, answer_free_rows)
    quartiles = length_quartiles(answer_free_rows)
    scorer = load_official_scorer(
        starter_dir, str(starter_config["scorer_py_sha256"])
    )
    candidates = load_candidates(config)
    hidden_size = int(config["features"]["hidden_size"])  # type: ignore[index]
    num_shards = int(plan["num_shards"])

    summaries: list[dict[str, object]] = []
    flattened_by_id: dict[str, list[dict[str, object]]] = {}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        candidate_dir = experiment_dir / "candidates" / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        records, runtimes = merge_candidate_records(
            experiment_dir=experiment_dir,
            candidate=candidate,
            manifest=manifest,
            rows=answer_free_rows,
            num_shards=num_shards,
            hidden_size=hidden_size,
        )
        arrays = feature_arrays(records, hidden_size)
        cache_path = candidate_dir / "features.npz"
        write_feature_cache(cache_path, arrays)
        examples, values, names, dialog_audit = build_candidate_matrix(
            source_rows=source_rows,
            answer_free_rows=answer_free_rows,
            records=records,
            fold_by_index=fold_by_index,
            candidate=candidate,
            cache_path=cache_path,
            config=config,
        )
        decisions, fold_details = cross_validate_neural_matrix(
            examples,
            values,
            names,
            folds=int(config["folds"]["count"]),  # type: ignore[index]
            calibration_fold_offset=int(
                config["folds"]["calibration_fold_offset"]  # type: ignore[index]
            ),
            seed=int(training["seed"]),
            max_iterations=int(training["max_iterations"]),
            l2_weights=[float(value) for value in training["l2_weights"]],  # type: ignore[index]
            l2_reduction=str(training["l2_reduction"]),  # type: ignore[arg-type]
        )
        predictions = prediction_rows(examples, decisions)
        validate_prediction_rows(source_rows, predictions)
        predictions_path = candidate_dir / "oof_predictions.jsonl"
        metrics_path = candidate_dir / "oof_metrics.json"
        write_jsonl(predictions_path, predictions)
        metrics = _run_scorer(
            starter_dir=starter_dir,
            input_path=input_path,
            predictions_path=predictions_path,
            metrics_path=metrics_path,
        )
        flattened = flatten_decisions(source_rows, predictions, list(range(len(source_rows))), quartiles)
        flattened_by_id[candidate_id] = flattened
        timing = timing_statistics(
            records,
            runtimes,
            session_limit_seconds=float(
                evaluation["per_session_model_inference_limit_seconds"]
            ),
        )
        summary = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "name": candidate["name"],
            "mechanism": candidate["mechanism"],
            "parameters": candidate["parameters"],
            "is_baseline": candidate["is_baseline"],
            "overall": metrics["overall"],
            "timing": timing,
            "deployable": timing["deployable"],
            "sessions": len(records),
            "chunks": len(examples),
            "feature_count": len(names),
            "head_parameters": len(names) + 1,
            "fold_details": fold_details,
            "stratified": stratified_statistics(scorer, flattened),
            "dialog_feature_audit": dialog_audit,
            "feature_cache_sha256": sha256_file(cache_path),
            "oof_predictions_sha256": sha256_file(predictions_path),
            "oof_metrics_sha256": sha256_file(metrics_path),
        }
        write_json(candidate_dir / "oof_summary.json", summary)
        summaries.append(summary)

    baselines = [summary for summary in summaries if summary["is_baseline"]]
    if len(baselines) != 1:
        raise ValueError("D4.2 evaluation requires exactly one baseline")
    baseline = baselines[0]
    expected_macro = float(evaluation["baseline_oof_macro_f1"])
    if float(baseline["overall"]["macro_f1"]) != expected_macro:  # type: ignore[index]
        raise ValueError(
            "D4.2 baseline OOF did not reproduce D3-D: "
            f"{baseline['overall']['macro_f1']} != {expected_macro}"  # type: ignore[index]
        )
    if baseline["oof_predictions_sha256"] != evaluation["baseline_oof_predictions_sha256"]:
        raise ValueError("D4.2 baseline OOF prediction SHA256 did not reproduce D3-D")
    if baseline["oof_metrics_sha256"] != evaluation["baseline_oof_metrics_sha256"]:
        raise ValueError("D4.2 baseline OOF metrics SHA256 did not reproduce D3-D")

    baseline_rows = flattened_by_id[str(baseline["candidate_id"])]
    for summary in summaries:
        candidate_rows = flattened_by_id[str(summary["candidate_id"])]
        summary["delta_macro_f1_vs_baseline"] = round(
            float(summary["overall"]["macro_f1"])  # type: ignore[index]
            - float(baseline["overall"]["macro_f1"]),  # type: ignore[index]
            6,
        )
        summary["decision_changes_vs_baseline"] = decision_change_statistics(
            candidate_rows, baseline_rows
        )
        summary["session_bootstrap_vs_baseline"] = paired_session_bootstrap(
            candidate_rows,
            baseline_rows,
            repetitions=int(evaluation["bootstrap_repetitions"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
        write_json(
            experiment_dir / "candidates" / str(summary["candidate_id"]) / "oof_summary.json",
            summary,
        )

    ranking = list(rank_summaries(summaries))
    deployable = [summary for summary in ranking if summary["deployable"]]
    if not deployable:
        raise ValueError("D4.2 has no candidate under the 300-second session limit")
    winner = deployable[0]
    winner_id = str(winner["candidate_id"])
    winner_candidate = next(
        candidate for candidate in candidates if candidate["candidate_id"] == winner_id
    )

    winner_records, _ = merge_candidate_records(
        experiment_dir=experiment_dir,
        candidate=winner_candidate,
        manifest=manifest,
        rows=answer_free_rows,
        num_shards=num_shards,
        hidden_size=hidden_size,
    )
    winner_cache = experiment_dir / "candidates" / winner_id / "features.npz"
    examples, values, names, _ = build_candidate_matrix(
        source_rows=source_rows,
        answer_free_rows=answer_free_rows,
        records=winner_records,
        fold_by_index=fold_by_index,
        candidate=winner_candidate,
        cache_path=winner_cache,
        config=config,
    )
    fold_details = winner["fold_details"]
    selected_l2 = [float(detail["selected_l2_weight"]) for detail in fold_details]  # type: ignore[index]
    thresholds = [float(detail["threshold_logit"]) for detail in fold_details]  # type: ignore[index]
    final_l2 = statistics.median(selected_l2)
    final_threshold = statistics.median(thresholds)
    labels = [example.gold_interrupt for example in examples]
    model = fit_linear_logistic(
        values,
        labels,
        seed=int(training["seed"]),
        max_iterations=int(training["max_iterations"]),
        l2_weight=final_l2,
        l2_reduction=str(training["l2_reduction"]),  # type: ignore[arg-type]
    )
    head = LinearDecisionHead(names, model, final_threshold)
    final_dir = experiment_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    head_path = final_dir / "decision_head.json"
    write_json(
        head_path,
        serialize_decision_head(
            head,
            {
                "experiment_id": config["experiment_id"],
                "classification": config["validation_policy"]["classification"],  # type: ignore[index]
                "candidate_id": winner_id,
                "candidate_name": winner["name"],
                "parameters": winner["parameters"],
                "mechanism": winner["mechanism"],
                "fit_sessions": len(source_rows),
                "fit_chunks": len(examples),
                "feature_variant": config["features"]["dialog_variant"],  # type: ignore[index]
                "selected_l2_by_fold": selected_l2,
                "calibration_threshold_by_fold": thresholds,
                "l2_weight": final_l2,
                "threshold_full_fit_predictions_used": False,
                "source_oof_predictions_sha256": winner["oof_predictions_sha256"],
            },
        ),
    )
    loaded = load_decision_head(_load_object(head_path))
    logits = predict_logits(loaded.model, values)
    final_decisions = {
        example.key: int(logit >= loaded.threshold_logit)
        for example, logit in zip(examples, logits)
    }
    final_predictions = prediction_rows(examples, final_decisions)
    validate_prediction_rows(source_rows, final_predictions)
    final_predictions_path = final_dir / "train_fit_predictions.jsonl"
    write_jsonl(final_predictions_path, final_predictions)
    write_jsonl(
        final_dir / "train_fit_records.jsonl",
        [
            {
                "input_index": example.feature.input_index,
                "video_path": example.feature.video_path,
                "chunk_index": example.feature.chunk_index,
                "logit": float(logit),
                "threshold_logit": final_threshold,
                "predicted_interrupt": final_decisions[example.key],
                "gold_interrupt": example.gold_interrupt,
            }
            for example, logit in zip(examples, logits)
        ],
    )
    final_metrics_path = final_dir / "train_fit_metrics.json"
    final_metrics = _run_scorer(
        starter_dir=starter_dir,
        input_path=input_path,
        predictions_path=final_predictions_path,
        metrics_path=final_metrics_path,
    )
    final = {
        "schema_version": 1,
        "status": "complete winner all-public-development refit",
        "classification": "train-fit sanity only; not held-out performance",
        "candidate_id": winner_id,
        "candidate_name": winner["name"],
        "parameters": winner["parameters"],
        "feature_count": len(names),
        "head_parameters": len(names) + 1,
        "l2_weight": final_l2,
        "selected_l2_by_fold": selected_l2,
        "threshold_logit": final_threshold,
        "calibration_threshold_by_fold": thresholds,
        "train_fit_internal": binary_metrics(
            labels, [final_decisions[example.key] for example in examples]
        ),
        "train_fit_official": final_metrics["overall"],
        "head_sha256": sha256_file(head_path),
        "train_fit_predictions_sha256": sha256_file(final_predictions_path),
    }
    write_json(final_dir / "summary.json", final)

    best = {
        "schema_version": 1,
        "claim": "D4.2 public-validation policy-matched OOF best",
        "classification": config["validation_policy"]["classification"],  # type: ignore[index]
        "candidate_id": winner_id,
        "name": winner["name"],
        "mechanism": winner["mechanism"],
        "parameters": winner["parameters"],
        "rank": 1,
        "official_metrics": winner["overall"],
        "delta_macro_f1_vs_baseline": winner["delta_macro_f1_vs_baseline"],
        "session_bootstrap_vs_baseline": winner["session_bootstrap_vs_baseline"],
        "deployable_under_300_second_limit": winner["deployable"],
        "experiment_config_sha256": config_sha256,
        "final_head_path": str(head_path),
        "final_head_sha256": final["head_sha256"],
        "train_fit_official": final["train_fit_official"],
    }
    comparison = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "experiment_config_sha256": config_sha256,
        "classification": config["validation_policy"]["classification"],  # type: ignore[index]
        "ranking_rule": config["evaluation"]["tie_break"],  # type: ignore[index]
        "baseline_reproduced": True,
        "ranking": ranking,
        "winner": best,
        "final_refit": final,
    }
    write_json(experiment_dir / "comparison.json", comparison)
    write_json(experiment_dir / "best_policy.json", best)
    _write_comparison_csv(experiment_dir / "comparison.csv", summaries)
    _write_report(experiment_dir / "report.md", ranking, best, final)
    return comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args(argv)
    result = evaluate_experiment(Path(args.experiment_dir))
    print(canonical_json(result))


if __name__ == "__main__":
    main()
