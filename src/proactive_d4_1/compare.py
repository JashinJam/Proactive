"""Merge D4.1 shards and rebuild official metrics, rankings, and statistics."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Sequence

import numpy as np

from proactive_d1.core import strip_answers
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import (
    INTERRUPT_TAG,
    load_jsonl,
    validate_prediction_rows,
    write_jsonl,
)

from .core import (
    BASELINE,
    canonical_json,
    object_sha256,
    percentile,
    rank_summaries,
    validate_shard_records,
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


def load_official_scorer(starter_dir: Path, expected_sha256: str) -> ModuleType:
    scorer_path = starter_dir / "run_evaluation.py"
    if sha256_file(scorer_path) != expected_sha256:
        raise ValueError("D4.1 official scorer SHA256 changed")
    spec = importlib.util.spec_from_file_location(
        "_d4_1_official_run_evaluation", scorer_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official scorer: {scorer_path}")
    module = importlib.util.module_from_spec(spec)
    inserted = str(starter_dir) not in sys.path
    if inserted:
        sys.path.insert(0, str(starter_dir))
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(str(starter_dir))
    return module


def official_score(
    scorer: ModuleType,
    golden: Sequence[dict[str, object]],
    predictions: Sequence[dict[str, object]],
) -> dict[str, object]:
    result = scorer.score_proactive(list(golden), list(predictions))
    if not isinstance(result, dict) or not isinstance(result.get("overall"), dict):
        raise ValueError("D4.1 official scorer returned an invalid result")
    overall = dict(result["overall"])
    support = int(overall["support"])
    overall["predicted_interrupt_rate"] = round(
        (int(overall["tp"]) + int(overall["fp"])) / support if support else 0.0,
        6,
    )
    result["overall"] = overall
    return result


def _decision(value: object) -> int:
    return int(str(value).lstrip().startswith(INTERRUPT_TAG))


def _position_bin(chunk_index: int) -> str:
    if chunk_index == 0:
        return "0:first"
    if chunk_index == 1:
        return "1:second"
    if chunk_index <= 4:
        return "2-4"
    if chunk_index <= 9:
        return "5-9"
    return "10+"


def flatten_decisions(
    golden: Sequence[dict[str, object]],
    predictions: Sequence[dict[str, object]],
    input_indices: Sequence[int],
    quartile_by_index: Mapping[int, int],
) -> list[dict[str, object]]:
    if len(golden) != len(predictions) or len(golden) != len(input_indices):
        raise ValueError("D4.1 flattened decisions require aligned sessions")
    result: list[dict[str, object]] = []
    for input_index, row, prediction in zip(input_indices, golden, predictions):
        gold_answers = row.get("answers")
        pred_answers = prediction.get("answers")
        if not isinstance(gold_answers, list) or not isinstance(pred_answers, list):
            raise ValueError("D4.1 flattened prediction lacks answers")
        if len(gold_answers) != len(pred_answers):
            raise ValueError("D4.1 refuses official scorer length truncation")
        for chunk_index, (gold, pred) in enumerate(zip(gold_answers, pred_answers)):
            result.append(
                {
                    "input_index": input_index,
                    "video_path": row["video_path"],
                    "domain": str(row.get("domain", "unknown")),
                    "task": str(row.get("task", "unknown")),
                    "chunk_index": chunk_index,
                    "position": _position_bin(chunk_index),
                    "session_chunks": len(gold_answers),
                    "session_length_quartile": f"Q{quartile_by_index[input_index]}",
                    "gold_interrupt": _decision(gold),
                    "predicted_interrupt": _decision(pred),
                }
            )
    return result


def _counts(rows: Sequence[Mapping[str, object]]) -> dict[str, int]:
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for row in rows:
        gold = int(row["gold_interrupt"])
        prediction = int(row["predicted_interrupt"])
        if gold and prediction:
            counts["tp"] += 1
        elif not gold and prediction:
            counts["fp"] += 1
        elif not gold and not prediction:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def grouped_official_metrics(
    scorer: ModuleType,
    rows: Sequence[Mapping[str, object]],
    key: Callable[[Mapping[str, object]], str],
) -> dict[str, object]:
    groups: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    result: dict[str, object] = {}
    for name, selected in sorted(groups.items()):
        counts = _counts(selected)
        metrics = dict(scorer.binary_metrics(**counts))
        support = int(metrics["support"])
        metrics["predicted_interrupt_rate"] = round(
            (counts["tp"] + counts["fp"]) / support if support else 0.0, 6
        )
        result[name] = metrics
    return result


def stratified_statistics(
    scorer: ModuleType, rows: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    return {
        "domain": grouped_official_metrics(scorer, rows, lambda row: str(row["domain"])),
        "task": grouped_official_metrics(scorer, rows, lambda row: str(row["task"])),
        "chunk_position": grouped_official_metrics(
            scorer, rows, lambda row: str(row["position"])
        ),
        "session_length_quartile": grouped_official_metrics(
            scorer, rows, lambda row: str(row["session_length_quartile"])
        ),
    }


def decision_change_statistics(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    if len(candidate) != len(baseline):
        raise ValueError("D4.1 decision-change rows are not aligned")
    counts = {
        "changed_decisions": 0,
        "corrected_errors": 0,
        "new_errors": 0,
        "interrupt_to_silent": 0,
        "silent_to_interrupt": 0,
    }
    for candidate_row, baseline_row in zip(candidate, baseline):
        identity = (candidate_row["input_index"], candidate_row["chunk_index"])
        if identity != (baseline_row["input_index"], baseline_row["chunk_index"]):
            raise ValueError("D4.1 decision-change row order differs")
        candidate_prediction = int(candidate_row["predicted_interrupt"])
        baseline_prediction = int(baseline_row["predicted_interrupt"])
        gold = int(candidate_row["gold_interrupt"])
        if candidate_prediction == baseline_prediction:
            continue
        counts["changed_decisions"] += 1
        if baseline_prediction and not candidate_prediction:
            counts["interrupt_to_silent"] += 1
        else:
            counts["silent_to_interrupt"] += 1
        if candidate_prediction == gold and baseline_prediction != gold:
            counts["corrected_errors"] += 1
        elif candidate_prediction != gold and baseline_prediction == gold:
            counts["new_errors"] += 1
    return counts


def paired_session_bootstrap(
    candidate: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
    *,
    repetitions: int,
    seed: int,
) -> dict[str, object]:
    if repetitions <= 0 or len(candidate) != len(baseline):
        raise ValueError("D4.1 bootstrap requires positive repetitions and aligned rows")
    sessions = sorted({int(row["input_index"]) for row in candidate})
    if not sessions:
        raise ValueError("D4.1 bootstrap has no sessions")
    position = {session: index for index, session in enumerate(sessions)}
    candidate_counts = np.zeros((len(sessions), 4), dtype=np.int64)
    baseline_counts = np.zeros((len(sessions), 4), dtype=np.int64)

    def add(array: np.ndarray, row: Mapping[str, object]) -> None:
        session = position[int(row["input_index"])]
        gold = int(row["gold_interrupt"])
        prediction = int(row["predicted_interrupt"])
        column = 0 if gold and prediction else 1 if not gold and prediction else 2 if not gold else 3
        array[session, column] += 1

    for candidate_row, baseline_row in zip(candidate, baseline):
        identity = (candidate_row["input_index"], candidate_row["chunk_index"])
        if identity != (baseline_row["input_index"], baseline_row["chunk_index"]):
            raise ValueError("D4.1 bootstrap row order differs")
        add(candidate_counts, candidate_row)
        add(baseline_counts, baseline_row)
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(sessions), size=(repetitions, len(sessions)))

    def macro(array: np.ndarray) -> np.ndarray:
        tp, fp, tn, fn = (array[:, column] for column in range(4))
        interrupt_denominator = 2 * tp + fp + fn
        silent_denominator = 2 * tn + fn + fp
        interrupt = np.divide(
            2 * tp,
            interrupt_denominator,
            out=np.zeros_like(tp, dtype=np.float64),
            where=interrupt_denominator != 0,
        )
        silent = np.divide(
            2 * tn,
            silent_denominator,
            out=np.zeros_like(tn, dtype=np.float64),
            where=silent_denominator != 0,
        )
        return (interrupt + silent) / 2

    candidate_macro = macro(candidate_counts[sampled].sum(axis=1))
    baseline_macro = macro(baseline_counts[sampled].sum(axis=1))
    delta = candidate_macro - baseline_macro
    lower, median, upper = np.quantile(delta, [0.025, 0.5, 0.975])
    return {
        "unit": "session",
        "repetitions": repetitions,
        "seed": seed,
        "delta_macro_f1_p2_5": float(lower),
        "delta_macro_f1_median": float(median),
        "delta_macro_f1_p97_5": float(upper),
        "positive_fraction": float((delta > 0).mean()),
    }


def timing_statistics(
    records: Sequence[Mapping[str, object]],
    runtimes: Sequence[Mapping[str, object]],
    *,
    session_limit_seconds: float,
) -> dict[str, object]:
    if not records:
        raise ValueError("D4.1 timing statistics require session records")
    session_generation = [
        float(record["timing"]["generation_seconds"]) for record in records  # type: ignore[index]
    ]
    session_decision = [
        float(record["timing"]["decision_feature_seconds"]) for record in records  # type: ignore[index]
    ]
    session_model = [
        float(record["timing"]["model_inference_seconds"]) for record in records  # type: ignore[index]
    ]
    session_wall = [
        float(record["timing"]["session_wall_seconds"]) for record in records  # type: ignore[index]
    ]

    def distribution(values: Sequence[float]) -> dict[str, float]:
        return {
            "total": sum(values),
            "median": percentile(values, 0.5),
            "p95": percentile(values, 0.95),
            "max": max(values),
        }

    over_limit = [
        int(record["input_index"])
        for record, seconds in zip(records, session_model)
        if seconds > session_limit_seconds
    ]
    return {
        "generation_seconds": distribution(session_generation),
        "decision_feature_seconds": distribution(session_decision),
        "model_inference_seconds": distribution(session_model),
        "session_wall_seconds": distribution(session_wall),
        "total_model_inference_seconds": sum(session_model),
        "gpu_seconds": sum(float(runtime["wall_time_seconds_this_attempt"]) for runtime in runtimes),
        "peak_gpu_memory_bytes": max(int(runtime["peak_gpu_memory_bytes"]) for runtime in runtimes),
        "session_limit_seconds": session_limit_seconds,
        "sessions_over_limit": over_limit,
        "deployable": not over_limit,
    }


def merge_variant(
    *,
    experiment_dir: Path,
    stage_plan: Mapping[str, object],
    variant: Mapping[str, object],
    source_rows: Sequence[dict[str, object]],
    scorer: ModuleType,
    quartile_by_index: Mapping[int, int],
    session_limit_seconds: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    stage = str(stage_plan["stage"])
    variant_id = str(variant["variant_id"])
    variant_dir = experiment_dir / "runs" / stage / variant_id
    records: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    planned_shards = stage_plan["shards"]
    if not isinstance(planned_shards, list) or not planned_shards:
        raise ValueError(f"D4.1 stage {stage} has no planned shards")
    for shard_id, expected_indices in enumerate(planned_shards):
        shard_dir = variant_dir / f"shard_{shard_id:03d}"
        status = _load_object(shard_dir / "status.json")
        if status.get("status") != "complete":
            raise ValueError(f"D4.1 shard is not complete: {shard_dir}")
        shard_records = load_jsonl(shard_dir / "session_records.jsonl")
        validate_shard_records(
            shard_records,
            list(expected_indices),
            source_rows,
            require_complete=True,
        )
        records.extend(shard_records)
        runtimes.append(_load_object(shard_dir / "runtime.json"))
    records.sort(key=lambda record: int(record["input_index"]))
    expected_stage_indices = list(stage_plan["indices"])
    if [int(record["input_index"]) for record in records] != expected_stage_indices:
        raise ValueError(f"D4.1 merged {stage}/{variant_id} coverage or order changed")
    predictions = [record["prediction"] for record in records]
    golden = [source_rows[index] for index in expected_stage_indices]
    validate_prediction_rows(strip_answers(golden), predictions)  # type: ignore[arg-type]
    write_jsonl(variant_dir / "session_records.jsonl", records)
    write_jsonl(variant_dir / "predictions.jsonl", predictions)  # type: ignore[arg-type]
    write_jsonl(variant_dir / "evaluation_golden.jsonl", golden)
    metrics = official_score(scorer, golden, predictions)  # type: ignore[arg-type]
    write_json(variant_dir / "metrics.json", metrics)
    flattened = flatten_decisions(
        golden,
        predictions,  # type: ignore[arg-type]
        expected_stage_indices,
        quartile_by_index,
    )
    strata = stratified_statistics(scorer, flattened)
    write_json(variant_dir / "stratified_metrics.json", strata)
    timing = timing_statistics(
        records, runtimes, session_limit_seconds=session_limit_seconds
    )
    summary = {
        "stage": stage,
        "variant_id": variant_id,
        "parameters": variant["parameters"],
        "is_baseline": variant["is_baseline"],
        "origin": variant["origin"],
        "overall": metrics["overall"],
        "timing": timing,
        "deployable": timing["deployable"],
        "sessions": len(records),
        "chunks": len(flattened),
        "predictions_sha256": sha256_file(variant_dir / "predictions.jsonl"),
        "session_records_sha256": sha256_file(variant_dir / "session_records.jsonl"),
        "official_scorer_sha256": sha256_file(
            _resolve(_load_object(experiment_dir / "config.json")["starter_kit"]["path"])  # type: ignore[index]
            / "run_evaluation.py"
        ),
        "stratified": strata,
    }
    write_json(variant_dir / "summary.json", summary)
    return summary, flattened


def _read_variants(path: Path) -> list[dict[str, object]]:
    variants = load_jsonl(path)
    if len({str(variant["variant_id"]) for variant in variants}) != len(variants):
        raise ValueError("D4.1 variants.jsonl contains duplicate IDs")
    return variants


def _add_baseline_comparisons(
    summaries: list[dict[str, object]],
    flattened: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    repetitions: int,
    seed: int,
) -> None:
    baselines = [summary for summary in summaries if summary["is_baseline"]]
    if len(baselines) != 1:
        raise ValueError("D4.1 completed stage must contain one baseline")
    baseline = baselines[0]
    baseline_rows = flattened[str(baseline["variant_id"])]
    for summary in summaries:
        candidate_rows = flattened[str(summary["variant_id"])]
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
            repetitions=repetitions,
            seed=seed,
        )


def _write_csv(path: Path, stage_summaries: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    fields = [
        "stage",
        "rank",
        "variant_id",
        *BASELINE.to_dict().keys(),
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
        "gpu_seconds",
        "peak_gpu_memory_bytes",
        "deployable",
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for stage in ("search", "confirmation", "full"):
            summaries = stage_summaries.get(stage, [])
            for rank, summary in enumerate(rank_summaries(summaries), start=1):
                overall = summary["overall"]
                timing = summary["timing"]
                writer.writerow(
                    {
                        "stage": stage,
                        "rank": rank,
                        "variant_id": summary["variant_id"],
                        **summary["parameters"],
                        **{name: overall[name] for name in fields if name in overall},
                        "delta_macro_f1_vs_baseline": summary[
                            "delta_macro_f1_vs_baseline"
                        ],
                        "total_model_inference_seconds": timing[
                            "total_model_inference_seconds"
                        ],
                        "gpu_seconds": timing["gpu_seconds"],
                        "peak_gpu_memory_bytes": timing["peak_gpu_memory_bytes"],
                        "deployable": summary["deployable"],
                    }
                )
    temporary.replace(path)


def _write_report(
    path: Path,
    comparison: Mapping[str, object],
    best: Mapping[str, object] | None,
) -> None:
    lines = [
        "# D4.1 推理输入策略搜索报告",
        "",
        "本报告是 val-supervised public-validation 输入策略审计，不是 hidden-test 或独立泛化证据。",
        "",
    ]
    stages = comparison["stages"]
    for stage in ("search", "confirmation", "full"):
        value = stages.get(stage)  # type: ignore[union-attr]
        if not isinstance(value, Mapping):
            continue
        if value.get("status") != "complete":
            lines.extend(
                [
                    f"## {stage}",
                    "",
                    f"状态：`incomplete`，剩余任务数 `{len(value.get('remaining_tasks', []))}`。",
                    "",
                ]
            )
            continue
        lines.extend([f"## {stage}", "", "| Rank | Variant | Macro F1 | G-mean F1 | Model seconds | Deployable |", "|---:|---|---:|---:|---:|---|"])
        for rank, summary in enumerate(value["ranking"], start=1):  # type: ignore[index]
            lines.append(
                f"| {rank} | `{summary['variant_id']}` | "
                f"{summary['overall']['macro_f1']:.4f} | "
                f"{summary['overall']['gmean_f1']:.4f} | "
                f"{summary['timing']['total_model_inference_seconds']:.3f} | "
                f"{summary['deployable']} |"
            )
        lines.append("")
    if best is not None:
        lines.extend(
            [
                "## 最佳策略",
                "",
                f"`{best['variant_id']}` 是 D4.1 public-validation 最佳输入策略，参数为 `{canonical_json(best['parameters'])}`。",
                f"相对 D4 baseline 的 Macro F1 差值为 `{best['delta_macro_f1_vs_d4_baseline']:+.4f}`。",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def compare_experiment(experiment_dir: Path) -> dict[str, object]:
    experiment_dir = experiment_dir.expanduser().resolve()
    config = _load_object(experiment_dir / "config.json")
    config_sha256 = object_sha256(config)
    identity = _load_object(experiment_dir / "experiment_identity.json")
    if identity.get("experiment_config_sha256") != config_sha256:
        raise ValueError("D4.1 compare found a changed experiment config")
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    evaluation = dict(config["evaluation"])  # type: ignore[arg-type]
    source_rows = load_jsonl(_resolve(data_config["input"]))
    manifest = _load_object(experiment_dir / "sample_manifest.json")
    quartile_by_index = {
        int(entry["input_index"]): int(entry["length_quartile"])
        for entry in manifest["all_sessions"]  # type: ignore[index]
    }
    variants = _read_variants(experiment_dir / "variants.jsonl")
    by_id = {str(variant["variant_id"]): variant for variant in variants}
    scorer = load_official_scorer(
        _resolve(starter_config["path"]), str(starter_config["scorer_py_sha256"])
    )
    stage_summaries: dict[str, list[dict[str, object]]] = {}
    stages: dict[str, object] = {}
    for stage in ("search", "confirmation", "full"):
        plan_path = experiment_dir / "stage_plans" / f"{stage}.json"
        if not plan_path.exists():
            continue
        stage_plan = _load_object(plan_path)
        incomplete: list[str] = []
        for variant_id in stage_plan["variant_ids"]:  # type: ignore[index]
            for shard_id in range(len(stage_plan["shards"])):  # type: ignore[arg-type]
                status_path = (
                    experiment_dir
                    / "runs"
                    / stage
                    / str(variant_id)
                    / f"shard_{shard_id:03d}"
                    / "status.json"
                )
                if (
                    not status_path.exists()
                    or _load_object(status_path).get("status") != "complete"
                ):
                    incomplete.append(f"{variant_id}/shard_{shard_id:03d}")
        if incomplete:
            stages[stage] = {
                "status": "incomplete",
                "remaining_tasks": incomplete,
                "variant_count": len(stage_plan["variant_ids"]),  # type: ignore[arg-type]
            }
            continue
        summaries: list[dict[str, object]] = []
        flattened: dict[str, list[dict[str, object]]] = {}
        for variant_id in stage_plan["variant_ids"]:  # type: ignore[index]
            summary, rows = merge_variant(
                experiment_dir=experiment_dir,
                stage_plan=stage_plan,
                variant=by_id[str(variant_id)],
                source_rows=source_rows,
                scorer=scorer,
                quartile_by_index=quartile_by_index,
                session_limit_seconds=float(
                    evaluation["per_session_model_inference_limit_seconds"]
                ),
            )
            summaries.append(summary)
            flattened[str(variant_id)] = rows
        _add_baseline_comparisons(
            summaries,
            flattened,
            repetitions=int(evaluation["bootstrap_repetitions"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
        for summary in summaries:
            write_json(
                experiment_dir
                / "runs"
                / stage
                / str(summary["variant_id"])
                / "summary.json",
                summary,
            )
        ranking = rank_summaries(summaries)
        stage_summaries[stage] = summaries
        stages[stage] = {
            "status": "complete",
            "variant_count": len(summaries),
            "ranking": ranking,
            "baseline_variant_id": next(
                str(summary["variant_id"])
                for summary in summaries
                if summary["is_baseline"]
            ),
        }
    comparison = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "experiment_config_sha256": config_sha256,
        "ranking_rule": [
            "official_macro_f1_desc",
            "official_gmean_f1_desc",
            "total_model_inference_seconds_asc",
            "stable_variant_id_asc",
        ],
        "stages": stages,
    }
    write_json(experiment_dir / "comparison.json", comparison)
    _write_csv(experiment_dir / "comparison.csv", stage_summaries)
    best: dict[str, object] | None = None
    if (
        isinstance(stages.get("full"), Mapping)
        and stages["full"].get("status") == "complete"  # type: ignore[union-attr]
    ):
        deployable = rank_summaries(stage_summaries["full"], require_deployable=True)
        if deployable:
            winner = deployable[0]
            baseline = next(
                summary for summary in stage_summaries["full"] if summary["is_baseline"]
            )
            best = {
                "schema_version": 1,
                "claim": "D4.1 public-validation best input policy",
                "classification": "val-supervised; not hidden-test or independent generalization evidence",
                "variant_id": winner["variant_id"],
                "parameters": winner["parameters"],
                "source_stage": "full",
                "rank": 1,
                "experiment_config_sha256": config_sha256,
                "official_metrics": winner["overall"],
                "d4_baseline_variant_id": baseline["variant_id"],
                "delta_macro_f1_vs_d4_baseline": round(
                    float(winner["overall"]["macro_f1"])  # type: ignore[index]
                    - float(baseline["overall"]["macro_f1"]),  # type: ignore[index]
                    6,
                ),
                "deployable_under_300_second_limit": True,
                "timing": winner["timing"],
            }
            write_json(experiment_dir / "best_inference.json", best)
    _write_report(experiment_dir / "report.md", comparison, best)
    return comparison


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args(argv)
    result = compare_experiment(Path(args.experiment_dir))
    print(canonical_json(result))


if __name__ == "__main__":
    main()
