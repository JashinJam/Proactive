"""Evaluate the frozen visual temporal residual on the D4 session OOF folds."""

from __future__ import annotations

import argparse
import json
import logging
import math
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from proactive_d1.core import prediction_rows, strip_answers
from proactive_d4_1.compare import (
    decision_change_statistics,
    flatten_decisions,
    grouped_official_metrics,
    load_official_scorer,
    paired_session_bootstrap,
    stratified_statistics,
)
from proactive_d4_1.core import object_sha256
from proactive_d4_2.evaluate import build_candidate_matrix, length_quartiles
from proactive_r0.artifacts import code_snapshot, environment_snapshot, sha256_file, write_json
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer

from .session_folds import validate_session_fold_manifest
from .run_multiscale_evaluate import _previous_gold_rows, promotion_audit
from .temporal import temporal_residual_oof


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("proactive_d5.visual_temporal_evaluate")


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _check_hash(path: Path, expected: object) -> str:
    actual = sha256_file(path)
    if actual != str(expected):
        raise ValueError(f"D5 frozen artifact mismatch: {path}: {actual}")
    return actual


def merge_visual_shards(
    config: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
    visual = config["visual"]
    if not isinstance(visual, Mapping) or not isinstance(visual.get("shards"), list):
        raise ValueError("D5 visual shard plan is malformed")
    root = _resolve(visual["shard_root"])
    width = int(visual["width"])
    vectors: list[list[float]] = []
    records: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    covered: list[int] = []
    for shard in visual["shards"]:  # type: ignore[union-attr]
        if not isinstance(shard, Mapping):
            raise ValueError("D5 visual shard entry is malformed")
        expected = list(range(int(shard["first_index"]), int(shard["last_index"]) + 1))
        shard_dir = root / f"shard_{int(shard['id']):03d}"
        shard_records = load_jsonl(shard_dir / "session_records.jsonl")
        if len(shard_records) != len(expected):
            raise ValueError(f"D5 visual shard is incomplete: {shard_dir}")
        effective = _load_object(shard_dir / "config.json")
        runtime_config = effective.get("runtime")
        if not isinstance(runtime_config, Mapping) or runtime_config.get("session_indices") != expected:
            raise ValueError(f"D5 visual shard indices changed: {shard_dir}")
        for position, record in enumerate(shard_records):
            input_index = expected[position]
            if record.get("input_index") != input_index or record.get("video_path") != rows[input_index].get("video_path"):
                raise ValueError(f"D5 visual shard identity changed: {shard_dir}")
            chunks = record.get("chunks")
            intervals = rows[input_index].get("video_intervals")
            if not isinstance(chunks, list) or not isinstance(intervals, list) or len(chunks) != len(intervals):
                raise ValueError(f"D5 visual shard chunk coverage changed: {shard_dir}")
            for chunk_index, chunk in enumerate(chunks):
                if not isinstance(chunk, Mapping) or chunk.get("chunk_index") != chunk_index:
                    raise ValueError(f"D5 visual chunk order changed: {shard_dir}")
                if chunk.get("frame_sampling") != "causal_multiscale_16_8_8_v1":
                    raise ValueError(f"D5 visual frame policy changed: {shard_dir}")
                vector = [float(value) for value in chunk["vision_state"]]  # type: ignore[index]
                norm = math.sqrt(sum(value * value for value in vector))
                if len(vector) != width or not all(math.isfinite(value) for value in vector) or abs(norm - 1.0) > 1e-5:
                    raise ValueError(f"D5 visual state is invalid: {shard_dir}")
                vectors.append(vector)
        runtime = _load_object(shard_dir / "runtime.json")
        if runtime.get("status") != "complete answer-free visual feature shard":
            raise ValueError(f"D5 visual shard runtime is incomplete: {shard_dir}")
        records.extend(shard_records)
        runtimes.append(runtime)
        covered.extend(expected)
    if covered != list(range(len(rows))):
        raise ValueError("D5 visual shards do not exactly cover source sessions")
    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[1:] != (width,):
        raise ValueError("D5 merged visual matrix width changed")
    return values, records, runtimes


def _configure(output_dir: Path) -> None:
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (
        logging.StreamHandler(),
        logging.FileHandler(output_dir / "run.log", mode="w", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def _write_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join(
        [sys.executable, "-m", "proactive_d5.run_visual_temporal_evaluate", *argv]
    )
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def run(
    config_path: Path, output_dir: Path, raw_argv: Sequence[str], device: str
) -> dict[str, object]:
    started = time.monotonic()
    config = _load_object(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D5 visual temporal output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure(output_dir)
    write_json(output_dir / "config.json", {**config, "runtime": {"training_device": device}})
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    tracked = [
        *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d4_2").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d5").glob("*.py")),
        config_path,
        PROJECT_ROOT / "Agent.md",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
    ]
    write_json(output_dir / "code_state.txt", code_snapshot(PROJECT_ROOT, tracked))
    protocol = config["protocol"]
    data = config["data"]
    starter = config["starter_kit"]
    base = config["base"]
    baseline = config["baseline"]
    folds = config["folds"]
    if not all(isinstance(value, Mapping) for value in (protocol, data, starter, base, baseline, folds)):
        raise ValueError("D5 visual temporal config sections are malformed")
    input_path = _resolve(data["input"])
    starter_dir = _resolve(starter["path"])
    fingerprints = {
        "config_object": object_sha256(config),
        "input": _check_hash(input_path, data["input_sha256"]),
        "protocol": _check_hash(_resolve(protocol["path"]), protocol["sha256"]),
        "base_records": _check_hash(_resolve(base["records"]), base["records_sha256"]),
        "base_cache": _check_hash(_resolve(base["cache"]), base["cache_sha256"]),
        "fold_manifest": _check_hash(_resolve(folds["manifest"]), folds["manifest_sha256"]),
        "baseline_summary": _check_hash(_resolve(baseline["summary"]), baseline["summary_sha256"]),
        "baseline_predictions": _check_hash(_resolve(baseline["predictions"]), baseline["predictions_sha256"]),
        "official_scorer": _check_hash(starter_dir / "run_evaluation.py", starter["scorer_py_sha256"]),
    }
    source_rows = load_jsonl(input_path)
    answer_free_rows = strip_answers(source_rows)
    fold_manifest = _load_object(_resolve(folds["manifest"]))
    fold_by_index, split_audit = validate_session_fold_manifest(fold_manifest, answer_free_rows)
    vision_values, visual_records, visual_runtimes = merge_visual_shards(config, answer_free_rows)
    base_records = load_jsonl(_resolve(base["records"]))
    candidate = {"name": "history8_visual_temporal", "parameters": base["parameters"]}
    examples, base_values, names, dialog_audit = build_candidate_matrix(
        source_rows=source_rows,
        answer_free_rows=answer_free_rows,
        records=base_records,
        fold_by_index=fold_by_index,
        candidate=candidate,
        cache_path=_resolve(base["cache"]),
        config=config,
    )
    if vision_values.shape[0] != len(examples):
        raise ValueError("D5 visual features do not align with base examples")
    feature_path = output_dir / "features.npz"
    with feature_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            vision_state=vision_values,
            input_index=np.asarray([example.feature.input_index for example in examples], dtype=np.int32),
            chunk_index=np.asarray([example.feature.chunk_index for example in examples], dtype=np.int32),
        )
    decisions, logits, fold_details = temporal_residual_oof(
        examples,
        base_values,
        vision_values,
        folds=int(folds["count"]),
        calibration_fold_offset=int(folds["calibration_fold_offset"]),
        base_config=config["base_training"],  # type: ignore[arg-type]
        temporal_config=config["temporal_training"],  # type: ignore[arg-type]
        device=device,
    )
    predictions = prediction_rows(examples, decisions)
    validate_prediction_rows(source_rows, predictions)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    write_jsonl(predictions_path, predictions)
    _run_official_scorer(starter_dir, input_path, predictions_path, metrics_path, output_dir / "scorer.log")
    metrics = _load_object(metrics_path)
    write_jsonl(
        output_dir / "oof_logits.jsonl",
        [
            {
                "input_index": example.feature.input_index,
                "chunk_index": example.feature.chunk_index,
                "logit": logits[example.key],
                "decision_interrupt": decisions[example.key],
            }
            for example in examples
        ],
    )
    scorer = load_official_scorer(starter_dir, str(starter["scorer_py_sha256"]))
    quartiles = length_quartiles(answer_free_rows)
    candidate_flat = flatten_decisions(source_rows, predictions, list(range(len(source_rows))), quartiles)
    baseline_predictions = load_jsonl(_resolve(baseline["predictions"]))
    baseline_flat = flatten_decisions(source_rows, baseline_predictions, list(range(len(source_rows))), quartiles)
    candidate_strata = stratified_statistics(scorer, candidate_flat)
    baseline_strata = stratified_statistics(scorer, baseline_flat)
    candidate_previous = grouped_official_metrics(
        scorer,
        _previous_gold_rows(candidate_flat, source_rows),
        lambda row: str(row["previous_gold"]),
    )
    baseline_previous = grouped_official_metrics(
        scorer,
        _previous_gold_rows(baseline_flat, source_rows),
        lambda row: str(row["previous_gold"]),
    )
    evaluation = config["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise ValueError("D5 visual temporal evaluation config is malformed")
    bootstrap = paired_session_bootstrap(
        candidate_flat,
        baseline_flat,
        repetitions=int(evaluation["bootstrap_repetitions"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    baseline_summary = _load_object(_resolve(baseline["summary"]))
    audit = promotion_audit(
        candidate_overall=metrics["overall"],  # type: ignore[arg-type]
        baseline_overall=baseline_summary["overall"],  # type: ignore[arg-type]
        candidate_flat=candidate_flat,
        baseline_flat=baseline_flat,
        candidate_strata=candidate_strata,
        baseline_strata=baseline_strata,
        candidate_previous=candidate_previous,
        baseline_previous=baseline_previous,
        fold_by_index=fold_by_index,
        scorer=scorer,
        gates=evaluation["promotion_gates"],  # type: ignore[arg-type]
        bootstrap=bootstrap,
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "classification": evaluation["classification"],
        "overall": metrics["overall"],
        "baseline_overall": baseline_summary["overall"],
        "promotion_audit": audit,
        "session_bootstrap_vs_baseline": bootstrap,
        "decision_changes_vs_baseline": decision_change_statistics(candidate_flat, baseline_flat),
        "fold_details": fold_details,
        "stratified": candidate_strata,
        "baseline_stratified": baseline_strata,
        "previous_gold_stratified": candidate_previous,
        "baseline_previous_gold_stratified": baseline_previous,
        "split_audit": split_audit,
        "dialog_feature_audit": dialog_audit,
        "sessions": len(visual_records),
        "chunks": len(examples),
        "feature_count": len(names),
        "total_parameters": config["model_accounting"]["total_parameters"],  # type: ignore[index]
        "visual_feature_sha256": sha256_file(feature_path),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "visual_shard_runtime": visual_runtimes,
        "fingerprints": fingerprints,
        "training_device": device,
        "wall_time_seconds": time.monotonic() - started,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "data_manifest.json",
        {
            "data": data,
            "protocol": protocol,
            "base": base,
            "visual": config["visual"],
            "baseline": baseline,
            "folds": folds,
            "fingerprints": fingerprints,
            "split_audit": split_audit,
            "visual_features_are_answer_free": True,
            "external_data_used": False,
            "supervision": evaluation["classification"],
        },
    )
    overall = metrics["overall"]
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：已完成冻结视觉时序残差的 D4 session-level OOF。",
                "",
                f"- Sessions/chunks: `{len(visual_records)}/{len(examples)}`",
                f"- Official Macro/G-mean: `{overall['macro_f1']:.4f}/{overall['gmean_f1']:.4f}`",  # type: ignore[index]
                f"- Macro delta vs D4-fold history8: `{audit['macro_delta']:+.4f}`",
                f"- Promotion gates passed: `{audit['passed']}`",
                f"- Predictions SHA256: `{summary['predictions_sha256']}`",
                "- 证据类型：post-selection、val-supervised；不是独立泛化或 hidden-test 证据。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "Visual temporal OOF complete: Macro=%.4f delta=%+.4f promoted=%s",
        float(overall["macro_f1"]),  # type: ignore[index]
        float(audit["macro_delta"]),
        audit["passed"],
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(raw_argv)
    summary = run(_resolve(args.config), _resolve(args.experiment_dir), raw_argv, args.device)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
