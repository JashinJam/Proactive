"""Evaluate frozen causal-multiscale features on the D4 session OOF folds."""

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

from proactive_d1.core import prediction_rows, strip_answers
from proactive_d1.neural_core import cross_validate_neural_matrix
from proactive_d4_1.compare import (
    decision_change_statistics,
    flatten_decisions,
    grouped_official_metrics,
    load_official_scorer,
    paired_session_bootstrap,
    stratified_statistics,
)
from proactive_d4_1.core import object_sha256
from proactive_d4_2.core import validate_feature_records
from proactive_d4_2.evaluate import (
    build_candidate_matrix,
    feature_arrays,
    length_quartiles,
    write_feature_cache,
)
from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    write_json,
)
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer

from .session_folds import validate_session_fold_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("proactive_d5.multiscale_evaluate")


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


def _configure_logging(output_dir: Path) -> None:
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
        [sys.executable, "-m", "proactive_d5.run_multiscale_evaluate", *argv]
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


def _tracked_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d4_1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d4_2").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d5").glob("*.py")),
        config_path,
        PROJECT_ROOT / "Agent.md",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
    ]


def merge_feature_shards(
    config: Mapping[str, object], rows: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Validate all answer-free shard outputs before returning merged features."""
    multiscale = config["multiscale"]
    if not isinstance(multiscale, Mapping):
        raise ValueError("D5 multiscale config is malformed")
    shard_root = _resolve(multiscale["shard_root"])
    shards = multiscale["shards"]
    parameters = multiscale["parameters"]
    if not isinstance(shards, list) or not isinstance(parameters, Mapping):
        raise ValueError("D5 multiscale shard plan is malformed")
    records: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    covered: list[int] = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("D5 multiscale shard entry is malformed")
        shard_id = int(shard["id"])
        expected = list(range(int(shard["first_index"]), int(shard["last_index"]) + 1))
        runtime_expected = list(
            range(
                int(shard.get("runtime_first_index", shard["first_index"])),
                int(shard.get("runtime_last_index", shard["last_index"])) + 1,
            )
        )
        shard_dir = shard_root / f"shard_{shard_id:03d}"
        shard_records = load_jsonl(shard_dir / "session_records.jsonl")
        # The shared validator checks identity, complete chunk coverage, hidden width,
        # finite values, and the feature fields required by the cache builder.
        try:
            validate_feature_records(
                shard_records,
                expected,
                rows,
                hidden_size=int(config["features"]["hidden_size"]),  # type: ignore[index]
                require_complete=True,
            )
        except ValueError as error:
            # run_deploy does not persist D4.2's timing-only field. Re-run the
            # substantive validation below and reject every other mismatch.
            if "model_inference_seconds" not in str(error):
                raise
            if len(shard_records) != len(expected):
                raise ValueError(f"Incomplete D5 feature shard: {shard_dir}") from error
        effective = _load_object(shard_dir / "config.json")
        inference = effective.get("inference")
        runtime_config = effective.get("runtime")
        if not isinstance(inference, Mapping) or not isinstance(runtime_config, Mapping):
            raise ValueError(f"D5 shard lacks effective inference metadata: {shard_dir}")
        if inference.get("frame_sampling") != parameters.get("frame_sampling"):
            raise ValueError(f"D5 shard frame policy changed: {shard_dir}")
        if runtime_config.get("session_indices") != runtime_expected:
            raise ValueError(f"D5 shard selected indices changed: {shard_dir}")
        for position, record in enumerate(shard_records):
            input_index = expected[position]
            if int(record.get("input_index", -1)) != input_index:
                raise ValueError(f"D5 shard order changed: {shard_dir}")
            if record.get("video_path") != rows[input_index].get("video_path"):
                raise ValueError(f"D5 shard identity changed: {shard_dir}")
            chunks = record.get("chunks")
            intervals = rows[input_index].get("video_intervals")
            if not isinstance(chunks, list) or not isinstance(intervals, list) or len(chunks) != len(intervals):
                raise ValueError(f"D5 shard chunk coverage changed: {shard_dir}")
            for chunk_index, chunk in enumerate(chunks):
                if not isinstance(chunk, Mapping) or chunk.get("chunk_index") != chunk_index:
                    raise ValueError(f"D5 shard chunk order changed: {shard_dir}")
                hidden = chunk.get("hidden_state")
                if not isinstance(hidden, list) or len(hidden) != int(config["features"]["hidden_size"]):  # type: ignore[index]
                    raise ValueError(f"D5 shard hidden width changed: {shard_dir}")
                if not all(math.isfinite(float(value)) for value in hidden):
                    raise ValueError(f"D5 shard hidden values are non-finite: {shard_dir}")
                if chunk.get("frame_sampling") != parameters.get("frame_sampling"):
                    raise ValueError(f"D5 chunk frame policy changed: {shard_dir}")
                if not 0 < int(chunk.get("model_input_frames", 0)) <= int(parameters["max_frames"]):
                    raise ValueError(f"D5 chunk frame budget changed: {shard_dir}")
        runtime_path = shard_dir / "runtime.json"
        if runtime_path.exists():
            runtime = _load_object(runtime_path)
            if runtime.get("status") != "complete online dialog_stage_fused deployment run":
                raise ValueError(f"D5 shard runtime is incomplete: {shard_dir}")
        elif shard.get("intentional_prefix") is True:
            runtime = {
                "status": "intentionally stopped after frozen non-overlapping prefix",
                "sessions": len(shard_records),
                "chunks": sum(len(record["chunks"]) for record in shard_records),  # type: ignore[arg-type]
            }
        else:
            raise ValueError(f"D5 shard runtime is missing: {shard_dir}")
        records.extend(shard_records)
        runtimes.append(runtime)
        covered.extend(expected)
    if covered != list(range(len(rows))):
        raise ValueError("D5 feature shards do not exactly cover source sessions")
    return records, runtimes


def _previous_gold_rows(
    flat: Sequence[Mapping[str, object]], source_rows: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in flat:
        chunk_index = int(row["chunk_index"])
        if chunk_index == 0:
            continue
        answers = source_rows[int(row["input_index"])]["answers"]
        if not isinstance(answers, list):
            raise ValueError("D5 previous-gold stratum lacks answers")
        copy = dict(row)
        copy["previous_gold"] = (
            "interrupt"
            if str(answers[chunk_index - 1]).lstrip().startswith("$interrupt$")
            else "silent"
        )
        result.append(copy)
    return result


def promotion_audit(
    *,
    candidate_overall: Mapping[str, object],
    baseline_overall: Mapping[str, object],
    candidate_flat: Sequence[Mapping[str, object]],
    baseline_flat: Sequence[Mapping[str, object]],
    candidate_strata: Mapping[str, object],
    baseline_strata: Mapping[str, object],
    candidate_previous: Mapping[str, object],
    baseline_previous: Mapping[str, object],
    fold_by_index: Mapping[int, int],
    scorer: object,
    gates: Mapping[str, object],
    bootstrap: Mapping[str, object],
) -> dict[str, object]:
    candidate_fold = grouped_official_metrics(
        scorer, candidate_flat, lambda row: str(fold_by_index[int(row["input_index"])])
    )
    baseline_fold = grouped_official_metrics(
        scorer, baseline_flat, lambda row: str(fold_by_index[int(row["input_index"])])
    )
    fold_delta = {
        name: round(
            float(candidate_fold[name]["macro_f1"]) - float(baseline_fold[name]["macro_f1"]),  # type: ignore[index]
            6,
        )
        for name in candidate_fold
    }
    candidate_domain = candidate_strata["domain"]
    baseline_domain = baseline_strata["domain"]
    domain_delta = {
        name: round(
            float(candidate_domain[name]["macro_f1"]) - float(baseline_domain[name]["macro_f1"]),  # type: ignore[index]
            6,
        )
        for name in candidate_domain  # type: ignore[union-attr]
    }
    previous_delta = {
        name: round(
            float(candidate_previous[name]["macro_f1"]) - float(baseline_previous[name]["macro_f1"]),  # type: ignore[index]
            6,
        )
        for name in ("interrupt", "silent")
    }
    macro_delta = round(
        float(candidate_overall["macro_f1"]) - float(baseline_overall["macro_f1"]),
        6,
    )
    predicted_rate = (
        int(candidate_overall["tp"]) + int(candidate_overall["fp"])
    ) / int(candidate_overall["support"])
    checks = {
        "minimum_macro_improvement": macro_delta
        >= float(gates["minimum_macro_improvement"]),
        "paired_session_interval_lower_above_zero": float(
            bootstrap["delta_macro_f1_p2_5"]
        )
        > 0.0,
        "positive_folds": sum(value > 0 for value in fold_delta.values())
        >= int(gates["positive_folds"]),
        "positive_domains": sum(value > 0 for value in domain_delta.values())
        >= int(gates["positive_domains"]),
        "previous_interrupt_and_silent_nondecline": all(
            value >= 0 for value in previous_delta.values()
        ),
        "no_class_collapse": 0.0 < predicted_rate < 1.0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "macro_delta": macro_delta,
        "fold_delta": fold_delta,
        "domain_delta": domain_delta,
        "previous_gold_delta": previous_delta,
        "candidate_fold_metrics": candidate_fold,
        "baseline_fold_metrics": baseline_fold,
        "candidate_predicted_interrupt_rate": predicted_rate,
    }


def run(config_path: Path, output_dir: Path, raw_argv: Sequence[str]) -> dict[str, object]:
    started = time.monotonic()
    config_path = config_path.resolve()
    config = _load_object(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D5 multiscale evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_logging(output_dir)
    write_json(output_dir / "config.json", config)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(output_dir / "code_state.txt", code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)))

    protocol = config["protocol"]
    data = config["data"]
    starter = config["starter_kit"]
    baseline_config = config["baseline"]
    folds_config = config["folds"]
    if not all(isinstance(value, Mapping) for value in (protocol, data, starter, baseline_config, folds_config)):
        raise ValueError("D5 multiscale evaluation config sections are malformed")
    input_path = _resolve(data["input"])
    starter_dir = _resolve(starter["path"])
    fingerprints = {
        "config_object": object_sha256(config),
        "input": _check_hash(input_path, data["input_sha256"]),
        "protocol": _check_hash(_resolve(protocol["path"]), protocol["sha256"]),
        "fold_manifest": _check_hash(_resolve(folds_config["manifest"]), folds_config["manifest_sha256"]),
        "baseline_summary": _check_hash(_resolve(baseline_config["summary"]), baseline_config["summary_sha256"]),
        "baseline_predictions": _check_hash(_resolve(baseline_config["predictions"]), baseline_config["predictions_sha256"]),
        "official_scorer": _check_hash(starter_dir / "run_evaluation.py", starter["scorer_py_sha256"]),
    }

    # Shard validation uses answer-free rows. Gold labels are not attached to
    # examples until every frozen feature shard passes complete-coverage checks.
    source_rows = load_jsonl(input_path)
    answer_free_rows = strip_answers(source_rows)
    records, runtimes = merge_feature_shards(config, answer_free_rows)
    LOGGER.info("Validated multiscale features: sessions=%d", len(records))
    fold_manifest = _load_object(_resolve(folds_config["manifest"]))
    fold_by_index, split_audit = validate_session_fold_manifest(
        fold_manifest, answer_free_rows
    )
    arrays = feature_arrays(records, int(config["features"]["hidden_size"]))  # type: ignore[index]
    cache_path = output_dir / "features.npz"
    write_feature_cache(cache_path, arrays)
    candidate = {
        "name": "causal_multiscale_16_8_8",
        "parameters": config["multiscale"]["parameters"],  # type: ignore[index]
    }
    examples, values, names, dialog_audit = build_candidate_matrix(
        source_rows=source_rows,
        answer_free_rows=answer_free_rows,
        records=records,
        fold_by_index=fold_by_index,
        candidate=candidate,
        cache_path=cache_path,
        config=config,
    )
    training = config["training"]
    if not isinstance(training, Mapping):
        raise ValueError("D5 multiscale training config is malformed")
    decisions, fold_details = cross_validate_neural_matrix(
        examples,
        values,
        names,
        folds=int(folds_config["count"]),
        calibration_fold_offset=int(folds_config["calibration_fold_offset"]),
        seed=int(training["seed"]),
        max_iterations=int(training["max_iterations"]),
        l2_weights=[float(value) for value in training["l2_weights"]],  # type: ignore[index]
        l2_reduction=str(training["l2_reduction"]),
    )
    predictions = prediction_rows(examples, decisions)
    validate_prediction_rows(source_rows, predictions)
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    write_jsonl(predictions_path, predictions)
    _run_official_scorer(
        starter_dir, input_path, predictions_path, metrics_path, output_dir / "scorer.log"
    )
    metrics = _load_object(metrics_path)
    baseline_predictions = load_jsonl(_resolve(baseline_config["predictions"]))
    validate_prediction_rows(source_rows, baseline_predictions)
    quartiles = length_quartiles(answer_free_rows)
    candidate_flat = flatten_decisions(source_rows, predictions, list(range(len(source_rows))), quartiles)
    baseline_flat = flatten_decisions(source_rows, baseline_predictions, list(range(len(source_rows))), quartiles)
    scorer = load_official_scorer(starter_dir, str(starter["scorer_py_sha256"]))
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
    if not isinstance(evaluation, Mapping) or not isinstance(evaluation.get("promotion_gates"), Mapping):
        raise ValueError("D5 multiscale evaluation gates are malformed")
    bootstrap = paired_session_bootstrap(
        candidate_flat,
        baseline_flat,
        repetitions=int(evaluation["bootstrap_repetitions"]),
        seed=int(evaluation["bootstrap_seed"]),
    )
    baseline_summary = _load_object(_resolve(baseline_config["summary"]))
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
        "sessions": len(records),
        "chunks": len(examples),
        "feature_count": len(names),
        "total_parameters": config["model_accounting"]["total_parameters"],  # type: ignore[index]
        "feature_cache_sha256": sha256_file(cache_path),
        "predictions_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "shard_runtime": runtimes,
        "fingerprints": fingerprints,
        "wall_time_seconds": time.monotonic() - started,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "data_manifest.json",
        {
            "data": data,
            "protocol": protocol,
            "starter_kit": starter,
            "baseline": baseline_config,
            "folds": folds_config,
            "fingerprints": fingerprints,
            "split_audit": split_audit,
            "labels_used_only_after_feature_shard_validation": True,
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
                "状态：已完成冻结 causal multiscale 的 D4 session-level OOF。",
                "",
                f"- Sessions/chunks: `{len(records)}/{len(examples)}`",
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
        "Multiscale OOF complete: Macro=%.4f delta=%+.4f promoted=%s",
        float(overall["macro_f1"]),  # type: ignore[index]
        float(audit["macro_delta"]),
        audit["passed"],
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "d5_internvl35_1b_causal_multiscale_session_oof_v1.json"),
    )
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args(raw_argv)
    summary = run(_resolve(args.config), _resolve(args.experiment_dir), raw_argv)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
