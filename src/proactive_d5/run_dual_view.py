"""Evaluate the frozen uniform/multiscale dual-view linear heads."""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import shutil
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
from proactive_d4_2.evaluate import build_candidate_matrix, length_quartiles
from proactive_r0.artifacts import code_snapshot, environment_snapshot, sha256_file, write_json
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer

from .dual_view import build_dual_view_matrices
from .session_folds import validate_session_fold_manifest
from .run_multiscale_evaluate import (
    _previous_gold_rows,
    merge_feature_shards,
    promotion_audit,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("proactive_d5.dual_view")


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
    command = shlex.join([sys.executable, "-m", "proactive_d5.run_dual_view", *argv])
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def run(config_path: Path, output_dir: Path, raw_argv: Sequence[str]) -> dict[str, object]:
    started = time.monotonic()
    config = _load_object(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D5 dual-view output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure(output_dir)
    write_json(output_dir / "config.json", config)
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
    uniform = config["uniform"]
    multiscale = config["multiscale"]
    baseline = config["baseline"]
    folds = config["folds"]
    if not all(isinstance(value, Mapping) for value in (protocol, data, starter, uniform, multiscale, baseline, folds)):
        raise ValueError("D5 dual-view config sections are malformed")
    input_path = _resolve(data["input"])
    starter_dir = _resolve(starter["path"])
    multiscale_dir = _resolve(multiscale["evaluation_dir"])
    multiscale_summary = _load_object(multiscale_dir / "summary.json")
    if multiscale_summary.get("status") != "complete":
        raise ValueError("D5 dual-view requires complete multiscale evaluation artifacts")
    cache_path = _resolve(multiscale["cache"])
    if sha256_file(cache_path) != multiscale_summary.get("feature_cache_sha256"):
        raise ValueError("D5 multiscale cache differs from its evaluation summary")
    fingerprints = {
        "config_object": object_sha256(config),
        "input": _check_hash(input_path, data["input_sha256"]),
        "protocol": _check_hash(_resolve(protocol["path"]), protocol["sha256"]),
        "uniform_records": _check_hash(_resolve(uniform["records"]), uniform["records_sha256"]),
        "uniform_cache": _check_hash(_resolve(uniform["cache"]), uniform["cache_sha256"]),
        "fold_manifest": _check_hash(_resolve(folds["manifest"]), folds["manifest_sha256"]),
        "baseline_summary": _check_hash(_resolve(baseline["summary"]), baseline["summary_sha256"]),
        "baseline_predictions": _check_hash(_resolve(baseline["predictions"]), baseline["predictions_sha256"]),
        "multiscale_cache": sha256_file(cache_path),
        "multiscale_summary": sha256_file(multiscale_dir / "summary.json"),
        "official_scorer": _check_hash(starter_dir / "run_evaluation.py", starter["scorer_py_sha256"]),
    }
    source_rows = load_jsonl(input_path)
    answer_free_rows = strip_answers(source_rows)
    fold_manifest = _load_object(_resolve(folds["manifest"]))
    fold_by_index, split_audit = validate_session_fold_manifest(fold_manifest, answer_free_rows)
    multiscale_config = _load_object(_resolve(multiscale["evaluation_config"]))
    multiscale_records, _ = merge_feature_shards(multiscale_config, answer_free_rows)
    uniform_records = load_jsonl(_resolve(uniform["records"]))
    uniform_candidate = {"name": "uniform", "parameters": uniform["parameters"]}
    multiscale_candidate = {"name": "multiscale", "parameters": multiscale["parameters"]}
    examples, uniform_values, names, dialog_audit = build_candidate_matrix(
        source_rows=source_rows,
        answer_free_rows=answer_free_rows,
        records=uniform_records,
        fold_by_index=fold_by_index,
        candidate=uniform_candidate,
        cache_path=_resolve(uniform["cache"]),
        config=config,
    )
    multiscale_examples, multiscale_values, multiscale_names, _ = build_candidate_matrix(
        source_rows=source_rows,
        answer_free_rows=answer_free_rows,
        records=multiscale_records,
        fold_by_index=fold_by_index,
        candidate=multiscale_candidate,
        cache_path=cache_path,
        config=config,
    )
    if [example.key for example in examples] != [example.key for example in multiscale_examples] or names != multiscale_names:
        raise ValueError("D5 dual views are not aligned")
    matrices = build_dual_view_matrices(
        uniform_values,
        multiscale_values,
        names,
        gate_feature=str(config["features"]["gate_feature"]),  # type: ignore[index]
    )
    candidate_specs = config["candidates"]
    if not isinstance(candidate_specs, list):
        raise ValueError("D5 dual-view candidates are malformed")
    training = config["training"]
    evaluation = config["evaluation"]
    if not isinstance(training, Mapping) or not isinstance(evaluation, Mapping):
        raise ValueError("D5 dual-view training/evaluation config is malformed")
    scorer = load_official_scorer(starter_dir, str(starter["scorer_py_sha256"]))
    quartiles = length_quartiles(answer_free_rows)
    baseline_predictions = load_jsonl(_resolve(baseline["predictions"]))
    baseline_summary = _load_object(_resolve(baseline["summary"]))
    baseline_flat = flatten_decisions(source_rows, baseline_predictions, list(range(len(source_rows))), quartiles)
    baseline_strata = stratified_statistics(scorer, baseline_flat)
    baseline_previous = grouped_official_metrics(
        scorer,
        _previous_gold_rows(baseline_flat, source_rows),
        lambda row: str(row["previous_gold"]),
    )

    results: dict[str, dict[str, object]] = {}
    flat_by_name: dict[str, list[dict[str, object]]] = {}
    for spec in candidate_specs:
        if not isinstance(spec, Mapping):
            raise ValueError("D5 dual-view candidate entry is malformed")
        name = str(spec["name"])
        values, candidate_names = matrices[name]
        if values.shape[1] != int(spec["feature_count"]) or len(candidate_names) + 1 != int(spec["head_parameters"]):
            raise ValueError(f"D5 dual-view parameter accounting changed: {name}")
        decisions, fold_details = cross_validate_neural_matrix(
            examples,
            values,
            candidate_names,
            folds=int(folds["count"]),
            calibration_fold_offset=int(folds["calibration_fold_offset"]),
            seed=int(training["seed"]),
            max_iterations=int(training["max_iterations"]),
            l2_weights=[float(value) for value in training["l2_weights"]],  # type: ignore[index]
            l2_reduction=str(training["l2_reduction"]),
        )
        predictions = prediction_rows(examples, decisions)
        validate_prediction_rows(source_rows, predictions)
        candidate_dir = output_dir / "candidates" / name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = candidate_dir / "predictions.jsonl"
        metrics_path = candidate_dir / "metrics.json"
        write_jsonl(predictions_path, predictions)
        _run_official_scorer(starter_dir, input_path, predictions_path, metrics_path, candidate_dir / "scorer.log")
        metrics = _load_object(metrics_path)
        flat = flatten_decisions(source_rows, predictions, list(range(len(source_rows))), quartiles)
        flat_by_name[name] = flat
        strata = stratified_statistics(scorer, flat)
        previous = grouped_official_metrics(
            scorer,
            _previous_gold_rows(flat, source_rows),
            lambda row: str(row["previous_gold"]),
        )
        bootstrap = paired_session_bootstrap(
            flat,
            baseline_flat,
            repetitions=int(evaluation["bootstrap_repetitions"]),
            seed=int(evaluation["bootstrap_seed"]),
        )
        audit = promotion_audit(
            candidate_overall=metrics["overall"],  # type: ignore[arg-type]
            baseline_overall=baseline_summary["overall"],  # type: ignore[arg-type]
            candidate_flat=flat,
            baseline_flat=baseline_flat,
            candidate_strata=strata,
            baseline_strata=baseline_strata,
            candidate_previous=previous,
            baseline_previous=baseline_previous,
            fold_by_index=fold_by_index,
            scorer=scorer,
            gates=evaluation["promotion_gates"],  # type: ignore[arg-type]
            bootstrap=bootstrap,
        )
        result = {
            "name": name,
            "overall": metrics["overall"],
            "feature_count": len(candidate_names),
            "head_parameters": len(candidate_names) + 1,
            "added_head_parameters": spec["added_head_parameters"],
            "fold_details": fold_details,
            "stratified": strata,
            "previous_gold_stratified": previous,
            "session_bootstrap_vs_baseline": bootstrap,
            "decision_changes_vs_baseline": decision_change_statistics(flat, baseline_flat),
            "promotion_audit": audit,
            "predictions_sha256": sha256_file(predictions_path),
            "metrics_sha256": sha256_file(metrics_path),
        }
        write_json(candidate_dir / "summary.json", result)
        results[name] = result
        LOGGER.info("Dual-view %s complete: Macro=%.4f", name, float(metrics["overall"]["macro_f1"]))  # type: ignore[index]

    selection = config["selection"]
    if not isinstance(selection, Mapping):
        raise ValueError("D5 dual-view selection rule is malformed")
    shared_macro = float(results["shared_delta"]["overall"]["macro_f1"])  # type: ignore[index]
    gated_macro = float(results["dialog_gated_delta"]["overall"]["macro_f1"])  # type: ignore[index]
    selected_name = (
        "dialog_gated_delta"
        if gated_macro - shared_macro >= float(selection["gated_minimum_macro_advantage"])
        else str(selection["default"])
    )
    selected = results[selected_name]
    selected_dir = output_dir / "candidates" / selected_name
    shutil.copyfile(selected_dir / "predictions.jsonl", output_dir / "predictions.jsonl")
    shutil.copyfile(selected_dir / "metrics.json", output_dir / "metrics.json")
    summary = {
        "schema_version": 1,
        "status": "complete",
        "classification": evaluation["classification"],
        "selected_candidate": selected_name,
        "selection_margin_gated_minus_shared": gated_macro - shared_macro,
        "selected_promoted": selected["promotion_audit"]["passed"],  # type: ignore[index]
        "candidates": results,
        "baseline_overall": baseline_summary["overall"],
        "split_audit": split_audit,
        "dialog_feature_audit": dialog_audit,
        "sessions": len(source_rows),
        "chunks": len(examples),
        "maximum_total_parameters": config["model_accounting"]["maximum_total_parameters"],  # type: ignore[index]
        "fingerprints": fingerprints,
        "predictions_sha256": sha256_file(output_dir / "predictions.jsonl"),
        "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        "wall_time_seconds": time.monotonic() - started,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(
        output_dir / "data_manifest.json",
        {
            "data": data,
            "protocol": protocol,
            "uniform": uniform,
            "multiscale": multiscale,
            "baseline": baseline,
            "folds": folds,
            "fingerprints": fingerprints,
            "split_audit": split_audit,
            "external_data_used": False,
            "supervision": evaluation["classification"],
        },
    )
    selected_overall = selected["overall"]
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：已完成冻结双视图候选的 D4 session-level OOF。",
                "",
                f"- Selected candidate: `{selected_name}`",
                f"- Official Macro/G-mean: `{selected_overall['macro_f1']:.4f}/{selected_overall['gmean_f1']:.4f}`",  # type: ignore[index]
                f"- Promotion gates passed: `{summary['selected_promoted']}`",
                f"- Predictions SHA256: `{summary['predictions_sha256']}`",
                "- 证据类型：post-selection、val-supervised；不是独立泛化或 hidden-test 证据。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "d5_internvl35_1b_dual_view_session_oof_v1.json"),
    )
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args(raw_argv)
    summary = run(_resolve(args.config), _resolve(args.experiment_dir), raw_argv)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
