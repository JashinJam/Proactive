"""Evaluate the frozen equal-mix robustness head on the D4 session OOF folds."""

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

from proactive_d1.core import prediction_rows, serialize_decision_head, strip_answers
from proactive_d4_1.compare import (
    decision_change_statistics,
    flatten_decisions,
    grouped_official_metrics,
    load_official_scorer,
    official_score,
    paired_session_bootstrap,
    stratified_statistics,
)
from proactive_d4_1.core import object_sha256
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

from .session_folds import validate_session_fold_manifest
from .robust import cross_validate_multiview_linear, static_promotion_gate


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGGER = logging.getLogger("proactive_d5.robust_evaluate")


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
        raise ValueError(f"D5 robust frozen artifact mismatch: {path}: {actual}")
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
        [sys.executable, "-m", "proactive_d5.run_robust_evaluate", *argv]
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


def _validate_records(
    records: Sequence[Mapping[str, object]],
    expected_indices: Sequence[int],
    rows: Sequence[Mapping[str, object]],
    *,
    hidden_size: int,
    frame_sampling: str,
) -> None:
    if len(records) != len(expected_indices):
        raise ValueError("D5 robust feature shard is incomplete")
    for position, (record, input_index) in enumerate(zip(records, expected_indices)):
        if int(record.get("input_index", -1)) != input_index:
            raise ValueError(f"D5 robust shard order changed at {position}")
        if record.get("video_path") != rows[input_index].get("video_path"):
            raise ValueError(f"D5 robust shard identity changed at {position}")
        chunks = record.get("chunks")
        intervals = rows[input_index].get("video_intervals")
        if not isinstance(chunks, list) or not isinstance(intervals, list):
            raise ValueError("D5 robust shard lacks chunks/intervals")
        if len(chunks) != len(intervals):
            raise ValueError("D5 robust shard chunk coverage changed")
        for chunk_index, chunk in enumerate(chunks):
            if not isinstance(chunk, Mapping) or int(chunk.get("chunk_index", -1)) != chunk_index:
                raise ValueError("D5 robust chunk order changed")
            hidden = chunk.get("hidden_state")
            if not isinstance(hidden, list) or len(hidden) != hidden_size:
                raise ValueError("D5 robust hidden width changed")
            if not all(math.isfinite(float(value)) for value in hidden):
                raise ValueError("D5 robust hidden state contains non-finite values")
            if chunk.get("frame_sampling") != frame_sampling:
                raise ValueError("D5 robust frame policy changed")
            for name in (
                "interval",
                "model_input_frames",
                "tag_margin",
                "silent_log_probability",
                "interrupt_log_probability",
                "prompt_tokens",
                "raw_response",
                "r0_answer",
            ):
                if name not in chunk:
                    raise ValueError(f"D5 robust chunk lacks {name}")


def _merge_shards(
    spec: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    *,
    hidden_size: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shard_root = _resolve(spec["shard_root"])
    shards = spec.get("shards")
    parameters = spec.get("parameters")
    if not isinstance(shards, list) or not isinstance(parameters, Mapping):
        raise ValueError("D5 robust shard spec is malformed")
    frame_sampling = str(parameters["frame_sampling"])
    _check_hash(_resolve(spec["feature_config"]), spec["feature_config_sha256"])
    records: list[dict[str, object]] = []
    runtimes: list[dict[str, object]] = []
    covered: list[int] = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise ValueError("D5 robust shard entry is malformed")
        shard_id = int(shard["id"])
        expected = list(range(int(shard["first_index"]), int(shard["last_index"]) + 1))
        shard_dir = shard_root / f"shard_{shard_id:03d}"
        _check_hash(shard_dir / "session_records.jsonl", shard["records_sha256"])
        _check_hash(shard_dir / "config.json", shard["config_sha256"])
        _check_hash(shard_dir / "runtime.json", shard["runtime_sha256"])
        shard_records = load_jsonl(shard_dir / "session_records.jsonl")
        _validate_records(
            shard_records,
            expected,
            rows,
            hidden_size=hidden_size,
            frame_sampling=frame_sampling,
        )
        effective = _load_object(shard_dir / "config.json")
        inference = effective.get("inference")
        runtime = effective.get("runtime")
        if not isinstance(inference, Mapping) or not isinstance(runtime, Mapping):
            raise ValueError("D5 robust shard lacks effective metadata")
        for name in (
            "max_frames",
            "frames_per_interval",
            "max_history_turns",
            "max_new_tokens",
            "frame_sampling",
        ):
            if inference.get(name) != parameters.get(name):
                raise ValueError(f"D5 robust shard parameter changed: {name}")
        if runtime.get("session_indices") != expected:
            raise ValueError("D5 robust shard runtime indices changed")
        runtime_summary = _load_object(shard_dir / "runtime.json")
        if runtime_summary.get("status") != "complete online dialog_stage_fused deployment run":
            raise ValueError("D5 robust shard runtime is incomplete")
        records.extend(shard_records)
        runtimes.append(runtime_summary)
        covered.extend(expected)
    if covered != list(range(len(rows))):
        raise ValueError("D5 robust shards do not exactly cover source sessions")
    return records, runtimes


def _fold_metrics(
    scorer: object,
    flat: Sequence[Mapping[str, object]],
    fold_by_index: Mapping[int, int],
) -> dict[str, object]:
    return grouped_official_metrics(
        scorer, flat, lambda row: str(fold_by_index[int(row["input_index"])])
    )


def run(config_path: Path, output_dir: Path, raw_argv: Sequence[str]) -> dict[str, object]:
    started = time.monotonic()
    config = _load_object(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"D5 robust evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "predictions").mkdir()
    (output_dir / "view_metrics").mkdir()
    (output_dir / "view_features").mkdir()
    _configure_logging(output_dir)
    write_json(output_dir / "config.json", config)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(output_dir / "code_state.txt", code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)))

    protocol = config.get("protocol")
    data = config.get("data")
    starter = config.get("starter_kit")
    folds = config.get("folds")
    baseline = config.get("baseline")
    views_config = config.get("views")
    features_config = config.get("features")
    training = config.get("training")
    evaluation = config.get("evaluation")
    sections = (protocol, data, starter, folds, baseline, views_config, features_config, training, evaluation)
    if not all(isinstance(value, Mapping) for value in sections):
        raise ValueError("D5 robust evaluation config sections are malformed")
    input_path = _resolve(data["input"])  # type: ignore[index]
    starter_dir = _resolve(starter["path"])  # type: ignore[index]
    fingerprints = {
        "config_object": object_sha256(config),
        "input": _check_hash(input_path, data["input_sha256"]),  # type: ignore[index]
        "protocol": _check_hash(_resolve(protocol["path"]), protocol["sha256"]),  # type: ignore[index]
        "fold_manifest": _check_hash(_resolve(folds["manifest"]), folds["manifest_sha256"]),  # type: ignore[index]
        "baseline_summary": _check_hash(_resolve(baseline["summary"]), baseline["summary_sha256"]),  # type: ignore[index]
        "baseline_predictions": _check_hash(_resolve(baseline["predictions"]), baseline["predictions_sha256"]),  # type: ignore[index]
        "official_scorer": _check_hash(starter_dir / "run_evaluation.py", starter["scorer_py_sha256"]),  # type: ignore[index]
    }
    source_rows = load_jsonl(input_path)
    answer_free_rows = strip_answers(source_rows)
    fold_manifest = _load_object(_resolve(folds["manifest"]))  # type: ignore[index]
    fold_by_index, split_audit = validate_session_fold_manifest(
        fold_manifest, answer_free_rows
    )
    hidden_size = int(features_config["hidden_size"])  # type: ignore[index]

    prepared_views = {}
    view_runtimes = {}
    view_cache_sha256 = {}
    for view_name in ("clean_history8", "history4", "assistant_drop", "frame_jitter"):
        spec = views_config.get(view_name)  # type: ignore[union-attr]
        if not isinstance(spec, Mapping):
            raise ValueError(f"D5 robust view is missing: {view_name}")
        view_input_path = _resolve(spec["input"])
        _check_hash(view_input_path, spec["input_sha256"])
        view_rows = load_jsonl(view_input_path)
        if any("answers" in row for row in view_rows):
            view_rows = strip_answers(view_rows)
        if view_name in ("clean_history8", "history4"):
            records_path = _resolve(spec["records"])
            cache_path = _resolve(spec["cache"])
            _check_hash(records_path, spec["records_sha256"])
            _check_hash(cache_path, spec["cache_sha256"])
            records = load_jsonl(records_path)
            runtimes: list[dict[str, object]] = []
        else:
            records, runtimes = _merge_shards(spec, view_rows, hidden_size=hidden_size)
            cache_path = output_dir / "view_features" / f"{view_name}.npz"
            write_feature_cache(cache_path, feature_arrays(records, hidden_size))
        prepared_views[view_name] = (view_rows, records, cache_path)
        view_runtimes[view_name] = runtimes
        view_cache_sha256[view_name] = sha256_file(cache_path)
        LOGGER.info("Validated robust artifacts for view %s", view_name)

    # Gold labels are first attached here, after every answer-free view and
    # feature shard has passed fingerprint and complete-coverage validation.
    view_examples = {}
    view_values = {}
    view_audits = {}
    names: tuple[str, ...] | None = None
    for view_name in ("clean_history8", "history4", "assistant_drop", "frame_jitter"):
        spec = views_config[view_name]  # type: ignore[index]
        if not isinstance(spec, Mapping):
            raise ValueError(f"D5 robust view is malformed: {view_name}")
        view_rows, records, cache_path = prepared_views[view_name]
        candidate = {"name": view_name, "parameters": spec["parameters"]}
        examples, values, current_names, dialog_audit = build_candidate_matrix(
            source_rows=source_rows,
            answer_free_rows=view_rows,
            records=records,
            fold_by_index=fold_by_index,
            candidate=candidate,
            cache_path=cache_path,
            config=config,
        )
        if names is None:
            names = current_names
        if current_names != names:
            raise ValueError(f"D5 robust feature names changed: {view_name}")
        if [example.key for example in examples] != [
            example.key for example in view_examples.get("clean_history8", examples)
        ]:
            raise ValueError(f"D5 robust example order changed: {view_name}")
        if [example.gold_interrupt for example in examples] != [
            example.gold_interrupt for example in view_examples.get("clean_history8", examples)
        ]:
            raise ValueError(f"D5 robust labels changed across views: {view_name}")
        view_examples[view_name] = examples
        view_values[view_name] = values
        view_audits[view_name] = dialog_audit
        LOGGER.info("Validated robust view %s: shape=%s", view_name, values.shape)
    if names is None:
        raise RuntimeError("D5 robust evaluation built no views")

    decisions, fold_details, heads = cross_validate_multiview_linear(
        view_examples["clean_history8"],
        view_values,
        names,
        clean_view="clean_history8",
        training_views=("clean_history8", "history4", "assistant_drop", "frame_jitter"),
        folds=int(folds["count"]),  # type: ignore[index]
        calibration_fold_offset=int(folds["calibration_fold_offset"]),  # type: ignore[index]
        seed=int(training["seed"]),  # type: ignore[index]
        max_iterations=int(training["max_iterations"]),  # type: ignore[index]
        l2_weights=[float(value) for value in training["l2_weights"]],  # type: ignore[index]
        l2_reduction=str(training["l2_reduction"]),  # type: ignore[arg-type,index]
    )
    scorer = load_official_scorer(starter_dir, str(starter["scorer_py_sha256"]))  # type: ignore[index]
    quartiles = length_quartiles(answer_free_rows)
    metrics_by_method: dict[str, dict[str, dict[str, object]]] = {
        "standard": {},
        "robust": {},
    }
    predictions_by_method = {"standard": {}, "robust": {}}
    flat_by_method = {"standard": {}, "robust": {}}
    stratified = {"standard": {}, "robust": {}}
    fold_metrics = {"standard": {}, "robust": {}}
    for method in ("standard", "robust"):
        for view_name in view_values:
            predictions = prediction_rows(
                view_examples[view_name], decisions[method][view_name]  # type: ignore[index]
            )
            validate_prediction_rows(source_rows, predictions)
            path = output_dir / "predictions" / f"{method}_{view_name}.jsonl"
            write_jsonl(path, predictions)
            metric = official_score(scorer, source_rows, predictions)
            write_json(output_dir / "view_metrics" / f"{method}_{view_name}.json", metric)
            flat = flatten_decisions(
                source_rows, predictions, list(range(len(source_rows))), quartiles
            )
            predictions_by_method[method][view_name] = predictions
            metrics_by_method[method][view_name] = metric
            flat_by_method[method][view_name] = flat
            stratified[method][view_name] = stratified_statistics(scorer, flat)
            fold_metrics[method][view_name] = _fold_metrics(scorer, flat, fold_by_index)

    clean_standard_path = output_dir / "predictions" / "standard_clean_history8.jsonl"
    reproduced_sha = sha256_file(clean_standard_path)
    if reproduced_sha != str(baseline["predictions_sha256"]):  # type: ignore[index]
        raise ValueError(
            "D5 robust clean comparator failed D4-fold baseline reproduction: "
            f"{reproduced_sha}"
        )
    gates = evaluation.get("static_gates")  # type: ignore[union-attr]
    if not isinstance(gates, Mapping):
        raise ValueError("D5 robust static gates are malformed")
    static_audit = static_promotion_gate(
        metrics_by_method,
        clean_view="clean_history8",
        perturbation_views=("history4", "assistant_drop", "frame_jitter"),
        maximum_clean_drop=float(gates["maximum_clean_drop"]),
        minimum_perturbation_gain=float(gates["minimum_each_perturbation_gain"]),
    )
    comparisons = {}
    for view_name in view_values:
        candidate_flat = flat_by_method["robust"][view_name]
        standard_flat = flat_by_method["standard"][view_name]
        comparisons[view_name] = {
            "macro_delta": round(
                float(metrics_by_method["robust"][view_name]["overall"]["macro_f1"])  # type: ignore[index]
                - float(metrics_by_method["standard"][view_name]["overall"]["macro_f1"]),  # type: ignore[index]
                6,
            ),
            "session_bootstrap": paired_session_bootstrap(
                candidate_flat,
                standard_flat,
                repetitions=int(evaluation["bootstrap_repetitions"]),  # type: ignore[index]
                seed=int(evaluation["bootstrap_seed"]),  # type: ignore[index]
            ),
            "decision_changes": decision_change_statistics(candidate_flat, standard_flat),
        }
    serialized_heads = {
        method: {
            str(fold): serialize_decision_head(
                head,
                {
                    "experiment_id": config["experiment_id"],
                    "method": method,
                    "test_fold": fold,
                    "classification": evaluation["classification"],  # type: ignore[index]
                },
            )
            for fold, head in method_heads.items()
        }
        for method, method_heads in heads.items()
    }
    write_json(output_dir / "fold_heads.json", serialized_heads)

    robust_clean_predictions = predictions_by_method["robust"]["clean_history8"]
    write_jsonl(output_dir / "predictions.jsonl", robust_clean_predictions)
    write_json(output_dir / "metrics.json", metrics_by_method["robust"]["clean_history8"])
    status = (
        "static complete; eligible for self-fed"
        if static_audit["self_fed_eligible"]
        else "complete; stopped before self-fed by frozen static gate"
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "classification": evaluation["classification"],  # type: ignore[index]
        "metrics": metrics_by_method,
        "static_promotion_audit": static_audit,
        "comparisons": comparisons,
        "fold_details": fold_details,
        "fold_metrics": fold_metrics,
        "stratified": stratified,
        "split_audit": split_audit,
        "dialog_feature_audit": view_audits,
        "view_cache_sha256": view_cache_sha256,
        "view_runtime": view_runtimes,
        "sessions": len(source_rows),
        "chunks": len(view_examples["clean_history8"]),
        "feature_count": len(names),
        "head_parameters": len(names) + 1,
        "total_parameters": config["model_accounting"]["total_parameters"],
        "clean_comparator_reproduced_sha256": reproduced_sha,
        "predictions_sha256": sha256_file(output_dir / "predictions.jsonl"),
        "metrics_sha256": sha256_file(output_dir / "metrics.json"),
        "fold_heads_sha256": sha256_file(output_dir / "fold_heads.json"),
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
            "folds": folds,
            "baseline": baseline,
            "views": views_config,
            "fingerprints": fingerprints,
            "split_audit": split_audit,
            "labels_attached_after_feature_validation": True,
            "external_data_used": False,
            "supervision": evaluation["classification"],  # type: ignore[index]
        },
    )
    clean_overall = metrics_by_method["robust"]["clean_history8"]["overall"]
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                f"状态：{status}。",
                "",
                f"- Sessions/chunks: `{len(source_rows)}/{len(view_examples['clean_history8'])}`",
                f"- Robust clean Macro/G-mean: `{clean_overall['macro_f1']:.4f}/{clean_overall['gmean_f1']:.4f}`",  # type: ignore[index]
                f"- Clean delta: `{static_audit['clean_delta']:+.4f}`",
                f"- Worst static perturbation delta: `{static_audit['worst_perturbation_delta']:+.4f}`",
                f"- Self-fed eligible: `{static_audit['self_fed_eligible']}`",
                f"- Predictions SHA256: `{summary['predictions_sha256']}`",
                "- 证据类型：post-selection、val-supervised robustness audit；不是 hidden-test 证据。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    LOGGER.info(
        "Robust static OOF complete: clean=%.4f worst_delta=%+.4f self_fed=%s",
        float(clean_overall["macro_f1"]),  # type: ignore[index]
        float(static_audit["worst_perturbation_delta"]),
        static_audit["self_fed_eligible"],
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/d5_internvl35_1b_robust_session_oof_v1.json"
    )
    parser.add_argument("--experiment-dir", required=True)
    args = parser.parse_args(raw_argv)
    summary = run(_resolve(args.config), _resolve(args.experiment_dir), raw_argv)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
