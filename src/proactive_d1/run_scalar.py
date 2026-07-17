"""Run D1 scalar controls with rotating session-held-out calibration folds."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    write_json,
)
from proactive_r0.core import load_jsonl, validate_prediction_rows, write_jsonl
from proactive_r0.run import _run_official_scorer, _validate_static_files

from .core import (
    FEATURE_VARIANTS,
    attach_gold_labels,
    build_label_free_chunks,
    cross_validate_linear,
    decisions_from_feature,
    feature_names,
    make_fold_manifest,
    metrics_for_subset,
    paired_session_bootstrap,
    prediction_rows,
    strip_answers,
    validate_fold_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "d1_internvl35_1b_scalar_oof.json"


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_d1.run_scalar", *argv])
    content = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n"
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _tracked_paths(config_path: Path) -> list[Path]:
    return [
        *sorted((PROJECT_ROOT / "src" / "proactive_r0").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_r0f").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1").glob("*.py")),
        *sorted((PROJECT_ROOT / "src" / "proactive_d1" / "tests").glob("*.py")),
        config_path,
        PROJECT_ROOT / "configs" / "r0_internvl35_1b_no_plan.json",
        PROJECT_ROOT / "configs" / "r0f_internvl35_1b_response_intent_repair.json",
        PROJECT_ROOT / "models" / "internvl35_1b_hf.json",
        PROJECT_ROOT / "CURRENT_ROUTE.md",
        PROJECT_ROOT / "C1_SPEC.md",
        PROJECT_ROOT / "Agent.md",
    ]


def _verify_reference(reference: dict[str, object]) -> tuple[Path, dict[str, str]]:
    directory = _resolve(reference["experiment_dir"])
    actual = {
        "predictions_sha256": sha256_file(directory / "predictions.jsonl"),
        "session_records_sha256": sha256_file(directory / "session_records.jsonl"),
        "metrics_sha256": sha256_file(directory / "metrics.json"),
    }
    expected = {key: str(reference[key]) for key in actual}
    if actual != expected:
        raise ValueError(f"Frozen reference mismatch for {directory}: {actual} != {expected}")
    return directory, actual


def _fold_summary(manifest: dict[str, object]) -> dict[str, object]:
    sessions = manifest["sessions"]
    assert isinstance(sessions, list)
    by_fold: Counter[int] = Counter()
    chunks_by_fold: Counter[int] = Counter()
    domain_by_fold: Counter[str] = Counter()
    for entry in sessions:
        assert isinstance(entry, dict)
        fold = int(entry["fold"])
        domain = str(entry["domain"])
        by_fold[fold] += 1
        chunks_by_fold[fold] += int(entry["chunks"])
        domain_by_fold[f"{fold}:{domain}"] += 1
    return {
        "sessions_by_fold": dict(sorted(by_fold.items())),
        "chunks_by_fold": dict(sorted(chunks_by_fold.items())),
        "domain_sessions_by_fold": dict(sorted(domain_by_fold.items())),
    }


def _readme(
    config: dict[str, object],
    status: str,
    comparison: dict[str, object] | None = None,
) -> str:
    lines = [
        f"# {config['experiment_id']}",
        "",
        f"Status: **{status}**",
        "",
        str(config["hypothesis"]),
        "",
        "## Boundary",
        "",
        "- Five-fold session-level OOF development on public validation labels",
        "- Label-independent domain-stratified fold assignment",
        "- Three fit folds, one threshold-calibration fold, one test fold per rotation",
        "- No new video/model inference; features use frozen R0 records and causal metadata",
        "- Val-supervised development, not hidden-test evidence",
    ]
    if comparison:
        variants = comparison.get("variants")
        if isinstance(variants, dict):
            lines.extend(["", "## Official OOF Results", ""])
            for name, payload in variants.items():
                if isinstance(payload, dict):
                    lines.append(
                        f"- `{name}`: Macro F1 `{payload['macro_f1']}`, "
                        f"delta vs R0-F `{payload['delta_macro_f1_vs_r0f']}`, "
                        f"promoted `{payload['promotion_gate_passed']}`"
                    )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    started_at = time.monotonic()
    config_path = _resolve(args.config)
    config = _load_json(config_path)
    required = {
        "experiment_id",
        "hypothesis",
        "model",
        "data",
        "starter_kit",
        "r0_reference",
        "r0f_reference",
        "split",
        "features",
        "training",
        "evaluation",
        "validation_policy",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"D1 config missing keys: {sorted(missing)}")
    data_config = dict(config["data"])  # type: ignore[arg-type]
    starter_config = dict(config["starter_kit"])  # type: ignore[arg-type]
    split_config = dict(config["split"])  # type: ignore[arg-type]
    feature_config = dict(config["features"])  # type: ignore[arg-type]
    training_config = dict(config["training"])  # type: ignore[arg-type]
    evaluation_config = dict(config["evaluation"])  # type: ignore[arg-type]
    model_config = dict(config["model"])  # type: ignore[arg-type]
    r0_reference = dict(config["r0_reference"])  # type: ignore[arg-type]
    r0f_reference = dict(config["r0f_reference"])  # type: ignore[arg-type]

    input_path = _resolve(data_config["input"])
    starter_dir = _resolve(starter_config["path"])
    output_dir = _resolve(
        args.output_dir or f"output/experiments/{config['experiment_id']}"
    )
    if output_dir.exists():
        raise FileExistsError(f"D1 output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    fingerprints = _validate_static_files(config, input_path, starter_dir)
    r0_dir, r0_hashes = _verify_reference(r0_reference)
    r0f_dir, r0f_hashes = _verify_reference(r0f_reference)
    source_rows = load_jsonl(input_path)
    r0_records = load_jsonl(r0_dir / "session_records.jsonl")
    if len(source_rows) != 700 or len(r0_records) != 700:
        raise ValueError("D1 scalar control requires all 700 R0 sessions")

    folds = int(split_config["folds"])
    if split_config.get("algorithm") != "domain_stratified_sha256_round_robin":
        raise ValueError("D1 config split algorithm is not supported")
    manifest = make_fold_manifest(
        strip_answers(source_rows), folds=folds, seed=str(split_config["seed"])
    )
    fold_by_index = validate_fold_manifest(manifest, strip_answers(source_rows))
    write_json(output_dir / "split_manifest.json", manifest)
    fold_summary = _fold_summary(manifest)

    label_free = build_label_free_chunks(
        strip_answers(source_rows),
        r0_records,
        fold_by_index,
        max_history_turns=int(feature_config["max_history_turns"]),
    )
    if len(label_free) != 9935:
        raise ValueError(f"Unexpected D1 chunk count: {len(label_free)}")
    domains = sorted({feature.domain for feature in label_free})
    configured_variants = tuple(str(value) for value in feature_config["variants"])
    if any(variant not in FEATURE_VARIANTS for variant in configured_variants):
        raise ValueError(f"Unsupported D1 feature variants: {configured_variants}")
    feature_audit = {
        "sessions": len(source_rows),
        "chunks": len(label_free),
        "domains": domains,
        "fold_summary": fold_summary,
        "feature_names": {
            variant: feature_names(variant, domains) for variant in configured_variants
        },
        "feature_rows_contain_gold": False,
        "fold_assignment_reads_gold": False,
    }
    write_json(output_dir / "feature_audit.json", feature_audit)

    effective = json.loads(json.dumps(config))
    effective["runtime"] = {
        "config_path": str(config_path),
        "input_path": str(input_path),
        "r0_reference_dir": str(r0_dir),
        "r0f_reference_dir": str(r0f_dir),
        "output_dir": str(output_dir),
        "audit_only": args.audit_only,
        "gpu_used": False,
        "model_inference_rerun": False,
    }
    write_json(output_dir / "config.json", effective)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(PROJECT_ROOT, _tracked_paths(config_path)),
    )
    write_json(
        output_dir / "data_manifest.json",
        {
            "source": {
                "path": str(input_path),
                "sha256": fingerprints["input_sha256"],
                "sessions": 700,
                "chunks": 9935,
                "top_level_license": data_config["license"],
            },
            "r0_reference": {"path": str(r0_dir), **r0_hashes},
            "r0f_reference": {"path": str(r0f_dir), **r0f_hashes},
            "starter_kit_sha256": fingerprints,
            "supervision": config["validation_policy"],
        },
    )
    if args.audit_only:
        runtime = {
            "status": "audit_only",
            "completed_at": datetime.now().astimezone().isoformat(),
            "wall_time_seconds": round(time.monotonic() - started_at, 3),
            "gpu_used": False,
            "labels_attached": False,
        }
        write_json(output_dir / "runtime.json", runtime)
        (output_dir / "README.md").write_text(
            _readme(config, "audit only; label-free features verified"),
            encoding="utf-8",
        )
        print(json.dumps({"status": "audit_only", **feature_audit}, sort_keys=True))
        return

    examples = attach_gold_labels(label_free, source_rows)
    r0f_decisions = decisions_from_feature(examples, "r0f_decision_interrupt")
    r0f_rebuilt = prediction_rows(examples, r0f_decisions)
    r0f_rebuilt_path = output_dir / "r0f_rebuilt_predictions.jsonl"
    write_jsonl(r0f_rebuilt_path, r0f_rebuilt)
    rebuilt_sha = sha256_file(r0f_rebuilt_path)
    if rebuilt_sha != r0f_reference["predictions_sha256"]:
        raise ValueError(
            "Label-free D1 reconstruction does not reproduce frozen R0-F: "
            f"{rebuilt_sha} != {r0f_reference['predictions_sha256']}"
        )

    variants_comparison: dict[str, object] = {}
    variant_metrics: dict[str, dict[str, object]] = {}
    max_head_parameters = 0
    for variant in configured_variants:
        names = feature_names(variant, domains)  # type: ignore[arg-type]
        max_head_parameters = max(max_head_parameters, len(names) + 1)
        decisions, fold_details = cross_validate_linear(
            examples,
            names,
            folds=folds,
            calibration_fold_offset=int(split_config["calibration_fold_offset"]),
            seed=int(training_config["seed"]),
            max_iterations=int(training_config["max_iterations"]),
            l2_weight=float(training_config["l2_weight"]),
        )
        predictions = prediction_rows(examples, decisions)
        validation = validate_prediction_rows(source_rows, predictions)
        variant_dir = output_dir / "variants" / variant
        variant_dir.mkdir(parents=True)
        predictions_path = variant_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        internal = metrics_for_subset(examples, decisions, include_first=True)
        non_first = metrics_for_subset(examples, decisions, include_first=False)
        r0f_non_first = metrics_for_subset(
            examples, r0f_decisions, include_first=False
        )
        bootstrap = paired_session_bootstrap(
            examples,
            decisions,
            r0f_decisions,
            repetitions=int(evaluation_config["bootstrap_repetitions"]),
            seed=int(evaluation_config["bootstrap_seed"]),
        )
        diagnostics = {
            **validation,
            "feature_variant": variant,
            "feature_names": names,
            "head_parameters_per_fold": len(names) + 1,
            "fold_details": fold_details,
            "internal_oof_metrics": internal,
            "non_first_chunk_metrics": non_first,
            "r0f_non_first_chunk_metrics": r0f_non_first,
            "paired_session_bootstrap_vs_r0f": bootstrap,
            "predictions_sha256": sha256_file(predictions_path),
        }
        write_json(variant_dir / "diagnostics.json", diagnostics)
        write_jsonl(
            variant_dir / "oof_records.jsonl",
            [
                {
                    "input_index": example.feature.input_index,
                    "video_path": example.feature.video_path,
                    "fold": example.feature.fold,
                    "chunk_index": example.feature.chunk_index,
                    "gold_interrupt": example.gold_interrupt,
                    "predicted_interrupt": decisions[example.key],
                    "r0f_interrupt": r0f_decisions[example.key],
                }
                for example in examples
            ],
        )
        metrics_path = variant_dir / "metrics.json"
        _run_official_scorer(
            starter_dir,
            input_path,
            predictions_path,
            metrics_path,
            variant_dir / "scorer.log",
        )
        metrics = _load_json(metrics_path)
        overall = metrics["overall"]
        assert isinstance(overall, dict)
        variant_metrics[variant] = overall
        r0f_macro = float(r0f_reference["macro_f1"])
        delta = float(overall["macro_f1"]) - r0f_macro
        non_first_gain = float(non_first["macro_f1"]) > float(
            r0f_non_first["macro_f1"]
        )
        promotion = (
            delta >= float(evaluation_config["promotion_min_delta_vs_r0f"])
            and float(bootstrap["delta_macro_f1_p2_5"]) > 0
            and float(overall["interrupt_f1"]) > 0
            and float(overall["silent_f1"]) > 0
            and non_first_gain
        )
        variants_comparison[variant] = {
            "macro_f1": overall["macro_f1"],
            "interrupt_f1": overall["interrupt_f1"],
            "silent_f1": overall["silent_f1"],
            "predicted_interrupt_rate": internal["predicted_interrupt_rate"],
            "delta_macro_f1_vs_r0f": round(delta, 6),
            "non_first_chunk_macro_f1": non_first["macro_f1"],
            "r0f_non_first_chunk_macro_f1": r0f_non_first["macro_f1"],
            "bootstrap": bootstrap,
            "promotion_gate_passed": promotion,
        }

    r0f_metrics = _load_json(r0f_dir / "metrics.json")
    comparison = {
        "classification": "val-supervised session-level out-of-fold development",
        "r0f_reference": r0f_metrics["overall"],
        "variants": variants_comparison,
        "promotion_rule": {
            "min_delta_macro_f1_vs_r0f": evaluation_config[
                "promotion_min_delta_vs_r0f"
            ],
            "positive_session_bootstrap_lower_bound": True,
            "non_first_chunk_gain": True,
            "both_class_f1_nonzero": True,
        },
    }
    write_json(output_dir / "comparison.json", comparison)
    total_parameters = int(model_config["total_parameters"]) + max_head_parameters
    if total_parameters > 2_000_000_000:
        raise ValueError("D1 linear head would exceed the Small parameter limit")
    runtime = {
        "status": "complete scalar OOF controls",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": round(time.monotonic() - started_at, 3),
        "gpu_used": False,
        "model_inference_rerun": False,
        "sessions": 700,
        "chunks": 9935,
        "folds": folds,
        "max_head_parameters": max_head_parameters,
        "max_total_parameters_with_one_deployed_head": total_parameters,
        "small_limit_parameters": 2_000_000_000,
    }
    write_json(output_dir / "runtime.json", runtime)
    (output_dir / "README.md").write_text(
        _readme(config, "complete scalar OOF controls", comparison),
        encoding="utf-8",
    )
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()

