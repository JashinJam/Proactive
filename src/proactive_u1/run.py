"""Run fixed-D1-gate forced generation on the frozen U1 sample."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    verify_model_snapshot,
    write_json,
)
from proactive_r0.core import (
    CausalInferenceConfig,
    build_messages,
    load_jsonl,
    load_starter_kit,
    subsample_frames,
    validate_prediction_rows,
    write_jsonl,
)
from proactive_r0.run import _run_official_scorer
from proactive_u1.core import (
    controlled_messages,
    normalize_continuation,
    oracle_state_block,
    validate_oracle_annotations,
    validate_decision_invariance,
)
from proactive_u1.internvl import PrefillInternVLProactiveModel
from proactive_u1.prepare import strip_current_answers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_GENERATED_VARIANTS = {
    "forced_no_state",
    "forced_oracle_step",
    "forced_oracle_full",
}


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"U1 SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _annotations_by_input(
    rows: Sequence[dict[str, object]],
) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        input_index = int(row["input_index"])
        if input_index in result:
            raise ValueError(f"Duplicate U1 oracle annotation: {input_index}")
        result[input_index] = row
    return result


def _load_annotation_file(path: Path) -> list[dict[str, object]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("U1 oracle-state file must contain a JSON array of objects")
    return value


def _replace_variant_content(
    reference: Sequence[dict[str, object]],
    content_rows: Sequence[dict[str, object]],
    variant: str,
) -> list[dict[str, object]]:
    predictions = copy.deepcopy(list(reference))
    selected = [row for row in content_rows if row["variant"] == variant]
    for row in selected:
        input_index = int(row["input_index"])
        chunk_index = int(row["chunk_index"])
        answers = predictions[input_index]["answers"]
        assert isinstance(answers, list)
        answers[chunk_index] = row["answer"]
    validate_decision_invariance(reference, predictions)
    return predictions


def _metric_identity(
    frozen: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    keys = (
        "macro_f1",
        "interrupt_f1",
        "silent_f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "support",
    )
    differences = {
        key: [frozen[key], candidate[key]]
        for key in keys
        if frozen[key] != candidate[key]
    }
    if differences:
        raise ValueError(f"U1 official metrics changed: {differences}")
    return {"identical": True, "checked_fields": list(keys)}


def run(
    config_path: Path,
    output_dir: Path,
    device: str,
    variants: Sequence[str],
    smoke_only: bool,
    oracle_states_path: Path | None,
    require_exclusive_gpu: bool,
) -> dict[str, object]:
    unknown = set(variants) - ALLOWED_GENERATED_VARIANTS
    if unknown or not variants:
        raise ValueError(f"Unsupported or empty U1 variants: {sorted(unknown)}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"U1 output directory is not empty: {output_dir}")
    started = time.monotonic()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_config = config["model"]
    sources = config["sources"]
    prepared = config["prepared_sample"]
    starter_config = config["starter_kit"]
    inference_values = config["inference"]

    source_paths = {name: _resolve(value["path"]) for name, value in sources.items()}
    for name, value in sources.items():
        _check_hash(source_paths[name], value["sha256"])
    sample_path = _resolve(prepared["items"])
    _check_hash(sample_path, prepared["items_sha256"])
    _check_hash(_resolve(prepared["manifest"]), prepared["manifest_sha256"])
    _check_hash(_resolve(prepared["protocol"]), prepared["protocol_sha256"])
    starter_dir = _resolve(starter_config["path"])
    _check_hash(starter_dir / "model.py", starter_config["model_py_sha256"])
    _check_hash(
        starter_dir / "run_generate_proactive.py",
        starter_config["proactive_py_sha256"],
    )
    _check_hash(
        starter_dir / "run_evaluation.py", starter_config["scorer_py_sha256"]
    )

    model_path = Path(model_config["local_path"]).expanduser().resolve()
    model_audit = verify_model_snapshot(model_path, model_config)
    video_folder = _resolve(config["video_folder"])
    all_source_rows = load_jsonl(source_paths["gold_container"])
    label_free_sources = strip_current_answers(all_source_rows)
    frozen_predictions = load_jsonl(source_paths["d1_predictions"])
    r0_rows = load_jsonl(source_paths["r0_session_records"])
    samples = load_jsonl(sample_path)
    if smoke_only:
        samples = [row for row in samples if row["is_smoke"]]
    expected_samples = 16 if smoke_only else 80
    if len(samples) != expected_samples:
        raise ValueError(f"U1 expected {expected_samples} samples, got {len(samples)}")
    sample_keys = {
        (int(row["input_index"]), int(row["chunk_index"])) for row in samples
    }
    if len(sample_keys) != len(samples):
        raise ValueError("U1 sample contains duplicate chunks")

    oracle_annotations: dict[int, dict[str, object]] = {}
    oracle_validation: dict[str, object] | None = None
    if any(variant.startswith("forced_oracle") for variant in variants):
        if oracle_states_path is None:
            raise ValueError("Oracle U1 variants require --oracle-states")
        annotation_rows = _load_annotation_file(oracle_states_path)
        oracle_validation = validate_oracle_annotations(annotation_rows, samples)
        oracle_annotations = _annotations_by_input(annotation_rows)
        if smoke_only and "smoke_oracle_states_sha256" in prepared:
            _check_hash(
                oracle_states_path, str(prepared["smoke_oracle_states_sha256"])
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_config = copy.deepcopy(config)
    effective_config["runtime"] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "device": device,
        "variants": list(variants),
        "smoke_only": smoke_only,
        "oracle_states": str(oracle_states_path) if oracle_states_path else None,
        "require_exclusive_gpu": require_exclusive_gpu,
    }
    write_json(output_dir / "config.json", effective_config)
    _write_text(
        output_dir / "command.sh",
        " ".join(["PYTHONNOUSERSITE=1", "PYTHONPATH=src", *sys.argv]) + "\n",
    )
    (output_dir / "command.sh").chmod(0o755)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(
            PROJECT_ROOT,
            [
                config_path,
                PROJECT_ROOT / "src/proactive_u1/core.py",
                PROJECT_ROOT / "src/proactive_u1/internvl.py",
                PROJECT_ROOT / "src/proactive_u1/prepare.py",
                PROJECT_ROOT / "src/proactive_u1/run.py",
                PROJECT_ROOT / "src/proactive_u1/tests/test_core.py",
                PROJECT_ROOT / "src/proactive_u1/tests/test_prepare.py",
            ],
        ),
    )

    inference = CausalInferenceConfig(
        frames_per_interval=int(inference_values["frames_per_interval"]),
        max_frames=int(inference_values["max_frames"]),
        max_history_turns=int(inference_values["max_history_turns"]),
        max_new_tokens=int(inference_values["max_new_tokens"]),
    )
    starter = load_starter_kit(starter_dir)
    model = PrefillInternVLProactiveModel(
        model_path=str(model_path),
        device=device,
        dtype_name="bfloat16",
        attention_implementation="sdpa",
        seed=int(inference_values["seed"]),
        require_exclusive_gpu=require_exclusive_gpu,
        video_frame_size=int(inference_values["video_frame_size"]),
        pad_token_id=int(inference_values["pad_token_id"]),
    )

    samples_by_input: dict[int, list[dict[str, object]]] = defaultdict(list)
    for sample in samples:
        samples_by_input[int(sample["input_index"])].append(sample)
    content_rows: list[dict[str, object]] = []
    replay_rows: list[dict[str, object]] = []
    for input_index in sorted(samples_by_input):
        row = label_free_sources[input_index]
        selected_by_chunk = {
            int(sample["chunk_index"]): sample
            for sample in samples_by_input[input_index]
        }
        intervals = [
            (float(interval[0]), float(interval[1]))
            for interval in row["video_intervals"]  # type: ignore[index]
        ]
        cumulative_frames: list[object] = []
        max_chunk = max(selected_by_chunk)
        for chunk_index, interval in enumerate(intervals[: max_chunk + 1]):
            current_frames = starter.extract_frames(
                str(video_folder / str(row["video_path"])),
                intervals=[interval],
                frames_per_interval=inference.frames_per_interval,
            )
            cumulative_frames.extend(current_frames)
            if chunk_index not in selected_by_chunk:
                continue
            sample = selected_by_chunk[chunk_index]
            frames = subsample_frames(cumulative_frames, inference.max_frames)
            messages = build_messages(
                row=row,
                chunk_index=chunk_index,
                system_prompt=starter.system_prompt,
                normalize_dialog_turns=starter.normalize_dialog_turns,
                max_history_turns=inference.max_history_turns,
            )
            replay_started = time.monotonic()
            replay_raw = model.generate(
                frames, messages, max_new_tokens=inference.max_new_tokens
            )
            replay_seconds = time.monotonic() - replay_started
            expected_chunk = r0_rows[input_index]["chunks"][chunk_index]  # type: ignore[index]
            expected_raw = str(expected_chunk["raw_response"])
            replay = {
                "sample_id": sample["sample_id"],
                "input_index": input_index,
                "chunk_index": chunk_index,
                "current_interval_frames": len(current_frames),
                "model_input_frames": len(frames),
                "expected_current_interval_frames": expected_chunk.get(
                    "current_interval_frames"
                ),
                "expected_model_input_frames": expected_chunk.get("model_input_frames"),
                "expected_raw_response": expected_raw,
                "replayed_raw_response": replay_raw,
                "exact_match": replay_raw == expected_raw,
                "wall_time_seconds": replay_seconds,
            }
            replay_rows.append(replay)
            if not replay["exact_match"]:
                raise ValueError(
                    f"U1 R0 replay mismatch for {sample['sample_id']}: "
                    f"{replay_raw!r} != {expected_raw!r}"
                )
            if len(frames) != int(expected_chunk["model_input_frames"]):
                raise ValueError(f"U1 frame-count mismatch for {sample['sample_id']}")
            if len(current_frames) != int(expected_chunk["current_interval_frames"]):
                raise ValueError(
                    f"U1 current-frame-count mismatch for {sample['sample_id']}"
                )

            content_rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "input_index": input_index,
                    "chunk_index": chunk_index,
                    "domain": sample["domain"],
                    "position_bin": sample["position_bin"],
                    "variant": "current_fallback",
                    "answer": sample["current_output"],
                    "content": str(sample["current_output"]).removeprefix(
                        "$interrupt$"
                    ),
                    "used_fallback": True,
                    "wall_time_seconds": 0.0,
                }
            )
            for variant in variants:
                state_block = None
                if variant.startswith("forced_oracle"):
                    annotation = oracle_annotations.get(input_index)
                    if annotation is None:
                        raise ValueError(
                            f"Missing U1 oracle annotation for input {input_index}"
                        )
                    state_block = oracle_state_block(
                        annotation, str(sample["sample_id"]), variant
                    )
                generation_messages = controlled_messages(messages, state_block)
                generation_started = time.monotonic()
                generated = model.generate_prefilled(
                    frames,
                    generation_messages,
                    assistant_prefix=str(inference_values["assistant_prefix"]),
                    max_new_tokens=inference.max_new_tokens,
                )
                generation_seconds = time.monotonic() - generation_started
                normalized = normalize_continuation(generated["continuation"])
                content_rows.append(
                    {
                        "sample_id": sample["sample_id"],
                        "input_index": input_index,
                        "chunk_index": chunk_index,
                        "domain": sample["domain"],
                        "position_bin": sample["position_bin"],
                        "variant": variant,
                        **generated,
                        **normalized,
                        "state_block": state_block,
                        "wall_time_seconds": generation_seconds,
                    }
                )

    if len(replay_rows) != len(samples) or not all(
        row["exact_match"] for row in replay_rows
    ):
        raise ValueError("U1 R0 replay did not cover the complete selected sample")
    expected_content_rows = len(samples) * (1 + len(variants))
    if len(content_rows) != expected_content_rows:
        raise ValueError("U1 content generation coverage mismatch")
    write_jsonl(output_dir / "r0_replay_records.jsonl", replay_rows)
    write_jsonl(output_dir / "content_records.jsonl", content_rows)

    frozen_metrics = json.loads(
        source_paths["d1_metrics_summary"].read_text(encoding="utf-8")
    )["overall"]
    variant_summaries: dict[str, object] = {}
    for variant in ["current_fallback", *variants]:
        variant_dir = output_dir / "variants" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        predictions = _replace_variant_content(
            frozen_predictions, content_rows, variant
        )
        validation = validate_prediction_rows(all_source_rows, predictions)
        predictions_path = variant_dir / "predictions.jsonl"
        write_jsonl(predictions_path, predictions)
        metrics_path = variant_dir / "metrics.json"
        _run_official_scorer(
            starter_dir,
            source_paths["gold_container"],
            predictions_path,
            metrics_path,
            variant_dir / "scorer.log",
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        identity = _metric_identity(frozen_metrics, metrics["overall"])
        selected_rows = [row for row in content_rows if row["variant"] == variant]
        diagnostics = {
            "samples": len(selected_rows),
            "used_fallback": sum(bool(row["used_fallback"]) for row in selected_rows),
            "empty_continuation": sum(
                bool(row.get("empty_continuation", False)) for row in selected_rows
            ),
            "extra_interrupt_tag": sum(
                bool(row.get("extra_interrupt_tag", False)) for row in selected_rows
            ),
            "generated_silent_tag": sum(
                bool(row.get("generated_silent_tag", False)) for row in selected_rows
            ),
            "normalization_reasons": dict(
                Counter(
                    str(row["normalization"])
                    for row in selected_rows
                    if row.get("normalization")
                )
            ),
            "mean_generation_seconds": (
                sum(float(row["wall_time_seconds"]) for row in selected_rows)
                / len(selected_rows)
            ),
        }
        write_json(variant_dir / "diagnostics.json", diagnostics)
        variant_summaries[variant] = {
            "predictions_sha256": sha256_file(predictions_path),
            "metrics_sha256": sha256_file(metrics_path),
            "official_metric_identity": identity,
            "diagnostics": diagnostics,
        }

    runtime = {
        "wall_time_seconds": time.monotonic() - started,
        "peak_gpu_memory_bytes": model.peak_memory_bytes(),
        "preexisting_gpu_processes": model.preexisting_gpu_processes,
        "model_parameters": model.parameter_count,
        "device": device,
    }
    diagnostics = {
        "status": "engineering smoke complete" if smoke_only else "pilot generation complete",
        "smoke_only": smoke_only,
        "samples": len(samples),
        "sessions": len(samples_by_input),
        "r0_replay_exact": sum(bool(row["exact_match"]) for row in replay_rows),
        "r0_replay_total": len(replay_rows),
        "decision_invariance": True,
        "official_macro_f1": frozen_metrics["macro_f1"],
        "variants": variant_summaries,
        "runtime": runtime,
    }
    write_json(output_dir / "diagnostics.json", diagnostics)
    write_json(output_dir / "runtime.json", runtime)
    write_json(
        output_dir / "data_manifest.json",
        {
            "source_hashes": {
                name: value["sha256"] for name, value in sources.items()
            },
            "sample_items_sha256": prepared["items_sha256"],
            "protocol_sha256": prepared["protocol_sha256"],
            "generation_reads_current_gold_answers": False,
            "generation_reads_future_video_or_dialog": False,
            "official_scorer_reads_gold_after_generation": True,
            "external_data_used": False,
            "oracle_used": bool(oracle_annotations),
            "oracle_states_path": (
                str(oracle_states_path) if oracle_states_path else None
            ),
            "oracle_states_sha256": (
                sha256_file(oracle_states_path) if oracle_states_path else None
            ),
            "oracle_validation": oracle_validation,
            "model_audit": model_audit,
        },
    )
    _write_text(
        output_dir / "README.md",
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                f"Status: {diagnostics['status']}",
                "",
                "本实验固定 D1 的全部二元决策，只替换冻结样本的 utterance 正文。",
                f"R0 replay: {len(replay_rows)}/{len(replay_rows)} exact.",
                f"Official Macro F1: {frozen_metrics['macro_f1']} for every variant.",
                "人工内容评价尚未进行；smoke 不构成科学效果结论。",
                "",
            ]
        ),
    )
    _write_text(
        output_dir / "run.log",
        json.dumps(diagnostics, indent=2, ensure_ascii=True) + "\n",
    )
    return diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--variants", nargs="+", default=["forced_no_state"])
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--oracle-states", default=None)
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(
        config_path=_resolve(args.config),
        output_dir=_resolve(args.output_dir),
        device=args.device,
        variants=args.variants,
        smoke_only=args.smoke_only,
        oracle_states_path=(
            _resolve(args.oracle_states) if args.oracle_states else None
        ),
        require_exclusive_gpu=args.require_exclusive_gpu,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
