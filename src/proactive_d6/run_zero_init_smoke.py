"""Run the frozen 102-chunk D6 zero-init equivalence and causality smoke."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Mapping, Sequence

import torch

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl, load_starter_kit

from .contract import gpu_resource_audit, inference_config, load_experiment
from .run_fold import _build_model, _extract_records, _mapping, _write_contract
from .runtime import process_session


DEFAULT_CONFIG = Path("configs/d6_internvl35_1b_query_memory_lora_oof_v1.json")
D4_3_RECORDS = Path(
    "output/experiments/20260721_internvl35_1b_d4_3_history8_gpu_equivalence_v1/"
    "session_records.jsonl"
)
D4_3_RECORDS_SHA256 = "e12310419db40eb1957ded6ae513872e151bf47fd62eb2c0ff2c05bbc337970e"


def _by_index(records: Sequence[dict[str, object]]) -> dict[int, dict[str, object]]:
    result = {int(record["input_index"]): record for record in records}
    if len(result) != len(records):
        raise ValueError("D6 smoke reference has duplicate sessions")
    return result


def _compare(
    actual: Sequence[dict[str, object]],
    expected: Mapping[int, dict[str, object]],
) -> dict[str, object]:
    exact_fields = ("raw_response", "prompt_tokens", "model_input_frames")
    numeric_fields = (
        "tag_margin",
        "silent_log_probability",
        "interrupt_log_probability",
    )
    exact = {field: 0 for field in exact_fields}
    maximum = {field: 0.0 for field in (*numeric_fields, "hidden_state")}
    residual_maximum = 0.0
    update_maximum = 0.0
    chunks = 0
    for record in actual:
        input_index = int(record["input_index"])
        expected_record = expected[input_index]
        if record["video_path"] != expected_record["video_path"]:
            raise ValueError("D6 smoke video identity changed")
        actual_chunks = record["chunks"]
        expected_chunks = expected_record["chunks"]
        if not isinstance(actual_chunks, list) or not isinstance(expected_chunks, list):
            raise ValueError("D6 smoke chunks are malformed")
        if len(actual_chunks) != len(expected_chunks):
            raise ValueError("D6 smoke chunk coverage changed")
        for actual_chunk, expected_chunk in zip(actual_chunks, expected_chunks):
            if not isinstance(actual_chunk, Mapping) or not isinstance(expected_chunk, Mapping):
                raise ValueError("D6 smoke chunk is malformed")
            for field in exact_fields:
                if actual_chunk[field] == expected_chunk[field]:
                    exact[field] += 1
            for field in numeric_fields:
                maximum[field] = max(
                    maximum[field],
                    abs(float(actual_chunk[field]) - float(expected_chunk[field])),
                )
            actual_hidden = actual_chunk["hidden_state"]
            expected_hidden = expected_chunk["hidden_state"]
            if not isinstance(actual_hidden, list) or not isinstance(expected_hidden, list):
                raise ValueError("D6 smoke hidden state is malformed")
            if len(actual_hidden) != 1024 or len(expected_hidden) != 1024:
                raise ValueError("D6 smoke hidden width changed")
            maximum["hidden_state"] = max(
                maximum["hidden_state"],
                max(abs(float(left) - float(right)) for left, right in zip(actual_hidden, expected_hidden)),
            )
            residual_maximum = max(
                residual_maximum, float(actual_chunk["memory_residual_norm"])
            )
            update_maximum = max(
                update_maximum,
                float(actual_chunk["candidate_memory_update_max_abs_difference"]),
            )
            chunks += 1
    return {
        "chunks": chunks,
        "exact_match_counts": exact,
        "maximum_abs_differences": maximum,
        "maximum_memory_residual_norm": residual_maximum,
        "maximum_candidate_update_difference": update_maximum,
    }


def _first_chunk_signature(record: Mapping[str, object]) -> dict[str, object]:
    chunks = record.get("chunks")
    if not isinstance(chunks, list) or len(chunks) != 1 or not isinstance(chunks[0], Mapping):
        raise ValueError("D6 causality replay did not produce exactly one chunk")
    chunk = chunks[0]
    return {
        "prompt_tokens": chunk["prompt_tokens"],
        "model_input_frames": chunk["model_input_frames"],
        "frame_source_indices": chunk["frame_source_indices"],
        "tag_margin": chunk["tag_margin"],
        "silent_log_probability": chunk["silent_log_probability"],
        "interrupt_log_probability": chunk["interrupt_log_probability"],
        "hidden_state": chunk["hidden_state"],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--model-path")
    args = parser.parse_args(argv)

    inputs = load_experiment(
        Path(args.config),
        model_path_override=Path(args.model_path) if args.model_path else None,
    )
    source_indices = [int(value) for value in inputs.config["smoke"]["source_indices"]]  # type: ignore[index]
    expected_chunks = int(inputs.config["smoke"]["chunks"])  # type: ignore[index]
    d4_records_path = D4_3_RECORDS.resolve()
    if sha256_file(d4_records_path) != D4_3_RECORDS_SHA256:
        raise ValueError("D6 frozen D4.3 smoke reference SHA256 changed")
    expected = _by_index(load_jsonl(d4_records_path))
    if set(expected) != set(source_indices):
        raise ValueError("D6 frozen D4.3 smoke source sessions changed")

    resources = _mapping(inputs.config["resources"], "resources")
    previous_summary: dict[str, object] | None = None
    previous_path = Path(args.output_dir).resolve() / "summary.json"
    if previous_path.exists():
        value = json.loads(previous_path.read_text(encoding="utf-8"))
        if (
            isinstance(value, dict)
            and value.get("kind") == "d6_zero_init_causality_smoke"
            and value.get("config_sha256") == inputs.config_sha256
        ):
            previous_summary = value
    resource = gpu_resource_audit(args.device, float(resources["minimum_free_memory_gib"]))
    output_dir = Path(args.output_dir).resolve()
    _write_contract(output_dir, inputs, resource)
    torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    model = _build_model(inputs, args.device)
    records, timing = _extract_records(
        path=output_dir / "session_records.jsonl",
        session_indices=source_indices,
        inputs=inputs,
        model=model,
        memory_enabled=True,
        lora_enabled=True,
    )
    comparison = _compare(records, expected)
    if comparison["chunks"] != expected_chunks:
        raise ValueError("D6 zero-init smoke chunk count changed")

    causal_index = source_indices[0]
    original = copy.deepcopy(inputs.answer_free_rows[causal_index])
    mutated = copy.deepcopy(original)
    mutated["video_intervals"] = [
        mutated["video_intervals"][0],  # type: ignore[index]
        *[[float(value[0]) + 0.001, float(value[1]) + 0.001] for value in mutated["video_intervals"][1:]],  # type: ignore[index]
    ]
    for dialog_snapshot in mutated["dialog"][1:]:  # type: ignore[index]
        if isinstance(dialog_snapshot, list):
            dialog_snapshot.append({"role": "assistant", "text": "FUTURE_ONLY_MUTATION"})
    starter = load_starter_kit(inputs.starter_dir)
    inference = inference_config(inputs.config)
    with torch.no_grad():
        original_prefix = process_session(
            row=original,
            input_index=causal_index,
            video_folder=inputs.video_folder,
            model=model,
            starter=starter,
            inference=inference,
            reference=inputs.references[causal_index],
            maximum_chunks=1,
        )
        mutated_prefix = process_session(
            row=mutated,
            input_index=causal_index,
            video_folder=inputs.video_folder,
            model=model,
            starter=starter,
            inference=inference,
            reference=inputs.references[causal_index],
            maximum_chunks=1,
        )
    future_mutation_exact = _first_chunk_signature(original_prefix) == _first_chunk_signature(
        mutated_prefix
    )

    peak_gib = torch.cuda.max_memory_allocated(model.device) / 2**30
    if previous_summary is not None:
        previous_timing = previous_summary.get("timing")
        if isinstance(previous_timing, Mapping):
            timing["resume_wall_seconds"] = float(
                previous_timing["resume_wall_seconds"]
            )
        peak_gib = max(peak_gib, float(previous_summary["peak_allocated_gib"]))
    numeric_zero = all(
        float(value) == 0.0
        for value in comparison["maximum_abs_differences"].values()  # type: ignore[union-attr]
    )
    exact_counts = comparison["exact_match_counts"]
    gates = {
        "all_102_chunks_covered": comparison["chunks"] == expected_chunks,
        "all_discrete_fields_exact": all(
            int(value) == expected_chunks for value in exact_counts.values()  # type: ignore[union-attr]
        ),
        "hidden_and_tag_differences_zero": numeric_zero,
        "zero_memory_residual": comparison["maximum_memory_residual_norm"] == 0.0,
        "candidate_memory_updates_identical": comparison[
            "maximum_candidate_update_difference"
        ]
        == 0.0,
        "future_mutation_preserves_historical_logits": future_mutation_exact,
        "peak_allocated_at_most_70_gib": peak_gib
        <= float(resources["maximum_peak_allocated_gib"]),
        "maximum_session_model_at_most_240_seconds": float(
            timing["maximum_session_model_seconds"]
        )
        <= float(resources["maximum_smoke_session_model_seconds"]),
    }
    summary = {
        "schema_version": 1,
        "kind": "d6_zero_init_causality_smoke",
        "status": "pass" if all(gates.values()) else "fail",
        "classification": "zero-init equivalence and causality audit; no efficacy conclusion",
        "config_sha256": inputs.config_sha256,
        "d4_3_reference_sha256": D4_3_RECORDS_SHA256,
        "comparison": comparison,
        "future_mutation_exact": future_mutation_exact,
        "timing": timing,
        "peak_allocated_gib": peak_gib,
        "session_records_sha256": sha256_file(output_dir / "session_records.jsonl"),
        "gates": gates,
    }
    write_json(output_dir / "summary.json", summary)
    if not all(gates.values()):
        raise RuntimeError(f"D6 zero-init/causality smoke failed: {gates}")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
