"""Run the frozen U2 six-view early-chunk utterance grounding audit."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

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
    write_jsonl,
)
from proactive_u1.core import controlled_messages, normalize_continuation
from proactive_u1.internvl import PrefillInternVLProactiveModel
from proactive_u1.prepare import strip_current_answers
from proactive_u2.core import (
    VIEWS,
    analyze_records,
    build_review_packages,
    fact_extraction_messages,
    normalize_visual_facts,
    remove_assistant_history,
    visual_fact_block,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"U2 SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty U2 CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _utterance_rating_rows(
    blind_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    fields = (
        "review_id",
        "sample_id",
        "candidate",
        "reviewer_slot",
        "correctness_1_5",
        "specificity_1_5",
        "actionability_1_5",
        "groundedness_1_5",
        "current_visual_support_1_5",
        "plan_consistency_1_5",
        "conciseness_1_5",
        "safety_1_5",
        "generic_flag",
        "hallucination_flag",
        "unsupported_visual_claim_flag",
        "stale_history_flag",
        "unsafe_flag",
        "primary_error_type",
        "notes",
    )
    result: list[dict[str, str]] = []
    for item in blind_rows:
        for slot in ("A", "B"):
            row = {field: "" for field in fields}
            row.update(
                {
                    "review_id": str(item["review_id"]),
                    "sample_id": str(item["sample_id"]),
                    "candidate": str(item["candidate"]),
                    "reviewer_slot": slot,
                }
            )
            result.append(row)
    return result


def _fact_rating_rows(fact_rows: list[dict[str, object]]) -> list[dict[str, str]]:
    fields = (
        "fact_review_id",
        "sample_id",
        "reviewer_slot",
        "fact_correctness_1_5",
        "fact_completeness_1_5",
        "unsupported_fact_flag",
        "inferred_completion_flag",
        "missed_salient_fact_flag",
        "notes",
    )
    result: list[dict[str, str]] = []
    for item in fact_rows:
        for slot in ("A", "B"):
            row = {field: "" for field in fields}
            row.update(
                {
                    "fact_review_id": str(item["fact_review_id"]),
                    "sample_id": str(item["sample_id"]),
                    "reviewer_slot": slot,
                }
            )
            result.append(row)
    return result


def _validate_samples(
    samples: list[dict[str, object]],
    source_rows: list[dict[str, object]],
    expected_samples: int,
) -> None:
    if len(samples) != expected_samples:
        raise ValueError(f"U2 expected {expected_samples} samples, got {len(samples)}")
    if len({str(row["sample_id"]) for row in samples}) != len(samples):
        raise ValueError("U2 sample contains duplicate sample_id values")
    keys: set[tuple[int, int]] = set()
    for sample in samples:
        if "answers" in sample:
            raise ValueError("U2 sample contains answers")
        input_index = int(sample["input_index"])
        chunk_index = int(sample["chunk_index"])
        key = (input_index, chunk_index)
        if key in keys:
            raise ValueError(f"U2 sample contains duplicate chunk: {key}")
        keys.add(key)
        if chunk_index < 1:
            raise ValueError("U2 no-current-video view requires non-first chunks")
        source = source_rows[input_index]
        if "answers" in source:
            raise ValueError("U2 generation source contains answers")
        if str(sample["video_path"]) != str(source["video_path"]):
            raise ValueError(f"U2 sample/source video mismatch at {key}")
        interval = [float(value) for value in source["video_intervals"][chunk_index]]  # type: ignore[index]
        if interval != [float(value) for value in sample["interval"]]:  # type: ignore[union-attr]
            raise ValueError(f"U2 sample/source interval mismatch at {key}")
        if sample["prior_dialog"] != source["dialog"][chunk_index]:  # type: ignore[index]
            raise ValueError(f"U2 sample/source dialog mismatch at {key}")


def _partial_state(
    records: list[dict[str, object]],
    facts: list[dict[str, object]],
    replay: list[dict[str, object]],
) -> set[tuple[int, int]]:
    views_by_key: dict[tuple[int, int], set[str]] = defaultdict(set)
    for row in records:
        key = (int(row["input_index"]), int(row["chunk_index"]))
        views_by_key[key].add(str(row["view"]))
    facts_by_key = {
        (int(row["input_index"]), int(row["chunk_index"])) for row in facts
    }
    replay_by_key = {
        (int(row["input_index"]), int(row["chunk_index"])) for row in replay
    }
    incomplete = {key: views for key, views in views_by_key.items() if views != set(VIEWS)}
    if incomplete:
        raise ValueError(f"U2 partial output contains incomplete views: {incomplete}")
    if set(views_by_key) != facts_by_key or set(views_by_key) != replay_by_key:
        raise ValueError("U2 partial records/facts/replay coverage differs")
    return set(views_by_key)


def run(
    config_path: Path,
    output_dir: Path,
    *,
    device: str,
    require_exclusive_gpu: bool,
    resume: bool,
) -> dict[str, object]:
    started = time.monotonic()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["views"]) != VIEWS:
        raise ValueError("U2 view order differs from the frozen implementation")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"U2 output directory is not empty: {output_dir}")

    protocol_path = _resolve(config["protocol"]["path"])
    _check_hash(protocol_path, str(config["protocol"]["sha256"]))
    source_paths = {
        name: _resolve(value["path"]) for name, value in config["sources"].items()
    }
    source_hashes = {
        name: _check_hash(source_paths[name], str(value["sha256"]))
        for name, value in config["sources"].items()
    }
    sample_config = config["prepared_sample"]
    sample_path = _resolve(sample_config["items"])
    sample_key_path = _resolve(sample_config["key"])
    sample_manifest_path = _resolve(sample_config["manifest"])
    _check_hash(sample_path, str(sample_config["items_sha256"]))
    _check_hash(sample_key_path, str(sample_config["key_sha256"]))
    _check_hash(sample_manifest_path, str(sample_config["manifest_sha256"]))

    starter_config = config["starter_kit"]
    starter_dir = _resolve(starter_config["path"])
    _check_hash(starter_dir / "model.py", str(starter_config["model_py_sha256"]))
    _check_hash(
        starter_dir / "run_generate_proactive.py",
        str(starter_config["proactive_py_sha256"]),
    )
    _check_hash(
        starter_dir / "run_evaluation.py", str(starter_config["scorer_py_sha256"])
    )
    model_path = Path(config["model"]["local_path"]).expanduser().resolve()
    model_audit = verify_model_snapshot(model_path, config["model"])

    label_free_sources = strip_current_answers(load_jsonl(source_paths["gold_container"]))
    if any("answers" in row for row in label_free_sources):
        raise ValueError("U2 generation source still contains answers")
    r0_rows = load_jsonl(source_paths["r0_session_records"])
    samples = load_jsonl(sample_path)
    _validate_samples(samples, label_free_sources, int(sample_config["samples"]))

    output_dir.mkdir(parents=True, exist_ok=True)
    effective_config = copy.deepcopy(config)
    effective_config["runtime"] = {
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "device": device,
        "require_exclusive_gpu": require_exclusive_gpu,
        "resume": resume,
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
                protocol_path,
                PROJECT_ROOT / "src/proactive_u1/core.py",
                PROJECT_ROOT / "src/proactive_u1/internvl.py",
                PROJECT_ROOT / "src/proactive_u2/core.py",
                PROJECT_ROOT / "src/proactive_u2/prepare.py",
                PROJECT_ROOT / "src/proactive_u2/run.py",
                PROJECT_ROOT / "src/proactive_u2/tests/test_core.py",
                PROJECT_ROOT / "src/proactive_u2/tests/test_prepare.py",
            ],
        ),
    )

    partial_records_path = output_dir / "partial_content_records.jsonl"
    partial_facts_path = output_dir / "partial_fact_records.jsonl"
    partial_replay_path = output_dir / "partial_r0_replay_records.jsonl"
    content_records = (
        load_jsonl(partial_records_path)
        if resume and partial_records_path.is_file()
        else []
    )
    fact_records = (
        load_jsonl(partial_facts_path) if resume and partial_facts_path.is_file() else []
    )
    replay_records = (
        load_jsonl(partial_replay_path)
        if resume and partial_replay_path.is_file()
        else []
    )
    completed_keys = _partial_state(content_records, fact_records, replay_records)

    inference_values = config["inference"]
    inference = CausalInferenceConfig(
        frames_per_interval=int(inference_values["frames_per_interval"]),
        max_frames=int(inference_values["max_frames"]),
        max_history_turns=int(inference_values["max_history_turns"]),
        max_new_tokens=int(inference_values["utterance_max_new_tokens"]),
    )
    starter = load_starter_kit(starter_dir)
    model = PrefillInternVLProactiveModel(
        model_path=str(model_path),
        device=device,
        dtype_name=str(inference_values["dtype"]),
        attention_implementation=str(inference_values["attention_implementation"]),
        seed=int(inference_values["seed"]),
        require_exclusive_gpu=require_exclusive_gpu,
        video_frame_size=int(inference_values["video_frame_size"]),
        pad_token_id=int(inference_values["pad_token_id"]),
    )

    video_folder = _resolve(config["video_folder"])
    for sample in sorted(samples, key=lambda row: int(row["input_index"])):
        input_index = int(sample["input_index"])
        chunk_index = int(sample["chunk_index"])
        key = (input_index, chunk_index)
        if key in completed_keys:
            continue
        source = label_free_sources[input_index]
        intervals = [
            (float(interval[0]), float(interval[1]))
            for interval in source["video_intervals"]
        ]
        cumulative_frames: list[object] = []
        previous_frames: list[object] = []
        current_frames: list[object] = []
        for current_index, interval in enumerate(intervals[: chunk_index + 1]):
            previous_frames = list(cumulative_frames)
            extracted = starter.extract_frames(
                str(video_folder / str(source["video_path"])),
                intervals=[interval],
                frames_per_interval=inference.frames_per_interval,
            )
            cumulative_frames.extend(extracted)
            if current_index == chunk_index:
                current_frames = extracted
        full_frames = subsample_frames(cumulative_frames, inference.max_frames)
        previous_only_frames = subsample_frames(previous_frames, inference.max_frames)
        current_only_frames = subsample_frames(current_frames, inference.max_frames)
        if not previous_only_frames or not current_only_frames:
            raise ValueError(f"U2 causal frame view is empty at {key}")
        expected_chunk = r0_rows[input_index]["chunks"][chunk_index]
        if len(current_frames) != int(expected_chunk["current_interval_frames"]):
            raise ValueError(f"U2 current frame count differs from frozen R0 at {key}")
        if len(full_frames) != int(expected_chunk["model_input_frames"]):
            raise ValueError(f"U2 full frame count differs from frozen R0 at {key}")

        base_messages = build_messages(
            row=source,
            chunk_index=chunk_index,
            system_prompt=starter.system_prompt,
            normalize_dialog_turns=starter.normalize_dialog_turns,
            max_history_turns=inference.max_history_turns,
        )
        query_only_messages = remove_assistant_history(base_messages)
        replay_started = time.monotonic()
        replay_raw = model.generate(
            full_frames, base_messages, max_new_tokens=inference.max_new_tokens
        )
        expected_raw = str(expected_chunk["raw_response"])
        replay = {
            "sample_id": sample["sample_id"],
            "input_index": input_index,
            "chunk_index": chunk_index,
            "expected_raw_response": expected_raw,
            "replayed_raw_response": replay_raw,
            "exact_match": replay_raw == expected_raw,
            "current_interval_frames": len(current_frames),
            "full_model_input_frames": len(full_frames),
            "wall_time_seconds": time.monotonic() - replay_started,
        }
        if not replay["exact_match"]:
            raise ValueError(
                f"U2 R0 replay mismatch at {key}: {replay_raw!r} != {expected_raw!r}"
            )

        fact_started = time.monotonic()
        raw_facts = model.generate(
            current_only_frames,
            fact_extraction_messages(source["query"]),
            max_new_tokens=int(inference_values["fact_max_new_tokens"]),
        )
        normalized_facts = normalize_visual_facts(raw_facts)
        fact_record = {
            "sample_id": sample["sample_id"],
            "input_index": input_index,
            "chunk_index": chunk_index,
            "domain": sample["domain"],
            "position_bin": sample["position_bin"],
            **normalized_facts,
            "model_input_frames": len(current_only_frames),
            "wall_time_seconds": time.monotonic() - fact_started,
        }
        fact_block = visual_fact_block(fact_record["visual_facts"])
        messages_by_view = {
            "full_history": controlled_messages(base_messages),
            "no_current_video": controlled_messages(base_messages),
            "query_only_full_video": controlled_messages(query_only_messages),
            "query_only_current_video": controlled_messages(query_only_messages),
            "facts_full_history": controlled_messages(base_messages, fact_block),
            "facts_query_current": controlled_messages(query_only_messages, fact_block),
        }
        frames_by_view = {
            "full_history": full_frames,
            "no_current_video": previous_only_frames,
            "query_only_full_video": full_frames,
            "query_only_current_video": current_only_frames,
            "facts_full_history": full_frames,
            "facts_query_current": current_only_frames,
        }
        sample_records: list[dict[str, object]] = []
        for view in VIEWS:
            generation_started = time.monotonic()
            generated = model.generate_prefilled(
                frames_by_view[view],
                messages_by_view[view],
                assistant_prefix=str(inference_values["assistant_prefix"]),
                max_new_tokens=inference.max_new_tokens,
            )
            normalized = normalize_continuation(generated["continuation"])
            sample_records.append(
                {
                    "sample_id": sample["sample_id"],
                    "input_index": input_index,
                    "chunk_index": chunk_index,
                    "domain": sample["domain"],
                    "position_bin": sample["position_bin"],
                    "view": view,
                    **generated,
                    **normalized,
                    "visual_facts": fact_record["visual_facts"],
                    "fact_block_injected": view in {
                        "facts_full_history",
                        "facts_query_current",
                    },
                    "current_interval_frames": len(current_frames),
                    "model_input_frames": len(frames_by_view[view]),
                    "assistant_history_turns": sum(
                        message.get("role") == "assistant"
                        for message in messages_by_view[view]
                    ),
                    "wall_time_seconds": time.monotonic() - generation_started,
                }
            )
        replay_records.append(replay)
        fact_records.append(fact_record)
        content_records.extend(sample_records)
        write_jsonl(partial_replay_path, replay_records)
        write_jsonl(partial_facts_path, fact_records)
        write_jsonl(partial_records_path, content_records)

    expected_samples = int(sample_config["samples"])
    if len(replay_records) != expected_samples or not all(
        bool(row["exact_match"]) for row in replay_records
    ):
        raise ValueError("U2 R0 replay did not exactly cover the frozen sample")
    if len(fact_records) != expected_samples:
        raise ValueError("U2 fact generation coverage mismatch")
    if len(content_records) != expected_samples * len(VIEWS):
        raise ValueError("U2 six-view content coverage mismatch")

    analysis, discordant = analyze_records(
        content_records, fact_records, config["review_priority_thresholds"]
    )
    blind, review_key, fact_blind, review_summary = build_review_packages(
        content_records,
        fact_records,
        samples,
        seed=str(config["review"]["blind_seed"]),
    )
    content_records.sort(
        key=lambda row: (
            int(row["input_index"]),
            int(row["chunk_index"]),
            VIEWS.index(str(row["view"])),
        )
    )
    fact_records.sort(key=lambda row: (int(row["input_index"]), int(row["chunk_index"])))
    replay_records.sort(key=lambda row: (int(row["input_index"]), int(row["chunk_index"])))
    content_path = output_dir / "content_records.jsonl"
    facts_path = output_dir / "fact_records.jsonl"
    replay_path = output_dir / "r0_replay_records.jsonl"
    write_jsonl(content_path, content_records)
    write_jsonl(facts_path, fact_records)
    write_jsonl(replay_path, replay_records)
    write_json(output_dir / "analysis.json", analysis)
    write_jsonl(output_dir / "discordant_cases.jsonl", discordant)

    review_dir = output_dir / "review"
    utterance_blind_path = review_dir / "utterance_items_blind.jsonl"
    utterance_key_path = review_dir / "utterance_key.jsonl"
    fact_blind_path = review_dir / "fact_items_blind.jsonl"
    write_jsonl(utterance_blind_path, blind)
    write_jsonl(utterance_key_path, review_key)
    write_jsonl(fact_blind_path, fact_blind)
    _write_csv(review_dir / "utterance_ratings_template.csv", _utterance_rating_rows(blind))
    _write_csv(review_dir / "fact_ratings_template.csv", _fact_rating_rows(fact_blind))
    _write_text(
        review_dir / "RUBRIC.md",
        """# U2 视觉支持盲评补充规则

六个 utterance 候选必须独立评分，不能强制选赢家，也不能查看 variant key。
`current_visual_support_1_5` 只判断候选中的对象、动作和状态是否得到当前 interval 画面
直接支持；历史中出现但当前画面无法确认的事实不能自动给高分。

- `unsupported_visual_claim_flag=yes`：候选含当前画面不支持的具体视觉断言。
- `stale_history_flag=yes`：候选主要沿用旧 assistant 指导，和当前画面进度不一致。
- hallucination、unsafe 和其他内容维度沿用 U0/U1 冻结定义。

事实包单独判断 predicted facts：correctness 看每条事实是否直接可见；completeness 只看是否
遗漏当前最显著且任务相关的事实；不得用未来画面补充判断。正式双人评分前需先用少量
独立 calibration examples 对齐上述边界，原始分不得被仲裁覆盖。
""",
    )
    review_manifest = {
        **review_summary,
        "utterance_items_sha256": sha256_file(utterance_blind_path),
        "utterance_key_sha256": sha256_file(utterance_key_path),
        "fact_items_sha256": sha256_file(fact_blind_path),
        "utterance_ratings_template_sha256": sha256_file(
            review_dir / "utterance_ratings_template.csv"
        ),
        "fact_ratings_template_sha256": sha256_file(
            review_dir / "fact_ratings_template.csv"
        ),
        "rubric_sha256": sha256_file(review_dir / "RUBRIC.md"),
    }
    write_json(review_dir / "manifest.json", review_manifest)

    runtime = {
        "wall_time_seconds": time.monotonic() - started,
        "samples": expected_samples,
        "r0_replay_exact": sum(bool(row["exact_match"]) for row in replay_records),
        "fact_generations": len(fact_records),
        "utterance_generations": len(content_records),
        "peak_gpu_memory_bytes": model.peak_memory_bytes(),
        "preexisting_gpu_processes": model.preexisting_gpu_processes,
        "model_parameters": model.parameter_count,
        "device": device,
    }
    write_json(output_dir / "runtime.json", runtime)
    write_json(
        output_dir / "data_manifest.json",
        {
            "schema_version": 1,
            "source_hashes": source_hashes,
            "protocol_sha256": config["protocol"]["sha256"],
            "sample_items_sha256": sample_config["items_sha256"],
            "sample_key_sha256": sample_config["key_sha256"],
            "sample_manifest_sha256": sample_config["manifest_sha256"],
            "content_records_sha256": sha256_file(content_path),
            "fact_records_sha256": sha256_file(facts_path),
            "r0_replay_records_sha256": sha256_file(replay_path),
            "generation_reads_current_or_future_gold_answers": False,
            "generation_reads_future_video_or_dialog": False,
            "d4_gate_changed": False,
            "official_scorer_invoked": False,
            "reviewer_scores_used_to_select_prompts": False,
            "external_data_used": False,
            "model_audit": model_audit,
        },
    )
    _write_text(
        output_dir / "README.md",
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "Status: complete automatic U2 mechanism audit; human quality review pending.",
                "",
                f"Samples: {analysis['samples']}; views: {len(VIEWS)}; "
                f"R0 replay: {runtime['r0_replay_exact']}/{expected_samples} exact.",
                "D4 decisions are frozen; no official metric or gate changes were made.",
                "Fallback and text sensitivity are not grounding-quality evidence.",
                "",
            ]
        ),
    )
    _write_text(
        output_dir / "run.log",
        json.dumps(
            {"analysis": analysis, "runtime": runtime, "review": review_manifest},
            indent=2,
            ensure_ascii=True,
        )
        + "\n",
    )
    for path in (partial_records_path, partial_facts_path, partial_replay_path):
        if path.exists():
            path.unlink()
    result = {"analysis": analysis, "runtime": runtime, "review": review_manifest}
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--require-exclusive-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(
        _resolve(args.config),
        _resolve(args.output_dir),
        device=args.device,
        require_exclusive_gpu=args.require_exclusive_gpu,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
