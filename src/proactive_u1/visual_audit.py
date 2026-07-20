"""Run the frozen U1 visual/dialog reliance counterfactual audit."""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
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
    write_jsonl,
)
from proactive_u1.core import controlled_messages, normalize_continuation
from proactive_u1.internvl import PrefillInternVLProactiveModel
from proactive_u1.prepare import strip_current_answers


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIEWS = (
    "full",
    "no_assistant_history",
    "no_current_interval_video",
    "masked_video",
)
GENERATED_VIEWS = VIEWS[1:]
COMPLETION_CLAIM = re.compile(
    r"(?:\b(?:you\s+are|you're)\s+(?:done|finished|all\s+set)\b"
    r"|\b(?:you\s+have|you've)\s+(?:finished|completed)\b"
    r"|\b(?:it|this|that)\s+(?:is|looks)\s+"
    r"(?:done|finished|complete|completed|ready|good)\b"
    r"|\ball\s+set\b|\b(?:great|good)\s+job\b)",
    flags=re.IGNORECASE,
)


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _check_hash(path: Path, expected: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"U1-V SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def remove_assistant_history(
    messages: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    """Remove only prior assistant turns while preserving system/query/user turns."""
    result = [copy.deepcopy(row) for row in messages if row.get("role") != "assistant"]
    if not result or result[0].get("role") != "system":
        raise ValueError("U1-V messages must begin with a system prompt")
    return result


def mask_frames(frames: Sequence[object], pixel_value: int) -> list[object]:
    """Replace PIL frame content while retaining frame count and dimensions."""
    from PIL import Image

    if not 0 <= pixel_value <= 255:
        raise ValueError("U1-V mask pixel value must be in [0, 255]")
    result: list[object] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Image.Image):
            raise TypeError(f"U1-V frame {index} is not a PIL image")
        result.append(Image.new("RGB", frame.size, (pixel_value,) * 3))
    return result


def _normalized_text(value: object) -> str:
    return " ".join(str(value).lower().split())


def _word_count(value: object) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", str(value)))


def _summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    if not rows:
        raise ValueError("U1-V cannot summarize an empty group")
    fallback = sum(bool(row["used_fallback"]) for row in rows)
    completion = sum(bool(COMPLETION_CLAIM.search(str(row["content"]))) for row in rows)
    return {
        "samples": len(rows),
        "fallback": fallback,
        "fallback_rate": fallback / len(rows),
        "nonempty": len(rows) - fallback,
        "nonempty_rate": (len(rows) - fallback) / len(rows),
        "mean_word_count": sum(_word_count(row["content"]) for row in rows)
        / len(rows),
        "completion_claims": completion,
        "completion_claim_rate": completion / len(rows),
    }


def _paired_summary(
    rows: Sequence[dict[str, object]],
    full_by_key: dict[tuple[int, int], dict[str, object]],
) -> dict[str, object]:
    if not rows:
        raise ValueError("U1-V cannot compare an empty group")
    similarities: list[float] = []
    answer_exact = 0
    content_exact = 0
    fallback_agreement = 0
    full_nonfallback_to_fallback = 0
    full_fallback_to_nonfallback = 0
    for row in rows:
        key = (int(row["input_index"]), int(row["chunk_index"]))
        full = full_by_key[key]
        answer_exact += int(str(row["answer"]) == str(full["answer"]))
        left = _normalized_text(full["content"])
        right = _normalized_text(row["content"])
        content_exact += int(left == right)
        similarities.append(SequenceMatcher(None, left, right).ratio())
        full_fallback = bool(full["used_fallback"])
        candidate_fallback = bool(row["used_fallback"])
        fallback_agreement += int(full_fallback == candidate_fallback)
        full_nonfallback_to_fallback += int(not full_fallback and candidate_fallback)
        full_fallback_to_nonfallback += int(full_fallback and not candidate_fallback)
    candidate = _summary(rows)
    full = _summary([full_by_key[(int(row["input_index"]), int(row["chunk_index"]))] for row in rows])
    return {
        "samples": len(rows),
        "answer_exact": answer_exact,
        "answer_exact_rate": answer_exact / len(rows),
        "content_exact": content_exact,
        "content_exact_rate": content_exact / len(rows),
        "fallback_agreement": fallback_agreement,
        "fallback_agreement_rate": fallback_agreement / len(rows),
        "mean_text_similarity": sum(similarities) / len(similarities),
        "fallback_rate_delta_vs_full": float(candidate["fallback_rate"])
        - float(full["fallback_rate"]),
        "completion_claim_rate_delta_vs_full": float(
            candidate["completion_claim_rate"]
        )
        - float(full["completion_claim_rate"]),
        "full_nonfallback_to_candidate_fallback": full_nonfallback_to_fallback,
        "full_fallback_to_candidate_nonfallback": full_fallback_to_nonfallback,
    }


def _grouped(
    rows: Sequence[dict[str, object]],
    field: str,
    summarize,
) -> dict[str, object]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {name: summarize(groups[name]) for name in sorted(groups)}


def analyze_records(
    records: Sequence[dict[str, object]],
    thresholds: dict[str, object],
) -> dict[str, object]:
    """Validate paired coverage and compute frozen sensitivity diagnostics."""
    by_view: dict[str, list[dict[str, object]]] = defaultdict(list)
    seen: set[tuple[str, int, int]] = set()
    for row in records:
        view = str(row.get("view"))
        if view not in VIEWS:
            raise ValueError(f"Unsupported U1-V record view: {view}")
        key = (view, int(row["input_index"]), int(row["chunk_index"]))
        if key in seen:
            raise ValueError(f"Duplicate U1-V record: {key}")
        seen.add(key)
        by_view[view].append(row)
    counts = {view: len(by_view[view]) for view in VIEWS}
    if len(set(counts.values())) != 1 or not counts["full"]:
        raise ValueError(f"U1-V view coverage differs: {counts}")
    full_by_key = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in by_view["full"]
    }
    if len(full_by_key) != counts["full"]:
        raise ValueError("U1-V full view contains duplicate paired keys")
    expected_keys = set(full_by_key)
    for view in GENERATED_VIEWS:
        keys = {
            (int(row["input_index"]), int(row["chunk_index"]))
            for row in by_view[view]
        }
        if keys != expected_keys:
            raise ValueError(f"U1-V paired keys differ for {view}")

    view_summaries: dict[str, object] = {}
    paired: dict[str, object] = {}
    for view in VIEWS:
        rows = by_view[view]
        view_summaries[view] = {
            "overall": _summary(rows),
            "by_position": _grouped(rows, "position_bin", _summary),
            "by_domain": _grouped(rows, "domain", _summary),
        }
        if view == "full":
            continue
        pair_fn = lambda group: _paired_summary(group, full_by_key)
        paired[view] = {
            "overall": pair_fn(rows),
            "by_position": _grouped(rows, "position_bin", pair_fn),
            "by_domain": _grouped(rows, "domain", pair_fn),
        }

    history = paired["no_assistant_history"]["overall"]
    current = paired["no_current_interval_video"]["overall"]
    masked = paired["masked_video"]["overall"]
    gates = {
        "history_necessary": float(history["fallback_rate_delta_vs_full"])
        >= float(thresholds["history_fallback_rate_increase"]),
        "current_visual_material": (
            float(current["fallback_rate_delta_vs_full"])
            >= float(thresholds["current_video_fallback_rate_increase"])
            or float(current["mean_text_similarity"])
            < float(thresholds["current_video_mean_similarity_below"])
        ),
        "any_visual_material": (
            float(masked["fallback_rate_delta_vs_full"])
            >= float(thresholds["masked_video_fallback_rate_increase"])
            or float(masked["mean_text_similarity"])
            < float(thresholds["masked_video_mean_similarity_below"])
        ),
    }
    discordant: list[dict[str, object]] = []
    for view in GENERATED_VIEWS:
        for row in by_view[view]:
            key = (int(row["input_index"]), int(row["chunk_index"]))
            full = full_by_key[key]
            similarity = SequenceMatcher(
                None,
                _normalized_text(full["content"]),
                _normalized_text(row["content"]),
            ).ratio()
            if str(full["answer"]) != str(row["answer"]):
                discordant.append(
                    {
                        "view": view,
                        "sample_id": row["sample_id"],
                        "input_index": row["input_index"],
                        "chunk_index": row["chunk_index"],
                        "domain": row["domain"],
                        "position_bin": row["position_bin"],
                        "similarity": similarity,
                        "full_used_fallback": full["used_fallback"],
                        "candidate_used_fallback": row["used_fallback"],
                        "full_content": full["content"],
                        "candidate_content": row["content"],
                    }
                )
    discordant.sort(key=lambda row: (str(row["view"]), float(row["similarity"])))
    return {
        "status": "complete U1-V automatic sensitivity audit",
        "classification": (
            "development-set mechanism diagnostic; text changes are not quality, "
            "grounding, or promotion evidence"
        ),
        "samples": counts["full"],
        "views": view_summaries,
        "paired_vs_full": paired,
        "preregistered_thresholds": thresholds,
        "diagnostic_gates": gates,
        "discordant_cases": len(discordant),
        "reviewer_a_scores_used": False,
        "reviewer_b_files_read": False,
        "completion_claim_is_lexical_diagnostic_only": True,
        "discordant_records": discordant,
    }


def _full_records(
    source_rows: Sequence[dict[str, object]],
    samples: Sequence[dict[str, object]],
) -> dict[tuple[int, int], dict[str, object]]:
    sample_by_key = {
        (int(row["input_index"]), int(row["chunk_index"])): row for row in samples
    }
    result: dict[tuple[int, int], dict[str, object]] = {}
    for source in source_rows:
        if source.get("variant") != "forced_no_state":
            continue
        key = (int(source["input_index"]), int(source["chunk_index"]))
        sample = sample_by_key.get(key)
        if sample is None:
            raise ValueError(f"U1-V full source contains an unexpected key: {key}")
        if key in result:
            raise ValueError(f"U1-V full source contains a duplicate key: {key}")
        if str(source["sample_id"]) != str(sample["sample_id"]):
            raise ValueError(f"U1-V full source sample ID differs at {key}")
        result[key] = {
            **copy.deepcopy(source),
            "source_variant": source["variant"],
            "view": "full",
            "generated_in_current_run": False,
        }
    if set(result) != set(sample_by_key):
        raise ValueError("U1-V frozen full source does not exactly cover the sample")
    return result


def run(
    config_path: Path,
    output_dir: Path,
    device: str,
    require_exclusive_gpu: bool,
    resume: bool,
) -> dict[str, object]:
    started = time.monotonic()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["views"]) != VIEWS or tuple(config["generated_views"]) != GENERATED_VIEWS:
        raise ValueError("U1-V view order differs from the frozen implementation")
    if output_dir.exists() and any(output_dir.iterdir()) and not resume:
        raise FileExistsError(f"U1-V output directory is not empty: {output_dir}")

    protocol_path = _resolve(config["protocol"]["path"])
    _check_hash(protocol_path, config["protocol"]["sha256"])
    source_paths = {
        name: _resolve(value["path"]) for name, value in config["sources"].items()
    }
    for name, value in config["sources"].items():
        _check_hash(source_paths[name], value["sha256"])
    prepared = config["prepared_sample"]
    sample_path = _resolve(prepared["items"])
    _check_hash(sample_path, prepared["items_sha256"])
    _check_hash(_resolve(prepared["manifest"]), prepared["manifest_sha256"])
    starter_config = config["starter_kit"]
    starter_dir = _resolve(starter_config["path"])
    _check_hash(starter_dir / "model.py", starter_config["model_py_sha256"])
    _check_hash(
        starter_dir / "run_generate_proactive.py",
        starter_config["proactive_py_sha256"],
    )
    _check_hash(starter_dir / "run_evaluation.py", starter_config["scorer_py_sha256"])
    model_path = Path(config["model"]["local_path"]).expanduser().resolve()
    model_audit = verify_model_snapshot(model_path, config["model"])

    samples = load_jsonl(sample_path)
    if len(samples) != int(prepared["chunks"]):
        raise ValueError(f"U1-V expected {prepared['chunks']} samples, got {len(samples)}")
    if any(int(sample["chunk_index"]) < 1 for sample in samples):
        raise ValueError("U1-V no-current-video view requires non-first chunks")
    label_free_sources = strip_current_answers(load_jsonl(source_paths["gold_container"]))
    if any("answers" in row for row in label_free_sources):
        raise ValueError("U1-V generation source still contains answers")
    r0_rows = load_jsonl(source_paths["r0_session_records"])
    frozen_full = _full_records(
        load_jsonl(source_paths["full_content_records"]), samples
    )

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
                PROJECT_ROOT / "src/proactive_u1/prepare.py",
                PROJECT_ROOT / "src/proactive_u1/visual_audit.py",
                PROJECT_ROOT / "src/proactive_u1/tests/test_visual_audit.py",
            ],
        ),
    )

    partial_path = output_dir / "partial_content_records.jsonl"
    records = load_jsonl(partial_path) if resume and partial_path.is_file() else []
    completed_keys: set[tuple[int, int]] = set()
    if records:
        partial_views: dict[tuple[int, int], set[str]] = defaultdict(set)
        for record in records:
            key = (int(record["input_index"]), int(record["chunk_index"]))
            partial_views[key].add(str(record["view"]))
        invalid = {key: views for key, views in partial_views.items() if views != set(VIEWS)}
        if invalid:
            raise ValueError(f"U1-V partial records contain incomplete samples: {invalid}")
        completed_keys = set(partial_views)

    inference_values = config["inference"]
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
        dtype_name=str(inference_values["dtype"]),
        attention_implementation=str(inference_values["attention_implementation"]),
        seed=int(inference_values["seed"]),
        require_exclusive_gpu=require_exclusive_gpu,
        video_frame_size=int(inference_values["video_frame_size"]),
        pad_token_id=int(inference_values["pad_token_id"]),
    )

    samples_by_input: dict[int, list[dict[str, object]]] = defaultdict(list)
    for sample in samples:
        samples_by_input[int(sample["input_index"])].append(sample)
    video_folder = _resolve(config["video_folder"])
    for input_index in sorted(samples_by_input):
        source = label_free_sources[input_index]
        selected_by_chunk = {
            int(sample["chunk_index"]): sample for sample in samples_by_input[input_index]
        }
        intervals = [
            (float(interval[0]), float(interval[1]))
            for interval in source["video_intervals"]
        ]
        cumulative_frames: list[object] = []
        for chunk_index, interval in enumerate(intervals[: max(selected_by_chunk) + 1]):
            previous_frames = list(cumulative_frames)
            current_frames = starter.extract_frames(
                str(video_folder / str(source["video_path"])),
                intervals=[interval],
                frames_per_interval=inference.frames_per_interval,
            )
            cumulative_frames.extend(current_frames)
            if chunk_index not in selected_by_chunk:
                continue
            key = (input_index, chunk_index)
            if key in completed_keys:
                continue
            sample = selected_by_chunk[chunk_index]
            full_frames = subsample_frames(cumulative_frames, inference.max_frames)
            previous_only_frames = subsample_frames(previous_frames, inference.max_frames)
            if not previous_only_frames:
                raise ValueError(f"U1-V previous-only view is empty at {key}")
            masked = mask_frames(full_frames, int(inference_values["mask_pixel_value"]))
            if len(masked) != len(full_frames):
                raise ValueError(f"U1-V masked frame count changed at {key}")
            expected_chunk = r0_rows[input_index]["chunks"][chunk_index]
            if len(current_frames) != int(expected_chunk["current_interval_frames"]):
                raise ValueError(f"U1-V current frame count differs from frozen R0 at {key}")
            if len(full_frames) != int(expected_chunk["model_input_frames"]):
                raise ValueError(f"U1-V full frame count differs from frozen R0 at {key}")

            base_messages = build_messages(
                row=source,
                chunk_index=chunk_index,
                system_prompt=starter.system_prompt,
                normalize_dialog_turns=starter.normalize_dialog_turns,
                max_history_turns=inference.max_history_turns,
            )
            messages_by_view = {
                "no_assistant_history": remove_assistant_history(base_messages),
                "no_current_interval_video": base_messages,
                "masked_video": base_messages,
            }
            frames_by_view = {
                "no_assistant_history": full_frames,
                "no_current_interval_video": previous_only_frames,
                "masked_video": masked,
            }
            full_record = {
                **copy.deepcopy(frozen_full[key]),
                "current_interval_frames": len(current_frames),
                "model_input_frames": len(full_frames),
                "assistant_history_turns": sum(
                    message.get("role") == "assistant" for message in base_messages
                ),
            }
            sample_records = [full_record]
            for view in GENERATED_VIEWS:
                generation_started = time.monotonic()
                generated = model.generate_prefilled(
                    frames_by_view[view],
                    controlled_messages(messages_by_view[view]),
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
                        "generated_in_current_run": True,
                        **generated,
                        **normalized,
                        "current_interval_frames": len(current_frames),
                        "model_input_frames": len(frames_by_view[view]),
                        "assistant_history_turns": sum(
                            message.get("role") == "assistant"
                            for message in messages_by_view[view]
                        ),
                        "wall_time_seconds": time.monotonic() - generation_started,
                    }
                )
            records.extend(sample_records)
            write_jsonl(partial_path, records)

    analysis = analyze_records(records, config["diagnostic_thresholds"])
    discordant = analysis.pop("discordant_records")
    records.sort(
        key=lambda row: (
            str(row["domain"]),
            int(row["input_index"]),
            int(row["chunk_index"]),
            VIEWS.index(str(row["view"])),
        )
    )
    content_path = output_dir / "content_records.jsonl"
    write_jsonl(content_path, records)
    write_jsonl(output_dir / "discordant_cases.jsonl", discordant)
    write_json(output_dir / "analysis.json", analysis)
    runtime = {
        "wall_time_seconds": time.monotonic() - started,
        "generated_samples": sum(
            bool(row.get("generated_in_current_run")) for row in records
        ),
        "reused_full_samples": sum(row["view"] == "full" for row in records),
        "peak_gpu_memory_bytes": model.peak_memory_bytes(),
        "preexisting_gpu_processes": model.preexisting_gpu_processes,
        "model_parameters": model.parameter_count,
        "device": device,
    }
    write_json(output_dir / "runtime.json", runtime)
    write_json(
        output_dir / "data_manifest.json",
        {
            "source_hashes": {
                name: value["sha256"] for name, value in config["sources"].items()
            },
            "sample_items_sha256": prepared["items_sha256"],
            "protocol_sha256": config["protocol"]["sha256"],
            "content_records_sha256": sha256_file(content_path),
            "generation_reads_current_gold_answers": False,
            "generation_reads_future_video_or_dialog": False,
            "reviewer_a_scores_used": False,
            "reviewer_b_files_read": False,
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
                "Status: complete U1-V development-set mechanism audit.",
                "",
                "本实验复用冻结 full 输出，并生成三个严格配对的视觉/对话反事实视图。",
                "自动文本差异只表示输入敏感性，不表示 utterance 质量或视觉 grounding。",
                f"Samples: {analysis['samples']}; views: {len(VIEWS)}.",
                "",
            ]
        ),
    )
    if partial_path.exists():
        partial_path.unlink()
    result = {"analysis": analysis, "runtime": runtime}
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
        config_path=_resolve(args.config),
        output_dir=_resolve(args.output_dir),
        device=args.device,
        require_exclusive_gpu=args.require_exclusive_gpu,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
