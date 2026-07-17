"""Prepare the label-independent U1 sample and oracle annotation template."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl, write_jsonl
from proactive_u0.core import FALLBACK_ANSWER, position_bin


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGET_POSITIONS = ("1:second", "2-4", "5-9", "10+")


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _stable_hash(seed: str, *parts: object) -> str:
    text = "|".join([seed, *(str(part) for part in parts)])
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_current_answers(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: value for key, value in row.items() if key != "answers"} for row in rows]


def prepare_sample(
    label_free_sources: Sequence[dict[str, object]],
    predictions: Sequence[dict[str, object]],
    r0_rows: Sequence[dict[str, object]],
    seed: str,
    sessions_per_domain: int,
    excluded_input_indices: set[int],
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    if not (len(label_free_sources) == len(predictions) == len(r0_rows) == 700):
        raise ValueError("U1 preparation requires 700 aligned sessions")
    candidates: dict[tuple[str, int], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for input_index, (source, prediction, r0) in enumerate(
        zip(label_free_sources, predictions, r0_rows)
    ):
        if "answers" in source:
            raise ValueError("U1 sample selection must receive label-free source rows")
        video_path = str(source["video_path"])
        if prediction.get("video_path") != video_path or r0.get("video_path") != video_path:
            raise ValueError(f"U1 source order mismatch at session {input_index}")
        if int(r0.get("input_index", -1)) != input_index:
            raise ValueError(f"U1 R0 input index mismatch at session {input_index}")
        intervals = source["video_intervals"]
        dialog = source["dialog"]
        pred_answers = prediction["answers"]
        chunks = r0["chunks"]
        if not (
            isinstance(intervals, list)
            and isinstance(dialog, list)
            and isinstance(pred_answers, list)
            and isinstance(chunks, list)
            and len(intervals) == len(dialog) == len(pred_answers) == len(chunks)
        ):
            raise ValueError(f"U1 chunk alignment mismatch at session {input_index}")
        if input_index in excluded_input_indices:
            continue
        domain = str(source["domain"])
        for chunk_index, (interval, answer, chunk) in enumerate(
            zip(intervals, pred_answers, chunks)
        ):
            current_position = position_bin(chunk_index)
            if current_position not in TARGET_POSITIONS:
                continue
            if str(answer) != FALLBACK_ANSWER:
                continue
            if not isinstance(chunk, dict) or str(chunk.get("raw_response", "")).strip() != "$silent$":
                continue
            candidates[(domain, input_index)][current_position].append(
                {
                    "input_index": input_index,
                    "video_path": video_path,
                    "query": str(source["query"]),
                    "task": str(source["task"]),
                    "domain": domain,
                    "chunk_index": chunk_index,
                    "position_bin": current_position,
                    "interval": [float(interval[0]), float(interval[1])],
                    "observed_through_sec": float(interval[1]),
                    "video_intervals_so_far": intervals[: chunk_index + 1],
                    "prior_dialog": dialog[chunk_index],
                    "frozen_decision": "$interrupt$",
                    "current_output": FALLBACK_ANSWER,
                }
            )

    domains = sorted({str(row["domain"]) for row in label_free_sources})
    selected_sessions: dict[str, list[int]] = {}
    for domain in domains:
        eligible = [
            input_index
            for (candidate_domain, input_index), by_position in candidates.items()
            if candidate_domain == domain
            and all(position in by_position for position in TARGET_POSITIONS)
        ]
        eligible.sort(key=lambda index: _stable_hash(seed, "session", domain, index))
        if len(eligible) < sessions_per_domain:
            raise ValueError(
                f"U1 domain {domain} has only {len(eligible)} fully covered sessions"
            )
        selected_sessions[domain] = eligible[:sessions_per_domain]

    rows: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for domain in domains:
        for session_rank, input_index in enumerate(selected_sessions[domain]):
            source = label_free_sources[input_index]
            session_rows: list[dict[str, object]] = []
            for current_position in TARGET_POSITIONS:
                pool = candidates[(domain, input_index)][current_position]
                chosen = min(
                    pool,
                    key=lambda row: _stable_hash(
                        seed, "chunk", input_index, current_position, row["chunk_index"]
                    ),
                )
                row = dict(chosen)
                row["sample_id"] = (
                    f"U1-{domain.replace(' ', '_').replace('&', 'and')}-"
                    f"S{session_rank + 1:02d}-{current_position.replace(':', '_').replace('+', 'plus')}"
                )
                row["is_smoke"] = session_rank == 0
                row["sample_hash"] = _stable_hash(
                    seed, input_index, row["chunk_index"]
                )
                rows.append(row)
                session_rows.append(row)
            annotations.append(
                {
                    "schema_version": 1,
                    "status": "pending",
                    "input_index": input_index,
                    "video_path": source["video_path"],
                    "query": source["query"],
                    "task": source["task"],
                    "domain": domain,
                    "is_smoke_session": session_rank == 0,
                    "provenance": {
                        "plan_inputs": ["task", "query"],
                        "chunk_inputs": [
                            "task",
                            "query",
                            "dialog_at_chunk",
                            "video_through_interval_end",
                        ],
                        "excluded_inputs": [
                            "answers",
                            "future_dialog",
                            "future_video",
                            "R0/D1 errors",
                        ],
                        "annotation_type": "evaluation_only_oracle_non_deployable",
                    },
                    "goal": source["query"],
                    "steps": [],
                    "sampled_chunk_states": [
                        {
                            "sample_id": row["sample_id"],
                            "chunk_index": row["chunk_index"],
                            "observed_through_sec": row["observed_through_sec"],
                            "current_step_id": "",
                            "progress": "",
                            "completion_evidence": [],
                            "incompletion_or_error_evidence": [],
                            "next_step_id": "",
                            "recovery_action": "",
                            "confidence": None,
                        }
                        for row in session_rows
                    ],
                }
            )

    rows.sort(key=lambda row: (str(row["domain"]), int(row["input_index"]), TARGET_POSITIONS.index(str(row["position_bin"]))))
    manifest = {
        "schema_version": 1,
        "sample_id": "u1_fixed_gate_forced_generation_v1",
        "selection": {
            "seed": seed,
            "labels_read": False,
            "gold_utterances_read": False,
            "eligibility": (
                "D1 fused OOF answer is the exact fallback and frozen R0 raw "
                "response is exactly $silent$; exclude first chunks and old R1 sessions."
            ),
            "domains": domains,
            "sessions_per_domain": sessions_per_domain,
            "positions_per_session": list(TARGET_POSITIONS),
            "excluded_input_indices": sorted(excluded_input_indices),
            "selected_sessions": selected_sessions,
            "sessions": len({int(row["input_index"]) for row in rows}),
            "chunks": len(rows),
            "smoke_sessions": sum(annotation["is_smoke_session"] for annotation in annotations),
            "smoke_chunks": sum(bool(row["is_smoke"]) for row in rows),
        },
        "coverage": {
            "by_domain": dict(Counter(str(row["domain"]) for row in rows)),
            "by_position": dict(Counter(str(row["position_bin"]) for row in rows)),
            "smoke_by_domain": dict(
                Counter(str(row["domain"]) for row in rows if row["is_smoke"])
            ),
            "smoke_by_position": dict(
                Counter(str(row["position_bin"]) for row in rows if row["is_smoke"])
            ),
        },
    }
    return rows, manifest, annotations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config_path = _resolve(args.config)
    output_dir = _resolve(args.output_dir)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sources = config["sources"]
    paths = {name: _resolve(str(value["path"])) for name, value in sources.items()}
    for name, value in sources.items():
        actual = sha256_file(paths[name])
        if actual != value["sha256"]:
            raise ValueError(f"U1 source hash mismatch for {name}: {actual}")
    all_sources = load_jsonl(paths["gold_container"])
    label_free = strip_current_answers(all_sources)
    predictions = load_jsonl(paths["d1_predictions"])
    r0_rows = load_jsonl(paths["r0_session_records"])
    rows, manifest, annotations = prepare_sample(
        label_free,
        predictions,
        r0_rows,
        seed=str(config["sampling"]["seed"]),
        sessions_per_domain=int(config["sampling"]["sessions_per_domain"]),
        excluded_input_indices={
            int(value) for value in config["sampling"]["excluded_input_indices"]
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "sample_items.jsonl", rows)
    write_json(output_dir / "manifest.json", manifest)
    write_json(output_dir / "oracle_states.template.json", annotations)
    write_json(
        output_dir / "data_manifest.json",
        {
            "sources": {
                name: {
                    "path": str(paths[name]),
                    "sha256": value["sha256"],
                    "role": value["role"],
                }
                for name, value in sources.items()
            },
            "selection_reads_gold_answers": False,
            "selection_reads_gold_utterances": False,
            "past_official_dialog_is_inference_visible": True,
            "external_data_used": False,
        },
    )
    summary = {
        "output_dir": str(output_dir),
        "sample_items_sha256": sha256_file(output_dir / "sample_items.jsonl"),
        "manifest_sha256": sha256_file(output_dir / "manifest.json"),
        "annotation_template_sha256": sha256_file(
            output_dir / "oracle_states.template.json"
        ),
        "selection": manifest["selection"],
        "coverage": manifest["coverage"],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
