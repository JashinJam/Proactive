"""Build the static evidence bundle for the 2026-07-23 D1-D6 experiment log."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np

from proactive_d1.core import (
    load_decision_head,
    predict_logits,
    strip_answers,
    validate_fold_manifest,
)
from proactive_d3.dialog_control_core import DIALOG_POLICY_NAMES
from proactive_d4_1.compare import load_official_scorer
from proactive_d4_2.core import load_candidates
from proactive_d4_2.evaluate import build_candidate_matrix, merge_candidate_records
from proactive_r0.core import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = Path(
    "/data1/wearable_ai_challenge_data/egoproactive/"
    "wearable_ai_2026_egoproactive_val_700.jsonl"
)
INPUT_SHA256 = "feef69ddee605e7070ad0f133636c35739c6964514a46d76da294b6bf1964740"
D42_EXPERIMENT = Path(
    "output/experiments/"
    "20260721_internvl35_1b_d4_2_adapted_input_policy_oof_v1"
)
D6_EXPERIMENT = Path(
    "output/experiments/"
    "20260722_internvl35_1b_d6_query_memory_lora_oof_v1"
)
SCALAR_NAMES = (
    "is_first_chunk",
    "log1p_chunk_number",
    "log1p_observed_end_sec",
    "log1p_interval_duration",
    "log1p_gap_from_previous",
    "history_turn_fraction",
    "model_input_frame_fraction",
    "domain=Arts and Crafts",
    "domain=Chef",
    "domain=Handyman",
    "domain=Tutorial",
    "r0_decision_interrupt",
    "r0f_decision_interrupt",
    "raw_explicit_interrupt",
    "raw_explicit_silent",
    "raw_malformed_nonempty",
    "raw_empty",
    "log1p_raw_length",
)
REQUIRED_CATALOG_FIELDS = (
    "id",
    "label",
    "stage",
    "comparison_group",
    "evidence_class",
    "status",
    "architecture_id",
    "parameters",
    "metrics_source",
    "report_source",
    "claim_boundary",
)
REQUIRED_STAGE_IDS = ("d1-d2", "d3", "d3d-d4", "d42", "d5", "d6")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _select(payload: Mapping[str, Any], selector: str | None) -> Any:
    value: Any = payload
    if selector:
        for part in selector.split("."):
            if not isinstance(value, Mapping) or part not in value:
                raise ValueError(f"Metrics selector does not exist: {selector}")
            value = value[part]
    return value


def parse_decision(answer: object) -> int:
    return int(str(answer).lstrip().startswith("$interrupt$"))


def utterance(answer: object) -> str:
    value = str(answer).strip()
    return value[len("$interrupt$") :].strip() if value.startswith("$interrupt$") else ""


def validate_source(rows: Sequence[Mapping[str, Any]], input_path: Path) -> dict[str, Any]:
    actual_sha = sha256_file(input_path)
    if actual_sha != INPUT_SHA256:
        raise ValueError(f"Unexpected public-val SHA256: {actual_sha}")
    if len(rows) != 700:
        raise ValueError(f"Expected 700 sessions, got {len(rows)}")
    seen: set[str] = set()
    chunks = 0
    domains: Counter[str] = Counter()
    for input_index, row in enumerate(rows):
        video = row.get("video_path")
        intervals = row.get("video_intervals")
        answers = row.get("answers")
        dialog = row.get("dialog")
        if not isinstance(video, str) or Path(video).name != video or not video.endswith(".mp4"):
            raise ValueError(f"Invalid video_path at input {input_index}")
        if video in seen:
            raise ValueError(f"Duplicate video_path at input {input_index}")
        seen.add(video)
        if not all(isinstance(value, list) for value in (intervals, answers, dialog)):
            raise ValueError(f"Missing aligned chunk arrays at input {input_index}")
        if not (len(intervals) == len(answers) == len(dialog)) or not intervals:
            raise ValueError(f"Chunk arrays are not aligned at input {input_index}")
        previous_start = -math.inf
        for chunk_index, interval in enumerate(intervals):
            if not isinstance(interval, list) or len(interval) != 2:
                raise ValueError(f"Invalid interval at {(input_index, chunk_index)}")
            start, end = (float(interval[0]), float(interval[1]))
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                raise ValueError(f"Non-finite interval at {(input_index, chunk_index)}")
            if start < previous_start:
                raise ValueError(f"Interval order changed at {(input_index, chunk_index)}")
            previous_start = start
            if not isinstance(dialog[chunk_index], list):
                raise ValueError(f"Dialog alignment changed at {(input_index, chunk_index)}")
            if str(answers[chunk_index]).lstrip().startswith(("$silent$", "$interrupt$")) is False:
                raise ValueError(f"Invalid gold answer at {(input_index, chunk_index)}")
        chunks += len(intervals)
        domains[str(row.get("domain"))] += 1
    if chunks != 9935:
        raise ValueError(f"Expected 9,935 chunks, got {chunks}")
    if len(domains) != 4:
        raise ValueError(f"Expected four domains, got {sorted(domains)}")
    return {
        "path": str(input_path),
        "sha256": actual_sha,
        "sessions": len(rows),
        "chunks": chunks,
        "domains": dict(sorted(domains.items())),
        "video_allowlist": sorted(seen),
    }


def validate_catalog(payload: Mapping[str, Any], architecture: Mapping[str, Any]) -> list[dict[str, Any]]:
    experiments = payload.get("experiments")
    variants = architecture.get("variants")
    if not isinstance(experiments, list) or not isinstance(variants, Mapping):
        raise ValueError("Catalog or architecture source is malformed")
    ids: set[str] = set()
    result: list[dict[str, Any]] = []
    for raw in experiments:
        if not isinstance(raw, dict):
            raise ValueError("Catalog entries must be objects")
        missing = [field for field in REQUIRED_CATALOG_FIELDS if field not in raw]
        if missing:
            raise ValueError(f"Catalog entry lacks fields: {missing}")
        experiment_id = str(raw["id"])
        if experiment_id in ids:
            raise ValueError(f"Duplicate catalog id: {experiment_id}")
        if str(raw["architecture_id"]) not in variants:
            raise ValueError(f"Unknown architecture id for {experiment_id}")
        if raw.get("case_browsable") and not raw.get("predictions_source"):
            raise ValueError(f"Case-browsable entry lacks predictions: {experiment_id}")
        if raw.get("evidence_class") == "aggregate_only" and raw.get("case_browsable"):
            raise ValueError(f"Aggregate-only entry entered case browsing: {experiment_id}")
        ids.add(experiment_id)
        result.append(dict(raw))
    return result


def _official_scorer():
    starter = PROJECT_ROOT / "starter_kit"
    scorer_path = starter / "run_evaluation.py"
    return load_official_scorer(starter, sha256_file(scorer_path))


def validate_predictions(
    rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]], label: str
) -> None:
    if len(predictions) != len(rows):
        raise ValueError(f"{label}: expected 700 prediction rows")
    for input_index, (source, prediction) in enumerate(zip(rows, predictions)):
        if prediction.get("video_path") != source.get("video_path"):
            raise ValueError(f"{label}: source order changed at {input_index}")
        source_answers = source.get("answers")
        predicted_answers = prediction.get("answers")
        if not isinstance(source_answers, list) or not isinstance(predicted_answers, list):
            raise ValueError(f"{label}: answers are malformed at {input_index}")
        if len(predicted_answers) != len(source_answers):
            raise ValueError(f"{label}: chunk count changed at {input_index}")


def _metric_projection(metrics: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "macro_f1",
        "gmean_f1",
        "interrupt_precision",
        "interrupt_recall",
        "interrupt_f1",
        "silent_precision",
        "silent_recall",
        "silent_f1",
        "tp",
        "fp",
        "tn",
        "fn",
        "support",
    )
    result = {key: metrics[key] for key in fields if key in metrics}
    support = int(result.get("support", 0))
    if support:
        result["predicted_interrupt_rate"] = round(
            (int(result.get("tp", 0)) + int(result.get("fp", 0))) / support, 6
        )
    return result


def load_and_verify_experiments(
    entries: Sequence[dict[str, Any]], rows: Sequence[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, str]]:
    scorer = _official_scorer()
    generated: list[dict[str, Any]] = []
    predictions_by_id: dict[str, list[dict[str, Any]]] = {}
    artifact_hashes: dict[str, str] = {}
    for entry in entries:
        item = dict(entry)
        metrics_path = _resolve_project_path(str(entry["metrics_source"]))
        report_path = _resolve_project_path(str(entry["report_source"]))
        if not metrics_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"Missing catalog artifact for {entry['id']}")
        artifact_hashes[str(entry["metrics_source"])] = sha256_file(metrics_path)
        artifact_hashes[str(entry["report_source"])] = sha256_file(report_path)
        if entry["evidence_class"] == "aggregate_only":
            needle = str(entry.get("claim_needle", ""))
            if needle and needle not in report_path.read_text(encoding="utf-8"):
                raise ValueError(f"Aggregate claim not found for {entry['id']}: {needle}")
            item["metrics"] = _metric_projection(entry["aggregate_metrics"])
        elif entry["evidence_class"] == "running_snapshot_only":
            item["metrics"] = None
        else:
            source_payload = _read_json(metrics_path)
            frozen = _select(source_payload, entry.get("metrics_selector"))
            if not isinstance(frozen, Mapping):
                raise ValueError(f"Metrics payload is not an object: {entry['id']}")
            predictions_path = _resolve_project_path(str(entry["predictions_source"]))
            predictions = load_jsonl(predictions_path)
            validate_predictions(rows, predictions, str(entry["id"]))
            recomputed = scorer.score_proactive(list(rows), predictions)
            if recomputed.get("skipped_chunks") != 0:
                raise ValueError(f"Official scorer skipped chunks for {entry['id']}")
            expected_frozen = dict(recomputed["overall"])
            if "predicted_interrupt_rate" in frozen:
                expected_frozen["predicted_interrupt_rate"] = round(
                    (int(expected_frozen["tp"]) + int(expected_frozen["fp"]))
                    / int(expected_frozen["support"]),
                    6,
                )
            if expected_frozen != dict(frozen):
                raise ValueError(
                    f"Official scorer mismatch for {entry['id']}: "
                    f"{expected_frozen} != {dict(frozen)}"
                )
            item["metrics"] = _metric_projection(recomputed["overall"])
            predictions_by_id[str(entry["id"])] = predictions
            artifact_hashes[str(entry["predictions_source"])] = sha256_file(predictions_path)
        generated.append(item)
    return generated, predictions_by_id, artifact_hashes


def group_feature_names(names: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "temporal_response_scalar": [],
        "tag_margin": [],
        "hidden_block": [],
        "dialog_stage": [],
    }
    scalar = set(SCALAR_NAMES)
    dialog = set(DIALOG_POLICY_NAMES)
    for index, name in enumerate(names):
        if name in scalar:
            groups["temporal_response_scalar"].append(index)
        elif name == "tag_margin":
            groups["tag_margin"].append(index)
        elif re.fullmatch(r"hidden_\d{4}", name):
            groups["hidden_block"].append(index)
        elif name in dialog:
            groups["dialog_stage"].append(index)
        else:
            raise ValueError(f"Unknown D4.2 feature name: {name}")
    expected = {
        "temporal_response_scalar": 18,
        "tag_margin": 1,
        "hidden_block": 1024,
        "dialog_stage": 8,
    }
    actual = {key: len(value) for key, value in groups.items()}
    if actual != expected:
        raise ValueError(f"D4.2 feature groups changed: {actual}")
    if sum(actual.values()) != len(names) or len(set(names)) != len(names):
        raise ValueError("D4.2 feature grouping is incomplete or duplicated")
    return groups


def rebuild_winner_explanations(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    experiment_dir = (PROJECT_ROOT / D42_EXPERIMENT).resolve()
    config = _read_json(experiment_dir / "config.json")
    answer_free = strip_answers(rows)
    source_manifest = _read_json(experiment_dir / "source_manifest.json")
    fold_manifest = _read_json(experiment_dir / "fold_manifest.json")
    fold_by_index = validate_fold_manifest(fold_manifest, answer_free)
    candidate = next(
        value for value in load_candidates(config) if value["name"] == "history8"
    )
    plan = _read_json(experiment_dir / "feature_plan.json")
    records, _ = merge_candidate_records(
        experiment_dir=experiment_dir,
        candidate=candidate,
        manifest=source_manifest,
        rows=answer_free,
        num_shards=int(plan["num_shards"]),
        hidden_size=1024,
    )
    cache_path = experiment_dir / "candidates" / str(candidate["candidate_id"]) / "features.npz"
    examples, values, names, dialog_audit = build_candidate_matrix(
        source_rows=rows,
        answer_free_rows=answer_free,
        records=records,
        fold_by_index=fold_by_index,
        candidate=candidate,
        cache_path=cache_path,
        config=config,
    )
    head_path = experiment_dir / "final" / "decision_head.json"
    head = load_decision_head(_read_json(head_path))
    if tuple(names) != head.feature_names:
        raise ValueError("Rebuilt winner feature order differs from serialized head")
    groups = group_feature_names(names)
    matrix = np.asarray(values, dtype=np.float64)
    mean = np.asarray(head.model.mean, dtype=np.float64)
    scale = np.asarray(head.model.scale, dtype=np.float64)
    weight = np.asarray(head.model.weight, dtype=np.float64)
    contributions = ((matrix - mean) / scale) * weight
    grouped = {
        key: contributions[:, indices].sum(axis=1)
        for key, indices in groups.items()
    }
    reconstructed = np.full(len(examples), head.model.bias, dtype=np.float64)
    for values_by_group in grouped.values():
        reconstructed += values_by_group
    logits = np.asarray(predict_logits(head.model, matrix), dtype=np.float64)
    maximum_error = float(np.max(np.abs(reconstructed - logits)))
    if maximum_error > 1e-9:
        raise ValueError(f"Grouped contribution error is too large: {maximum_error}")
    fit_records = load_jsonl(experiment_dir / "final" / "train_fit_records.jsonl")
    if len(fit_records) != 9935 or len(examples) != 9935:
        raise ValueError("Full-refit record coverage changed")
    record_error = max(
        abs(float(logit) - float(record["logit"]))
        for logit, record in zip(logits, fit_records)
    )
    if record_error > 1e-9:
        raise ValueError(f"Rebuilt full-refit logits changed: {record_error}")
    non_hidden = groups["temporal_response_scalar"] + groups["tag_margin"] + groups["dialog_stage"]
    explanations: list[dict[str, Any]] = []
    for row_index, (example, logit, record) in enumerate(zip(examples, logits, fit_records)):
        ranked = sorted(non_hidden, key=lambda index: (-abs(contributions[row_index, index]), index))[:6]
        scalar_details = [
            {
                "name": names[index],
                "value": round(float(matrix[row_index, index]), 6),
                "contribution": round(float(contributions[row_index, index]), 9),
                "group": next(key for key, indices in groups.items() if index in indices),
            }
            for index in ranked
        ]
        explanations.append(
            {
                "input_index": example.feature.input_index,
                "chunk_index": example.feature.chunk_index,
                "logit": float(logit),
                "threshold": float(head.threshold_logit),
                "decision": int(record["predicted_interrupt"]),
                "margin": float(logit - head.threshold_logit),
                "contributions": {
                    "bias": float(head.model.bias),
                    **{key: float(value[row_index]) for key, value in grouped.items()},
                },
                "top_explainable_scalars": scalar_details,
            }
        )
    return {
        "explanations": explanations,
        "feature_groups": {key: len(value) for key, value in groups.items()},
        "maximum_reconstruction_error": maximum_error,
        "maximum_record_logit_error": record_error,
        "threshold": float(head.threshold_logit),
        "head_sha256": sha256_file(head_path),
        "dialog_audit": dialog_audit,
    }


def _fold_map() -> dict[int, int]:
    manifest = _read_json(PROJECT_ROOT / D42_EXPERIMENT / "fold_manifest.json")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or len(sessions) != 700:
        raise ValueError("D4.2 fold manifest no longer covers 700 sessions")
    result: dict[int, int] = {}
    for expected, session in enumerate(sessions):
        if int(session["input_index"]) != expected:
            raise ValueError("D4.2 fold manifest source order changed")
        result[expected] = int(session["fold"])
    return result


def _latest_assistant(dialog: object) -> str:
    if not isinstance(dialog, list):
        return ""
    for turn in reversed(dialog):
        if isinstance(turn, Mapping) and str(turn.get("role", "")).lower() == "assistant":
            return str(turn.get("text", ""))
    return ""


def _outcome(gold: int, predicted: int) -> str:
    return ("TP" if gold else "FP") if predicted else ("FN" if gold else "TN")


def choose_featured_cases(
    rows: Sequence[dict[str, Any]],
    baseline: Sequence[dict[str, Any]],
    history8: Sequence[dict[str, Any]],
    explanations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    cursor = 0
    for input_index, (source, base_row, history_row) in enumerate(zip(rows, baseline, history8)):
        gold_answers = source["answers"]
        base_answers = base_row["answers"]
        history_answers = history_row["answers"]
        repaired: list[int] = []
        regressed: list[int] = []
        common: list[int] = []
        common_margin: dict[int, float] = {}
        for chunk_index, (gold_answer, base_answer, history_answer) in enumerate(
            zip(gold_answers, base_answers, history_answers)
        ):
            gold = parse_decision(gold_answer)
            base = parse_decision(base_answer)
            history = parse_decision(history_answer)
            explanation = explanations[cursor]
            if (int(explanation["input_index"]), int(explanation["chunk_index"])) != (
                input_index,
                chunk_index,
            ):
                raise ValueError("Explanation source order changed")
            if base != gold and history == gold:
                repaired.append(chunk_index)
            if base == gold and history != gold:
                regressed.append(chunk_index)
            if base != gold and history != gold:
                common.append(chunk_index)
                common_margin[chunk_index] = abs(float(explanation["margin"]))
            cursor += 1
        stats.append(
            {
                "input_index": input_index,
                "domain": str(source["domain"]),
                "repaired": repaired,
                "regressed": regressed,
                "common": common,
                "common_margin": common_margin,
                "net": len(repaired) - len(regressed),
            }
        )
    used_domains: set[str] = set()

    def pick(candidates: list[dict[str, Any]], key) -> dict[str, Any]:
        different = [value for value in candidates if value["domain"] not in used_domains]
        pool = different or candidates
        if not pool:
            raise ValueError("No session satisfies a featured-case definition")
        selected = sorted(pool, key=key)[0]
        used_domains.add(str(selected["domain"]))
        return selected

    repair = pick(
        [value for value in stats if value["repaired"]],
        lambda value: (-int(value["net"]), -len(value["repaired"]), int(value["input_index"])),
    )
    regression = pick(
        [value for value in stats if value["regressed"]],
        lambda value: (-len(value["regressed"]), int(value["input_index"])),
    )
    common_hard = pick(
        [value for value in stats if value["common"]],
        lambda value: (
            -len(value["common"]),
            -max(value["common_margin"].values()),
            int(value["input_index"]),
        ),
    )
    common_chunk = sorted(
        common_hard["common"],
        key=lambda chunk: (-common_hard["common_margin"][chunk], chunk),
    )[0]
    return [
        {
            "type": "repair",
            "input_index": repair["input_index"],
            "chunk_index": min(repair["repaired"]),
            "domain": repair["domain"],
            "selection": "session net corrections desc, corrections desc, input_index asc; first qualifying chunk",
        },
        {
            "type": "regression",
            "input_index": regression["input_index"],
            "chunk_index": min(regression["regressed"]),
            "domain": regression["domain"],
            "selection": "new errors desc, input_index asc; first qualifying chunk",
        },
        {
            "type": "common_hard",
            "input_index": common_hard["input_index"],
            "chunk_index": common_chunk,
            "domain": common_hard["domain"],
            "selection": "common errors desc, full-refit abs margin desc, input_index asc",
        },
    ]


def build_session_data(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    predictions_by_id: Mapping[str, Sequence[dict[str, Any]]],
    explanations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    fold_by_index = _fold_map()
    index_rows: list[dict[str, Any]] = []
    cursor = 0
    config_ids = sorted(predictions_by_id)
    for input_index, source in enumerate(rows):
        chunks: list[dict[str, Any]] = []
        session_outcomes: dict[str, Counter[str]] = {
            config_id: Counter() for config_id in config_ids
        }
        disagreements = 0
        low_margin = 0
        for chunk_index, interval in enumerate(source["video_intervals"]):
            gold_answer = source["answers"][chunk_index]
            gold = parse_decision(gold_answer)
            config_predictions: dict[str, Any] = {}
            for config_id in config_ids:
                answer = predictions_by_id[config_id][input_index]["answers"][chunk_index]
                decision = parse_decision(answer)
                outcome = _outcome(gold, decision)
                session_outcomes[config_id][outcome] += 1
                config_predictions[config_id] = {
                    "decision": decision,
                    "outcome": outcome,
                    "answer": answer,
                    "utterance": utterance(answer),
                }
            explanation = explanations[cursor]
            if (explanation["input_index"], explanation["chunk_index"]) != (
                input_index,
                chunk_index,
            ):
                raise ValueError("Full-refit explanation order changed")
            disagreements += int(
                config_predictions["d42_baseline"]["decision"]
                != config_predictions["d42_history8"]["decision"]
            )
            low_margin += int(abs(float(explanation["margin"])) < 0.2)
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "interval": [float(interval[0]), float(interval[1])],
                    "gold": {
                        "decision": gold,
                        "answer": gold_answer,
                        "utterance": utterance(gold_answer),
                    },
                    "predictions": config_predictions,
                    "full_refit": explanation,
                    "latest_visible_assistant": _latest_assistant(source["dialog"][chunk_index]),
                }
            )
            cursor += 1
        session = {
            "schema_version": 1,
            "input_index": input_index,
            "video_name": source["video_path"],
            "media_url": f"/media/{source['video_path']}",
            "duration_in_sec": float(source.get("duration_in_sec", 0.0)),
            "domain": source.get("domain"),
            "task": source.get("task"),
            "query": source.get("query"),
            "fold": fold_by_index[input_index],
            "chunks": chunks,
        }
        _write_json(output_dir / "sessions" / f"{input_index}.json", session)
        index_rows.append(
            {
                "input_index": input_index,
                "video_name": source["video_path"],
                "domain": source.get("domain"),
                "task": source.get("task"),
                "query": source.get("query"),
                "fold": fold_by_index[input_index],
                "chunks": len(chunks),
                "outcomes": {
                    config_id: dict(sorted(counts.items()))
                    for config_id, counts in session_outcomes.items()
                },
                "d42_disagreements": disagreements,
                "low_margin_chunks": low_margin,
                "keywords": f"{source.get('task', '')} {source.get('query', '')}".lower(),
            }
        )
    if cursor != 9935:
        raise ValueError(f"Session writer covered {cursor} chunks")
    _write_json(output_dir / "sessions" / "index.json", {"sessions": index_rows})
    return index_rows


def _trim_summary(path: Path, selector: str | None = None) -> dict[str, Any]:
    payload = _read_json(path)
    selected = _select(payload, selector)
    if not isinstance(selected, Mapping):
        raise ValueError(f"Summary selector is not an object: {path}")
    return dict(selected)


def build_metrics(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(entry["id"]): entry for entry in entries}
    d42_ids = ["d42_history8", "d42_frames16", "d42_baseline", "d42_tokens16"]
    d42_summaries = {
        experiment_id: _read_json(_resolve_project_path(by_id[experiment_id]["metrics_source"]))
        for experiment_id in d42_ids
    }
    d5_ids = [
        "d5_history8_replay",
        "d5_multiscale",
        "d5_dual_view",
        "d5_visual_residual",
        "d5_robust",
    ]
    d42_comparison = []
    for experiment_id in d42_ids:
        summary = d42_summaries[experiment_id]
        d42_comparison.append(
            {
                "id": experiment_id,
                "label": by_id[experiment_id]["label"],
                "status": by_id[experiment_id]["status"],
                "parameters": by_id[experiment_id]["parameters"],
                "metrics": by_id[experiment_id]["metrics"],
                "timing": summary.get("timing"),
                "bootstrap": summary.get("session_bootstrap_vs_baseline"),
                "folds": [
                    {
                        "fold": detail["test_fold"],
                        "macro_f1": round(float(detail["test_metrics_internal"]["macro_f1"]), 4),
                    }
                    for detail in summary.get("fold_details", [])
                ],
                "domains": summary.get("stratified", {}).get("domain", {}),
            }
        )
    d5_funnel = [
        {
            "id": experiment_id,
            "label": by_id[experiment_id]["label"],
            "status": by_id[experiment_id]["status"],
            "metrics": by_id[experiment_id]["metrics"],
            "claim_boundary": by_id[experiment_id]["claim_boundary"],
        }
        for experiment_id in d5_ids
    ]
    history = d42_summaries["d42_history8"]
    baseline = d42_summaries["d42_baseline"]
    train_fit = _read_json(PROJECT_ROOT / D42_EXPERIMENT / "final" / "summary.json")
    winner_audit = {
        "oof": by_id["d42_history8"]["metrics"],
        "baseline_oof": by_id["d42_baseline"]["metrics"],
        "bootstrap": history["session_bootstrap_vs_baseline"],
        "decision_changes": history["decision_changes_vs_baseline"],
        "folds": [
            {
                "fold": current["test_fold"],
                "history8": current["test_metrics_internal"]["macro_f1"],
                "baseline": control["test_metrics_internal"]["macro_f1"],
            }
            for current, control in zip(history["fold_details"], baseline["fold_details"])
        ],
        "domains": {
            domain: {
                "history8": history["stratified"]["domain"][domain]["macro_f1"],
                "baseline": baseline["stratified"]["domain"][domain]["macro_f1"],
            }
            for domain in sorted(history["stratified"]["domain"])
        },
        "slices": {
            key: {
                "history8": value["macro_f1"],
                "baseline": baseline["stratified"]["chunk_position"][key]["macro_f1"],
                "support": value["support"],
            }
            for key, value in history["stratified"]["chunk_position"].items()
        },
        "train_fit_sanity": train_fit["train_fit_official"],
        "train_fit_label": "All-development train-fit sanity",
        "oof_label": "OOF held-out development estimate",
    }
    trajectory = [
        {
            "id": experiment_id,
            "label": by_id[experiment_id]["label"],
            "macro_f1": by_id[experiment_id]["metrics"]["macro_f1"],
            "evidence_class": by_id[experiment_id]["evidence_class"],
            "status": by_id[experiment_id]["status"],
        }
        for experiment_id in ("d1_fused", "d3_dynamics", "d42_history8")
    ]
    return {
        "schema_version": 1,
        "scientific_trajectory": trajectory,
        "d42_comparison": d42_comparison,
        "d5_funnel": d5_funnel,
        "winner_audit": winner_audit,
        "evidence_separation": {
            "oof": "OOF held-out development estimate: 0.6988",
            "train_fit": "All-development train-fit sanity: 0.7469",
            "rule": "never rank, trend, or color-code these as the same evidence class",
        },
    }


def capture_d6_status() -> dict[str, Any]:
    path = PROJECT_ROOT / D6_EXPERIMENT / "launcher" / "status.json"
    captured_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    if not path.is_file():
        return {
            "schema_version": 1,
            "captured_at": captured_at,
            "status": "launcher_status_missing",
            "efficacy_available": False,
            "source": str(path.relative_to(PROJECT_ROOT)),
        }
    payload = _read_json(path)
    running = payload.get("running", [])
    completed = payload.get("completed", [])
    failed = payload.get("failed", [])
    pending = payload.get("pending", [])
    return {
        "schema_version": 1,
        "captured_at": captured_at,
        "source": str(path.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256_file(path),
        "source_mtime": datetime.fromtimestamp(
            path.stat().st_mtime, ZoneInfo("Asia/Shanghai")
        ).isoformat(),
        "kind": payload.get("kind"),
        "config_sha256": payload.get("config_sha256"),
        "pending_folds": [int(value) for value in pending],
        "running_folds": [int(value["fold"]) for value in running if isinstance(value, Mapping)],
        "completed_folds": [
            int(value["fold"]) if isinstance(value, Mapping) else int(value)
            for value in completed
        ],
        "failed_folds": [
            {"fold": int(value["fold"]), "return_code": int(value.get("return_code", -1))}
            for value in failed
            if isinstance(value, Mapping)
        ],
        "launcher_wall_seconds": payload.get("wall_seconds"),
        "efficacy_available": False,
        "ranking_eligible": False,
        "interpretation": "build-time static launcher snapshot only; no efficacy conclusion",
    }


def validate_stage_content(source: Mapping[str, Any]) -> dict[str, Any]:
    stages = source.get("stages")
    if not isinstance(stages, list):
        raise ValueError("Experiment log stages must be a list")
    ids = tuple(str(stage.get("id")) for stage in stages if isinstance(stage, Mapping))
    if ids != REQUIRED_STAGE_IDS:
        raise ValueError(f"Experiment stage order changed: {ids}")
    common = ("id", "label", "eyebrow", "title", "status", "summary")
    for stage in stages:
        missing = [field for field in common if field not in stage]
        if missing:
            raise ValueError(f"Experiment stage lacks fields: {stage.get('id')} {missing}")
        if stage["id"] == "d1-d2":
            if (
                not stage.get("baseline_facts")
                or not stage.get("brief_review")
                or not stage.get("result_view")
                or not stage.get("conclusion")
            ):
                raise ValueError("D1/D2 brief review is incomplete")
            if any(field in stage for field in ("configuration", "reason", "analysis")):
                raise ValueError("D1/D2 brief review must not use the four-part experiment template")
            continue
        if not isinstance(stage.get("configuration"), list) or not stage["configuration"]:
            raise ValueError(f"Experiment stage lacks configuration rows: {stage['id']}")
        if not isinstance(stage.get("reason"), list) or not stage["reason"]:
            raise ValueError(f"Experiment stage lacks reason text: {stage['id']}")
        if stage["id"] == "d6":
            if "result_view" in stage or "analysis" in stage:
                raise ValueError("D6 must not expose results or analysis while OOF is running")
            if not stage.get("model_flow") or not stage.get("why_slow") or not stage.get("status_note"):
                raise ValueError("D6 must explain its runtime and current status")
        elif not stage.get("result_view") or not stage.get("analysis"):
            raise ValueError(f"Completed experiment stage lacks results or analysis: {stage['id']}")
    d42 = next(stage for stage in stages if stage["id"] == "d42")
    if not d42.get("limited_search", {}).get("items"):
        raise ValueError("D4.2 must explain why its configuration search was bounded")
    return json.loads(json.dumps(source, ensure_ascii=False))


def _directory_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda value: value.name):
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def build_dashboard(presentation_dir: Path, input_jsonl: Path = DEFAULT_INPUT) -> dict[str, Any]:
    presentation_dir = presentation_dir.expanduser().resolve()
    input_jsonl = input_jsonl.expanduser().resolve()
    catalog_source = _read_json(presentation_dir / "catalog.json")
    architecture = _read_json(presentation_dir / "architecture.json")
    stage_source = _read_json(presentation_dir / "stage_content.json")
    rows = load_jsonl(input_jsonl)
    source_summary = validate_source(rows, input_jsonl)
    entries = validate_catalog(catalog_source, architecture)
    stages = validate_stage_content(stage_source)
    generated_entries, predictions_by_id, artifact_hashes = load_and_verify_experiments(
        entries, rows
    )
    explanation_bundle = rebuild_winner_explanations(rows)
    full_fit_predictions_path = PROJECT_ROOT / D42_EXPERIMENT / "final" / "train_fit_predictions.jsonl"
    full_fit_predictions = load_jsonl(full_fit_predictions_path)
    validate_predictions(rows, full_fit_predictions, "d42_history8_full_refit")
    scorer = _official_scorer()
    full_fit_metrics = scorer.score_proactive(rows, full_fit_predictions)
    if full_fit_metrics["overall"]["macro_f1"] != 0.7469:
        raise ValueError("Full-refit official Macro F1 no longer reproduces 0.7469")
    predictions_by_id["d42_history8_full_refit"] = full_fit_predictions
    artifact_hashes[str(full_fit_predictions_path.relative_to(PROJECT_ROOT))] = sha256_file(
        full_fit_predictions_path
    )
    featured = choose_featured_cases(
        rows,
        predictions_by_id["d42_baseline"],
        predictions_by_id["d42_history8"],
        explanation_bundle["explanations"],
    )
    d6_status = capture_d6_status()
    dashboard_dir = presentation_dir / "dashboard"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="data-build-", dir=dashboard_dir))
    try:
        session_index = build_session_data(
            temporary,
            rows,
            predictions_by_id,
            explanation_bundle["explanations"],
        )
        generated_catalog = {
            "schema_version": 1,
            "experiments": generated_entries,
            "case_config_ids": sorted(predictions_by_id),
        }
        metrics = build_metrics(generated_entries)
        cases = {
            "schema_version": 1,
            "featured": featured,
            "selection_is_deterministic": True,
            "sessions": len(session_index),
            "chunks": 9935,
            "filters": ["TP", "FP", "TN", "FN", "domain", "fold", "disagreement", "low_margin", "keyword"],
            "oof_explanation_boundary": (
                "OOF decisions are frozen fold-head outputs. Logit, threshold, and contribution "
                "waterfalls are from the single all-development full-refit head only."
            ),
            "utterance_boundary": "Utterance content is not part of the project Macro-F1.",
        }
        _write_json(temporary / "catalog.json", generated_catalog)
        _write_json(temporary / "stages.json", stages)
        _write_json(temporary / "metrics.json", metrics)
        _write_json(temporary / "cases.json", cases)
        _write_json(temporary / "architecture.json", architecture)
        _write_json(temporary / "d6_status.json", d6_status)
        session_paths = list((temporary / "sessions").glob("[0-9]*.json"))
        built_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        key_files = [
            temporary / "catalog.json",
            temporary / "stages.json",
            temporary / "metrics.json",
            temporary / "cases.json",
            temporary / "architecture.json",
            temporary / "d6_status.json",
            temporary / "sessions" / "index.json",
        ]
        manifest = {
            "schema_version": 1,
            "built_at": built_at,
            "source": source_summary,
            "catalog_source": str((presentation_dir / "catalog.json").relative_to(PROJECT_ROOT)),
            "catalog_source_sha256": sha256_file(presentation_dir / "catalog.json"),
            "architecture_source_sha256": sha256_file(presentation_dir / "architecture.json"),
            "stage_source_sha256": sha256_file(presentation_dir / "stage_content.json"),
            "official_scorer_sha256": sha256_file(PROJECT_ROOT / "starter_kit" / "run_evaluation.py"),
            "fold_manifest_sha256": sha256_file(PROJECT_ROOT / D42_EXPERIMENT / "fold_manifest.json"),
            "full_refit_head_sha256": explanation_bundle["head_sha256"],
            "full_refit_official": full_fit_metrics["overall"],
            "contribution_audit": {
                "feature_groups": explanation_bundle["feature_groups"],
                "maximum_reconstruction_error": explanation_bundle["maximum_reconstruction_error"],
                "maximum_record_logit_error": explanation_bundle["maximum_record_logit_error"],
                "hidden_vectors_emitted": False,
            },
            "artifact_hashes": dict(sorted(artifact_hashes.items())),
            "files": {str(path.relative_to(temporary)): sha256_file(path) for path in key_files},
            "session_files": {
                "count": len(session_paths),
                "composite_sha256": _directory_digest(session_paths),
            },
            "evidence_policy": {
                "oof_label": "OOF held-out development estimate",
                "train_fit_label": "All-development train-fit sanity",
                "d6_in_rankings": False,
                "hidden_test_claimed": False,
            },
        }
        _write_json(temporary / "manifest.json", manifest)
        target = dashboard_dir / "data"
        if target.exists():
            shutil.rmtree(target)
        temporary.replace(target)
        temporary = target
        return manifest
    finally:
        if temporary.exists() and temporary.name.startswith("data-build-"):
            shutil.rmtree(temporary)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--presentation-dir", required=True, type=Path)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args(argv)
    manifest = build_dashboard(args.presentation_dir, args.input_jsonl)
    print(
        json.dumps(
            {
                "status": "complete",
                "built_at": manifest["built_at"],
                "sessions": manifest["source"]["sessions"],
                "chunks": manifest["source"]["chunks"],
                "data_dir": str(args.presentation_dir / "dashboard" / "data"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
