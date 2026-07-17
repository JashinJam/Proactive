"""Verify online fused deployment rows against frozen offline D1 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def verify(
    deploy_dir: Path,
    final_dir: Path,
    r0_dir: Path,
    cache_path: Path,
    tag_margin_tolerance: float,
    logit_tolerance: float,
    hidden_tolerance: float,
    require_hidden_state: bool,
) -> dict[str, object]:
    online_records = load_jsonl(deploy_dir / "session_records.jsonl")
    r0_records = {
        int(record["input_index"]): record
        for record in load_jsonl(r0_dir / "session_records.jsonl")
    }
    final_records = {
        (int(record["input_index"]), int(record["chunk_index"])): record
        for record in load_jsonl(final_dir / "train_fit_records.jsonl")
    }
    final_predictions = load_jsonl(final_dir / "train_fit_predictions.jsonl")
    with np.load(cache_path, allow_pickle=False) as archive:
        cache_margin = {
            (int(input_index), int(chunk_index)): float(tag_margin)
            for input_index, chunk_index, tag_margin in zip(
                archive["input_index"], archive["chunk_index"], archive["tag_margin"]
            )
        }
        cache_hidden = {
            (int(input_index), int(chunk_index)): hidden.astype(
                np.float32, copy=True
            )
            for input_index, chunk_index, hidden in zip(
                archive["input_index"], archive["chunk_index"], archive["hidden_state"]
            )
        }

    raw_equal = 0
    margin_equal = 0
    decision_equal = 0
    answer_equal = 0
    chunks = 0
    max_margin_difference = 0.0
    max_logit_difference = 0.0
    hidden_rows_recorded = 0
    hidden_exact_matches = 0
    max_hidden_difference = 0.0
    for online in online_records:
        input_index = int(online["input_index"])
        r0 = r0_records[input_index]
        online_chunks = online["chunks"]
        r0_chunks = r0["chunks"]
        if not isinstance(online_chunks, list) or not isinstance(r0_chunks, list):
            raise ValueError("D1 deployment or R0 record has no chunk list")
        if len(online_chunks) != len(r0_chunks):
            raise ValueError("D1 online and R0 chunk counts differ")
        online_prediction = online["prediction"]
        if not isinstance(online_prediction, dict):
            raise ValueError("D1 online record has no prediction")
        expected_answers = final_predictions[input_index]["answers"]
        actual_answers = online_prediction["answers"]
        if not isinstance(expected_answers, list) or not isinstance(actual_answers, list):
            raise ValueError("D1 online or final prediction has no answers")
        if len(expected_answers) != len(actual_answers):
            raise ValueError("D1 online and final answer counts differ")
        for chunk_index, (current, baseline) in enumerate(zip(online_chunks, r0_chunks)):
            if not isinstance(current, dict) or not isinstance(baseline, dict):
                raise ValueError("D1 chunk record is not an object")
            key = (input_index, chunk_index)
            final = final_records[key]
            chunks += 1
            raw_equal += int(current["raw_response"] == baseline["raw_response"])
            margin_difference = abs(float(current["tag_margin"]) - cache_margin[key])
            max_margin_difference = max(max_margin_difference, margin_difference)
            margin_equal += int(margin_difference == 0.0)
            logit_difference = abs(float(current["decision_logit"]) - float(final["logit"]))
            max_logit_difference = max(max_logit_difference, logit_difference)
            decision_equal += int(
                int(current["decision_interrupt"]) == int(final["predicted_interrupt"])
            )
            answer_equal += int(actual_answers[chunk_index] == expected_answers[chunk_index])
            if "hidden_state" in current:
                hidden = np.asarray(current["hidden_state"], dtype=np.float32)
                expected_hidden = cache_hidden[key]
                if hidden.shape != expected_hidden.shape:
                    raise ValueError("D1 online and cached hidden shapes differ")
                hidden_difference = float(np.max(np.abs(hidden - expected_hidden)))
                max_hidden_difference = max(max_hidden_difference, hidden_difference)
                hidden_rows_recorded += 1
                hidden_exact_matches += int(hidden_difference == 0.0)

    passed = (
        chunks > 0
        and raw_equal == chunks
        and max_margin_difference <= tag_margin_tolerance
        and decision_equal == chunks
        and answer_equal == chunks
        and max_logit_difference <= logit_tolerance
        and (
            not require_hidden_state
            or (
                hidden_rows_recorded == chunks
                and max_hidden_difference <= hidden_tolerance
            )
        )
    )
    result = {
        "status": "pass" if passed else "fail",
        "sessions": len(online_records),
        "chunks": chunks,
        "raw_response_exact_matches": raw_equal,
        "tag_margin_exact_matches": margin_equal,
        "decision_exact_matches": decision_equal,
        "answer_exact_matches": answer_equal,
        "max_tag_margin_abs_difference": max_margin_difference,
        "tag_margin_tolerance": tag_margin_tolerance,
        "max_logit_abs_difference": max_logit_difference,
        "logit_tolerance": logit_tolerance,
        "hidden_rows_recorded": hidden_rows_recorded,
        "hidden_exact_matches": hidden_exact_matches,
        "max_hidden_abs_difference": max_hidden_difference,
        "hidden_tolerance": hidden_tolerance,
        "hidden_state_required": require_hidden_state,
        "artifacts": {
            "online_records_sha256": sha256_file(deploy_dir / "session_records.jsonl"),
            "r0_records_sha256": sha256_file(r0_dir / "session_records.jsonl"),
            "neural_cache_sha256": sha256_file(cache_path),
            "final_records_sha256": sha256_file(final_dir / "train_fit_records.jsonl"),
            "final_predictions_sha256": sha256_file(
                final_dir / "train_fit_predictions.jsonl"
            ),
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deploy-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--r0-dir", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--tag-margin-tolerance", type=float, default=0.0)
    parser.add_argument("--logit-tolerance", type=float, default=1e-6)
    parser.add_argument("--hidden-tolerance", type=float, default=1e-6)
    parser.add_argument("--require-hidden-state", action="store_true")
    args = parser.parse_args()
    deploy_dir = _resolve(args.deploy_dir)
    result = verify(
        deploy_dir=deploy_dir,
        final_dir=_resolve(args.final_dir),
        r0_dir=_resolve(args.r0_dir),
        cache_path=_resolve(args.cache),
        tag_margin_tolerance=args.tag_margin_tolerance,
        logit_tolerance=args.logit_tolerance,
        hidden_tolerance=args.hidden_tolerance,
        require_hidden_state=args.require_hidden_state,
    )
    write_json(deploy_dir / "consistency_audit.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("D1 fused deployment consistency gate failed")


if __name__ == "__main__":
    main()
