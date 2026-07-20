"""Compare a GPU D4 smoke with frozen R0/cache/final artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from proactive_d1.neural_core import NeuralFeatureCache
from proactive_d3.dialog_control_core import DIALOG_POLICY_NAMES
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGIT_ABS_TOLERANCE = 1e-6


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_cache(path: Path) -> NeuralFeatureCache:
    with np.load(path, allow_pickle=False) as arrays:
        return NeuralFeatureCache(
            hidden_state=arrays["hidden_state"].astype(np.float32, copy=False),
            tag_margin=arrays["tag_margin"].astype(np.float32, copy=False),
            silent_log_probability=arrays["silent_log_probability"].astype(
                np.float32, copy=False
            ),
            interrupt_log_probability=arrays["interrupt_log_probability"].astype(
                np.float32, copy=False
            ),
            prompt_tokens=arrays["prompt_tokens"].astype(np.int32, copy=False),
            input_index=arrays["input_index"].astype(np.int32, copy=False),
            chunk_index=arrays["chunk_index"].astype(np.int32, copy=False),
        )


def verify(deployment_dir: Path, final_dir: Path) -> dict[str, object]:
    final_config = json.loads((final_dir / "config.json").read_text(encoding="utf-8"))
    sources = final_config["sources"]
    r0_path = _resolve(sources["r0_session_records"]["path"])
    cache_path = _resolve(sources["neural_features"]["path"])
    deployment_records = load_jsonl(deployment_dir / "session_records.jsonl")
    final_records = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in load_jsonl(final_dir / "train_fit_records.jsonl")
    }
    final_predictions = load_jsonl(final_dir / "train_fit_predictions.jsonl")
    r0_records = load_jsonl(r0_path)
    cache = _load_cache(cache_path)
    cache_rows = {
        (int(input_index), int(chunk_index)): row_index
        for row_index, (input_index, chunk_index) in enumerate(
            zip(cache.input_index, cache.chunk_index)
        )
    }
    counts = {
        "raw_response": 0,
        "tag_margin": 0,
        "prompt_tokens": 0,
        "hidden_state": 0,
        "dialog_features": 0,
        "decision": 0,
        "answer": 0,
    }
    maxima = {
        "tag_margin": 0.0,
        "hidden_state": 0.0,
        "dialog_features": 0.0,
        "logit": 0.0,
    }
    chunks = 0
    for session in deployment_records:
        input_index = int(session["input_index"])
        if session["video_path"] != r0_records[input_index]["video_path"]:
            raise ValueError(f"D4 smoke R0 session identity differs at {input_index}")
        expected_prediction = final_predictions[input_index]
        if session["prediction"]["video_path"] != expected_prediction["video_path"]:
            raise ValueError(f"D4 smoke final session identity differs at {input_index}")
        r0_chunks = r0_records[input_index]["chunks"]
        online_chunks = session["chunks"]
        if len(r0_chunks) != len(online_chunks):
            raise ValueError(f"D4 smoke chunk count differs at {input_index}")
        for chunk_index, online in enumerate(online_chunks):
            key = (input_index, chunk_index)
            row_index = cache_rows[key]
            final = final_records[key]
            r0 = r0_chunks[chunk_index]
            counts["raw_response"] += int(online["raw_response"] == r0["raw_response"])
            margin_difference = abs(
                float(online["tag_margin"]) - float(cache.tag_margin[row_index])
            )
            maxima["tag_margin"] = max(maxima["tag_margin"], margin_difference)
            counts["tag_margin"] += int(margin_difference == 0.0)
            counts["prompt_tokens"] += int(
                int(online["prompt_tokens"]) == int(cache.prompt_tokens[row_index])
            )
            online_hidden = np.asarray(online["hidden_state"], dtype=np.float32)
            hidden_difference = float(
                np.max(np.abs(online_hidden - cache.hidden_state[row_index]))
            )
            maxima["hidden_state"] = max(maxima["hidden_state"], hidden_difference)
            counts["hidden_state"] += int(hidden_difference == 0.0)
            dialog_difference = max(
                abs(float(online[name]) - float(final[name]))
                for name in DIALOG_POLICY_NAMES
            )
            maxima["dialog_features"] = max(
                maxima["dialog_features"], dialog_difference
            )
            counts["dialog_features"] += int(dialog_difference == 0.0)
            logit_difference = abs(
                float(online["decision_logit"]) - float(final["logit"])
            )
            maxima["logit"] = max(maxima["logit"], logit_difference)
            counts["decision"] += int(
                int(online["decision_interrupt"])
                == int(final["predicted_interrupt"])
            )
            counts["answer"] += int(
                online["answer"] == expected_prediction["answers"][chunk_index]
            )
            chunks += 1
    passed = (
        chunks > 0
        and all(value == chunks for value in counts.values())
        and maxima["logit"] <= LOGIT_ABS_TOLERANCE
    )
    return {
        "status": "pass" if passed else "fail",
        "classification": "GPU D4 deployment equivalence smoke; not a performance estimate",
        "sessions": len(deployment_records),
        "chunks": chunks,
        "exact_match_counts": counts,
        "max_abs_differences": maxima,
        "logit_abs_tolerance": LOGIT_ABS_TOLERANCE,
        "artifacts": {
            "deployment_records_sha256": sha256_file(
                deployment_dir / "session_records.jsonl"
            ),
            "deployment_predictions_sha256": sha256_file(
                deployment_dir / "predictions.jsonl"
            ),
            "final_head_sha256": sha256_file(final_dir / "decision_head.json"),
            "final_records_sha256": sha256_file(final_dir / "train_fit_records.jsonl"),
            "cache_sha256": sha256_file(cache_path),
            "r0_records_sha256": sha256_file(r0_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    args = parser.parse_args()
    deployment_dir = _resolve(args.deployment_dir)
    result = verify(deployment_dir, _resolve(args.final_dir))
    write_json(deployment_dir / "equivalence_audit.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("D4 GPU deployment equivalence smoke failed")


if __name__ == "__main__":
    main()
