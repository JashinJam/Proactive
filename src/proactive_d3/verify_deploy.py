"""Compare a GPU D3 deployment smoke with frozen R0/cache/final artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from proactive_d3.core import DYNAMIC_SCALAR_NAMES, build_causal_dynamics
from proactive_d1.neural_core import NeuralFeatureCache
from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGIT_ABS_TOLERANCE = 1e-6


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
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


def verify(deployment_dir: Path, final_dir: Path, cache_path: Path) -> dict[str, object]:
    final_config = json.loads((final_dir / "config.json").read_text(encoding="utf-8"))
    r0_dir = _resolve(final_config["r0_reference"]["experiment_dir"])
    deployment_records = load_jsonl(deployment_dir / "session_records.jsonl")
    final_records = {
        (int(row["input_index"]), int(row["chunk_index"])): row
        for row in load_jsonl(final_dir / "train_fit_records.jsonl")
    }
    final_predictions = load_jsonl(final_dir / "train_fit_predictions.jsonl")
    r0_records = load_jsonl(r0_dir / "session_records.jsonl")
    cache = _load_cache(cache_path)
    dynamics = build_causal_dynamics(cache)
    cache_rows = {
        (int(input_index), int(chunk_index)): row_index
        for row_index, (input_index, chunk_index) in enumerate(
            zip(cache.input_index, cache.chunk_index)
        )
    }

    raw_matches = 0
    margin_matches = 0
    prompt_matches = 0
    hidden_matches = 0
    scalar_matches = 0
    delta_matches = 0
    decision_matches = 0
    answer_matches = 0
    chunks = 0
    max_margin_difference = 0.0
    max_hidden_difference = 0.0
    max_scalar_difference = 0.0
    max_delta_difference = 0.0
    max_logit_difference = 0.0
    for session in deployment_records:
        input_index = int(session["input_index"])
        if session["video_path"] != r0_records[input_index]["video_path"]:
            raise ValueError(f"D3 smoke R0 session identity differs at {input_index}")
        expected_prediction = final_predictions[input_index]
        if session["prediction"]["video_path"] != expected_prediction["video_path"]:  # type: ignore[index]
            raise ValueError(f"D3 smoke final session identity differs at {input_index}")
        r0_chunks = r0_records[input_index]["chunks"]
        online_chunks = session["chunks"]
        if len(r0_chunks) != len(online_chunks):  # type: ignore[arg-type]
            raise ValueError(f"D3 smoke chunk count differs at {input_index}")
        for chunk_index, online in enumerate(online_chunks):  # type: ignore[assignment]
            key = (input_index, chunk_index)
            row_index = cache_rows[key]
            final = final_records[key]
            r0 = r0_chunks[chunk_index]  # type: ignore[index]
            raw_matches += int(online["raw_response"] == r0["raw_response"])
            margin_difference = abs(
                float(online["tag_margin"]) - float(cache.tag_margin[row_index])
            )
            max_margin_difference = max(max_margin_difference, margin_difference)
            margin_matches += int(margin_difference == 0.0)
            prompt_matches += int(
                int(online["prompt_tokens"]) == int(cache.prompt_tokens[row_index])
            )
            online_hidden = np.asarray(online["hidden_state"], dtype=np.float32)
            hidden_difference = float(
                np.max(np.abs(online_hidden - cache.hidden_state[row_index]))
            )
            max_hidden_difference = max(max_hidden_difference, hidden_difference)
            hidden_matches += int(hidden_difference == 0.0)
            scalar_difference = max(
                abs(float(online[name]) - float(dynamics.scalar[row_index, index]))
                for index, name in enumerate(DYNAMIC_SCALAR_NAMES)
            )
            max_scalar_difference = max(max_scalar_difference, scalar_difference)
            scalar_matches += int(scalar_difference == 0.0)
            online_delta = np.asarray(online["hidden_delta"], dtype=np.float32)
            delta_difference = float(
                np.max(np.abs(online_delta - dynamics.hidden_delta[row_index]))
            )
            max_delta_difference = max(max_delta_difference, delta_difference)
            delta_matches += int(delta_difference == 0.0)
            logit_difference = abs(
                float(online["decision_logit"]) - float(final["logit"])
            )
            max_logit_difference = max(max_logit_difference, logit_difference)
            decision_matches += int(
                int(online["decision_interrupt"])
                == int(final["predicted_interrupt"])
            )
            expected_answer = expected_prediction["answers"][chunk_index]
            answer_matches += int(online["answer"] == expected_answer)
            chunks += 1
    exact_counts = {
        "raw_response": raw_matches,
        "tag_margin": margin_matches,
        "prompt_tokens": prompt_matches,
        "hidden_state": hidden_matches,
        "dynamic_scalars": scalar_matches,
        "hidden_delta": delta_matches,
        "decision": decision_matches,
        "answer": answer_matches,
    }
    passed = (
        chunks > 0
        and all(count == chunks for count in exact_counts.values())
        and max_logit_difference <= LOGIT_ABS_TOLERANCE
    )
    return {
        "status": "pass" if passed else "fail",
        "classification": "GPU deployment equivalence smoke; not a performance estimate",
        "sessions": len(deployment_records),
        "chunks": chunks,
        "exact_match_counts": exact_counts,
        "max_abs_differences": {
            "tag_margin": max_margin_difference,
            "hidden_state": max_hidden_difference,
            "dynamic_scalars": max_scalar_difference,
            "hidden_delta": max_delta_difference,
            "logit": max_logit_difference,
        },
        "logit_abs_tolerance": LOGIT_ABS_TOLERANCE,
        "artifacts": {
            "deployment_records_sha256": sha256_file(
                deployment_dir / "session_records.jsonl"
            ),
            "deployment_predictions_sha256": sha256_file(
                deployment_dir / "predictions.jsonl"
            ),
            "final_head_sha256": sha256_file(final_dir / "decision_head.json"),
            "final_records_sha256": sha256_file(
                final_dir / "train_fit_records.jsonl"
            ),
            "cache_sha256": sha256_file(cache_path),
            "r0_records_sha256": sha256_file(r0_dir / "session_records.jsonl"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-dir", required=True)
    parser.add_argument("--final-dir", required=True)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    deployment_dir = _resolve(args.deployment_dir)
    result = verify(
        deployment_dir,
        _resolve(args.final_dir),
        _resolve(args.cache),
    )
    write_json(deployment_dir / "equivalence_audit.json", result)
    print(json.dumps(result, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit("D3 GPU deployment equivalence smoke failed")


if __name__ == "__main__":
    main()
