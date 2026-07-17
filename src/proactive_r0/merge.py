"""Merge completed R0 data-parallel shards and run the official full scorer."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

from .artifacts import code_snapshot, environment_snapshot, write_json
from .core import load_jsonl, validate_prediction_rows, write_jsonl
from .run import (
    PROJECT_ROOT,
    _diagnostics,
    _load_config,
    _readme_text,
    _run_official_scorer,
    _tracked_code_paths,
    _write_completed_report,
    contiguous_shard_bounds,
)


def _resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _write_command(path: Path, argv: list[str]) -> None:
    command = shlex.join([sys.executable, "-m", "proactive_r0.merge", *argv])
    text = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n"
    )
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "r0_internvl35_1b_no_plan.json"),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--shard-dir", action="append", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    started_at = time.monotonic()
    config_path = _resolve(args.config)
    base_config = _load_config(config_path)
    data_config: dict[str, object] = base_config["data"]  # type: ignore[assignment]
    starter_config: dict[str, object] = base_config["starter_kit"]  # type: ignore[assignment]
    input_path = _resolve(str(data_config["input"]))
    starter_dir = _resolve(str(starter_config["path"]))
    output_dir = _resolve(args.output_dir)
    shard_dirs = [_resolve(path) for path in args.shard_dir]
    if output_dir.exists():
        raise FileExistsError(f"Merge output already exists: {output_dir}")

    source_rows = load_jsonl(input_path)
    num_shards = len(shard_dirs)
    records: list[dict[str, object]] = []
    shard_summaries: list[dict[str, object]] = []
    first_manifest: dict[str, object] | None = None
    seen_shard_indices: set[int] = set()
    invariant_keys = [
        "model",
        "data",
        "starter_kit",
        "inference",
        "evaluation",
        "validation_policy",
    ]

    for shard_dir in shard_dirs:
        shard_config = json.loads(
            (shard_dir / "config.json").read_text(encoding="utf-8")
        )
        for key in invariant_keys:
            if shard_config[key] != base_config[key]:
                raise ValueError(
                    f"Shard {shard_dir} differs in frozen config key {key}"
                )
        shard_runtime_config = shard_config["runtime"]
        shard_index = int(shard_runtime_config["shard_index"])
        if int(shard_runtime_config["num_shards"]) != num_shards:
            raise ValueError(f"Shard {shard_dir} has inconsistent num_shards")
        if shard_index in seen_shard_indices:
            raise ValueError(f"Duplicate shard index {shard_index}")
        seen_shard_indices.add(shard_index)

        start, stop = contiguous_shard_bounds(
            len(source_rows), num_shards, shard_index
        )
        shard_records = load_jsonl(shard_dir / "session_records.jsonl")
        expected_indices = list(range(start, stop))
        actual_indices = [int(record["input_index"]) for record in shard_records]
        if actual_indices != expected_indices:
            raise ValueError(
                f"Shard {shard_index} index coverage mismatch: expected "
                f"{start}:{stop}, got {actual_indices[:3]}..."
            )
        shard_predictions = [record["prediction"] for record in shard_records]
        validate_prediction_rows(  # type: ignore[arg-type]
            source_rows[start:stop], shard_predictions
        )
        records.extend(shard_records)

        runtime = json.loads(
            (shard_dir / "runtime.json").read_text(encoding="utf-8")
        )
        if runtime.get("status") != "complete":
            raise ValueError(f"Shard {shard_index} is not complete")
        diagnostics = json.loads(
            (shard_dir / "diagnostics.json").read_text(encoding="utf-8")
        )
        shard_summaries.append(
            {
                "shard_index": shard_index,
                "path": str(shard_dir),
                "selection_start": start,
                "selection_stop": stop,
                "runtime": runtime,
                "diagnostics": diagnostics,
            }
        )
        if first_manifest is None:
            first_manifest = json.loads(
                (shard_dir / "data_manifest.json").read_text(encoding="utf-8")
            )

    if seen_shard_indices != set(range(num_shards)):
        missing = set(range(num_shards)) - seen_shard_indices
        raise ValueError(f"Missing shard indices: {missing}")
    records.sort(key=lambda record: int(record["input_index"]))
    merged_indices = [int(record["input_index"]) for record in records]
    if merged_indices != list(range(len(source_rows))):
        raise ValueError("Merged records do not cover every source row exactly once")

    predictions = [record["prediction"] for record in records]
    validation = validate_prediction_rows(  # type: ignore[arg-type]
        source_rows, predictions
    )
    if validation["sessions"] != 700 or validation["chunks"] != 9935:
        raise ValueError(f"Unexpected full-set shape: {validation}")

    output_dir.mkdir(parents=True)
    write_jsonl(output_dir / "session_records.jsonl", records)
    predictions_path = output_dir / "predictions.jsonl"
    write_jsonl(predictions_path, predictions)  # type: ignore[arg-type]
    diagnostics = _diagnostics(records, validation, predictions_path)
    write_json(output_dir / "diagnostics.json", diagnostics)

    metrics_path = output_dir / "metrics.json"
    _run_official_scorer(
        starter_dir,
        input_path,
        predictions_path,
        metrics_path,
        output_dir / "scorer.log",
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    ordered_summaries = sorted(
        shard_summaries, key=lambda item: int(item["shard_index"])
    )
    wall_times = [
        float(item["runtime"]["wall_time_seconds"])  # type: ignore[index]
        for item in ordered_summaries
    ]
    peak_memories = [
        int(item["runtime"]["peak_gpu_memory_bytes"] or 0)  # type: ignore[index]
        for item in ordered_summaries
    ]
    runtime = {
        "status": "complete",
        "completed_at": datetime.now().astimezone().isoformat(),
        "execution": f"{num_shards}-way contiguous session data parallel",
        "num_shards": num_shards,
        "wall_time_seconds": max(wall_times),
        "aggregate_gpu_process_seconds": sum(wall_times),
        "merge_and_score_seconds": round(time.monotonic() - started_at, 3),
        "peak_gpu_memory_bytes": max(peak_memories),
        "sessions": validation["sessions"],
        "chunks": validation["chunks"],
        "shards": ordered_summaries,
    }
    write_json(output_dir / "runtime.json", runtime)

    effective = json.loads(json.dumps(base_config))
    effective["runtime"] = {
        "mode": "merged_data_parallel_shards",
        "num_shards": num_shards,
        "shard_dirs": [str(path) for path in shard_dirs],
        "output_dir": str(output_dir),
    }
    write_json(output_dir / "config.json", effective)
    _write_command(output_dir / "command.sh", raw_argv)
    shard_environments = [
        {
            "shard_index": int(summary["shard_index"]),
            "environment": json.loads(
                (Path(str(summary["path"])) / "environment.txt").read_text(
                    encoding="utf-8"
                )
            ),
        }
        for summary in ordered_summaries
    ]
    write_json(
        output_dir / "environment.txt",
        {
            "merge_process": environment_snapshot(),
            "inference_shards": shard_environments,
        },
    )
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(PROJECT_ROOT, _tracked_code_paths(config_path)),
    )

    assert first_manifest is not None
    source_manifest = first_manifest["source"]
    if not isinstance(source_manifest, dict):
        raise ValueError("Invalid source manifest in first shard")
    source_manifest.update(
        {
            "sessions_selected": validation["sessions"],
            "chunks_selected": validation["chunks"],
            "full_public_validation": True,
            "selection_start": 0,
            "selection_stop": len(source_rows),
            "num_shards": num_shards,
            "shard_index": None,
        }
    )
    first_manifest["parallel_execution"] = {
        "strategy": "contiguous session shards",
        "shards": ordered_summaries,
    }
    supervision_manifest = first_manifest.get("supervision")
    if isinstance(supervision_manifest, dict):
        supervision_manifest[
            "official_scoring_reads_labels_after_predictions_are_frozen"
        ] = True
    write_json(output_dir / "data_manifest.json", first_manifest)

    readme = _readme_text(
        effective,
        "complete full public-validation R0",
        diagnostics,
        metrics,
        runtime,
    )
    readme += (
        "\n## Parallel Execution\n\n"
        f"The 700 sessions were split into {num_shards} contiguous shards. "
        "Each shard used the identical frozen model and inference config; only "
        "session-level throughput was parallelized. Global input indices were "
        "validated as an exact 0..699 cover before scoring.\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    _write_completed_report(effective, output_dir, diagnostics, metrics, runtime)


if __name__ == "__main__":
    main()
