"""Launch up to five frozen D6 folds under the selected GPU resource policy."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

from proactive_r0.artifacts import write_json

from .contract import SHARED_MINIMUM_FREE_GIB, load_experiment
from .run_fold import _load_gate, _mapping


DEFAULT_CONFIG = Path("configs/d6_internvl35_1b_query_memory_lora_oof_v1.json")


def _eligible_gpus(
    minimum_free_gib: float, *, require_exclusive: bool = True
) -> list[dict[str, object]]:
    import pynvml

    pynvml.nvmlInit()
    try:
        result: list[dict[str, object]] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gib = float(memory.free) / 2**30
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            if (
                (not require_exclusive or not processes)
                and free_gib >= minimum_free_gib
                and "A800" in str(name)
            ):
                result.append(
                    {
                        "index": index,
                        "uuid": str(pynvml.nvmlDeviceGetUUID(handle)),
                        "name": str(name),
                        "free_gib": free_gib,
                        "foreign_compute_processes": len(processes),
                    }
                )
        return sorted(result, key=lambda value: float(value["free_gib"]), reverse=True)
    finally:
        pynvml.nvmlShutdown()


def _fold_complete(experiment_dir: Path, fold: int, config_sha256: str) -> bool:
    path = experiment_dir / "folds" / f"fold_{fold}" / "fold_summary.json"
    if not path.exists():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return bool(
        isinstance(value, dict)
        and value.get("kind") == "d6_formal_oof_fold"
        and value.get("status") == "complete"
        and value.get("fold") == fold
        and value.get("config_sha256") == config_sha256
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument("--zero-init-summary", required=True)
    parser.add_argument("--trainability-summary", required=True)
    parser.add_argument("--maximum-gpus", type=int, default=5)
    parser.add_argument("--allow-shared-gpus", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.maximum_gpus <= 5:
        raise ValueError("D6 launcher allows one to five GPUs")

    inputs = load_experiment(Path(args.config))
    _load_gate(
        Path(args.zero_init_summary), inputs, "d6_zero_init_causality_smoke"
    )
    _load_gate(
        Path(args.trainability_summary),
        inputs,
        "d6_rotation0_trainability_smoke",
    )
    resources = _mapping(inputs.config["resources"], "resources")
    require_exclusive = not args.allow_shared_gpus
    minimum_free_gib = (
        SHARED_MINIMUM_FREE_GIB
        if args.allow_shared_gpus
        else float(resources["minimum_free_memory_gib"])
    )
    experiment_dir = Path(args.experiment_dir).resolve()
    launcher_dir = experiment_dir / "launcher"
    launcher_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        fold
        for fold in range(5)
        if not _fold_complete(experiment_dir, fold, inputs.config_sha256)
    ]
    initial_pending = list(pending)
    running: dict[int, dict[str, object]] = {}
    completed: list[int] = []
    failed: list[dict[str, object]] = []
    launches: list[dict[str, object]] = []
    started = time.monotonic()

    while pending or running:
        occupied = {int(value["gpu_index"]) for value in running.values()}
        eligible = [
            gpu
            for gpu in _eligible_gpus(
                minimum_free_gib, require_exclusive=require_exclusive
            )
            if int(gpu["index"]) not in occupied
        ]
        while pending and eligible and len(running) < args.maximum_gpus and not failed:
            fold = pending.pop(0)
            gpu = eligible.pop(0)
            fold_dir = experiment_dir / "folds" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            log_path = launcher_dir / f"fold_{fold}.log"
            log_handle = log_path.open("a", encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "proactive_d6.run_fold",
                "--config",
                str(Path(args.config).resolve()),
                "--output-dir",
                str(fold_dir),
                "--device",
                "cuda:0",
                "--fold",
                str(fold),
                "--formal",
                "--zero-init-summary",
                str(Path(args.zero_init_summary).resolve()),
                "--trainability-summary",
                str(Path(args.trainability_summary).resolve()),
            ]
            if args.allow_shared_gpus:
                command.append("--allow-shared-gpu")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
            environment["PYTHONNOUSERSITE"] = "1"
            source_root = str(Path(__file__).resolve().parents[1])
            environment["PYTHONPATH"] = source_root
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            launch = {
                "fold": fold,
                "gpu_index": gpu["index"],
                "gpu_uuid": gpu["uuid"],
                "gpu_name": gpu["name"],
                "prelaunch_free_gib": gpu["free_gib"],
                "prelaunch_foreign_compute_processes": gpu[
                    "foreign_compute_processes"
                ],
                "require_exclusive_gpu": require_exclusive,
                "pid": process.pid,
                "command": command,
                "log_path": str(log_path),
                "started_monotonic_seconds": time.monotonic() - started,
            }
            launches.append(launch)
            running[fold] = {
                **launch,
                "process": process,
                "log_handle": log_handle,
            }

        for fold, value in list(running.items()):
            process = value["process"]
            if not isinstance(process, subprocess.Popen):
                raise TypeError("D6 launcher process state changed")
            return_code = process.poll()
            if return_code is None:
                continue
            handle = value["log_handle"]
            handle.close()  # type: ignore[union-attr]
            del running[fold]
            if return_code == 0 and _fold_complete(
                experiment_dir, fold, inputs.config_sha256
            ):
                completed.append(fold)
            else:
                failed.append(
                    {
                        "fold": fold,
                        "return_code": return_code,
                        "log_path": value["log_path"],
                    }
                )
        write_json(
            launcher_dir / "status.json",
            {
                "schema_version": 1,
                "kind": "d6_five_fold_launcher",
                "config_sha256": inputs.config_sha256,
                "initial_pending": initial_pending,
                "pending": pending,
                "running": [
                    {
                        key: item
                        for key, item in value.items()
                        if key not in ("process", "log_handle")
                    }
                    for value in running.values()
                ],
                "completed": sorted(completed),
                "failed": failed,
                "launches": launches,
                "wall_seconds": time.monotonic() - started,
            },
        )
        if failed and not running:
            break
        if pending or running:
            time.sleep(10)

    if failed:
        raise RuntimeError(f"D6 formal fold launcher failed without new dispatch: {failed}")
    if any(
        not _fold_complete(experiment_dir, fold, inputs.config_sha256)
        for fold in range(5)
    ):
        raise RuntimeError("D6 launcher ended without all five complete folds")
    print(json.dumps({"status": "complete", "folds": list(range(5))}))


if __name__ == "__main__":
    main()
