"""Create and audit the frozen answer-free D5 robustness input transforms."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

from proactive_d1.core import strip_answers
from proactive_d4_1.core import object_sha256
from proactive_r0.artifacts import (
    code_snapshot,
    environment_snapshot,
    sha256_file,
    write_json,
)
from proactive_r0.core import load_jsonl, validate_source_rows, write_jsonl

from .robust import drop_assistant_history


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_command(path: Path, argv: Sequence[str]) -> None:
    command = shlex.join(
        [sys.executable, "-m", "proactive_d5.prepare_robust_inputs", *argv]
    )
    path.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd {shlex.quote(str(PROJECT_ROOT))}\n"
        "export PYTHONNOUSERSITE=1\n"
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src'))}\n"
        f"exec {command}\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def run(config_path: Path, output_dir: Path, raw_argv: Sequence[str]) -> dict[str, object]:
    started = time.monotonic()
    config = _load_object(config_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Robust input output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = config.get("data")
    protocol = config.get("protocol")
    if not isinstance(data, dict) or not isinstance(protocol, dict):
        raise ValueError("Robust input config sections are malformed")
    input_path = _resolve(data["input"])
    video_folder = _resolve(data["video_folder"])
    protocol_path = _resolve(protocol["path"])
    if sha256_file(input_path) != data["input_sha256"]:
        raise ValueError("Robust input source fingerprint mismatch")
    if sha256_file(protocol_path) != protocol["sha256"]:
        raise ValueError("Robust protocol fingerprint mismatch")

    source_rows = load_jsonl(input_path)
    answer_free = strip_answers(source_rows)
    if any("answers" in row for row in answer_free):
        raise RuntimeError("Robust source stripping failed")
    transformed, audit = drop_assistant_history(answer_free)
    validation = validate_source_rows(transformed, video_folder)
    if validation["sessions"] != 700 or validation["chunks"] != 9935:
        raise ValueError("Robust assistant-drop input coverage changed")
    derived_path = output_dir / "assistant_drop_input.jsonl"
    write_jsonl(derived_path, transformed)
    summary = {
        "schema_version": 1,
        "status": "complete answer-free robust input preparation",
        "completed_at": datetime.now().astimezone().isoformat(),
        "wall_time_seconds": time.monotonic() - started,
        "source_sha256": sha256_file(input_path),
        "assistant_drop_sha256": sha256_file(derived_path),
        "sessions": validation["sessions"],
        "chunks": validation["chunks"],
        "answers_present": False,
        "audit": audit,
        "config_object_sha256": object_sha256(config),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "config.json", config)
    _write_command(output_dir / "command.sh", raw_argv)
    write_json(output_dir / "environment.txt", environment_snapshot())
    write_json(
        output_dir / "code_state.txt",
        code_snapshot(
            PROJECT_ROOT,
            [
                PROJECT_ROOT / "src" / "proactive_d5" / "robust.py",
                PROJECT_ROOT / "src" / "proactive_d5" / "prepare_robust_inputs.py",
                config_path,
                protocol_path,
                PROJECT_ROOT / "Agent.md",
                PROJECT_ROOT / "CURRENT_ROUTE.md",
            ],
        ),
    )
    write_json(
        output_dir / "data_manifest.json",
        {
            "data": data,
            "protocol": protocol,
            "transformation": config["transformation"],
            "source_sha256": summary["source_sha256"],
            "assistant_drop_sha256": summary["assistant_drop_sha256"],
            "labels_read_by_transformation": False,
            "external_data_used": False,
        },
    )
    (output_dir / "run.log").write_text(
        "Robust answer-free input preparation completed.\n", encoding="utf-8"
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                f"# {config['experiment_id']}",
                "",
                "状态：已完成 answer-free assistant-drop 输入构造。",
                "",
                f"- Sessions/chunks: `{validation['sessions']}/{validation['chunks']}`",
                f"- Assistant turns removed: `{audit['assistant_turns_removed']}`",
                f"- Derived SHA256: `{summary['assistant_drop_sha256']}`",
                "- Labels read by transformation: `false`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/d5_internvl35_1b_robust_inputs_v1.json"
    )
    parser.add_argument("--experiment-dir")
    args = parser.parse_args(raw_argv)
    config_path = _resolve(args.config)
    config = _load_object(config_path)
    output_dir = _resolve(
        args.experiment_dir or f"output/experiments/{config['experiment_id']}"
    )
    print(json.dumps(run(config_path, output_dir, raw_argv), sort_keys=True))


if __name__ == "__main__":
    main()
