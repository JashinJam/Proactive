"""Experiment manifest and reproducibility helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from math import prod
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def count_safetensors(path: Path) -> dict[str, object]:
    from safetensors import safe_open

    handle = safe_open(str(path), framework="pt", device="cpu")
    by_component: dict[str, int] = {}
    dtypes: set[str] = set()
    total = 0
    keys = list(handle.keys())
    for key in keys:
        tensor_slice = handle.get_slice(key)
        count = prod(tensor_slice.get_shape())
        total += count
        component = key.split(".", 1)[0]
        by_component[component] = by_component.get(component, 0) + count
        dtypes.add(str(tensor_slice.get_dtype()))
    return {
        "stored_unique_parameters": total,
        "tensor_count": len(keys),
        "parameters_by_component": by_component,
        "dtypes": sorted(dtypes),
    }


def verify_model_snapshot(model_path: Path, model_config: dict[str, object]) -> dict[str, object]:
    weights_path = model_path / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Missing model weights: {weights_path}")
    audit = count_safetensors(weights_path)
    actual_sha = sha256_file(weights_path)
    expected_sha = str(model_config["weights_sha256"])
    expected_parameters = int(model_config["total_parameters"])
    if actual_sha != expected_sha:
        raise ValueError(f"Model SHA256 mismatch: {actual_sha} != {expected_sha}")
    if audit["stored_unique_parameters"] != expected_parameters:
        raise ValueError(
            "Model parameter mismatch: "
            f"{audit['stored_unique_parameters']} != {expected_parameters}"
        )
    if expected_parameters > 2_000_000_000:
        raise ValueError("Model exceeds the C1 Small 2B total-parameter limit")
    return {
        **audit,
        "weights_path": str(weights_path),
        "weights_bytes": weights_path.stat().st_size,
        "weights_sha256": actual_sha,
        "small_limit_parameters": 2_000_000_000,
        "small_limit_fraction": round(expected_parameters / 2_000_000_000, 6),
    }


def package_versions() -> dict[str, str]:
    packages = ["torch", "transformers", "safetensors", "cv2", "PIL", "numpy"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover - environment-specific
            versions[package] = f"unavailable: {type(exc).__name__}"
    return versions


def environment_snapshot() -> dict[str, object]:
    import torch

    cuda_available = torch.cuda.is_available()
    return {
        "timestamp_timezone": "Asia/Shanghai",
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": package_versions(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python_no_user_site": os.environ.get("PYTHONNOUSERSITE"),
    }


def _git_state(repo: Path) -> dict[str, object]:
    if not (repo / ".git").exists():
        return {"path": str(repo), "is_git_repository": False}

    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        return completed.stdout.rstrip()

    return {
        "path": str(repo),
        "is_git_repository": True,
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status_short": run("status", "--short"),
    }


def code_snapshot(project_root: Path, tracked_paths: Iterable[Path]) -> dict[str, object]:
    hashes = {
        str(path.relative_to(project_root)): sha256_file(path)
        for path in sorted(tracked_paths)
        if path.is_file()
    }
    return {
        "project_repository": _git_state(project_root),
        "file_sha256": hashes,
        "nested_repositories": [
            _git_state(project_root / "STRIDE"),
            _git_state(project_root / "wearable-ai-leaderboard"),
        ],
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
