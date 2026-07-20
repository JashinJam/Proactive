"""Write automatic, non-semantic diagnostics for the three U1 state variants."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

from proactive_r0.artifacts import sha256_file, write_json
from proactive_r0.core import load_jsonl
from proactive_u0.core import normalize_utterance
from proactive_u1.analyze import _summary
from proactive_u1.state_review import STATE_VARIANTS


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _variant_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_position: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_session: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row["domain"])].append(row)
        by_position[str(row["position_bin"])].append(row)
        by_session[int(row["input_index"])].append(row)
    repeated_rows = 0
    repeated_sessions = 0
    for session_rows in by_session.values():
        seen: Counter[str] = Counter()
        repeated = False
        for row in sorted(session_rows, key=lambda value: int(value["chunk_index"])):
            content = str(row.get("content", ""))
            if not content:
                continue
            seen[normalize_utterance(content)] += 1
            if seen[normalize_utterance(content)] > 1:
                repeated_rows += 1
                repeated = True
        repeated_sessions += int(repeated)
    return {
        "overall": {
            **_summary(rows),
            "used_fallback": sum(bool(row.get("used_fallback")) for row in rows),
            "exact_repeat_after_first": repeated_rows,
            "sessions_with_exact_repeat": repeated_sessions,
        },
        "by_domain": {
            key: _summary(value) for key, value in sorted(by_domain.items())
        },
        "by_position": {
            key: _summary(value) for key, value in sorted(by_position.items())
        },
    }


def analyze_state_content(
    sample_rows: Sequence[dict[str, object]],
    content_rows: Sequence[dict[str, object]],
) -> dict[str, object]:
    sample_ids = {str(row["sample_id"]) for row in sample_rows}
    by_variant: dict[str, dict[str, dict[str, object]]] = {
        variant: {} for variant in STATE_VARIANTS
    }
    for row in content_rows:
        variant = str(row.get("variant"))
        if variant not in by_variant:
            continue
        sample_id = str(row["sample_id"])
        if sample_id in by_variant[variant]:
            raise ValueError(f"Duplicate state diagnostic row: {variant}/{sample_id}")
        by_variant[variant][sample_id] = row
    for variant, rows in by_variant.items():
        if set(rows) != sample_ids:
            raise ValueError(f"State diagnostic coverage mismatch for {variant}")
    variants = {
        variant: _variant_summary(list(rows.values()))
        for variant, rows in by_variant.items()
    }
    pairwise: dict[str, object] = {}
    for reference, target in (
        ("forced_no_state", "forced_oracle_step"),
        ("forced_no_state", "forced_oracle_full"),
        ("forced_oracle_step", "forced_oracle_full"),
    ):
        exact = 0
        both_empty = 0
        reference_only = 0
        target_only = 0
        both_nonempty_changed = 0
        for sample_id in sample_ids:
            left = str(by_variant[reference][sample_id].get("content", ""))
            right = str(by_variant[target][sample_id].get("content", ""))
            exact += int(left == right)
            both_empty += int(not left and not right)
            reference_only += int(bool(left) and not right)
            target_only += int(not left and bool(right))
            both_nonempty_changed += int(bool(left) and bool(right) and left != right)
        pairwise[f"{target}_vs_{reference}"] = {
            "samples": len(sample_ids),
            "exact_content_equal": exact,
            "content_changed": len(sample_ids) - exact,
            "both_empty": both_empty,
            "reference_only_nonempty": reference_only,
            "target_only_nonempty": target_only,
            "both_nonempty_but_changed": both_nonempty_changed,
        }
    return {
        "schema_version": 1,
        "status": "automatic non-semantic diagnostics complete; human state review pending",
        "samples": len(sample_ids),
        "variants": variants,
        "pairwise": pairwise,
        "warning": (
            "Nonempty, lexical, and exact-change statistics do not establish "
            "correctness, grounding, safety, or state benefit."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--content-records", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sample_path = _resolve(args.samples)
    content_paths = [_resolve(value) for value in args.content_records]
    result = analyze_state_content(
        load_jsonl(sample_path),
        [row for path in content_paths for row in load_jsonl(path)],
    )
    result["sources"] = {
        "samples": {"path": str(sample_path), "sha256": sha256_file(sample_path)},
        "content_records": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in content_paths
        ],
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
