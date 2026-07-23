"""Evidence-bounded data builder and read-only server for the D1-D6 review."""

from pathlib import Path
from typing import Any


def build_dashboard(presentation_dir: Path, input_jsonl: Path | None = None) -> dict[str, Any]:
    """Load the builder lazily so ``python -m proactive_presentation.build`` stays clean."""
    from .build import DEFAULT_INPUT, build_dashboard as _build_dashboard

    return _build_dashboard(presentation_dir, input_jsonl or DEFAULT_INPUT)


__all__ = ["build_dashboard"]
