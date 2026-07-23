"""Pre-registered low-parameter dual-view feature construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def build_dual_view_matrices(
    uniform: np.ndarray,
    multiscale: np.ndarray,
    feature_names: Sequence[str],
    *,
    gate_feature: str,
) -> dict[str, tuple[np.ndarray, tuple[str, ...]]]:
    if uniform.shape != multiscale.shape or uniform.ndim != 2:
        raise ValueError("D5 dual-view matrices must be aligned and two-dimensional")
    names = tuple(feature_names)
    if uniform.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("D5 dual-view feature names are not canonical")
    hidden_names = tuple(name for name in names if name.startswith("hidden_"))
    if hidden_names != tuple(f"hidden_{index:04d}" for index in range(1024)):
        raise ValueError("D5 dual-view requires the canonical 1,024 hidden features")
    difference_names = ("tag_margin", *hidden_names)
    difference_indices = [names.index(name) for name in difference_names]
    difference = multiscale[:, difference_indices] - uniform[:, difference_indices]
    if not np.isfinite(difference).all():
        raise ValueError("D5 dual-view difference contains non-finite values")
    difference_feature_names = tuple(f"multiscale_delta={name}" for name in difference_names)
    shared = np.concatenate([uniform, difference], axis=1)

    if gate_feature not in names:
        raise ValueError(f"D5 dual-view gate feature is missing: {gate_feature}")
    gate = uniform[:, names.index(gate_feature)]
    if not np.isin(gate, [0.0, 1.0]).all():
        raise ValueError("D5 dual-view gate must be binary")
    inactive = difference * (1.0 - gate[:, None])
    active = difference * gate[:, None]
    gated = np.concatenate([uniform, inactive, active], axis=1)
    gated_names = (
        *names,
        *(f"no_assistant_add={name}" for name in difference_feature_names),
        *(f"assistant_add={name}" for name in difference_feature_names),
    )
    return {
        "shared_delta": (shared, (*names, *difference_feature_names)),
        "dialog_gated_delta": (gated, gated_names),
    }
