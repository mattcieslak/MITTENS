"""Compositional summaries of transition probabilities (CoDI, CoAsy)."""
from __future__ import annotations

import numpy as np

from ._core import neighbor_offsets


def _clr(x: np.ndarray, zero_policy: str, delta: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if zero_policy == "replace":
        # Multiplicative replacement: zeros become delta, the rest shrink so
        # each composition keeps its total.
        zeros = x <= 0
        total = x.sum(axis=-1, keepdims=True)
        n_zero = zeros.sum(axis=-1, keepdims=True)
        x = np.where(zeros, delta * total, x * (1 - n_zero * delta))
    elif zero_policy != "nan":
        raise ValueError("zero_policy must be 'nan' or 'replace'")
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.log(np.where(x > 0, x, np.nan))
    return logs - logs.mean(axis=-1, keepdims=True)


def aitchison_distance(x, ref, zero_policy: str = "nan", delta: float = 1e-6) -> np.ndarray:
    """Aitchison distance between each row of ``x`` and ``ref``.

    Equal to MITTENS v1's pairwise form ``sqrt(1/D Σ_{i<j} (log(x_i/x_j) -
    log(r_i/r_j))^2)`` but O(D). Rows with a zero part are NaN unless
    ``zero_policy="replace"``.
    """
    return np.sqrt(np.sum((_clr(x, zero_policy, delta) - _clr(ref, zero_policy, delta)) ** 2, axis=-1))


def coasy(probs, zero_policy: str = "nan", delta: float = 1e-6) -> np.ndarray:
    """Compositional asymmetry: Aitchison distance between the probabilities
    toward the 13 neighbors on one side and toward their opposites."""
    probs = np.asarray(probs)
    half = np.arange(13)
    return aitchison_distance(probs[:, half], probs[:, 25 - half], zero_policy, delta)


def kl_divergence(ref, x) -> np.ndarray:
    """``Σ ref log(ref / x)`` for each row of ``x``; inf where ``x`` is 0 and ``ref`` is not."""
    ref = np.asarray(ref, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(ref > 0, ref * np.log(ref / x), 0.0)
    return terms.sum(axis=-1)


def odf_asymmetry_index(amplitudes, antipode: np.ndarray) -> np.ndarray:
    """``sqrt(1 - r^2)`` with ``r = Σ a(θ) a(-θ) / Σ a(θ)^2``: 0 for an
    antipodally symmetric ODF, 1 when no lobe has an antipodal partner."""
    a = np.clip(np.asarray(amplitudes, dtype=np.float64), 0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (a * a[:, antipode]).sum(-1) / (a * a).sum(-1)
    return np.sqrt(np.clip(1 - r * r, 0, 1))


def neighbor_labels(affine: np.ndarray) -> list[str]:
    """Anatomical labels (r/l, a/p, s/i) of the 26 neighbor offsets for a
    voxel-to-RAS affine, e.g. ``"ras"`` for right-anterior-superior."""
    labels = []
    for off in neighbor_offsets():
        w = np.asarray(affine)[:3, :3] @ off
        keep = np.abs(w) > 0.25 * np.abs(w).max()
        label = ""
        for axis, (pos, neg) in enumerate(("rl", "ap", "si")):
            if keep[axis]:
                label += pos if w[axis] > 0 else neg
        labels.append(label)
    return labels
