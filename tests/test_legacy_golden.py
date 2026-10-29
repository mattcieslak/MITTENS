"""The runtime path table must reproduce the v1 generated Fortran exactly."""
from pathlib import Path

import numpy as np

from mittens import PathTable, neighbor_offsets

GOLDEN = Path(__file__).parent / "data" / "legacy_odf8_ss087_am35.npz"


def _legacy_terms():
    g = np.load(GOLDEN)
    terms = {}
    for off, chain, coef in zip(g["offsets"], g["chains"], g["coefs"]):
        terms[(tuple(int(x) for x in off), tuple(int(c) for c in chain if c >= 0))] = coef
    return g["vertices"], terms


def _rust_terms(table):
    em = table.emissions()
    offsets = neighbor_offsets()
    return {
        (tuple(int(x) for x in offsets[n]), tuple(int(c) for c in chain if c >= 0)): coef
        for chain, n, coef in zip(em["chain"], em["neighbor"], em["volume"])
    }


def test_folded_table_matches_fortran_coefficients():
    vertices, legacy = _legacy_terms()
    # v1 applied the steps to the raw vertex coordinates in voxel units.
    table = PathTable(vertices, np.sqrt(3) / 2, 35.0, np.eye(3)).fold()
    rust = _rust_terms(table)

    assert set(rust) <= set(legacy)
    missing = {k: v for k, v in legacy.items() if k not in rust}
    # Only floating-point slivers (one box side ~1e-17) are dropped.
    assert max(missing.values(), default=0.0) <= 1e-12
    diffs = np.array([abs(rust[k] - legacy[k]) for k in rust])
    assert diffs.max() < 1e-12
    assert table.max_hops == 3


def test_lps_affine_reproduces_v1_neighbor_labels():
    """With DSI Studio's LPS index axes under a RAS affine, the r/l and a/p
    swap v1 did by hand falls out of the affine."""
    vertices, legacy = _legacy_terms()
    flip = np.diag([-1.0, -1.0, 1.0])
    table = PathTable(vertices @ flip.T, np.sqrt(3) / 2, 35.0, flip).fold()
    rust = _rust_terms(table)
    diffs = np.array([abs(rust[k] - legacy[k]) for k in rust])
    assert diffs.max() < 1e-12
