"""The Rust kernel against an independent float64 evaluation of v1's model.

The oracle uses the legacy Fortran terms (golden table) with v1's dense
turning matrix ``G = W a / (W a)`` built in numpy, extended with the stop
mass. It shares no code with the Rust kernel.
"""
import numpy as np
import pytest

from mittens import OdfData, TransitionModel

from .conftest import grid_ijk, random_odfs

H = 321


def legacy_weights(vertices, angle_max=35.0, scheme="flat", power=1.0):
    """v1's get_transition_analysis_matrices, folded to the hemisphere."""
    u = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    ang = np.degrees(np.arccos(np.clip(u @ u.T, -1, 1)))
    ok = ang < angle_max
    if scheme == "flat":
        w = ok.astype(float)
    elif scheme == "weighted":
        w = ((180 - ang) / (180 + ang)) ** power * ok
    else:
        w = np.exp(power * (np.cos(np.radians(ang)) - 1)) * ok
    w /= w.sum(1, keepdims=True)
    return w[:H, :H] + w[H:, :H]


def oracle(golden, a, a_nbrs, wf, tau, tau0):
    """Single and raw double probabilities for one voxel (v1 neighbor order
    via the golden offsets)."""
    n_full = 2 * H

    def first_hop(x):
        return x / (2 * x.sum() + n_full * tau0)

    def turning(x):
        num = wf * x[None, :]
        return num / (num.sum(1, keepdims=True) + tau)

    x, g = first_hop(a), turning(a)
    cone = (wf > 0).astype(float)
    single, double = {}, {}
    for off, chain, coef in zip(golden["offsets"], golden["chains"], golden["coefs"]):
        chain = [c for c in chain if c >= 0]
        p = coef * x[chain[0]]
        for i, j in zip(chain[:-1], chain[1:]):
            p *= g[i, j]
        key = tuple(off)
        y = cone[chain[-1]] @ first_hop(a_nbrs[key])
        single[key] = single.get(key, 0.0) + p
        double[key] = double.get(key, 0.0) + p * y
    return single, double


@pytest.mark.parametrize("scheme,power,tau", [
    ("flat", 1.0, 0.0),
    ("flat", 1.0, 0.02),
    ("weighted", 2.0, 0.0),
    ("vmf", 6.0, 0.01),
])
def test_rust_kernel_matches_legacy_formulas(golden, odf8, scheme, power, tau):
    ijk = grid_ijk(3)
    odfs = random_odfs(27, H, seed=11)
    data = OdfData.from_amplitudes(odfs, ijk, (3, 3, 3), np.eye(4), odf8)
    step = np.sqrt(3) / 2
    turning = {"flat": {}, "weighted": {"angle_weighting_power": power},
               "vmf": {"persistence_length_mm": power * step}}[scheme]
    model = TransitionModel(step_mm=step, turning=scheme, tau=tau, tau0=tau, **turning)
    res = model.fit(data)

    center = 13
    nbrs = {tuple(c - ijk[center]): odfs[i] for i, c in enumerate(ijk) if i != center}
    wf = legacy_weights(odf8, scheme=scheme, power=power)
    single, double = oracle(golden, odfs[center], nbrs, wf, tau, tau)
    for n, off in enumerate(res.neighbor_offsets):
        assert res.single[center, n] == pytest.approx(single.get(tuple(off), 0.0), abs=1e-12)
        assert res.double[center, n] == pytest.approx(double.get(tuple(off), 0.0), abs=1e-12)
    if tau == 0:
        assert res.single[center].sum() == pytest.approx(1.0, abs=1e-12)


def test_float32_fortran_coefficients_differ_by_float_precision(golden):
    """What v1 actually computed: gfortran read every literal as REAL(4)."""
    c64 = golden["coefs"]
    c32 = c64.astype(np.float32).astype(np.float64)
    rel = np.abs(c32 - c64) / c64
    assert 1e-8 < rel.max() < 1e-7
