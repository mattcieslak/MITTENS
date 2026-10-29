"""Model-level invariants: stop mass, symmetry handling, SH input, geometry."""
import numpy as np
import pytest

from mittens import OdfData, TransitionModel, _core
from mittens.io import antipodes, default_sphere, fibonacci_sphere, sh_basis_matrix

from .conftest import grid_ijk, random_odfs

AFFINE = np.diag([-2.0, -2.0, 2.0, 1.0])


@pytest.fixture(scope="module")
def sphere():
    return default_sphere()


def hemi_data(n=4, seed=0, scale=1.0):
    v = default_sphere()
    odfs = scale * random_odfs(n ** 3, len(v) // 2, seed=seed)
    return OdfData.from_amplitudes(odfs, grid_ijk(n), (n,) * 3, AFFINE, v)


def test_single_sums_to_one_and_is_scale_invariant():
    model = TransitionModel()
    a, b = model.fit(hemi_data()), model.fit(hemi_data(scale=37.0))
    np.testing.assert_allclose(a.single.sum(1), 1.0, atol=1e-12)
    np.testing.assert_allclose(a.single, b.single, atol=1e-12)
    np.testing.assert_allclose(a.double, b.double, atol=1e-12)


def test_stop_mass_conserves_probability_and_grows_with_tau():
    data = hemi_data()
    prev = np.zeros(data.n_voxels)
    for tau in (0.0, 0.01, 0.05, 0.2):
        res = TransitionModel(tau=tau).fit(data)
        np.testing.assert_allclose(res.single.sum(1) + res.p_terminate_single, 1.0, atol=1e-12)
        assert np.all(res.p_terminate_single >= prev - 1e-12)
        prev = res.p_terminate_single
    assert prev.min() > 0


def test_stop_mass_tracks_fiber_density():
    """With tau > 0 a weak voxel stops more often than a dense one with the
    same shape: amplitudes are no longer thrown away."""
    data = hemi_data(n=3)
    data.odfs[13] *= 0.05
    res = TransitionModel(tau=0.01).fit(data)
    shape_only = TransitionModel().fit(data)
    assert res.p_terminate_single[13] > 5 * np.median(res.p_terminate_single)
    np.testing.assert_allclose(shape_only.single.sum(1), 1.0, atol=1e-12)


def test_full_sphere_symmetric_matches_hemisphere(sphere):
    hemi = hemi_data()
    full = OdfData.from_amplitudes(np.hstack([hemi.odfs, hemi.odfs]), hemi.ijk, hemi.dims, AFFINE, sphere)
    assert full.is_symmetric() and not full.hemisphere
    model = TransitionModel()
    a, b = model.fit(hemi), model.fit(full)
    np.testing.assert_allclose(a.single, b.single, atol=1e-12)
    np.testing.assert_allclose(a.double, b.double, atol=1e-12)


def test_one_sided_aodf_never_moves_backwards(sphere):
    """A lobe within 20 degrees of +i (index space) cannot reach any neighbor
    with di = -1, and asymmetric data is never folded."""
    n = 3
    world_i = AFFINE[:3, 0] / np.linalg.norm(AFFINE[:3, 0])
    lobe = (sphere @ world_i > np.cos(np.radians(20))).astype(float)
    odfs = np.tile(lobe, (n ** 3, 1)) * (1 + random_odfs(n ** 3, len(sphere), seed=2))
    data = OdfData.from_amplitudes(odfs, grid_ijk(n), (n,) * 3, AFFINE, sphere)
    assert not data.is_symmetric()
    model = TransitionModel()
    table, _ = model._table_and_source(data)
    assert not table.folded
    res = model.fit(data)
    back = res.neighbor_offsets[:, 0] == -1
    assert res.single[:, back].max() == 0.0
    np.testing.assert_allclose(res.single.sum(1), 1.0, atol=1e-12)


def test_bending_aodf_is_direction_dependent(sphere):
    n = 3
    rng = np.random.default_rng(5)
    odfs = rng.random((n ** 3, len(sphere))) ** 8
    data = OdfData.from_amplitudes(odfs, grid_ijk(n), (n,) * 3, AFFINE, sphere)
    res = TransitionModel().fit(data)
    # P(u -> v) and P(v -> u) for face neighbors along i.
    u, v = 13, 13 + 9
    k = int(np.flatnonzero((res.neighbor_offsets == [1, 0, 0]).all(1))[0])
    assert abs(res.single[u, k] - res.single[v, 25 - k]) > 1e-3


def test_sh_input_matches_sampled_amplitudes(sphere):
    n, lmax = 3, 8
    ncoef = (lmax + 1) * (lmax + 2) // 2
    rng = np.random.default_rng(3)
    coefs = rng.normal(size=(n ** 3, ncoef))
    coefs[:, 0] = 3.0
    sh = OdfData.from_sh(coefs, grid_ijk(n), (n,) * 3, AFFINE, lmax, "tournier07")
    b = sh_basis_matrix(sphere[:321], lmax, "tournier07", False)
    amps = OdfData.from_amplitudes(coefs @ b.T, grid_ijk(n), (n,) * 3, AFFINE, sphere)
    model = TransitionModel()
    np.testing.assert_allclose(model.fit(sh).single, model.fit(amps).single, atol=1e-10)


def test_sh_on_a_denser_quadrature_sphere():
    n, lmax = 3, 8
    ncoef = (lmax + 1) * (lmax + 2) // 2
    rng = np.random.default_rng(4)
    coefs = rng.normal(size=(n ** 3, ncoef))
    coefs[:, 0] = 3.0
    sh = OdfData.from_sh(coefs, grid_ijk(n), (n,) * 3, AFFINE, lmax, "tournier07")
    coarse = TransitionModel().fit(sh)
    fine = TransitionModel(sphere=1281).fit(sh)
    np.testing.assert_allclose(fine.single.sum(1), 1.0, atol=1e-12)
    # Same model, finer quadrature: close but not identical.
    assert 0 < np.abs(fine.single - coarse.single).max() < 0.05


def test_full_basis_sh_detects_asymmetry():
    n, lmax = 2, 4
    ncoef = (lmax + 1) ** 2
    coefs = np.zeros((n ** 3, ncoef))
    coefs[:, 0] = 1.0
    sym = OdfData.from_sh(coefs, grid_ijk(n), (n,) * 3, AFFINE, lmax, "descoteaux07", full_basis=True)
    assert sym.is_symmetric()
    coefs[:, 2] = 0.3  # an l = 1 coefficient
    asym = OdfData.from_sh(coefs, grid_ijk(n), (n,) * 3, AFFINE, lmax, "descoteaux07", full_basis=True)
    assert not asym.is_symmetric()


def test_min_radius_fixes_the_physical_fiber_model():
    small = hemi_data()
    big = OdfData.from_amplitudes(small.odfs, small.ijk, small.dims, np.diag([-1.0, -1.0, 1.0, 1.0]), small.vertices)
    model = TransitionModel(min_radius_mm=3.0)
    for data in (small, big):
        step = model.step_for(data)
        theta = np.radians(model.angle_for(step))
        assert step / (2 * np.sin(theta / 2)) == pytest.approx(3.0)
    assert model.angle_for(model.step_for(small)) > model.angle_for(model.step_for(big))
    # v1 parameters imply a radius of ~1.44 voxels.
    v1 = TransitionModel()
    s = v1.step_for(big)
    assert s / (2 * np.sin(np.radians(v1.angle_for(s)) / 2)) == pytest.approx(1.44, abs=0.01)


def test_null_probabilities_and_metrics():
    data = hemi_data()
    res = TransitionModel().fit(data)
    assert res.null_single.sum() == pytest.approx(1.0)
    iso = OdfData.from_amplitudes(np.ones_like(data.odfs), data.ijk, data.dims, AFFINE, data.vertices)
    iso_res = TransitionModel().fit(iso)
    center = int(np.flatnonzero((data.ijk == 1).all(1))[0])
    np.testing.assert_allclose(iso_res.single[center], res.null_single, atol=1e-12)
    assert iso_res.codi_single[center] == pytest.approx(0.0, abs=1e-9)
    assert np.nanmin(res.codi_single) > 0


def test_aitchison_matches_v1_pairwise_formula():
    from mittens.metrics import aitchison_distance

    rng = np.random.default_rng(1)
    x = rng.dirichlet(np.ones(26), size=5)
    ref = rng.dirichlet(np.ones(26))
    i, j = np.triu_indices(26, 1)
    v1 = np.sqrt(np.sum((np.log(x[:, i] / x[:, j]) - np.log(ref[i] / ref[j])) ** 2, axis=1) / 26)
    np.testing.assert_allclose(aitchison_distance(x, ref), v1, rtol=1e-12)


def test_antipodes_of_default_sphere(sphere):
    anti = antipodes(sphere)
    np.testing.assert_array_equal(anti[:321], np.arange(321, 642))
    assert antipodes(fibonacci_sphere(10)) is not None


@pytest.mark.slow
def test_first_order_probabilities_are_exact_for_the_hop_process(sphere):
    """Monte Carlo streamlines of the hop process from the center of a 3x3x3
    block whose other voxels are empty, so a streamline stops right after its
    first voxel change: the fraction entering each neighbor is the single-ODF
    probability and the rest is the termination probability."""
    n = 3
    a = np.zeros((n ** 3, 642))
    center = 13
    a[center] = random_odfs(1, 642, seed=1)[0]
    a[center] += a[center, np.r_[321:642, 0:321]]
    data = OdfData.from_amplitudes(a, grid_ijk(n), (n,) * 3, np.eye(4), sphere)
    model = TransitionModel(step_mm=1.0, tau=0.01)
    res = model.fit(data)
    table, src = model._table_and_source(data, signed_only=True)
    sim = _core.simulate(ijk=data.ijk, dims=data.dims, table=table, seeds=np.array([center], dtype=np.uint32),
                         streamlines_per_seed=1_000_000, seed=7, **src, **model._params())
    first = np.array([sim["state_entries"][res.neighbor_table[center, k], k] for k in range(26)])
    observed = np.r_[first, 1 - first.sum()]
    expected = np.r_[res.single[center], res.p_terminate_single[center]]
    sigma = np.sqrt(expected * (1 - expected) / sim["n_streamlines"]) + 1e-9
    assert (np.abs(observed - expected) < 5 * sigma).all()
