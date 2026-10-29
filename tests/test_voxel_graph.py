"""Voxel-graph tractography on igraph, checked against scipy and brute force."""
import nibabel as nib
import numpy as np
import pytest
from scipy import sparse
from scipy.sparse import csgraph

from mittens import OdfData, TransitionModel, VoxelGraph
from mittens.io import default_sphere

from .conftest import grid_ijk, random_odfs

FLIP = np.r_[321:642, 0:321]
AFFINE = np.array([[-2.0, 0, 0, 10], [0, 2.0, 0, -20], [0, 0, 2.0, 5], [0, 0, 0, 1]])
N = 4


@pytest.fixture(scope="module")
def result():
    a = random_odfs(N ** 3, 642, seed=3, sharpness=6)
    data = OdfData.from_amplitudes(a + a[:, FLIP], grid_ijk(N), (N,) * 3, AFFINE, default_sphere())
    return TransitionModel().fit(data)


@pytest.fixture(scope="module")
def vg(result):
    return VoxelGraph.from_transitions(result)


def region(*coords):
    mask = np.zeros((N,) * 3, dtype=np.uint8)
    for c in coords:
        mask[c] = 1
    return mask


def csr(vg):
    return sparse.csr_array((vg.length, (vg.source, vg.target)), shape=(vg.n_voxels,) * 2)


def test_edges_follow_the_transition_probabilities(result, vg):
    nt = result.neighbor_table
    p = result.double_normalized
    u, k = np.nonzero(nt >= 0)
    expected = {(a, nt[a, b]): p[a, b] for a, b in zip(u, k) if p[a, b] > 0}
    got = dict(zip(zip(vg.source, vg.target), vg.p))
    assert got.keys() == expected.keys()
    np.testing.assert_allclose([got[e] for e in expected], list(expected.values()))
    # An interior voxel keeps all of its probability on edges.
    center = int(np.flatnonzero((result.ijk == [1, 1, 1]).all(1))[0])
    assert vg.p[vg.source == center].sum() == pytest.approx(1.0)

    scaled = VoxelGraph.from_transitions(result, weights="minus_iso_scaled")
    sums = np.bincount(scaled.source, scaled.p, minlength=scaled.n_voxels)
    np.testing.assert_allclose(sums[sums > 0], 1.0)


@pytest.mark.parametrize("toward", [False, True])
def test_shortest_paths_match_scipy_and_their_own_paths(vg, toward):
    nodes = vg.region_nodes(region((0, 0, 0), (0, 1, 0)))
    sp = vg.shortest_paths_to(nodes_mask(vg, nodes)) if toward else vg.shortest_paths_from(nodes_mask(vg, nodes))
    g = csr(vg).T if toward else csr(vg)
    dist = csgraph.dijkstra(g, directed=True, indices=nodes, min_only=True)
    np.testing.assert_allclose(sp.probability, np.exp(-dist), rtol=1e-10)

    edge = dict(zip(zip(vg.source, vg.target), vg.p))
    offsets = {tuple(o): i for i, o in enumerate(np.array([o for o in np.ndindex(3, 3, 3) if o != (1, 1, 1)]) - 1)}
    for v in np.flatnonzero(sp.reached):
        path = sp.path(v)
        assert len(path) == sp.hops[v] + 1
        assert (path[-1] if toward else path[0]) in nodes
        steps = list(zip(path[:-1], path[1:]))
        assert np.prod([edge[s] for s in steps]) == pytest.approx(sp.probability[v], rel=1e-10)
        null = np.prod([vg.null[offsets[tuple(vg.ijk[b] - vg.ijk[a])]] for a, b in steps])
        assert null == pytest.approx(sp.null_probability[v], rel=1e-10)


def nodes_mask(vg, nodes):
    m = np.zeros(vg.n_voxels, dtype=bool)
    m[nodes] = True
    return m


def test_backprop_is_the_max_over_paths_through_each_voxel(vg):
    sp = vg.shortest_paths_from(region((0, 0, 0)))
    values = sp.mean_probability
    brute = np.zeros(vg.n_voxels)
    for v in np.flatnonzero(sp.reached):
        for w in sp.path(v):
            brute[w] = max(brute[w], values[v])
    np.testing.assert_allclose(sp.backprop(values), brute)
    assert sp.mean_probability[sp.hops == 0].tolist() == [1.0]


def test_regions_from_nifti_and_atlas(vg, tmp_path):
    path = tmp_path / "roi.nii.gz"
    nib.Nifti1Image(region((1, 2, 3)), AFFINE).to_filename(path)
    node = vg.region_nodes(str(path))
    assert vg.ijk[node].tolist() == [[1, 2, 3]]
    nib.Nifti1Image(region((1, 2, 3)), np.eye(4)).to_filename(tmp_path / "bad.nii.gz")
    with pytest.raises(ValueError, match="not on the ODF grid"):
        vg.region_nodes(str(tmp_path / "bad.nii.gz"))

    atlas = np.zeros((N,) * 3, dtype=np.int16)
    atlas[0] = 1
    atlas[3] = 2
    vg.add_atlas(atlas)
    conn = vg.atlas_connectivity()
    assert conn["labels"].tolist() == [1, 2]
    np.testing.assert_allclose(np.diag(conn["probability"]), 1.0)
    from_1 = vg.shortest_paths_from(1)
    assert conn["probability"][0, 1] == pytest.approx(from_1.probability[vg.atlas == 2].max())
    assert conn["hops"][0, 1] >= 3


def test_mask_keeps_the_edges_between_kept_voxels(vg):
    keep = np.zeros((N,) * 3, dtype=bool)
    keep[:, :2] = True
    sub = vg.mask(keep)
    assert sub.n_voxels == keep.sum()
    full = {(tuple(vg.ijk[a]), tuple(vg.ijk[b])): p for a, b, p in zip(vg.source, vg.target, vg.p)
            if keep[tuple(vg.ijk[a])] and keep[tuple(vg.ijk[b])]}
    part = {(tuple(sub.ijk[a]), tuple(sub.ijk[b])): p for a, b, p in zip(sub.source, sub.target, sub.p)}
    assert full == part


def test_sampled_betweenness_with_every_source_is_exact(vg):
    exact = vg.betweenness()
    np.testing.assert_allclose(vg.betweenness(n_samples=vg.n_voxels), exact, atol=1e-9)
    assert exact.max() > 0


def test_max_flow_equals_min_cut(vg):
    src, dst = region((0, 0, 0), (0, 0, 1)), region((3, 3, 3), (3, 3, 2))
    f = vg.max_flow(src, dst)
    s_nodes, t_nodes = vg.region_nodes(src), vg.region_nodes(dst)
    side = f["sink_side"]
    cut = vg.p[~side[vg.source] & side[vg.target]].sum() + side[s_nodes].sum() + (~side[t_nodes]).sum()
    assert f["value"] == pytest.approx(cut)
    inner = np.setdiff1d(np.arange(vg.n_voxels), np.r_[s_nodes, t_nodes])
    np.testing.assert_allclose(f["inflow"][inner], f["outflow"][inner], atol=1e-12)
    net = (f["outflow"] - f["inflow"])[s_nodes].sum()
    assert net == pytest.approx(f["value"])
    assert 0 < f["value"] <= len(s_nodes)


def test_components_and_save_load(vg, tmp_path):
    assert (vg.components() == 0).all()
    vg.save(tmp_path / "g.npz")
    back = VoxelGraph.load(tmp_path / "g.npz")
    for name in ("source", "target", "p", "ijk", "affine", "null"):
        np.testing.assert_array_equal(getattr(back, name), getattr(vg, name))
    assert back.dims == vg.dims and back.weights == vg.weights


@pytest.mark.parametrize("ext", [".tck", ".trk"])
def test_streamlines_and_nifti_outputs(vg, tmp_path, ext):
    sp = vg.shortest_paths_to(region((0, 0, 0)))
    paths = sp.paths(np.flatnonzero(sp.hops >= 2))
    vg.save_streamlines(paths, tmp_path / f"paths{ext}")
    loaded = nib.streamlines.load(str(tmp_path / f"paths{ext}")).streamlines
    assert len(loaded) == len(paths)
    np.testing.assert_allclose(loaded[0], nib.affines.apply_affine(AFFINE, vg.ijk[paths[0]]), atol=1e-4)

    vg.to_nifti(sp.mean_probability, tmp_path / "mean.nii.gz")
    img = nib.load(tmp_path / "mean.nii.gz")
    np.testing.assert_allclose(img.affine, AFFINE)
    i, j, k = vg.ijk[5]
    assert img.get_fdata()[i, j, k] == pytest.approx(sp.mean_probability[5], rel=1e-6)
