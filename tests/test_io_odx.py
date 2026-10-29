"""Loading through odx and writing results back out."""
import json

import nibabel as nib
import numpy as np
import odx
import pytest

from mittens import OdfData, TransitionModel, load_odf_data

from .conftest import random_odfs

AFFINE = np.array([[-2.0, 0, 0, 10], [0, -2.0, 0, 20], [0, 0, 2.0, -5], [0, 0, 0, 1]])
DIMS = (4, 3, 5)


def _mask():
    mask = np.zeros(DIMS, dtype=np.uint8)
    mask[1:4, :, 1:5] = 1
    return mask


@pytest.fixture
def amplitude_odx(tmp_path):
    mask = _mask()
    verts, faces = odx.spheres.dsistudio_odf8()
    full = np.ascontiguousarray(odx.spheres.dsistudio_odf8_full(), dtype=np.float32)
    n = int(mask.sum())
    amps = random_odfs(n, 321, seed=8).astype(np.float32)
    builder = odx.OdxBuilder(AFFINE, DIMS, mask.ravel())
    builder.skip_all_peaks()
    builder.set_sphere(full, np.ascontiguousarray(faces, dtype=np.uint32))
    builder.set_odf("amplitudes", amps)
    path = tmp_path / "amps.odx"
    builder.finalize().save(str(path))
    return path, amps


@pytest.fixture
def sh_odx():
    mask = _mask()
    rng = np.random.default_rng(2)
    sh = rng.normal(size=DIMS + (45,)).astype(np.float32) * 0.2
    sh[..., 0] = 1.0
    return odx.from_sh_coefficients(sh, mask, AFFINE, basis="tournier07", sh_order=8)


def test_amplitude_odx_roundtrip(amplitude_odx, tmp_path):
    path, amps = amplitude_odx
    data = load_odf_data(path)
    assert data.kind == "amplitudes" and data.hemisphere
    assert data.n_voxels == _mask().sum()
    np.testing.assert_array_equal(data.odfs, amps)
    # Rows are the mask in C order.
    np.testing.assert_array_equal(data.ijk, np.argwhere(_mask()))

    res = TransitionModel(tau=0.01).fit(data)
    out = tmp_path / "result.odx"
    res.to_odx(out, dtype=np.float64)
    back = odx.load(str(out))
    np.testing.assert_allclose(back.dpv("mittens_single"), res.single)
    np.testing.assert_allclose(back.dpv("mittens_p_terminate_single").ravel(), res.p_terminate_single)
    assert back.metadata["mittens"]["neighbor_labels"] == res.neighbor_labels


def test_sh_odx(sh_odx):
    data = load_odf_data(sh_odx)
    assert data.kind == "sh" and data.sh_order == 8 and data.is_symmetric()
    res = TransitionModel().fit(data)
    np.testing.assert_allclose(res.single.sum(1), 1.0, atol=1e-10)


def test_to_nifti(amplitude_odx, tmp_path):
    path, _ = amplitude_odx
    data = load_odf_data(path)
    res = TransitionModel().fit(data)
    files = res.to_nifti(str(tmp_path / "sub"))
    img = nib.load(str(tmp_path / "sub_single.nii.gz"))
    assert img.shape == DIMS + (26,)
    np.testing.assert_allclose(img.affine, AFFINE)
    i, j, k = data.ijk[7]
    np.testing.assert_allclose(img.get_fdata()[i, j, k], res.single[7], atol=1e-6)
    sidecar = json.load(open(tmp_path / "sub_transitions.json"))
    assert len(sidecar["neighbor_labels"]) == 26
    assert any(f.endswith("codi_single.nii.gz") for f in files)


def test_lps_affine_labels():
    data = OdfData.from_amplitudes(np.ones((1, 321)), [[0, 0, 0]], (1, 1, 1), AFFINE,
                                   odx.spheres.dsistudio_odf8_full())
    res = TransitionModel().fit(data)
    labels = dict(zip(map(tuple, res.neighbor_offsets.tolist()), res.neighbor_labels))
    assert labels[(1, 0, 0)] == "l" and labels[(0, 1, 0)] == "p" and labels[(0, 0, 1)] == "s"
