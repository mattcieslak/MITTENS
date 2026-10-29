from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "data" / "legacy_odf8_ss087_am35.npz"


@pytest.fixture(scope="session")
def golden():
    g = np.load(GOLDEN)
    return {k: g[k] for k in g.files}


@pytest.fixture(scope="session")
def odf8(golden):
    """odf8 vertices exactly as v1 used them (LPS, float32 precision)."""
    return golden["vertices"]


def grid_ijk(n):
    return np.indices((n, n, n)).reshape(3, -1).T.astype(np.int64)


def random_odfs(n_vox, n_dirs, seed=0, sharpness=6):
    rng = np.random.default_rng(seed)
    return rng.random((n_vox, n_dirs)) ** sharpness
