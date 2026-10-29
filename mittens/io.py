"""Loading ODFs through odx.

MITTENS works on either ODF amplitudes sampled on a sphere (DSI Studio) or
spherical-harmonic coefficients (MRtrix, dipy, cs-dmri). SH data is never
resampled up front: the kernels evaluate it on whatever quadrature sphere the
model asks for, so a denser sphere costs compute but not memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import odx
except ImportError:  # pragma: no cover - odx is a hard dependency
    odx = None


@dataclass
class OdfData:
    """ODFs for the voxels of a mask.

    Rows of ``odfs`` follow ``ijk``. For ``kind == "amplitudes"`` the columns
    are ``vertices[:n_cols]``; when ``hemisphere`` is true the ODF is
    antipodally symmetric and ``vertices[n_cols:]`` are the antipodes. For
    ``kind == "sh"`` the columns are SH coefficients in ``sh_basis``.
    """

    odfs: np.ndarray
    ijk: np.ndarray
    dims: tuple[int, int, int]
    affine: np.ndarray
    kind: str = "amplitudes"
    vertices: np.ndarray | None = None
    hemisphere: bool = False
    sh_order: int | None = None
    sh_basis: str | None = None
    full_basis: bool = False
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.odfs = np.ascontiguousarray(self.odfs)
        if self.odfs.dtype not in (np.float32, np.float64):
            self.odfs = self.odfs.astype(np.float32)
        self.ijk = np.ascontiguousarray(self.ijk, dtype=np.int64)
        self.dims = tuple(int(d) for d in self.dims)
        self.affine = np.asarray(self.affine, dtype=np.float64)
        if self.odfs.ndim != 2 or self.odfs.shape[0] != self.ijk.shape[0]:
            raise ValueError("odfs must be (n_voxels, n) with one row per ijk")
        if self.kind == "amplitudes":
            if self.vertices is None:
                raise ValueError("amplitude ODFs need sphere vertices")
            self.vertices = np.asarray(self.vertices, dtype=np.float64)
            n_cols, n_vert = self.odfs.shape[1], len(self.vertices)
            if n_cols * 2 == n_vert:
                self.hemisphere = True
            elif n_cols != n_vert:
                raise ValueError(f"{n_cols} amplitude columns for a {n_vert}-vertex sphere")
        elif self.kind == "sh":
            if self.sh_order is None or self.sh_basis is None:
                raise ValueError("SH ODFs need sh_order and sh_basis")
        else:
            raise ValueError(f"unknown ODF kind {self.kind!r}")

    @property
    def n_voxels(self) -> int:
        return self.odfs.shape[0]

    @property
    def voxel_sizes(self) -> np.ndarray:
        return np.linalg.norm(self.affine[:3, :3], axis=0)

    def is_symmetric(self, rtol: float = 1e-6) -> bool:
        """True when every ODF satisfies a(θ) = a(-θ)."""
        if self.kind == "sh":
            if not self.full_basis:
                return True
            odd = _odd_order_columns(self.sh_order)
            scale = np.abs(self.odfs).max() or 1.0
            return bool(np.abs(self.odfs[:, odd]).max(initial=0.0) <= rtol * scale)
        if self.hemisphere:
            return True
        anti = antipodes(self.vertices)
        if anti is None:
            return False
        scale = np.abs(self.odfs).max() or 1.0
        return bool(np.abs(self.odfs - self.odfs[:, anti]).max() <= rtol * scale)

    @classmethod
    def from_amplitudes(cls, amplitudes, ijk, dims, affine, vertices, **kw) -> "OdfData":
        return cls(amplitudes, ijk, dims, affine, kind="amplitudes", vertices=vertices, **kw)

    @classmethod
    def from_sh(cls, coefs, ijk, dims, affine, sh_order, sh_basis="tournier07",
                full_basis=False, **kw) -> "OdfData":
        return cls(coefs, ijk, dims, affine, kind="sh", sh_order=sh_order,
                   sh_basis=sh_basis, full_basis=full_basis, **kw)

    @classmethod
    def from_odx(cls, ox, prefer: str = "sh") -> "OdfData":
        """Wrap an ``odx.Odx``. With both SH and amplitudes present, ``prefer``
        picks which one to use."""
        sh_names = list(ox.sh_names())
        odf_names = list(ox.odf_names())
        common = dict(
            ijk=np.asarray(ox.compact_to_ijk, dtype=np.int64),
            dims=tuple(int(d) for d in ox.dimensions),
            affine=np.asarray(ox.affine, dtype=np.float64),
            metadata=dict(ox.metadata),
        )
        use_sh = bool(sh_names) and ox.sh_order is not None and (prefer == "sh" or not odf_names)
        if use_sh:
            name = "coefficients" if "coefficients" in sh_names else sh_names[0]
            return cls(np.asarray(ox.sh(name)), kind="sh", sh_order=int(ox.sh_order),
                       sh_basis=ox.dipy_basis_name or ox.sh_basis,
                       full_basis=bool(ox.sh_full_basis), **common)
        if not odf_names:
            raise ValueError("ODX has neither ODF amplitudes nor SH coefficients")
        name = "amplitudes" if "amplitudes" in odf_names else odf_names[0]
        vertices = ox.sphere_vertices
        if vertices is None:
            vertices = default_sphere()
        return cls(np.asarray(ox.odf(name)), kind="amplitudes",
                   vertices=np.asarray(vertices, dtype=np.float64), **common)


def load_odf_data(src, *, affine=None, prefer: str = "sh") -> OdfData:
    """Load ODFs from an ``.odx`` file or directory, a DSI Studio ``.fib.gz``
    / ``.fz``, a pyAFQ aODF directory, or an ``odx.Odx``.

    ``affine`` is only used for ``.fib.gz`` files without a ``trans`` matrix.
    MRtrix ``.mif`` files should be converted first with ``odx convert``.
    """
    if isinstance(src, OdfData):
        return src
    if odx is None:
        raise ImportError("odx is required to load ODF files")
    if isinstance(src, odx.Odx):
        return OdfData.from_odx(src, prefer=prefer)
    path = Path(src)
    name = path.name.lower()
    if name.endswith(".odx") or (path.is_dir() and (path / "header.json").exists()):
        ox = odx.load(str(path))
    elif name.endswith(".fib.gz") or name.endswith(".fib"):
        ox = odx.from_fibgz(str(path), affine=None if affine is None else np.asarray(affine, dtype=np.float64))
    elif name.endswith(".fz"):
        ox = odx.from_fz(str(path))
    elif path.is_dir():
        ox = odx.from_pyafq_aodf(str(path))
    elif name.endswith(".mif") or name.endswith(".mif.gz"):
        raise ValueError(f"convert {path} to ODX first: odx convert {path} out.odx")
    else:
        raise ValueError(f"don't know how to load {path}")
    return OdfData.from_odx(ox, prefer=prefer)


def default_sphere() -> np.ndarray:
    """DSI Studio's 642-direction odf8 sphere (RAS), hemisphere first."""
    if odx is None:
        raise ImportError("odx is required for the default sphere")
    return np.asarray(odx.spheres.dsistudio_odf8_full(), dtype=np.float64)


def fibonacci_sphere(n_half: int) -> np.ndarray:
    """Quasi-uniform sphere of ``2 * n_half`` directions, hemisphere first and
    antipodes second, for quadrature finer than odf8."""
    i = np.arange(n_half) + 0.5
    z = 1.0 - i / n_half
    r = np.sqrt(1.0 - z * z)
    phi = np.pi * (3.0 - np.sqrt(5.0)) * np.arange(n_half)
    half = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return np.vstack([half, -half])


def antipodes(vertices: np.ndarray, tol: float = 1e-6) -> np.ndarray | None:
    """Index of the antipode of every vertex, or None if some vertex has none."""
    v = vertices / np.linalg.norm(vertices, axis=1, keepdims=True)
    dots = v @ v.T
    idx = np.argmin(dots, axis=1)
    if np.any(dots[np.arange(len(v)), idx] > -1 + tol):
        return None
    return idx


def sh_basis_matrix(vertices: np.ndarray, sh_order: int, sh_basis: str, full_basis: bool) -> np.ndarray:
    """(n_vertices, n_coef) float64 matrix mapping SH coefficients to amplitudes."""
    if odx is None:
        raise ImportError("odx is required to evaluate SH bases")
    b = odx.compute_b_matrix(np.ascontiguousarray(vertices, dtype=np.float32), int(sh_order),
                             basis=sh_basis, full_basis=bool(full_basis))
    return np.ascontiguousarray(b, dtype=np.float64)


def _odd_order_columns(sh_order: int) -> np.ndarray:
    """Columns of a full SH basis (ordered by l, then m) with odd l."""
    ls = np.concatenate([np.full(2 * l + 1, l) for l in range(sh_order + 1)])
    return np.flatnonzero(ls % 2 == 1)
