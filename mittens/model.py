"""Transition-probability models built on the Rust core."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import cached_property

import numpy as np

from . import _core
from .io import OdfData, default_sphere, fibonacci_sphere, sh_basis_matrix
from .metrics import aitchison_distance, coasy, kl_divergence, neighbor_labels

_TABLES: dict = {}
_TURNINGS = ("flat", "weighted", "vmf")


@dataclass(frozen=True)
class TransitionModel:
    """Analytic transition probabilities between each voxel and its 26
    neighbors (Cieslak et al. 2018), with absolute FOD amplitudes and
    asymmetric ODFs.

    Geometry
    --------
    step_mm : hop length in world units. Default ``sqrt(3)/2`` of the
        smallest voxel dimension (the v1 default).
    angle_max : maximum turn between consecutive hops, degrees. Default 35.
    min_radius_mm : alternative to ``angle_max``: the tightest radius of
        curvature, so the fiber model is the same at every resolution
        (``angle_max = 2 asin(step / (2 r))``).
    turning : prior over turns inside the cone: ``"flat"``, ``"weighted"``
        (``((180-Δ)/(180+Δ))^angle_weighting_power``) or ``"vmf"`` (worm-like
        chain with ``persistence_length_mm``).
    sphere : quadrature directions for SH data: None for DSI Studio odf8
        (642 directions), an int ``n`` for a Fibonacci sphere of ``2n``
        directions, or an ``(n, 3)`` array with antipodes in the second half.

    Amplitudes (stop-mass model)
    ----------------------------
    tau, tau0 : stop mass at each turn and at the first hop, per direction,
        in the units of the ODF amplitudes. ``0`` reproduces v1: every voxel
        sends probability 1 to its neighbors regardless of fiber density.
        ``tau0`` defaults to ``tau``.
    odf_power : amplitudes are raised to this power before use.
    """

    step_mm: float | None = None
    angle_max: float | None = None
    min_radius_mm: float | None = None
    turning: str = "flat"
    angle_weighting_power: float = 1.0
    persistence_length_mm: float | None = None
    tau: float = 0.0
    tau0: float | None = None
    odf_power: float = 1.0
    sphere: object = None

    def __post_init__(self):
        if self.turning not in _TURNINGS:
            raise ValueError(f"turning must be one of {_TURNINGS}")
        if self.angle_max is not None and self.min_radius_mm is not None:
            raise ValueError("give angle_max or min_radius_mm, not both")
        if self.turning == "vmf" and not self.persistence_length_mm:
            raise ValueError("turning='vmf' needs persistence_length_mm")

    # -- geometry ---------------------------------------------------------

    def step_for(self, data: OdfData) -> float:
        if self.step_mm is not None:
            return float(self.step_mm)
        return float(np.sqrt(3) / 2 * data.voxel_sizes.min())

    def angle_for(self, step: float) -> float:
        if self.min_radius_mm is not None:
            ratio = step / (2 * self.min_radius_mm)
            if ratio >= 1:
                raise ValueError("min_radius_mm must exceed half the step")
            return float(np.degrees(2 * np.arcsin(ratio)))
        return 35.0 if self.angle_max is None else float(self.angle_max)

    def _turning_param(self, step: float) -> float:
        if self.turning == "weighted":
            return float(self.angle_weighting_power)
        if self.turning == "vmf":
            return float(self.persistence_length_mm / step)
        return 1.0

    def quadrature_vertices(self, data: OdfData) -> np.ndarray:
        if data.kind == "amplitudes":
            if self.sphere is not None:
                raise ValueError("amplitude ODFs are tied to their sphere; `sphere` applies to SH data")
            return data.vertices
        if self.sphere is None:
            return default_sphere()
        if isinstance(self.sphere, (int, np.integer)):
            return fibonacci_sphere(int(self.sphere))
        return np.asarray(self.sphere, dtype=np.float64)

    def path_tables(self, data: OdfData):
        """(signed table, folded table or None) for this data's geometry."""
        vertices = np.ascontiguousarray(self.quadrature_vertices(data), dtype=np.float64)
        step = self.step_for(data)
        theta = self.angle_for(step)
        m = np.ascontiguousarray(data.affine[:3, :3], dtype=np.float64)
        param = self._turning_param(step)
        key = (vertices.tobytes(), step, theta, self.turning, param, m.tobytes())
        if key not in _TABLES:
            signed = _core.PathTable(vertices, step, theta, m, turning=self.turning, turning_param=param)
            folded = signed.fold() if signed.hemisphere_map() is not None and theta < 90 else None
            if len(_TABLES) > 8:
                _TABLES.pop(next(iter(_TABLES)))
            _TABLES[key] = (signed, folded)
        return _TABLES[key]

    def _table_and_source(self, data: OdfData, signed_only: bool = False):
        signed, folded = self.path_tables(data)
        use_folded = folded is not None and not signed_only and data.is_symmetric()
        table = folded if use_folded else signed
        src = {"odfs": data.odfs}
        if data.kind == "sh":
            verts = table.vertices()[: table.n_dirs]
            src["basis"] = sh_basis_matrix(verts, data.sh_order, data.sh_basis, data.full_basis)
        elif use_folded:
            if not data.hemisphere:
                src["col_map"] = np.arange(table.n_dirs, dtype=np.uint32)
        elif data.hemisphere:
            hmap = signed.hemisphere_map()
            if hmap is None:
                raise ValueError("hemisphere-stored ODFs need a sphere with antipodes in its second half")
            src["col_map"] = hmap
        return table, src

    def _params(self) -> dict:
        return dict(tau=float(self.tau), tau0=float(self.tau if self.tau0 is None else self.tau0),
                    odf_power=float(self.odf_power))

    # -- first order ------------------------------------------------------

    def fit(self, data: OdfData) -> "TransitionResult":
        """Single- and double-ODF probabilities for every voxel."""
        table, src = self._table_and_source(data)
        raw = _core.transition_probabilities(ijk=data.ijk, dims=data.dims, table=table, **src, **self._params())
        null_single, null_double = self.null_probabilities(data)
        return TransitionResult(
            single=raw["single"], double=raw["double"],
            p_terminate_single=raw["p_terminate_single"], p_terminate_double=raw["p_terminate_double"],
            total_amplitude=raw["total_amplitude"], null_single=null_single, null_double=null_double,
            ijk=data.ijk, dims=data.dims, affine=data.affine, model=self,
        )

    def null_probabilities(self, data: OdfData) -> tuple[np.ndarray, np.ndarray]:
        """Probabilities for an isotropic ODF surrounded by isotropic ODFs
        (shape only: no stop mass)."""
        signed, folded = self.path_tables(data)
        table = folded or signed
        ijk = np.indices((3, 3, 3)).reshape(3, -1).T.astype(np.int64)
        ones = np.ones((27, table.n_dirs))
        raw = _core.transition_probabilities(ones, ijk, (3, 3, 3), table)
        return raw["single"][13], raw["double"][13]


@dataclass
class TransitionResult:
    """First-order probabilities, rows following ``ijk`` and columns
    following ``neighbor_offsets`` (index-space offsets)."""

    single: np.ndarray
    double: np.ndarray
    p_terminate_single: np.ndarray
    p_terminate_double: np.ndarray
    total_amplitude: np.ndarray
    null_single: np.ndarray
    null_double: np.ndarray
    ijk: np.ndarray
    dims: tuple
    affine: np.ndarray
    model: TransitionModel

    neighbor_offsets = property(lambda self: _core.neighbor_offsets())

    @property
    def neighbor_labels(self) -> list[str]:
        return neighbor_labels(self.affine)

    @cached_property
    def neighbor_table(self) -> np.ndarray:
        return _core.neighbor_table(self.ijk, self.dims)

    @property
    def double_normalized(self) -> np.ndarray:
        """Double-ODF probabilities conditioned on continuing (v1's default)."""
        return _normalize_rows(self.double)

    @cached_property
    def codi_single(self) -> np.ndarray:
        return aitchison_distance(self.single, self.null_single)

    @cached_property
    def codi_double(self) -> np.ndarray:
        return aitchison_distance(self.double_normalized, _normalize_rows(self.null_double[None])[0])

    @cached_property
    def coasy(self) -> np.ndarray:
        return coasy(self.double_normalized)

    @cached_property
    def kl_single(self) -> np.ndarray:
        return kl_divergence(self.null_single, self.single)

    def maps(self) -> dict[str, np.ndarray]:
        return {
            "single": self.single, "double": self.double,
            "p_terminate_single": self.p_terminate_single, "p_terminate_double": self.p_terminate_double,
            "total_amplitude": self.total_amplitude, "codi_single": self.codi_single,
            "codi_double": self.codi_double, "coasy": self.coasy,
        }

    def sidecar(self) -> dict:
        return {
            "neighbor_offsets_ijk": self.neighbor_offsets.tolist(),
            "neighbor_labels": self.neighbor_labels,
            "null_single": self.null_single.tolist(),
            "null_double": self.null_double.tolist(),
            "model": {k: (v if not isinstance(v, np.ndarray) else "array") for k, v in asdict(self.model).items()},
        }

    def to_nifti(self, prefix: str) -> list[str]:
        """Write each map as ``{prefix}_{name}.nii.gz`` (4D for the 26
        probabilities) plus ``{prefix}_transitions.json``."""
        return _write_niftis(prefix, self.maps(), self.ijk, self.dims, self.affine, self.sidecar())

    def to_odx(self, path: str, dtype=np.float32) -> None:
        """Write the maps as ``mittens_*`` per-voxel arrays in a new ODX."""
        _write_odx(path, self.maps(), self.ijk, self.dims, self.affine, self.sidecar(), dtype)


def _normalize_rows(x: np.ndarray, fill=np.nan) -> np.ndarray:
    s = x.sum(axis=-1, keepdims=True)
    out = np.full(x.shape, fill, dtype=np.float64)
    np.divide(x, s, out=out, where=s > 0)
    return out


def _write_niftis(prefix, maps, ijk, dims, affine, sidecar) -> list[str]:
    import nibabel as nib

    written = []
    for name, values in maps.items():
        values = np.asarray(values)
        vol = np.zeros(tuple(dims) + values.shape[1:], dtype=np.float32)
        vol[tuple(ijk.T)] = values
        path = f"{prefix}_{name}.nii.gz"
        nib.Nifti1Image(vol, affine).to_filename(path)
        written.append(path)
    path = f"{prefix}_transitions.json"
    with open(path, "w") as f:
        json.dump(sidecar, f, indent=2)
    written.append(path)
    return written


def _write_odx(path, maps, ijk, dims, affine, metadata, dtype) -> None:
    import odx

    flat = np.ravel_multi_index(tuple(ijk.T), dims)
    order = np.argsort(flat)
    mask = np.zeros(int(np.prod(dims)), dtype=np.uint8)
    mask[flat] = 1
    builder = odx.OdxBuilder(np.asarray(affine, dtype=np.float64), tuple(int(d) for d in dims), mask)
    builder.skip_all_peaks()
    for name, values in maps.items():
        builder.set_dpv(f"mittens_{name}", np.ascontiguousarray(np.asarray(values)[order], dtype=dtype))
    builder.set_metadata({"mittens": json.loads(json.dumps(metadata))})
    builder.finalize().save(str(path))
