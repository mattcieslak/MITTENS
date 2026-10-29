"""Type stubs for the Rust extension ``mittens._core``."""
import numpy as np
import numpy.typing as npt

__version__: str

Odfs = npt.NDArray[np.float32] | npt.NDArray[np.float64]

class PathTable:
    """Exact turning-angle-sequence table for one sphere, step and max angle."""

    def __init__(
        self,
        vertices: npt.NDArray[np.float64],
        step: float,
        theta_max: float,
        vox_to_world: npt.NDArray[np.float64],
        turning: str = "flat",
        turning_param: float = 1.0,
        tol: float = 1e-12,
        max_depth: int = 8,
        max_nodes: int = 50_000_000,
    ) -> None: ...
    def fold(self) -> PathTable: ...
    folded: bool
    n_dirs: int
    n_full: int
    n_nodes: int
    n_emissions: int
    max_hops: int
    step: float
    theta_max: float
    def vertices(self) -> npt.NDArray[np.float64]: ...
    def hemisphere_map(self) -> npt.NDArray[np.uint32] | None: ...
    def displacements(self) -> npt.NDArray[np.float64]: ...
    def emissions(self) -> dict[str, np.ndarray]: ...
    def turning_matrix(self) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32], npt.NDArray[np.float64]]: ...

def neighbor_offsets() -> npt.NDArray[np.int32]: ...
def neighbor_table(ijk: npt.NDArray[np.int64], dims: tuple[int, int, int]) -> npt.NDArray[np.int32]: ...
def transition_probabilities(
    odfs: Odfs,
    ijk: npt.NDArray[np.int64],
    dims: tuple[int, int, int],
    table: PathTable,
    basis: npt.NDArray[np.float64] | None = None,
    col_map: npt.NDArray[np.uint32] | None = None,
    tau: float = 0.0,
    tau0: float = 0.0,
    odf_power: float = 1.0,
) -> dict[str, np.ndarray]: ...
def simulate(
    odfs: Odfs,
    ijk: npt.NDArray[np.int64],
    dims: tuple[int, int, int],
    table: PathTable,
    seeds: npt.NDArray[np.uint32],
    streamlines_per_seed: int,
    basis: npt.NDArray[np.float64] | None = None,
    col_map: npt.NDArray[np.uint32] | None = None,
    tau: float = 0.0,
    tau0: float = 0.0,
    odf_power: float = 1.0,
    max_hops: int = 100_000,
    seed: int = 0,
) -> dict[str, object]: ...
