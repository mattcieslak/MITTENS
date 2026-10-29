//! Python bindings for `mittens-core` (`mittens._core`).

use mittens_core::grid::Grid;
use mittens_core::kernel::{first_order, DenseOdfs, ModelParams, OdfSource, ShOdfs};
use mittens_core::paths::{PathParams, PathTable as CoreTable};
use mittens_core::simulate::{simulate as core_simulate, SimParams};
use mittens_core::sphere::{Sphere, Turning};
use mittens_core::{Error, NEIGHBOR_OFFSETS, N_NEIGHBORS};
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn py_err(e: Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn rows3(arr: &PyReadonlyArray2<'_, f64>, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let a = arr.as_array();
    if a.ncols() != 3 {
        return Err(PyValueError::new_err(format!("{what} must have shape (n, 3)")));
    }
    Ok(a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

fn parse_turning(name: &str, param: f64) -> PyResult<Turning> {
    match name {
        "flat" => Ok(Turning::Flat),
        "weighted" => Ok(Turning::Weighted { power: param }),
        "vmf" => Ok(Turning::Vmf { kappa: param }),
        other => Err(PyValueError::new_err(format!("unknown turning prior {other:?}"))),
    }
}

/// Exact turning-angle-sequence table for one sphere, step and max angle.
#[pyclass(name = "PathTable", module = "mittens._core", frozen)]
struct PathTable {
    inner: CoreTable,
    sphere: Sphere,
}

#[pymethods]
impl PathTable {
    /// Enumerate sequences over the signed sphere.
    ///
    /// vertices: (n, 3) unit directions in world coordinates.
    /// step: hop length in world units. theta_max: degrees.
    /// vox_to_world: (3, 3) linear part of the voxel-to-world affine.
    /// turning: "flat", "weighted" (param = power) or "vmf" (param = kappa).
    #[new]
    #[pyo3(signature = (vertices, step, theta_max, vox_to_world, turning="flat", turning_param=1.0, tol=1e-12, max_depth=8, max_nodes=50_000_000))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        vertices: PyReadonlyArray2<'_, f64>,
        step: f64,
        theta_max: f64,
        vox_to_world: PyReadonlyArray2<'_, f64>,
        turning: &str,
        turning_param: f64,
        tol: f64,
        max_depth: usize,
        max_nodes: usize,
    ) -> PyResult<Self> {
        let verts = rows3(&vertices, "vertices")?;
        let m = rows3(&vox_to_world, "vox_to_world")?;
        if m.len() != 3 {
            return Err(PyValueError::new_err("vox_to_world must be (3, 3)"));
        }
        let mut params = PathParams::new(step, theta_max);
        params.turning = parse_turning(turning, turning_param)?;
        params.vox_to_world = [m[0], m[1], m[2]];
        params.tol = tol;
        params.max_depth = max_depth;
        params.max_nodes = max_nodes;
        let sphere = Sphere::new(&verts).map_err(py_err)?;
        let inner = py.detach(|| CoreTable::build(&sphere, params)).map_err(py_err)?;
        Ok(Self { inner, sphere })
    }

    /// Hemisphere-indexed copy with ± sequences merged (symmetric ODFs only).
    fn fold(&self, py: Python<'_>) -> PyResult<Self> {
        let inner = py.detach(|| self.inner.fold(&self.sphere)).map_err(py_err)?;
        Ok(Self { inner, sphere: self.sphere.clone() })
    }

    #[getter]
    fn folded(&self) -> bool {
        self.inner.folded
    }

    #[getter]
    fn n_dirs(&self) -> usize {
        self.inner.n_dirs()
    }

    #[getter]
    fn n_full(&self) -> usize {
        self.inner.n_full
    }

    #[getter]
    fn n_nodes(&self) -> usize {
        self.inner.n_nodes()
    }

    #[getter]
    fn n_emissions(&self) -> usize {
        self.inner.n_emissions()
    }

    #[getter]
    fn max_hops(&self) -> usize {
        self.inner.max_hops()
    }

    #[getter]
    fn step(&self) -> f64 {
        self.inner.params.step
    }

    #[getter]
    fn theta_max(&self) -> f64 {
        self.inner.params.theta_max_deg
    }

    /// Sphere vertices (normalized) the table was built on.
    fn vertices<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let flat: Vec<f64> = self.sphere.dirs().iter().flatten().copied().collect();
        flat.into_pyarray(py).reshape([self.sphere.len(), 3])
    }

    /// Hemisphere representative of every vertex, or None.
    fn hemisphere_map<'py>(&self, py: Python<'py>) -> Option<Bound<'py, numpy::PyArray1<u32>>> {
        self.sphere.hemisphere_rep().map(|r| r.into_pyarray(py))
    }

    /// Index-space displacement of each signed direction, (n_full, 3).
    fn displacements<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let flat: Vec<f64> = self.inner.disp.iter().flatten().copied().collect();
        flat.into_pyarray(py).reshape([self.inner.disp.len(), 3])
    }

    /// One row per emission: dict with `chain` (E, max_hops) int32 direction
    /// indices padded with -1, `neighbor` (E,) neighbor index, `volume` and
    /// `coef` (E,) float64.
    fn emissions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let t = &self.inner;
        let width = t.max_hops();
        let mut chain = Vec::with_capacity(t.n_emissions() * width);
        let mut neighbor = Vec::with_capacity(t.n_emissions());
        let mut path = Vec::new();
        for i in 0..t.n_nodes() {
            if t.emissions(i).is_empty() {
                continue;
            }
            path.clear();
            let mut node = i as u32;
            while node != mittens_core::paths::NO_PARENT {
                path.push(t.node_dir[node as usize] as i32);
                node = t.node_parent[node as usize];
            }
            path.reverse();
            for e in t.emissions(i) {
                chain.extend(path.iter().copied());
                chain.extend(std::iter::repeat_n(-1, width - path.len()));
                neighbor.push(t.emit_nbr[e]);
            }
        }
        let d = PyDict::new(py);
        d.set_item("chain", chain.into_pyarray(py).reshape([t.n_emissions(), width])?)?;
        d.set_item("neighbor", neighbor.into_pyarray(py))?;
        d.set_item("volume", t.emit_vol.clone().into_pyarray(py))?;
        d.set_item("coef", t.emit_coef.clone().into_pyarray(py))?;
        Ok(d)
    }

    /// Turning prior as CSR arrays (indptr, indices, data).
    #[allow(clippy::type_complexity)]
    fn turning_matrix<'py>(
        &self,
        py: Python<'py>,
    ) -> (Bound<'py, numpy::PyArray1<u32>>, Bound<'py, numpy::PyArray1<u32>>, Bound<'py, numpy::PyArray1<f64>>) {
        let w = &self.inner.turning;
        (w.indptr.clone().into_pyarray(py), w.indices.clone().into_pyarray(py), w.values.clone().into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "PathTable(n_dirs={}, folded={}, step={}, theta_max={}, nodes={}, emissions={}, max_hops={})",
            self.inner.n_dirs(),
            self.inner.folded,
            self.inner.params.step,
            self.inner.params.theta_max_deg,
            self.inner.n_nodes(),
            self.inner.n_emissions(),
            self.inner.max_hops()
        )
    }
}

fn make_grid(ijk: &PyReadonlyArray2<'_, i64>, dims: (usize, usize, usize)) -> PyResult<Grid> {
    let a = ijk.as_array();
    if a.ncols() != 3 {
        return Err(PyValueError::new_err("ijk must have shape (n, 3)"));
    }
    let coords: Vec<[i64; 3]> = a.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect();
    Grid::new(&coords, [dims.0, dims.1, dims.2]).map_err(py_err)
}

/// Wrap `odfs` as an ODF source and run `f` without holding the GIL. `odfs`
/// is float32 or float64, C-contiguous: amplitudes on the table's directions
/// (remapped by `col_map`), or SH coefficients when `basis` is given.
fn with_source<R: Send>(
    py: Python<'_>,
    odfs: &Bound<'_, PyAny>,
    basis: &Option<PyReadonlyArray2<'_, f64>>,
    col_map: &Option<PyReadonlyArray1<'_, u32>>,
    table: &CoreTable,
    f: impl Fn(&dyn OdfSource) -> Result<R, Error> + Send + Sync,
) -> PyResult<R> {
    let basis = basis
        .as_ref()
        .map(|b| b.as_slice().map_err(|_| PyValueError::new_err("basis must be C-contiguous")))
        .transpose()?;
    let cols = col_map
        .as_ref()
        .map(|c| c.as_slice().map_err(|_| PyValueError::new_err("col_map must be contiguous")))
        .transpose()?;
    macro_rules! run {
        ($arr:expr) => {{
            let n_cols = $arr.shape()[1];
            let data = $arr.as_slice().map_err(|_| PyValueError::new_err("odfs must be C-contiguous"))?;
            match basis {
                Some(b) => {
                    let src = ShOdfs::new(data, n_cols, b, table).map_err(py_err)?;
                    py.detach(|| f(&src)).map_err(py_err)
                }
                None => {
                    let src = DenseOdfs::new(data, n_cols, table, cols).map_err(py_err)?;
                    py.detach(|| f(&src)).map_err(py_err)
                }
            }
        }};
    }
    if let Ok(arr) = odfs.extract::<PyReadonlyArray2<'_, f32>>() {
        return run!(arr);
    }
    let arr = odfs
        .extract::<PyReadonlyArray2<'_, f64>>()
        .map_err(|_| PyValueError::new_err("odfs must be a 2-D float32 or float64 array"))?;
    run!(arr)
}

/// The 26 neighbor offsets (index space), shape (26, 3).
#[pyfunction]
fn neighbor_offsets(py: Python<'_>) -> PyResult<Bound<'_, PyArray2<i32>>> {
    let flat: Vec<i32> = NEIGHBOR_OFFSETS.iter().flatten().copied().collect();
    flat.into_pyarray(py).reshape([N_NEIGHBORS, 3])
}

/// (N, 26) int32 neighbor voxel ids, -1 outside the mask.
#[pyfunction]
fn neighbor_table<'py>(
    py: Python<'py>,
    ijk: PyReadonlyArray2<'py, i64>,
    dims: (usize, usize, usize),
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let grid = make_grid(&ijk, dims)?;
    let table = py.detach(|| grid.neighbor_table());
    table.into_pyarray(py).reshape([grid.n_voxels(), N_NEIGHBORS])
}

/// First-order single- and double-ODF transition probabilities.
#[pyfunction]
#[pyo3(signature = (odfs, ijk, dims, table, basis=None, col_map=None, tau=0.0, tau0=0.0, odf_power=1.0))]
#[allow(clippy::too_many_arguments)]
fn transition_probabilities<'py>(
    py: Python<'py>,
    odfs: &Bound<'py, PyAny>,
    ijk: PyReadonlyArray2<'py, i64>,
    dims: (usize, usize, usize),
    table: &PathTable,
    basis: Option<PyReadonlyArray2<'py, f64>>,
    col_map: Option<PyReadonlyArray1<'py, u32>>,
    tau: f64,
    tau0: f64,
    odf_power: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let grid = make_grid(&ijk, dims)?;
    let params = ModelParams { tau, tau0, odf_power };
    let res = with_source(py, odfs, &basis, &col_map, &table.inner, |src| {
        first_order(src, &grid, &table.inner, params)
    })?;
    let n = grid.n_voxels();
    let d = PyDict::new(py);
    d.set_item("single", res.single.into_pyarray(py).reshape([n, N_NEIGHBORS])?)?;
    d.set_item("double", res.double.into_pyarray(py).reshape([n, N_NEIGHBORS])?)?;
    d.set_item("p_terminate_single", res.p_terminate_single.into_pyarray(py))?;
    d.set_item("p_terminate_double", res.p_terminate_double.into_pyarray(py))?;
    d.set_item("total_amplitude", res.total_amplitude.into_pyarray(py))?;
    Ok(d)
}

/// Monte Carlo simulation of the hop process, for checking the analytic
/// probabilities.
#[pyfunction]
#[pyo3(signature = (odfs, ijk, dims, table, seeds, streamlines_per_seed, basis=None, col_map=None, tau=0.0, tau0=0.0, odf_power=1.0, max_hops=100_000, seed=0))]
#[allow(clippy::too_many_arguments)]
fn simulate<'py>(
    py: Python<'py>,
    odfs: &Bound<'py, PyAny>,
    ijk: PyReadonlyArray2<'py, i64>,
    dims: (usize, usize, usize),
    table: &PathTable,
    seeds: PyReadonlyArray1<'py, u32>,
    streamlines_per_seed: usize,
    basis: Option<PyReadonlyArray2<'py, f64>>,
    col_map: Option<PyReadonlyArray1<'py, u32>>,
    tau: f64,
    tau0: f64,
    odf_power: f64,
    max_hops: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyDict>> {
    let grid = make_grid(&ijk, dims)?;
    let params = ModelParams { tau, tau0, odf_power };
    let seeds = seeds.as_slice().map_err(|_| PyValueError::new_err("seeds must be contiguous"))?.to_vec();
    let sim = SimParams { streamlines_per_seed, max_hops, seed };
    let res = with_source(py, odfs, &basis, &col_map, &table.inner, |src| {
        core_simulate(src, &grid, &table.inner, params, &seeds, sim)
    })?;
    let n = grid.n_voxels();
    let d = PyDict::new(py);
    d.set_item("entries", res.entries.into_pyarray(py))?;
    d.set_item("state_entries", res.state_entries.into_pyarray(py).reshape([n, N_NEIGHBORS])?)?;
    d.set_item("n_streamlines", res.n_streamlines)?;
    d.set_item("stopped_fraction", res.stopped_fraction)?;
    Ok(d)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PathTable>()?;
    m.add_function(wrap_pyfunction!(neighbor_offsets, m)?)?;
    m.add_function(wrap_pyfunction!(neighbor_table, m)?)?;
    m.add_function(wrap_pyfunction!(transition_probabilities, m)?)?;
    m.add_function(wrap_pyfunction!(simulate, m)?)?;
    Ok(())
}
