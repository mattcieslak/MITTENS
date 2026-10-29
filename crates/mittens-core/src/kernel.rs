//! First-order transition probabilities with the stop-mass amplitude model.
//!
//! With absolute amplitudes `a`, `A = Σ a` over the full sphere and `N`
//! signed directions:
//!
//! * first hop: `p(θ0) = a(θ0) / (A + N τ0)`
//! * turn:      `p(θj | θi) = W_ij a(θj) / (Σ_k W_ik a(θk) + τ)`
//!
//! The missing mass in each choice is the probability that the fiber stops.
//! `τ = τ0 = 0` and `odf_power = 1` reproduce MITTENS v1 exactly.

use rayon::prelude::*;

use crate::grid::Grid;
use crate::paths::{PathTable, NO_PARENT};
use crate::{Error, Result, N_NEIGHBORS};

/// Amplitude element types accepted by the kernels.
pub trait Amplitude: Copy + Send + Sync {
    fn to_f64(self) -> f64;
}

impl Amplitude for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Amplitude for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
}

/// Source of ODF amplitudes for the directions of a path table.
pub trait OdfSource: Sync {
    fn n_voxels(&self) -> usize;
    fn n_dirs(&self) -> usize;
    /// Amplitude of every table direction for voxel `v` (not yet clamped).
    fn fill(&self, v: usize, out: &mut [f64]);
    /// Amplitude of table direction `d` for voxel `v`.
    fn value(&self, v: usize, d: usize) -> f64;
}

/// Amplitudes sampled on the sphere, row-major `(n_voxels, n_cols)`. A
/// hemisphere-stored ODF used with an unfolded table maps both `d` and `-d`
/// to the same column through `col_map`.
pub struct DenseOdfs<'a, T> {
    data: &'a [T],
    n_cols: usize,
    cols: Vec<u32>,
}

impl<'a, T: Amplitude> DenseOdfs<'a, T> {
    pub fn new(data: &'a [T], n_cols: usize, table: &PathTable, col_map: Option<&[u32]>) -> Result<Self> {
        if n_cols == 0 || !data.len().is_multiple_of(n_cols) {
            return Err(Error::InvalidInput("amplitude array is not (n_voxels, n_dirs)".into()));
        }
        let n = table.n_dirs();
        let cols = match col_map {
            Some(m) => {
                if m.len() != n || m.iter().any(|&c| c as usize >= n_cols) {
                    return Err(Error::InvalidInput(format!(
                        "column map must have {n} entries, each < {n_cols}"
                    )));
                }
                m.to_vec()
            }
            None if n_cols == n => (0..n as u32).collect(),
            None => {
                return Err(Error::InvalidInput(format!(
                    "amplitudes have {n_cols} columns but the path table has {n} directions; pass a column map"
                )))
            }
        };
        Ok(Self { data, n_cols, cols })
    }
}

impl<T: Amplitude> OdfSource for DenseOdfs<'_, T> {
    fn n_voxels(&self) -> usize {
        self.data.len() / self.n_cols
    }

    fn n_dirs(&self) -> usize {
        self.cols.len()
    }

    #[inline]
    fn fill(&self, v: usize, out: &mut [f64]) {
        let row = &self.data[v * self.n_cols..(v + 1) * self.n_cols];
        for (o, &c) in out.iter_mut().zip(&self.cols) {
            *o = row[c as usize].to_f64();
        }
    }

    #[inline]
    fn value(&self, v: usize, d: usize) -> f64 {
        self.data[v * self.n_cols + self.cols[d] as usize].to_f64()
    }
}

/// Spherical-harmonic coefficients, row-major `(n_voxels, n_coef)`,
/// evaluated on the table's directions through `basis` (row-major
/// `(n_dirs, n_coef)`). Memory stays `O(n_voxels · n_coef)` however dense
/// the quadrature sphere is.
pub struct ShOdfs<'a, T> {
    coefs: &'a [T],
    n_coef: usize,
    basis: &'a [f64],
}

impl<'a, T: Amplitude> ShOdfs<'a, T> {
    pub fn new(coefs: &'a [T], n_coef: usize, basis: &'a [f64], table: &PathTable) -> Result<Self> {
        if n_coef == 0 || !coefs.len().is_multiple_of(n_coef) {
            return Err(Error::InvalidInput("SH array is not (n_voxels, n_coef)".into()));
        }
        if basis.len() != table.n_dirs() * n_coef {
            return Err(Error::InvalidInput(format!(
                "SH basis must be ({}, {n_coef}) for this path table",
                table.n_dirs()
            )));
        }
        Ok(Self { coefs, n_coef, basis })
    }
}

impl<T: Amplitude> OdfSource for ShOdfs<'_, T> {
    fn n_voxels(&self) -> usize {
        self.coefs.len() / self.n_coef
    }

    fn n_dirs(&self) -> usize {
        self.basis.len() / self.n_coef
    }

    #[inline]
    fn fill(&self, v: usize, out: &mut [f64]) {
        let c = &self.coefs[v * self.n_coef..(v + 1) * self.n_coef];
        for (o, b) in out.iter_mut().zip(self.basis.chunks_exact(self.n_coef)) {
            *o = b.iter().zip(c).map(|(x, y)| x * y.to_f64()).sum();
        }
    }

    #[inline]
    fn value(&self, v: usize, d: usize) -> f64 {
        let c = &self.coefs[v * self.n_coef..(v + 1) * self.n_coef];
        let b = &self.basis[d * self.n_coef..(d + 1) * self.n_coef];
        b.iter().zip(c).map(|(x, y)| x * y.to_f64()).sum()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ModelParams {
    /// Stop mass at every turn, in per-direction amplitude units.
    pub tau: f64,
    /// Stop mass at the first hop, in per-direction amplitude units.
    pub tau0: f64,
    /// Amplitudes are raised to this power (PTT's data-support exponent).
    pub odf_power: f64,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self { tau: 0.0, tau0: 0.0, odf_power: 1.0 }
    }
}

impl ModelParams {
    pub fn validate(&self) -> Result<()> {
        if !(self.tau >= 0.0 && self.tau0 >= 0.0 && self.odf_power > 0.0) {
            return Err(Error::InvalidInput("tau and tau0 must be >= 0 and odf_power > 0".into()));
        }
        Ok(())
    }
}

/// Per-voxel quantities shared by all kernels.
pub(crate) struct Prep {
    /// Amplitude of each table direction (clamped at 0, raised to the power).
    pub a: Vec<f64>,
    /// `1 / (cone mass + τ)` for each previous direction.
    pub r: Vec<f64>,
    /// Probability that a trajectory started here leaves in the cone of each
    /// direction (double-ODF weight, paper eq. 5).
    pub y: Vec<f64>,
    /// `1 / (A + N τ0)`.
    pub z0: f64,
    /// Total amplitude over the full sphere.
    pub total: f64,
}

impl Prep {
    pub fn new(n: usize) -> Self {
        Self { a: vec![0.0; n], r: vec![0.0; n], y: vec![0.0; n], z0: 0.0, total: 0.0 }
    }

    pub fn compute<S: OdfSource + ?Sized>(&mut self, src: &S, v: usize, table: &PathTable, params: &ModelParams) {
        let n = self.a.len();
        src.fill(v, &mut self.a);
        for x in self.a.iter_mut() {
            *x = clamp_amplitude(*x, params);
        }
        let scale = table.n_full as f64 / n as f64;
        self.total = self.a.iter().sum::<f64>() * scale;
        let denom0 = self.total + table.n_full as f64 * params.tau0;
        self.z0 = if denom0 > 0.0 { 1.0 / denom0 } else { 0.0 };
        for i in 0..n {
            let (idx, val) = table.turning.row(i);
            let mut cone = 0.0;
            let mut mass = 0.0;
            for (&j, &w) in idx.iter().zip(val) {
                let aj = self.a[j as usize];
                cone += w * aj;
                mass += aj;
            }
            let denom = cone + params.tau;
            self.r[i] = if denom > 0.0 { 1.0 / denom } else { 0.0 };
            self.y[i] = mass * self.z0;
        }
    }
}

#[inline]
pub(crate) fn clamp_amplitude(x: f64, params: &ModelParams) -> f64 {
    let x = if x > 0.0 && x.is_finite() { x } else { 0.0 };
    if params.odf_power == 1.0 {
        x
    } else {
        x.powf(params.odf_power)
    }
}

/// Weight every sequence in `nodes` (a union of whole subtrees) and call
/// `visit(node, weight)` for the non-zero ones. A node's weight is its
/// parent's times `a[dir] · r[parent dir]`; roots get `root_weight(dir)`.
#[inline]
pub(crate) fn walk(
    table: &PathTable,
    a: &[f64],
    r: &[f64],
    nodes: std::ops::Range<usize>,
    root_weight: impl Fn(usize) -> f64,
    w: &mut [f64],
    mut visit: impl FnMut(usize, f64),
) {
    let mut i = nodes.start;
    while i < nodes.end {
        let dir = table.node_dir[i] as usize;
        let wi = match table.node_parent[i] {
            NO_PARENT => root_weight(dir),
            p => {
                let p = p as usize;
                w[p] * a[dir] * r[table.node_dir[p] as usize]
            }
        };
        if wi == 0.0 {
            i = table.node_end[i] as usize;
            continue;
        }
        w[i] = wi;
        visit(i, wi);
        i += 1;
    }
}

pub(crate) fn check_inputs<S: OdfSource + ?Sized>(
    src: &S,
    grid: &Grid,
    table: &PathTable,
    params: &ModelParams,
) -> Result<()> {
    params.validate()?;
    if src.n_voxels() != grid.n_voxels() {
        return Err(Error::InvalidInput(format!("{} ODF rows for {} voxels", src.n_voxels(), grid.n_voxels())));
    }
    if src.n_dirs() != table.n_dirs() {
        return Err(Error::InvalidInput(format!(
            "ODF source gives {} directions but the path table has {}",
            src.n_dirs(),
            table.n_dirs()
        )));
    }
    Ok(())
}

/// Row-major `(n_voxels, 26)` outputs, neighbors ordered as
/// [`crate::NEIGHBOR_OFFSETS`].
pub struct FirstOrder {
    pub single: Vec<f64>,
    /// Raw double-ODF probabilities (sub-stochastic; not renormalized).
    pub double: Vec<f64>,
    pub p_terminate_single: Vec<f64>,
    pub p_terminate_double: Vec<f64>,
    pub total_amplitude: Vec<f64>,
}

pub fn first_order<S: OdfSource + ?Sized>(
    src: &S,
    grid: &Grid,
    table: &PathTable,
    params: ModelParams,
) -> Result<FirstOrder> {
    check_inputs(src, grid, table, &params)?;
    let n_dirs = table.n_dirs();
    let n_nodes = table.n_nodes();
    let blocks = grid.blocks(8, None);

    type Row = (u32, [f64; N_NEIGHBORS], [f64; N_NEIGHBORS], f64);
    let rows: Vec<Vec<Row>> = blocks
        .par_iter()
        .map(|blk| {
            let preps: Vec<Prep> = blk
                .halo
                .iter()
                .map(|&v| {
                    let mut p = Prep::new(n_dirs);
                    p.compute(src, v as usize, table, &params);
                    p
                })
                .collect();
            let mut w = vec![0.0; n_nodes];
            blk.voxels
                .iter()
                .enumerate()
                .map(|(bi, &v)| {
                    let p = &preps[blk.self_slot[bi] as usize];
                    let nbr = &blk.nbr_slot[bi];
                    let mut single = [0.0; N_NEIGHBORS];
                    let mut double = [0.0; N_NEIGHBORS];
                    walk(table, &p.a, &p.r, 0..n_nodes, |d| p.z0 * p.a[d], &mut w, |node, wi| {
                        let dir = table.node_dir[node] as usize;
                        for e in table.emissions(node) {
                            let k = table.emit_nbr[e] as usize;
                            let c = table.emit_coef[e] * wi;
                            single[k] += c;
                            if nbr[k] >= 0 {
                                double[k] += c * preps[nbr[k] as usize].y[dir];
                            }
                        }
                    });
                    (v, single, double, p.total)
                })
                .collect()
        })
        .collect();

    let n = grid.n_voxels();
    let mut out = FirstOrder {
        single: vec![0.0; n * N_NEIGHBORS],
        double: vec![0.0; n * N_NEIGHBORS],
        p_terminate_single: vec![0.0; n],
        p_terminate_double: vec![0.0; n],
        total_amplitude: vec![0.0; n],
    };
    for (v, single, double, total) in rows.into_iter().flatten() {
        let v = v as usize;
        out.single[v * N_NEIGHBORS..(v + 1) * N_NEIGHBORS].copy_from_slice(&single);
        out.double[v * N_NEIGHBORS..(v + 1) * N_NEIGHBORS].copy_from_slice(&double);
        out.p_terminate_single[v] = (1.0 - single.iter().sum::<f64>()).max(0.0);
        out.p_terminate_double[v] = (1.0 - double.iter().sum::<f64>()).max(0.0);
        out.total_amplitude[v] = total;
    }
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::paths::PathParams;
    use crate::sphere::{fibonacci_sphere, Sphere};

    pub(crate) fn cube_grid(n: i64) -> Grid {
        let mut ijk = Vec::new();
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    ijk.push([i, j, k]);
                }
            }
        }
        Grid::new(&ijk, [n as usize; 3]).unwrap()
    }

    pub(crate) fn random_odfs(n_vox: usize, n_dirs: usize, seed: u64) -> Vec<f64> {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        (0..n_vox * n_dirs).map(|_| rng.random::<f64>().powi(6)).collect()
    }

    #[test]
    fn single_sums_to_one_and_stop_mass_is_conserved() {
        let sphere = Sphere::new(&fibonacci_sphere(60)).unwrap();
        let table = PathTable::build(&sphere, PathParams::new(3f64.sqrt() / 2.0, 35.0)).unwrap();
        let folded = table.fold(&sphere).unwrap();
        let grid = cube_grid(4);
        let odfs = random_odfs(grid.n_voxels(), 60, 7);
        let amps = DenseOdfs::new(&odfs, 60, &folded, None).unwrap();
        let res = first_order(&amps, &grid, &folded, ModelParams::default()).unwrap();
        for v in 0..grid.n_voxels() {
            let s: f64 = res.single[v * 26..(v + 1) * 26].iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "{s}");
        }
        let res = first_order(&amps, &grid, &folded, ModelParams { tau: 0.05, tau0: 0.05, odf_power: 1.0 }).unwrap();
        for v in 0..grid.n_voxels() {
            let s: f64 = res.single[v * 26..(v + 1) * 26].iter().sum();
            assert!(s < 1.0 && (s + res.p_terminate_single[v] - 1.0).abs() < 1e-12);
        }

        // The unfolded table on hemisphere data (via a column map) agrees.
        let rep = sphere.hemisphere_rep().unwrap();
        let mapped = DenseOdfs::new(&odfs, 60, &table, Some(&rep)).unwrap();
        let full = first_order(&mapped, &grid, &table, ModelParams::default()).unwrap();
        let fold = first_order(&amps, &grid, &folded, ModelParams::default()).unwrap();
        for (x, y) in full.single.iter().zip(&fold.single) {
            assert!((x - y).abs() < 1e-12);
        }
        for (x, y) in full.double.iter().zip(&fold.double) {
            assert!((x - y).abs() < 1e-12);
        }
    }
}
