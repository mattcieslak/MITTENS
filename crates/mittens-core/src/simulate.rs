//! Monte Carlo simulation of the hop process the analytic kernel describes,
//! used to check the closed-form probabilities.
//!
//! Streamlines start at a uniform position in a seed voxel, take hops of the
//! path table's step, and at every hop turn under the ODF of the voxel they
//! are in, keeping the incoming direction across voxel boundaries. The
//! fraction of streamlines whose first voxel change enters neighbor `n` is
//! the single-ODF probability `P(seed -> n)`.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::grid::Grid;
use crate::kernel::{check_inputs, clamp_amplitude, ModelParams, OdfSource};
use crate::{neighbor_index, Error, Result, N_NEIGHBORS};
use crate::paths::PathTable;

#[derive(Clone, Copy, Debug)]
pub struct SimParams {
    pub streamlines_per_seed: usize,
    pub max_hops: usize,
    pub seed: u64,
}

pub struct SimResult {
    /// Entries into each voxel per streamline.
    pub entries: Vec<f64>,
    /// Entries into each `(voxel, incoming neighbor direction)` state per
    /// streamline, row-major `(n_voxels, 26)`.
    pub state_entries: Vec<f64>,
    pub n_streamlines: usize,
    /// Streamlines that stopped (stop mass or no compatible direction)
    /// rather than leaving the mask.
    pub stopped_fraction: f64,
}

struct Acc {
    entries: Vec<f64>,
    states: Vec<f64>,
    stopped: u64,
}

impl Acc {
    fn new(n: usize) -> Self {
        Self { entries: vec![0.0; n], states: vec![0.0; n * N_NEIGHBORS], stopped: 0 }
    }

    fn merge(mut self, other: Acc) -> Acc {
        self.entries.iter_mut().zip(&other.entries).for_each(|(a, b)| *a += b);
        self.states.iter_mut().zip(&other.states).for_each(|(a, b)| *a += b);
        self.stopped += other.stopped;
        self
    }
}

/// Simulate from every voxel in `seeds`. The first hop is drawn from the
/// seed voxel's ODF (including the first-hop stop mass).
pub fn simulate<S: OdfSource + ?Sized>(
    src: &S,
    grid: &Grid,
    table: &PathTable,
    params: ModelParams,
    seeds: &[u32],
    sim: SimParams,
) -> Result<SimResult> {
    check_inputs(src, grid, table, &params)?;
    if table.folded || table.disp.len() != table.n_dirs() {
        return Err(Error::InvalidInput("simulation needs the unfolded (signed) path table".into()));
    }
    if seeds.iter().any(|&s| s as usize >= grid.n_voxels()) {
        return Err(Error::InvalidInput("seed voxel out of range".into()));
    }
    let n_dirs = table.n_dirs();
    let n_full = table.n_full as f64;
    let amp = |v: usize, d: usize| clamp_amplitude(src.value(v, d), &params);

    // Work items are batches of streamlines from one seed, so a single seed
    // still uses every thread. Each batch has its own deterministic stream.
    const BATCH: usize = 4096;
    let items: Vec<(usize, usize, usize)> = seeds
        .iter()
        .enumerate()
        .flat_map(|(si, _)| {
            (0..sim.streamlines_per_seed)
                .step_by(BATCH)
                .enumerate()
                .map(move |(bi, start)| (si, bi, BATCH.min(sim.streamlines_per_seed - start)))
        })
        .collect();
    let per_task = items.len().div_ceil(rayon::current_num_threads().max(1) * 4).max(1);
    let acc = items
        .par_chunks(per_task)
        .map(|chunk| {
            let mut acc = Acc::new(grid.n_voxels());
            let mut weights = vec![0.0; n_dirs];
            for &(si, bi, count) in chunk {
                let mut rng = SmallRng::seed_from_u64(
                    sim.seed ^ (si as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (bi as u64).wrapping_mul(0xD1B5_4A32_D192_ED03),
                );
                let u = seeds[si] as usize;
                for _ in 0..count {
                    // First hop direction.
                    let mut total = 0.0;
                    for (d, wd) in weights.iter_mut().enumerate() {
                        *wd = amp(u, d);
                        total += *wd;
                    }
                    let x = rng.random::<f64>() * (total + n_full * params.tau0);
                    let Some(mut dir) = pick(&weights, x, total) else {
                        acc.stopped += 1;
                        continue;
                    };
                    let c = grid.ijk()[u];
                    let mut pos = [
                        c[0] as f64 + rng.random::<f64>(),
                        c[1] as f64 + rng.random::<f64>(),
                        c[2] as f64 + rng.random::<f64>(),
                    ];
                    let mut cell = [c[0] as i64, c[1] as i64, c[2] as i64];
                    let mut voxel = u;
                    let mut hops = 0;
                    loop {
                        let d = table.disp[dir];
                        pos = [pos[0] + d[0], pos[1] + d[1], pos[2] + d[2]];
                        let next = [pos[0].floor() as i64, pos[1].floor() as i64, pos[2].floor() as i64];
                        if next != cell {
                            let Some(v) = grid.lookup(next[0], next[1], next[2]) else { break };
                            let k = neighbor_index([
                                (next[0] - cell[0]) as i32,
                                (next[1] - cell[1]) as i32,
                                (next[2] - cell[2]) as i32,
                            ])
                            .expect("hops never skip a voxel");
                            acc.entries[v] += 1.0;
                            acc.states[v * N_NEIGHBORS + k] += 1.0;
                            voxel = v;
                            cell = next;
                        }
                        hops += 1;
                        if hops >= sim.max_hops {
                            break;
                        }
                        // Turn under the current voxel's ODF.
                        let (idx, val) = table.turning.row(dir);
                        let mut total = 0.0;
                        for (slot, (&j, &w)) in idx.iter().zip(val).enumerate() {
                            weights[slot] = w * amp(voxel, j as usize);
                            total += weights[slot];
                        }
                        let x = rng.random::<f64>() * (total + params.tau);
                        match pick(&weights[..idx.len()], x, total) {
                            Some(slot) => dir = idx[slot] as usize,
                            None => {
                                acc.stopped += 1;
                                break;
                            }
                        }
                    }
                }
            }
            acc
        })
        .reduce(|| Acc::new(grid.n_voxels()), Acc::merge);

    let n_streamlines = seeds.len() * sim.streamlines_per_seed;
    let scale = if n_streamlines > 0 { 1.0 / n_streamlines as f64 } else { 0.0 };
    Ok(SimResult {
        entries: acc.entries.iter().map(|x| x * scale).collect(),
        state_entries: acc.states.iter().map(|x| x * scale).collect(),
        n_streamlines,
        stopped_fraction: acc.stopped as f64 * scale,
    })
}

/// Index of the categorical draw `x` over `weights`, `None` when `x` falls in
/// the stop mass beyond `total`.
fn pick(weights: &[f64], x: f64, total: f64) -> Option<usize> {
    if !(total > 0.0) || x >= total {
        return None;
    }
    let mut acc = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if x < acc {
            return Some(i);
        }
    }
    weights.iter().rposition(|&w| w > 0.0)
}
