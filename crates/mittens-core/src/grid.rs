//! Masked voxel grid: neighbor lookup and spatial blocking.

use crate::{Error, Result, NEIGHBOR_OFFSETS, N_NEIGHBORS};

/// Voxels of a mask, one row per voxel, with O(1) neighbor lookup.
#[derive(Clone, Debug)]
pub struct Grid {
    dims: [usize; 3],
    ijk: Vec<[usize; 3]>,
    index: Vec<i32>,
}

impl Grid {
    pub fn new(ijk: &[[i64; 3]], dims: [usize; 3]) -> Result<Self> {
        let total = dims[0]
            .checked_mul(dims[1])
            .and_then(|x| x.checked_mul(dims[2]))
            .ok_or_else(|| Error::InvalidInput("grid too large".into()))?;
        if ijk.len() > i32::MAX as usize {
            return Err(Error::InvalidInput("too many voxels".into()));
        }
        let mut index = vec![-1i32; total];
        let mut out = Vec::with_capacity(ijk.len());
        for (v, c) in ijk.iter().enumerate() {
            if (0..3).any(|a| c[a] < 0 || c[a] as usize >= dims[a]) {
                return Err(Error::InvalidInput(format!("voxel {v} at {c:?} is outside the grid {dims:?}")));
            }
            let c = [c[0] as usize, c[1] as usize, c[2] as usize];
            let flat = (c[0] * dims[1] + c[1]) * dims[2] + c[2];
            if index[flat] >= 0 {
                return Err(Error::InvalidInput(format!("voxel {c:?} appears twice")));
            }
            index[flat] = v as i32;
            out.push(c);
        }
        Ok(Self { dims, ijk: out, index })
    }

    pub fn n_voxels(&self) -> usize {
        self.ijk.len()
    }

    pub fn dims(&self) -> [usize; 3] {
        self.dims
    }

    pub fn ijk(&self) -> &[[usize; 3]] {
        &self.ijk
    }

    pub fn lookup(&self, i: i64, j: i64, k: i64) -> Option<usize> {
        let d = self.dims;
        if i < 0 || j < 0 || k < 0 || i as usize >= d[0] || j as usize >= d[1] || k as usize >= d[2] {
            return None;
        }
        let v = self.index[(i as usize * d[1] + j as usize) * d[2] + k as usize];
        (v >= 0).then_some(v as usize)
    }

    #[inline]
    pub fn neighbor(&self, voxel: usize, n: usize) -> Option<usize> {
        let c = self.ijk[voxel];
        let o = NEIGHBOR_OFFSETS[n];
        self.lookup(c[0] as i64 + o[0] as i64, c[1] as i64 + o[1] as i64, c[2] as i64 + o[2] as i64)
    }

    /// `(N, 26)` row-major neighbor ids, −1 where the neighbor is not in the mask.
    pub fn neighbor_table(&self) -> Vec<i32> {
        let mut out = vec![-1i32; self.n_voxels() * N_NEIGHBORS];
        for v in 0..self.n_voxels() {
            for n in 0..N_NEIGHBORS {
                if let Some(w) = self.neighbor(v, n) {
                    out[v * N_NEIGHBORS + n] = w as i32;
                }
            }
        }
        out
    }

    /// Group voxels into cubic blocks of `size` (in voxels), keeping only the
    /// voxels listed in `subset` when given.
    pub fn blocks(&self, size: usize, subset: Option<&[u32]>) -> Vec<Block> {
        let size = size.max(1);
        let nb = [self.dims[1].div_ceil(size), self.dims[2].div_ceil(size)];
        let key = |v: usize| {
            let c = self.ijk[v];
            ((c[0] / size) * nb[0] + c[1] / size) * nb[1] + c[2] / size
        };
        let mut ids: Vec<u32> = match subset {
            Some(s) => s.to_vec(),
            None => (0..self.n_voxels() as u32).collect(),
        };
        ids.sort_by_key(|&v| (key(v as usize), v));
        ids.dedup();
        let mut out: Vec<Block> = Vec::new();
        let mut start = 0;
        while start < ids.len() {
            let k = key(ids[start] as usize);
            let mut end = start;
            while end < ids.len() && key(ids[end] as usize) == k {
                end += 1;
            }
            out.push(Block::new(self, ids[start..end].to_vec()));
            start = end;
        }
        out
    }
}

/// A set of voxels plus the halo of their in-mask neighbors. Kernels compute
/// per-voxel quantities once for every halo voxel and look neighbors up by
/// slot.
pub struct Block {
    pub voxels: Vec<u32>,
    /// Sorted voxel ids of `voxels` and all their neighbors.
    pub halo: Vec<u32>,
    /// Slot in `halo` of each block voxel.
    pub self_slot: Vec<u32>,
    /// Slot in `halo` of each block voxel's neighbors, −1 if missing.
    pub nbr_slot: Vec<[i32; N_NEIGHBORS]>,
}

impl Block {
    fn new(grid: &Grid, voxels: Vec<u32>) -> Self {
        let mut halo: Vec<u32> = voxels.clone();
        for &v in &voxels {
            for n in 0..N_NEIGHBORS {
                if let Some(w) = grid.neighbor(v as usize, n) {
                    halo.push(w as u32);
                }
            }
        }
        halo.sort_unstable();
        halo.dedup();
        let slot = |v: u32| halo.binary_search(&v).unwrap() as u32;
        let self_slot = voxels.iter().map(|&v| slot(v)).collect();
        let nbr_slot = voxels
            .iter()
            .map(|&v| {
                let mut s = [-1i32; N_NEIGHBORS];
                for (n, item) in s.iter_mut().enumerate() {
                    if let Some(w) = grid.neighbor(v as usize, n) {
                        *item = slot(w as u32) as i32;
                    }
                }
                s
            })
            .collect();
        Self { voxels, halo, self_slot, nbr_slot }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbors_and_blocks() {
        let mut ijk = Vec::new();
        for i in 0..5 {
            for j in 0..4 {
                for k in 0..3 {
                    ijk.push([i, j, k]);
                }
            }
        }
        let g = Grid::new(&ijk, [5, 4, 3]).unwrap();
        let center = g.lookup(2, 2, 1).unwrap();
        assert!((0..26).all(|n| g.neighbor(center, n).is_some()));
        assert_eq!(g.neighbor(0, 0), None);
        let blocks = g.blocks(2, None);
        assert_eq!(blocks.iter().map(|b| b.voxels.len()).sum::<usize>(), 60);
        for b in &blocks {
            for (i, &v) in b.voxels.iter().enumerate() {
                assert_eq!(b.halo[b.self_slot[i] as usize], v);
                for n in 0..26 {
                    let expect = g.neighbor(v as usize, n).map(|w| w as u32);
                    let got = (b.nbr_slot[i][n] >= 0).then(|| b.halo[b.nbr_slot[i][n] as usize]);
                    assert_eq!(expect, got);
                }
            }
        }
        assert!(Grid::new(&[[0, 0, 0], [0, 0, 0]], [1, 1, 1]).is_err());
    }
}
