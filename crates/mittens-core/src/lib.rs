//! Analytic voxel-to-voxel transition probabilities (MITTENS core).
//!
//! A fiber leaving a voxel is modelled as a sequence of fixed-length hops
//! ("turning angle sequence", Cieslak et al. 2018). For a sequence σ the set
//! of starting points inside the voxel that follow σ and land in neighbor v is
//! an axis-aligned box, so its volume V(σ, v) is exact. [`paths::PathTable`]
//! enumerates every sequence once per (sphere, step, max angle, voxel shape)
//! and the kernels weight each sequence by the ODF of the voxel it crosses.

// `!(x > 0.0)` is used on purpose: it is also true for NaN.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

pub mod grid;
pub mod kernel;
pub mod paths;
pub mod simulate;
pub mod sphere;

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    InvalidInput(String),
    Geometry(String),
    TooComplex(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::Geometry(msg) => write!(f, "geometry error: {msg}"),
            Error::TooComplex(msg) => write!(f, "path enumeration too large: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub const N_NEIGHBORS: usize = 26;

const fn make_offsets() -> [[i32; 3]; N_NEIGHBORS] {
    let mut out = [[0i32; 3]; N_NEIGHBORS];
    let mut n = 0;
    let mut code = 0;
    while code < 27 {
        let o = [code / 9 - 1, (code / 3) % 3 - 1, code % 3 - 1];
        if !(o[0] == 0 && o[1] == 0 && o[2] == 0) {
            out[n] = o;
            n += 1;
        }
        code += 1;
    }
    out
}

/// Index-space offsets of the 26 neighbors, in lexicographic order of
/// `(di, dj, dk)`. `NEIGHBOR_OFFSETS[25 - n] == -NEIGHBOR_OFFSETS[n]`.
pub const NEIGHBOR_OFFSETS: [[i32; 3]; N_NEIGHBORS] = make_offsets();

/// Position of `offset` in [`NEIGHBOR_OFFSETS`].
pub fn neighbor_index(offset: [i32; 3]) -> Option<usize> {
    if offset.iter().any(|&x| !(-1..=1).contains(&x)) || offset == [0, 0, 0] {
        return None;
    }
    let code = ((offset[0] + 1) * 9 + (offset[1] + 1) * 3 + (offset[2] + 1)) as usize;
    Some(if code > 13 { code - 1 } else { code })
}

/// Index of the neighbor in the opposite direction.
pub const fn opposite_neighbor(n: usize) -> usize {
    N_NEIGHBORS - 1 - n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn offsets_roundtrip_and_opposites() {
        for (n, &o) in NEIGHBOR_OFFSETS.iter().enumerate() {
            assert_eq!(neighbor_index(o), Some(n));
            let neg = [-o[0], -o[1], -o[2]];
            assert_eq!(NEIGHBOR_OFFSETS[opposite_neighbor(n)], neg);
        }
        assert_eq!(neighbor_index([0, 0, 0]), None);
        assert_eq!(neighbor_index([2, 0, 0]), None);
    }
}
