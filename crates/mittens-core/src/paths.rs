//! Enumeration of turning-angle sequences and their exact exit volumes.

use std::collections::HashMap;

use crate::sphere::{fold_turning, turning_matrix, Csr, Sphere, Turning};
use crate::{neighbor_index, Error, Result};

pub const NO_PARENT: u32 = u32::MAX;

#[derive(Clone, Debug)]
pub struct PathParams {
    /// Hop length in world units (mm when the affine is in mm).
    pub step: f64,
    /// Maximum angle between consecutive hops, degrees (strict `<`).
    pub theta_max_deg: f64,
    pub turning: Turning,
    /// Linear part of the voxel-to-world affine (rows are world axes).
    pub vox_to_world: [[f64; 3]; 3],
    /// Exit volumes at or below this are dropped (round-off slivers).
    pub tol: f64,
    /// Maximum number of hops inside one voxel.
    pub max_depth: usize,
    pub max_nodes: usize,
}

impl PathParams {
    pub fn new(step: f64, theta_max_deg: f64) -> Self {
        Self {
            step,
            theta_max_deg,
            turning: Turning::Flat,
            vox_to_world: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            tol: 1e-12,
            max_depth: 8,
            max_nodes: 50_000_000,
        }
    }
}

/// Every turning-angle sequence that starts in a unit voxel, stored as a
/// prefix tree in preorder. Node `i` is the sequence ending with hop
/// `node_dir[i]`; its emissions are the neighbors that sequence can exit
/// into with volume `emit_vol` and coefficient `emit_coef = vol · Π W`
/// (the angle-prior part of each turn).
#[derive(Clone, Debug)]
pub struct PathTable {
    pub params: PathParams,
    /// Number of signed directions on the full sphere.
    pub n_full: usize,
    /// Directions are hemisphere indices with ± paths merged.
    pub folded: bool,
    /// Turning prior over the table's direction indices.
    pub turning: Csr,
    /// Index-space displacement of each signed direction (unfolded tables).
    pub disp: Vec<[f64; 3]>,
    pub node_dir: Vec<u32>,
    pub node_parent: Vec<u32>,
    pub node_root: Vec<u32>,
    pub node_depth: Vec<u8>,
    /// One past the last node of the subtree rooted at each node.
    pub node_end: Vec<u32>,
    pub emit_ptr: Vec<u32>,
    pub emit_nbr: Vec<u8>,
    pub emit_vol: Vec<f64>,
    pub emit_coef: Vec<f64>,
    /// Node index of the root for each direction (`NO_PARENT` if absent).
    pub root_node: Vec<u32>,
}

struct RawNode {
    parent: u32,
    dir: u32,
    root: u32,
    depth: u8,
}

struct Builder<'a> {
    turning: &'a Csr,
    disp: &'a [[f64; 3]],
    params: &'a PathParams,
    nodes: Vec<RawNode>,
    emissions: Vec<(u32, u8, f64, f64)>,
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn descend(
        &mut self,
        parent: u32,
        dir: usize,
        root: u32,
        depth: usize,
        lo: [f64; 3],
        hi: [f64; 3],
        cum: [f64; 3],
        wprod: f64,
    ) -> Result<()> {
        if self.nodes.len() >= self.params.max_nodes {
            return Err(Error::TooComplex(format!(
                "more than {} sequences; increase the step or decrease the max angle",
                self.params.max_nodes
            )));
        }
        let node = self.nodes.len() as u32;
        self.nodes.push(RawNode { parent, dir: dir as u32, root, depth: depth as u8 });
        let d = self.disp[dir];
        let cum = [cum[0] + d[0], cum[1] + d[1], cum[2] + d[2]];

        // Start points x in [lo, hi] land in cell o when x + cum ∈ [o, o + 1].
        let mut bounds = [[(0.0f64, 0.0f64); 3]; 3];
        for a in 0..3 {
            for (k, o) in [-1.0f64, 0.0, 1.0].into_iter().enumerate() {
                bounds[a][k] = (lo[a].max(o - cum[a]), hi[a].min(o + 1.0 - cum[a]));
            }
        }
        let len = |b: (f64, f64)| (b.1 - b.0).max(0.0);
        for (i, bi) in bounds[0].iter().enumerate() {
            for (j, bj) in bounds[1].iter().enumerate() {
                for (k, bk) in bounds[2].iter().enumerate() {
                    if i == 1 && j == 1 && k == 1 {
                        continue;
                    }
                    let vol = len(*bi) * len(*bj) * len(*bk);
                    if vol > self.params.tol {
                        let nbr = neighbor_index([i as i32 - 1, j as i32 - 1, k as i32 - 1]).unwrap();
                        self.emissions.push((node, nbr as u8, vol, vol * wprod));
                    }
                }
            }
        }
        let inside = [bounds[0][1], bounds[1][1], bounds[2][1]];
        if len(inside[0]) * len(inside[1]) * len(inside[2]) > self.params.tol {
            if depth + 1 >= self.params.max_depth {
                return Err(Error::TooComplex(format!(
                    "sequences longer than {} hops stay inside the voxel; the step is too small for this max angle",
                    self.params.max_depth
                )));
            }
            let new_lo = [inside[0].0, inside[1].0, inside[2].0];
            let new_hi = [inside[0].1, inside[1].1, inside[2].1];
            let (idx, val) = self.turning.row(dir);
            for (&next, &w) in idx.iter().zip(val) {
                self.descend(node, next as usize, root, depth + 1, new_lo, new_hi, cum, wprod * w)?;
            }
        }
        Ok(())
    }
}

impl PathTable {
    /// Enumerate all sequences over the signed sphere (unfolded table).
    pub fn build(sphere: &Sphere, params: PathParams) -> Result<Self> {
        if !(params.step > 0.0 && params.step.is_finite()) {
            return Err(Error::InvalidInput("step must be positive".into()));
        }
        let turning = turning_matrix(sphere, params.theta_max_deg, params.turning)?;
        let inv = invert3(&params.vox_to_world)?;
        let disp: Vec<[f64; 3]> = sphere
            .dirs()
            .iter()
            .map(|d| {
                let v = matvec(&inv, d);
                [v[0] * params.step, v[1] * params.step, v[2] * params.step]
            })
            .collect();
        let longest = disp.iter().flat_map(|d| d.iter().map(|x| x.abs())).fold(0.0, f64::max);
        if longest > 1.0 + 1e-9 {
            return Err(Error::Geometry(format!(
                "a hop moves {longest:.3} voxels along one axis; steps longer than one voxel \
                 can skip past the 26-neighborhood"
            )));
        }
        let mut builder = Builder { turning: &turning, disp: &disp, params: &params, nodes: Vec::new(), emissions: Vec::new() };
        for dir in 0..sphere.len() {
            builder.descend(NO_PARENT, dir, dir as u32, 0, [0.0; 3], [1.0; 3], [0.0; 3], 1.0)?;
        }
        let Builder { nodes, emissions, .. } = builder;
        Ok(assemble(params, sphere.len(), false, turning, disp, nodes, emissions))
    }

    /// Merge each sequence with its antipodal mirror so the table runs on
    /// hemisphere-indexed (symmetric) ODFs at half the cost.
    pub fn fold(&self, sphere: &Sphere) -> Result<Self> {
        if self.folded {
            return Ok(self.clone());
        }
        if sphere.len() != self.n_full {
            return Err(Error::InvalidInput("sphere does not match this path table".into()));
        }
        if self.params.theta_max_deg >= 90.0 {
            return Err(Error::Geometry("folding requires a max angle below 90 degrees".into()));
        }
        let h = sphere
            .hemisphere_size()
            .ok_or_else(|| Error::Geometry("sphere is not antipodally symmetric with a leading hemisphere".into()))?;
        let rep = sphere.hemisphere_rep().unwrap();

        let mut key: HashMap<(u32, u32), u32> = HashMap::new();
        let mut nodes: Vec<RawNode> = Vec::new();
        let mut map = vec![0u32; self.node_dir.len()];
        for s in 0..self.node_dir.len() {
            let parent = match self.node_parent[s] {
                NO_PARENT => NO_PARENT,
                p => map[p as usize],
            };
            let dir = rep[self.node_dir[s] as usize];
            let next = nodes.len() as u32;
            let idx = *key.entry((parent, dir)).or_insert_with(|| {
                nodes.push(RawNode { parent, dir, root: rep[self.node_root[s] as usize], depth: self.node_depth[s] });
                next
            });
            map[s] = idx;
        }
        let mut merged: HashMap<(u32, u8), (f64, f64)> = HashMap::new();
        for (s, &folded_node) in map.iter().enumerate() {
            for e in self.emissions(s) {
                let slot = merged.entry((folded_node, self.emit_nbr[e])).or_insert((0.0, 0.0));
                slot.0 += self.emit_vol[e];
                slot.1 += self.emit_coef[e];
            }
        }
        let emissions = merged.into_iter().map(|((n, nbr), (v, c))| (n, nbr, v, c)).collect();
        let turning = fold_turning(&self.turning, &rep, h);
        Ok(assemble(self.params.clone(), self.n_full, true, turning, self.disp.clone(), nodes, emissions))
    }

    pub fn n_dirs(&self) -> usize {
        self.turning.n_rows()
    }

    pub fn n_nodes(&self) -> usize {
        self.node_dir.len()
    }

    pub fn n_emissions(&self) -> usize {
        self.emit_nbr.len()
    }

    /// Longest sequence in hops.
    pub fn max_hops(&self) -> usize {
        self.node_depth.iter().map(|&d| d as usize + 1).max().unwrap_or(0)
    }

    /// Emission index range of node `i`.
    #[inline]
    pub fn emissions(&self, i: usize) -> std::ops::Range<usize> {
        self.emit_ptr[i] as usize..self.emit_ptr[i + 1] as usize
    }
}

/// Reorder nodes into preorder (children sorted by direction), compute
/// subtree extents and pack emissions as CSR.
fn assemble(
    params: PathParams,
    n_full: usize,
    folded: bool,
    turning: Csr,
    disp: Vec<[f64; 3]>,
    nodes: Vec<RawNode>,
    mut emissions: Vec<(u32, u8, f64, f64)>,
) -> PathTable {
    let n = nodes.len();
    let mut children: Vec<Vec<u32>> = vec![Vec::new(); n];
    let mut roots: Vec<u32> = Vec::new();
    for (i, node) in nodes.iter().enumerate() {
        match node.parent {
            NO_PARENT => roots.push(i as u32),
            p => children[p as usize].push(i as u32),
        }
    }
    roots.sort_by_key(|&i| nodes[i as usize].dir);
    for c in children.iter_mut() {
        c.sort_by_key(|&i| nodes[i as usize].dir);
    }
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut stack: Vec<u32> = roots.iter().rev().copied().collect();
    while let Some(i) = stack.pop() {
        order.push(i);
        stack.extend(children[i as usize].iter().rev());
    }
    let mut new_index = vec![0u32; n];
    for (k, &i) in order.iter().enumerate() {
        new_index[i as usize] = k as u32;
    }

    let n_dirs = turning.n_rows();
    let mut node_dir = Vec::with_capacity(n);
    let mut node_parent = Vec::with_capacity(n);
    let mut node_root = Vec::with_capacity(n);
    let mut node_depth = Vec::with_capacity(n);
    let mut root_node = vec![NO_PARENT; n_dirs];
    for (k, &i) in order.iter().enumerate() {
        let node = &nodes[i as usize];
        node_dir.push(node.dir);
        node_parent.push(if node.parent == NO_PARENT { NO_PARENT } else { new_index[node.parent as usize] });
        node_root.push(node.root);
        node_depth.push(node.depth);
        if node.parent == NO_PARENT {
            root_node[node.dir as usize] = k as u32;
        }
    }
    let mut node_end = vec![n as u32; n];
    let mut open: Vec<usize> = Vec::new();
    for k in 0..n {
        while let Some(&top) = open.last() {
            if node_depth[top] >= node_depth[k] {
                node_end[top] = k as u32;
                open.pop();
            } else {
                break;
            }
        }
        open.push(k);
    }

    for e in emissions.iter_mut() {
        e.0 = new_index[e.0 as usize];
    }
    emissions.sort_by_key(|e| (e.0, e.1));
    let mut emit_ptr = vec![0u32; n + 1];
    for e in &emissions {
        emit_ptr[e.0 as usize + 1] += 1;
    }
    for k in 0..n {
        emit_ptr[k + 1] += emit_ptr[k];
    }
    PathTable {
        params,
        n_full,
        folded,
        turning,
        disp,
        node_dir,
        node_parent,
        node_root,
        node_depth,
        node_end,
        emit_ptr,
        emit_nbr: emissions.iter().map(|e| e.1).collect(),
        emit_vol: emissions.iter().map(|e| e.2).collect(),
        emit_coef: emissions.iter().map(|e| e.3).collect(),
        root_node,
    }
}

fn matvec(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn invert3(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    if !(det.abs() > 1e-12) {
        return Err(Error::InvalidInput("voxel-to-world matrix is singular".into()));
    }
    let inv = 1.0 / det;
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv,
        ],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sphere::fibonacci_sphere;

    fn table(n_half: usize, step: f64) -> (Sphere, PathTable) {
        let sphere = Sphere::new(&fibonacci_sphere(n_half)).unwrap();
        let t = PathTable::build(&sphere, PathParams::new(step, 35.0)).unwrap();
        (sphere, t)
    }

    #[test]
    fn exit_volumes_partition_the_voxel() {
        // With a flat prior, Σ over sequences of V · Π W must be 1 for every
        // first direction: every start point exits somewhere.
        let (_, t) = table(60, 3f64.sqrt() / 2.0);
        let mut per_root = vec![0.0; t.n_dirs()];
        for i in 0..t.n_nodes() {
            for e in t.emissions(i) {
                per_root[t.node_root[i] as usize] += t.emit_coef[e];
            }
        }
        for total in per_root {
            assert!((total - 1.0).abs() < 1e-9, "{total}");
        }
    }

    #[test]
    fn preorder_and_subtrees() {
        let (_, t) = table(40, 0.8);
        for i in 0..t.n_nodes() {
            let p = t.node_parent[i];
            if p != NO_PARENT {
                assert!((p as usize) < i);
                assert!(t.node_end[p as usize] as usize >= t.node_end[i] as usize);
            }
            assert!(t.node_end[i] as usize > i);
        }
        assert!(t.max_hops() >= 2);
    }

    #[test]
    fn fold_preserves_coefficient_mass() {
        let (sphere, t) = table(50, 3f64.sqrt() / 2.0);
        let f = t.fold(&sphere).unwrap();
        assert_eq!(f.n_dirs(), 50);
        let a: f64 = t.emit_coef.iter().sum();
        let b: f64 = f.emit_coef.iter().sum();
        assert!((a - b).abs() < 1e-9);
        assert!(f.n_nodes() * 2 >= t.n_nodes() - 1);
    }

    #[test]
    fn rejects_steps_that_skip_neighbors() {
        let sphere = Sphere::new(&fibonacci_sphere(30)).unwrap();
        assert!(PathTable::build(&sphere, PathParams::new(1.5, 35.0)).is_err());
    }
}
