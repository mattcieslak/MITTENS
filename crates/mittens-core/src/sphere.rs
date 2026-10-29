//! Discrete spheres of fiber directions and turning-angle weights.

use crate::{Error, Result};

/// Unit directions (world coordinates). The full signed sphere is the
/// canonical representation: asymmetric ODFs need both `d` and `-d`.
///
/// Vertices within 1e-6 of unit length are kept exactly as given (sphere
/// files are often stored in float32); others are normalized.
#[derive(Clone, Debug)]
pub struct Sphere {
    dirs: Vec<[f64; 3]>,
    antipode: Option<Vec<u32>>,
}

impl Sphere {
    pub fn new(vertices: &[[f64; 3]]) -> Result<Self> {
        if vertices.is_empty() {
            return Err(Error::InvalidInput("sphere has no vertices".into()));
        }
        let mut dirs = Vec::with_capacity(vertices.len());
        for v in vertices {
            let norm = dot(v, v).sqrt();
            if !(norm.is_finite() && norm > 0.0) {
                return Err(Error::InvalidInput("sphere vertex with zero or non-finite length".into()));
            }
            let scale = if (norm - 1.0).abs() <= 1e-6 { 1.0 } else { 1.0 / norm };
            dirs.push([v[0] * scale, v[1] * scale, v[2] * scale]);
        }
        let antipode = detect_antipodes(&dirs);
        Ok(Self { dirs, antipode })
    }

    pub fn len(&self) -> usize {
        self.dirs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dirs.is_empty()
    }

    pub fn dirs(&self) -> &[[f64; 3]] {
        &self.dirs
    }

    /// `antipode()[i]` is the vertex pointing opposite to vertex `i`, if every
    /// vertex has one.
    pub fn antipode(&self) -> Option<&[u32]> {
        self.antipode.as_deref()
    }

    /// Size `h` of the leading hemisphere when vertices `0..h` have their
    /// antipodes in `h..2h` (the layout used by DSI Studio and ODX).
    pub fn hemisphere_size(&self) -> Option<usize> {
        let anti = self.antipode.as_ref()?;
        let n = self.dirs.len();
        if !n.is_multiple_of(2) {
            return None;
        }
        let h = n / 2;
        (0..h).all(|i| anti[i] as usize >= h).then_some(h)
    }

    /// Map from every vertex to its representative in the leading hemisphere.
    pub fn hemisphere_rep(&self) -> Option<Vec<u32>> {
        let h = self.hemisphere_size()?;
        let anti = self.antipode.as_ref()?;
        Some((0..self.dirs.len()).map(|i| if i < h { i as u32 } else { anti[i] }).collect())
    }
}

fn detect_antipodes(dirs: &[[f64; 3]]) -> Option<Vec<u32>> {
    let mut out = vec![0u32; dirs.len()];
    for (i, a) in dirs.iter().enumerate() {
        let mut best = (f64::INFINITY, 0usize);
        for (j, b) in dirs.iter().enumerate() {
            let c = dot(a, b);
            if c < best.0 {
                best = (c, j);
            }
        }
        if best.0 > -1.0 + 1e-6 {
            return None;
        }
        out[i] = best.1 as u32;
    }
    out.iter().enumerate().all(|(i, &j)| out[j as usize] as usize == i).then_some(out)
}

pub(crate) fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Angle between two directions in degrees, computed as in MITTENS v1.
pub fn angle_deg(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let na = dot(a, a).sqrt();
    let nb = dot(b, b).sqrt();
    let ua = [a[0] / na, a[1] / na, a[2] / na];
    let ub = [b[0] / nb, b[1] / nb, b[2] / nb];
    dot(&ua, &ub).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Prior over the turn between consecutive hops, restricted to
/// `angle < theta_max`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Turning {
    /// All compatible directions equally likely (paper eq. 2).
    Flat,
    /// `((180 - Δ) / (180 + Δ))^power`, Δ in degrees (v1 "weighted").
    Weighted { power: f64 },
    /// von Mises–Fisher `exp(kappa (cos Δ - 1))`. With `kappa = ℓp / step`
    /// the tangent decorrelates like a worm-like chain of persistence
    /// length ℓp.
    Vmf { kappa: f64 },
}

impl Turning {
    fn weight(&self, cos_angle: f64, angle_deg: f64) -> f64 {
        match *self {
            Turning::Flat => 1.0,
            Turning::Weighted { power } => ((180.0 - angle_deg) / (180.0 + angle_deg)).powf(power),
            Turning::Vmf { kappa } => (kappa * (cos_angle - 1.0)).exp(),
        }
    }
}

/// Compressed sparse rows.
#[derive(Clone, Debug, Default)]
pub struct Csr {
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
    pub values: Vec<f64>,
}

impl Csr {
    pub fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let (s, e) = (self.indptr[i] as usize, self.indptr[i + 1] as usize);
        (&self.indices[s..e], &self.values[s..e])
    }

    /// Value at `(i, j)`, zero when not stored.
    pub fn get(&self, i: usize, j: usize) -> f64 {
        let (idx, val) = self.row(i);
        idx.binary_search(&(j as u32)).map(|k| val[k]).unwrap_or(0.0)
    }
}

/// Row-normalized turning matrix `W[i, j] = P(next = j | previous = i)` over
/// all signed directions, before any ODF weighting.
pub fn turning_matrix(sphere: &Sphere, theta_max_deg: f64, turning: Turning) -> Result<Csr> {
    if !(theta_max_deg > 0.0 && theta_max_deg <= 180.0) {
        return Err(Error::InvalidInput(format!("theta_max must be in (0, 180], got {theta_max_deg}")));
    }
    let dirs = sphere.dirs();
    let mut csr = Csr { indptr: vec![0], ..Default::default() };
    let mut row: Vec<(u32, f64)> = Vec::new();
    for a in dirs {
        row.clear();
        for (j, b) in dirs.iter().enumerate() {
            let angle = angle_deg(a, b);
            if angle < theta_max_deg {
                row.push((j as u32, turning.weight(angle.to_radians().cos(), angle)));
            }
        }
        let total: f64 = row.iter().map(|x| x.1).sum();
        if !(total > 0.0) {
            return Err(Error::Geometry("turning prior has no mass in a cone".into()));
        }
        for &(j, w) in &row {
            csr.indices.push(j);
            csr.values.push(w / total);
        }
        csr.indptr.push(csr.indices.len() as u32);
    }
    Ok(csr)
}

/// Fold a signed turning matrix onto the leading hemisphere:
/// `Wf[i, rep(j)] = Σ W[i, j]` for `i < h`.
pub fn fold_turning(w: &Csr, rep: &[u32], h: usize) -> Csr {
    let mut csr = Csr { indptr: vec![0], ..Default::default() };
    let mut acc: Vec<(u32, f64)> = Vec::new();
    for i in 0..h {
        acc.clear();
        let (idx, val) = w.row(i);
        for (&j, &v) in idx.iter().zip(val) {
            acc.push((rep[j as usize], v));
        }
        acc.sort_by_key(|x| x.0);
        let mut k = 0;
        while k < acc.len() {
            let (j, mut v) = acc[k];
            k += 1;
            while k < acc.len() && acc[k].0 == j {
                v += acc[k].1;
                k += 1;
            }
            csr.indices.push(j);
            csr.values.push(v);
        }
        csr.indptr.push(csr.indices.len() as u32);
    }
    csr
}

/// Antipodally symmetric quasi-uniform sphere (Fibonacci hemisphere plus its
/// antipodes), used in tests and as a fallback.
pub fn fibonacci_sphere(n_half: usize) -> Vec<[f64; 3]> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    let mut half = Vec::with_capacity(n_half);
    for i in 0..n_half {
        let z = 1.0 - (i as f64 + 0.5) / n_half as f64;
        let r = (1.0 - z * z).sqrt();
        let phi = golden * i as f64;
        half.push([r * phi.cos(), r * phi.sin(), z]);
    }
    let mut out = half.clone();
    out.extend(half.iter().map(|v| [-v[0], -v[1], -v[2]]));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn antipodes_and_hemisphere() {
        let s = Sphere::new(&fibonacci_sphere(50)).unwrap();
        assert_eq!(s.hemisphere_size(), Some(50));
        let rep = s.hemisphere_rep().unwrap();
        assert!(rep.iter().all(|&r| (r as usize) < 50));
        assert_eq!(s.antipode().unwrap()[3], 53);
    }

    #[test]
    fn turning_rows_sum_to_one() {
        let s = Sphere::new(&fibonacci_sphere(80)).unwrap();
        for turning in [Turning::Flat, Turning::Weighted { power: 2.0 }, Turning::Vmf { kappa: 8.0 }] {
            let w = turning_matrix(&s, 35.0, turning).unwrap();
            for i in 0..w.n_rows() {
                let total: f64 = w.row(i).1.iter().sum();
                assert!((total - 1.0).abs() < 1e-12);
            }
        }
    }
}
