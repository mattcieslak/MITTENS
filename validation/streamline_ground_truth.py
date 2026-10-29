#!/usr/bin/env python
"""Validate MITTENS transition probabilities against streamlines.

A tractogram provides both the input and the answer:

* FODs are track-orientation densities: each voxel's histogram of streamline
  directions on the odf8 sphere, weighted by length and streamline weight.
* Ground truth comes from walking every streamline at the model's hop length,
  from a random phase, in both directions (tractograms are unoriented). Each
  change of voxel is a transition. This samples what MITTENS predicts: the
  next voxel after leaving ``u`` from a uniformly distributed position.
  Consecutive transitions show what a memoryless voxel chain misses: the
  next voxel given the voxel ``w`` the walk came from.

Errors are reported next to a split-half noise floor: the same statistic
between ground truth from odd and even streamlines, halved (two half-samples
differ by about twice the sampling noise of the full sample).

Example (TRXScan's sub-60501 phantom, 1M ACT + SIFT2 streamlines)::

    python validation/streamline_ground_truth.py \\
        ~/Library/Caches/trxscan/sub-60501/sub-60501_ses-01_space-ACPC_desc-actsift2_tracks.trx
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial import cKDTree

from mittens import OdfData, TransitionModel
from mittens.io import antipodes, default_sphere

N = 26


def load_streamlines(path, max_streamlines=None, seed=0):
    """Positions (P, 3) RAS mm, offsets (S + 1,), weights (S,) from a TRX file."""
    from trx.trx_file_memmap import load

    trx = load(str(path))
    positions = np.asarray(trx.streamlines._data, dtype=np.float32)
    starts = np.asarray(trx.streamlines._offsets, dtype=np.int64)
    lengths = np.asarray(trx.streamlines._lengths, dtype=np.int64)
    dps = trx.data_per_streamline
    weights = np.asarray(dps["sift2_weights"], dtype=np.float64).ravel() if "sift2_weights" in dps \
        else np.ones(len(starts))
    if max_streamlines is not None and max_streamlines < len(starts):
        keep = np.sort(np.random.default_rng(seed).choice(len(starts), max_streamlines, replace=False))
        idx = np.concatenate([np.arange(s, s + n) for s, n in zip(starts[keep], lengths[keep])])
        positions, lengths, weights = positions[idx], lengths[keep], weights[keep]
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    return positions, offsets, weights


def make_grid(positions, voxel_mm):
    """LPS-oriented grid (like DSI Studio and most dMRI data) covering the
    streamlines with a two-voxel margin."""
    lo, hi = positions.min(0).astype(np.float64), positions.max(0).astype(np.float64)
    affine = np.diag([-voxel_mm, -voxel_mm, voxel_mm, 1.0])
    affine[:3, 3] = [hi[0] + 2 * voxel_mm, hi[1] + 2 * voxel_mm, lo[2] - 2 * voxel_mm]
    dims = tuple(int(np.ceil((hi[a] - lo[a]) / voxel_mm)) + 5 for a in range(3))
    return affine, dims


def to_cells(points, affine):
    """Voxel index of each point (voxel i covers [i - 0.5, i + 0.5))."""
    inv = np.linalg.inv(affine)
    return np.floor(points @ inv[:3, :3].T + inv[:3, 3] + 0.5).astype(np.int64)


def resample(positions, offsets, step, rng, reverse=False):
    """Points every ``step`` mm along each streamline from a random phase.

    Returns the points (Q, 3) and the streamline id of each, ordered along the
    walk (from the end when ``reverse``)."""
    n_s = len(offsets) - 1
    counts = np.diff(offsets)
    seg = np.zeros(len(positions))
    seg[1:] = np.linalg.norm(np.diff(positions.astype(np.float64), axis=0), axis=1)
    seg[offsets[:-1]] = 0.0
    cum = np.cumsum(seg)
    cum -= np.repeat(cum[offsets[:-1]], counts)
    total = cum[offsets[1:] - 1]
    phase = rng.random(n_s) * step
    n_samples = np.where(total >= phase, np.floor((total - phase) / step).astype(np.int64) + 1, 0)
    sid = np.repeat(np.arange(n_s), n_samples)
    k = np.arange(len(sid)) - np.repeat(np.cumsum(n_samples) - n_samples, n_samples)
    arc = phase[sid] + k * step
    if reverse:
        arc = total[sid] - arc
    # Locate each arc length on its streamline (keys are monotone across streamlines).
    big = float(total.max()) + 1.0
    vertex_sid = np.repeat(np.arange(n_s), counts)
    seg_idx = np.searchsorted(vertex_sid * big + cum, sid * big + arc, side="right") - 1
    seg_idx = np.clip(seg_idx, offsets[sid], np.maximum(offsets[sid + 1] - 2, offsets[sid]))
    nxt = np.minimum(seg_idx + 1, offsets[sid + 1] - 1)
    span = cum[nxt] - cum[seg_idx]
    t = np.divide(arc - cum[seg_idx], span, out=np.zeros_like(arc), where=span > 0)[:, None]
    p = positions[seg_idx].astype(np.float64)
    return p + t * (positions[nxt] - positions[seg_idx]), sid


def neighbor_code(delta):
    """Index in mittens.neighbor_offsets() of each (di, dj, dk) in {-1, 0, 1}^3."""
    code = (delta[:, 0] + 1) * 9 + (delta[:, 1] + 1) * 3 + (delta[:, 2] + 1)
    return code - (code > 13)


def track_orientation_density(positions, offsets, weights, affine, dims, vertices, step, rng, chunk):
    """Symmetric TOD on ``vertices``: (voxel flat index, vertex) -> weighted length."""
    tree = cKDTree(vertices)
    anti = antipodes(vertices)
    keys, vals = [], []
    for s in range(0, len(offsets) - 1, chunk):
        e = min(s + chunk, len(offsets) - 1)
        sub = positions[offsets[s]:offsets[e]]
        pts, sid = resample(sub, offsets[s:e + 1] - offsets[s], step, rng)
        same = sid[1:] == sid[:-1]
        d = (pts[1:] - pts[:-1])[same]
        mid = 0.5 * (pts[1:] + pts[:-1])[same]
        w = weights[s + sid[:-1][same]] * np.linalg.norm(d, axis=1)
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        _, vtx = tree.query(d, workers=-1)
        flat = np.ravel_multi_index(to_cells(mid, affine).T, dims)
        for v in (vtx, anti[vtx]):
            k = flat * len(vertices) + v
            u, inv = np.unique(k, return_inverse=True)
            keys.append(u)
            vals.append(np.bincount(inv, weights=w))
    keys = np.concatenate(keys)
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.bincount(inv, weights=np.concatenate(vals))


def _reduce(keys, vals):
    """Sum ``vals`` over equal ``keys``."""
    u, inv = np.unique(keys, return_inverse=True)
    return u, np.bincount(inv, weights=vals)


def transition_counts(positions, offsets, weights, affine, dims, voxel_of, step, rng, chunk, half=None):
    """Ground-truth transitions from walks in both directions.

    Returns dense first-order counts ``(N, 26)`` (weighted and raw) and sparse
    second-order counts keyed by ``(u * 26 + k_in) * 26 + n_out`` (weighted
    and raw). ``half`` keeps only even (0) or odd (1) streamlines."""
    n_vox = int(voxel_of.max()) + 1
    first = np.zeros(n_vox * N)
    raw_first = np.zeros(n_vox * N)
    parts = []
    for s in range(0, len(offsets) - 1, chunk):
        e = min(s + chunk, len(offsets) - 1)
        sub = positions[offsets[s]:offsets[e]]
        local = offsets[s:e + 1] - offsets[s]
        for reverse in (False, True):
            pts, sid = resample(sub, local, step, rng, reverse=reverse)
            cells = to_cells(pts, affine)
            moved = (sid[1:] == sid[:-1]) & np.any(cells[1:] != cells[:-1], axis=1)
            src, dst, esid = cells[:-1][moved], cells[1:][moved], sid[:-1][moved]
            u = voxel_of[np.ravel_multi_index(src.T, dims)]
            v = voxel_of[np.ravel_multi_index(dst.T, dims)]
            n = neighbor_code(dst - src)
            w = weights[s + esid]
            keep = (u >= 0) & (v >= 0)
            if half is not None:
                keep &= (s + esid) % 2 == half
            idx = u * N + n
            first += np.bincount(idx[keep], weights=w[keep], minlength=n_vox * N)
            raw_first += np.bincount(idx[keep], minlength=n_vox * N)
            # Consecutive moves of one walk: w -> u (k_in) then u -> v (n_out).
            pair = (esid[1:] == esid[:-1]) & keep[1:] & keep[:-1]
            idx2 = ((u[1:] * N + n[:-1]) * N + n[1:])[pair]
            k2, w2 = _reduce(idx2, w[1:][pair])
            _, c2 = _reduce(idx2, np.ones(len(idx2)))
            parts.append((k2, w2, c2))
    keys = np.concatenate([p[0] for p in parts])
    k2, w2 = _reduce(keys, np.concatenate([p[1] for p in parts]))
    _, c2 = _reduce(keys, np.concatenate([p[2] for p in parts]))
    return first.reshape(n_vox, N), raw_first.reshape(n_vox, N), (k2, w2, c2)


def state_rows(second, states):
    """Dense (len(states), 26) weighted counts for the given ``u * 26 + k_in`` states."""
    keys, vals = second[0], second[1]
    row = np.searchsorted(states, keys // N)
    ok = (row < len(states)) & (states[np.minimum(row, len(states) - 1)] == keys // N)
    out = np.zeros((len(states), N))
    out[row[ok], keys[ok] % N] = vals[ok]
    return out


def state_totals(second, n_vox):
    """Raw transition-pair count of every state."""
    keys, counts = second[0], second[2]
    return np.bincount(keys // N, weights=counts, minlength=n_vox * N)


def normalize(x):
    s = x.sum(-1, keepdims=True)
    return np.divide(x, s, out=np.zeros_like(x, dtype=np.float64), where=s > 0)


def l1(a, b):
    return np.abs(normalize(a) - normalize(b)).sum(-1)


def summarize(name, err, floor):
    print(f"  {name:38s} median L1 {np.median(err):.3f}   90th pct {np.percentile(err, 90):.3f}"
          f"   (noise floor {np.median(floor):.3f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trx", help="tractogram (.trx) with optional dps/sift2_weights")
    ap.add_argument("--voxel-mm", type=float, default=2.0)
    ap.add_argument("--step-voxels", type=float, nargs="+", default=[np.sqrt(3) / 2, 1.0],
                    help="hop lengths to test, in voxels (default: v1's sqrt(3)/2 and 1)")
    ap.add_argument("--tod-step-mm", type=float, default=0.5)
    ap.add_argument("--max-streamlines", type=int, default=None)
    ap.add_argument("--min-count", type=int, default=400,
                    help="minimum ground-truth transitions for a voxel or state to be scored")
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    positions, offsets, weights = load_streamlines(args.trx, args.max_streamlines, args.seed)
    affine, dims = make_grid(positions, args.voxel_mm)
    print(f"{len(offsets) - 1:,} streamlines, {len(positions):,} points; grid {dims} at {args.voxel_mm} mm")

    vertices = default_sphere()
    keys, vals = track_orientation_density(positions, offsets, weights, affine, dims, vertices,
                                           args.tod_step_mm, rng, args.chunk)
    flat, vtx = np.divmod(keys, len(vertices))
    voxels = np.unique(flat)
    voxel_of = np.full(int(np.prod(dims)), -1, dtype=np.int64)
    voxel_of[voxels] = np.arange(len(voxels))
    full = np.zeros((len(voxels), len(vertices)), dtype=np.float32)
    full[voxel_of[flat], vtx] = vals
    h = len(vertices) // 2
    data = OdfData.from_amplitudes(full[:, :h], np.column_stack(np.unravel_index(voxels, dims)),
                                   dims, affine, vertices)
    del full
    print(f"TOD for {data.n_voxels:,} voxels ({time.time() - t0:.0f} s)")

    for step_vox in args.step_voxels:
        step = step_vox * args.voxel_mm
        print(f"\n=== hop length {step:.3f} mm ({step_vox:.3f} voxels) ===")
        t1 = time.time()
        gt = transition_counts(positions, offsets, weights, affine, dims, voxel_of, step, rng, args.chunk)
        halves = [transition_counts(positions, offsets, weights, affine, dims, voxel_of, step, rng,
                                    args.chunk, half=h_) for h_ in (0, 1)]
        print(f"ground truth: {gt[1].sum():,.0f} transitions, {gt[2][2].sum():,.0f} transition pairs "
              f"({time.time() - t1:.0f} s)")

        model = TransitionModel(step_mm=step)
        t2 = time.time()
        res = model.fit(data)
        print(f"MITTENS: {time.time() - t2:.1f} s")

        interior = (res.neighbor_table >= 0).all(1)
        rows = interior & (gt[1].sum(1) >= args.min_count) \
            & (halves[0][1].sum(1) > 0) & (halves[1][1].sum(1) > 0)
        floor1 = l1(halves[0][0][rows], halves[1][0][rows]) / 2
        print(f"one step, P(u -> v), {rows.sum():,} voxels with >= {args.min_count} transitions:")
        summarize("isotropic null", l1(np.broadcast_to(res.null_single, (rows.sum(), N)), gt[0][rows]), floor1)
        summarize("single-ODF", l1(res.single[rows], gt[0][rows]), floor1)
        summarize("double-ODF", l1(np.nan_to_num(res.double_normalized[rows]), gt[0][rows]), floor1)

        n_vox = data.n_voxels
        enough = state_totals(gt[2], n_vox) >= args.min_count
        seen = (state_totals(halves[0][2], n_vox) > 0) & (state_totals(halves[1][2], n_vox) > 0)
        states = np.flatnonzero(enough & seen & np.repeat(interior, N))
        if len(states) == 0:
            print("two steps: no states with enough transition pairs; use more streamlines or a lower --min-count")
            continue
        u = states // N
        truth = state_rows(gt[2], states)
        floor2 = l1(state_rows(halves[0][2], states), state_rows(halves[1][2], states)) / 2
        print(f"two steps, P(v | w -> u), {len(u):,} states with >= {args.min_count} transition pairs:")
        summarize("first-order single (ignores w)", l1(res.single[u], truth), floor2)
        summarize("first-order double (ignores w)", l1(np.nan_to_num(res.double_normalized[u]), truth), floor2)


if __name__ == "__main__":
    main()
