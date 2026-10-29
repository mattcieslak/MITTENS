"""Voxel-graph tractography (Cieslak et al. 2018) on igraph.

Nodes are the voxels of a :class:`~mittens.model.TransitionResult`, in the
same order. Edge ``u -> v`` joins a voxel to one of its 26 neighbors and
carries a probability ``p`` and a length ``-log p``, so a shortest path is
the most probable sequence of voxel-to-voxel moves.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import cached_property

import igraph as ig
import nibabel as nib
import numpy as np

WEIGHTS = ("transition probability", "minus_iso", "minus_iso_scaled")


def edge_probabilities(result, model: str = "double", weights: str = "transition probability"):
    """``(source, target, p, null)`` for every voxel-to-neighbor edge with
    ``p > 0``; ``null`` is the isotropic-ODF probability of each of the 26
    neighbor directions.

    weights
        ``"transition probability"``: ``P(u -> v)``.
        ``"minus_iso"``: ``P(u -> v)`` minus the probability for an isotropic
        ODF. Edges at or below it are dropped.
        ``"minus_iso_scaled"``: ``"minus_iso"`` rescaled so that each voxel's
        edges sum to 1.
    """
    if model not in ("single", "double"):
        raise ValueError("model must be 'single' or 'double'")
    if weights not in WEIGHTS:
        raise ValueError(f"weights must be one of {WEIGHTS}")
    if model == "double":
        probs, null = result.double_normalized, result.null_double / result.null_double.sum()
    else:
        probs, null = result.single, result.null_single
    p = np.nan_to_num(probs, nan=0.0)
    if weights != "transition probability":
        p = np.clip(p - null, 0, None)
        if weights == "minus_iso_scaled":
            s = p.sum(axis=1, keepdims=True)
            p = np.divide(p, s, out=np.zeros_like(p), where=s > 0)
    nt = result.neighbor_table
    u, k = np.nonzero((nt >= 0) & (p > 0))
    return u, nt[u, k].astype(np.int64), p[u, k], np.asarray(null, dtype=np.float64)


def _neighbor_index(offsets: np.ndarray) -> np.ndarray:
    """Position of each index-space offset in ``neighbor_offsets()``."""
    code = (offsets[:, 0] + 1) * 9 + (offsets[:, 1] + 1) * 3 + (offsets[:, 2] + 1)
    return code - (code > 13)


@dataclass
class ShortestPaths:
    """Most probable paths between a region and every voxel.

    probability : product of the edge probabilities along each voxel's path
        (1 inside the region, 0 where there is no path).
    null_probability : the same product for an isotropic ODF everywhere.
    hops : number of moves on the path (0 inside the region, -1 where there
        is no path).
    parent : the next voxel toward the region (-1 inside the region or where
        there is no path).
    toward : True when paths run from each voxel to the region, False when
        they run from the region to each voxel.
    """

    probability: np.ndarray
    null_probability: np.ndarray
    hops: np.ndarray
    parent: np.ndarray
    toward: bool
    levels: list[np.ndarray] = field(repr=False)

    @property
    def reached(self) -> np.ndarray:
        return self.hops >= 0

    @property
    def mean_probability(self) -> np.ndarray:
        """Geometric mean of the edge probabilities along each path (1 inside
        the region)."""
        out = np.where(self.hops == 0, 1.0, 0.0)
        m = self.hops > 0
        out[m] = self.probability[m] ** (1.0 / self.hops[m])
        return out

    def path(self, node: int) -> np.ndarray:
        """Voxels on the path of ``node``, in the direction of travel."""
        if self.hops[node] < 0:
            return np.empty(0, dtype=np.int64)
        out = [int(node)]
        while self.parent[out[-1]] >= 0:
            out.append(int(self.parent[out[-1]]))
        return np.array(out if self.toward else out[::-1])

    def paths(self, nodes) -> list[np.ndarray]:
        """Paths of every reached voxel in ``nodes``."""
        return [self.path(v) for v in np.asarray(nodes).ravel() if self.hops[v] >= 0]

    def backprop(self, values) -> np.ndarray:
        """For each voxel, the largest of ``values`` over every voxel whose
        path passes through it, itself included."""
        out = np.where(self.reached, np.asarray(values, dtype=np.float64), 0.0)
        for nodes in self.levels[:0:-1]:
            np.maximum.at(out, self.parent[nodes], out[nodes])
        return out


class VoxelGraph:
    """Directed graph from voxels to their 26 neighbors.

    Edge ``i`` runs from ``source[i]`` to ``target[i]`` with probability
    ``p[i]``. ``graph`` is the same graph as an ``igraph.Graph`` (edge order
    preserved, no attributes); pass ``weights=vg.length`` or
    ``capacity=vg.p.tolist()`` to igraph algorithms.
    """

    def __init__(self, source, target, p, ijk, dims, affine, null, weights: str = "transition probability"):
        self.source = np.asarray(source, dtype=np.int64)
        self.target = np.asarray(target, dtype=np.int64)
        self.p = np.asarray(p, dtype=np.float64)
        self.ijk = np.asarray(ijk, dtype=np.int64)
        self.dims = tuple(int(d) for d in dims)
        self.affine = np.asarray(affine, dtype=np.float64)
        self.null = np.asarray(null, dtype=np.float64)
        self.weights = weights
        self.atlas: np.ndarray | None = None
        n = len(self.ijk)
        if not (len(self.source) == len(self.target) == len(self.p)):
            raise ValueError("source, target and p must have the same length")
        if len(self.source) and (max(self.source.max(), self.target.max()) >= n or min(self.source.min(), self.target.min()) < 0):
            raise ValueError("edge endpoints must be voxel indices")

    @classmethod
    def from_transitions(cls, result, model: str = "double", weights: str = "transition probability") -> "VoxelGraph":
        """Graph whose edge probabilities come from ``result`` (see
        :func:`edge_probabilities`)."""
        u, v, p, null = edge_probabilities(result, model, weights)
        return cls(u, v, p, result.ijk, result.dims, result.affine, null, weights)

    def __repr__(self) -> str:
        return f"VoxelGraph({self.n_voxels} voxels, {len(self.p)} edges, weights={self.weights!r})"

    @property
    def n_voxels(self) -> int:
        return len(self.ijk)

    @property
    def length(self) -> np.ndarray:
        return -np.log(self.p)

    @cached_property
    def graph(self) -> ig.Graph:
        return ig.Graph(n=self.n_voxels, edges=np.column_stack([self.source, self.target]), directed=True)

    # -- regions ----------------------------------------------------------

    def voxel_values(self, image) -> np.ndarray:
        """The value of ``image`` at every voxel. ``image`` is a NIfTI path or
        image on the ODF grid, an array of shape ``dims``, or one value per
        voxel."""
        if isinstance(image, (str, os.PathLike)):
            image = nib.load(image)
        if isinstance(image, nib.spatialimages.SpatialImage):
            if image.shape[:3] != self.dims or not np.allclose(image.affine, self.affine, atol=1e-4):
                raise ValueError("image is not on the ODF grid")
            image = np.asanyarray(image.dataobj)
        arr = np.asarray(image)
        if arr.shape[:1] == (self.n_voxels,) and arr.ndim == 1:
            return arr
        if arr.shape[:3] != self.dims:
            raise ValueError(f"expected shape {self.dims} or ({self.n_voxels},), got {arr.shape}")
        return arr[tuple(self.ijk.T)]

    def add_atlas(self, atlas) -> None:
        """Label every voxel from an integer atlas image, so regions can be
        given as labels."""
        self.atlas = np.rint(self.voxel_values(atlas)).astype(np.int64)

    def region_nodes(self, region) -> np.ndarray:
        """Voxels in ``region``: an atlas label, or a mask accepted by
        :meth:`voxel_values` (nonzero voxels)."""
        if isinstance(region, (int, np.integer)):
            if self.atlas is None:
                raise ValueError("integer regions need add_atlas() first")
            nodes = np.flatnonzero(self.atlas == region)
        else:
            nodes = np.flatnonzero(self.voxel_values(region))
        if len(nodes) == 0:
            raise ValueError("region contains no voxels of the graph")
        return nodes

    def mask(self, mask) -> "VoxelGraph":
        """The subgraph of voxels that are nonzero in ``mask``."""
        keep = self.voxel_values(mask).astype(bool)
        new_index = np.cumsum(keep) - 1
        e = keep[self.source] & keep[self.target]
        out = VoxelGraph(new_index[self.source[e]], new_index[self.target[e]], self.p[e], self.ijk[keep],
                         self.dims, self.affine, self.null, self.weights)
        if self.atlas is not None:
            out.atlas = self.atlas[keep]
        return out

    # -- shortest paths ---------------------------------------------------

    def shortest_paths_from(self, region) -> ShortestPaths:
        """The most probable path from ``region`` to every voxel."""
        return self._shortest_paths(self.region_nodes(region), toward=False)

    def shortest_paths_to(self, region) -> ShortestPaths:
        """The most probable path from every voxel to ``region``."""
        return self._shortest_paths(self.region_nodes(region), toward=True)

    def _shortest_paths(self, nodes: np.ndarray, toward: bool) -> ShortestPaths:
        # Dijkstra from a virtual node joined to the region by zero-length edges.
        n = self.n_voxels
        root = np.full(len(nodes), n)
        a = np.concatenate([self.source, nodes if toward else root])
        b = np.concatenate([self.target, root if toward else nodes])
        length = np.concatenate([self.length, np.zeros(len(nodes))])
        mode = "in" if toward else "out"
        g = self.graph.copy()
        g.add_vertices(1)
        g.add_edges(np.column_stack([a[len(self.p):], b[len(self.p):]]))
        dist = np.asarray(g.distances(source=n, weights=length, mode=mode)[0])

        # Keep the edges that lie on a shortest path; a breadth-first search
        # over them picks, for every voxel, the fewest-move shortest path.
        start, end = (b, a) if toward else (a, b)
        via = dist[start] + length
        tight = np.isfinite(via) & (np.abs(via - dist[end]) <= 1e-12 * np.maximum(1.0, dist[end]))
        tree = ig.Graph(n=n + 1, edges=np.column_stack([a[tight], b[tight]]), directed=True)
        vids, layers, parents = tree.bfs(n, mode=mode)
        vids, parent = np.asarray(vids), np.asarray(parents[:n])
        parent[(parent < 0) | (parent == n)] = -1
        levels = [vids[layers[i]:layers[i + 1]] for i in range(1, len(layers) - 1)]
        hops = np.full(n, -1)
        for h, level in enumerate(levels):
            hops[level] = h

        # Null probability of each voxel's move to or from its parent.
        moved = np.flatnonzero(parent >= 0)
        offsets = self.ijk[parent[moved]] - self.ijk[moved]
        with np.errstate(divide="ignore"):
            log_null = np.log(self.null)
        step = np.zeros(n)
        step[moved] = log_null[_neighbor_index(offsets if toward else -offsets)]
        acc = np.zeros(n)
        for level in levels[1:]:
            acc[level] = acc[parent[level]] + step[level]
        reached = hops >= 0
        return ShortestPaths(
            probability=np.where(reached, np.exp(-dist[:n]), 0.0),
            null_probability=np.where(reached, np.exp(acc), 0.0),
            hops=hops, parent=parent, toward=toward, levels=levels,
        )

    def atlas_connectivity(self, labels=None) -> dict[str, np.ndarray]:
        """Region-to-region scores from the most probable paths.

        Entry ``[i, j]`` describes the best path from region ``labels[i]`` to
        any voxel of region ``labels[j]``: ``probability`` and
        ``mean_probability`` take the largest value over those voxels and
        ``hops`` the fewest moves (-1 if none is reached).
        """
        if self.atlas is None:
            raise ValueError("call add_atlas() first")
        labels = np.unique(self.atlas[self.atlas > 0]) if labels is None else np.asarray(labels)
        members = [np.flatnonzero(self.atlas == label) for label in labels]
        r = len(labels)
        prob, mean, hops = np.zeros((r, r)), np.zeros((r, r)), np.full((r, r), -1)
        for i, nodes in enumerate(members):
            if len(nodes) == 0:
                continue
            sp = self._shortest_paths(nodes, toward=False)
            mp = sp.mean_probability
            for j, target in enumerate(members):
                h = sp.hops[target]
                if (h >= 0).any():
                    prob[i, j] = sp.probability[target].max()
                    mean[i, j] = mp[target].max()
                    hops[i, j] = h[h >= 0].min()
        return {"labels": labels, "probability": prob, "mean_probability": mean, "hops": hops}

    # -- graph measures ---------------------------------------------------

    def components(self) -> np.ndarray:
        """Weakly connected component of each voxel."""
        return np.asarray(self.graph.connected_components(mode="weak").membership)

    def betweenness(self, n_samples: int | None = None, seed: int = 0) -> np.ndarray:
        """Shortest-path betweenness of each voxel, with edge lengths
        ``-log p``. Exact when ``n_samples`` is None (all pairs; slow for a
        whole brain); otherwise estimated from that many random source voxels,
        following one shortest path per pair."""
        if n_samples is None:
            return np.asarray(self.graph.betweenness(directed=True, weights=self.length))
        rng = np.random.default_rng(seed)
        sources = rng.choice(self.n_voxels, size=min(n_samples, self.n_voxels), replace=False)
        total = np.zeros(self.n_voxels)
        for s in sources:
            sp = self._shortest_paths(np.array([s]), toward=False)
            # Paths from s run through every ancestor of their endpoint.
            below = sp.reached.astype(np.float64)
            for nodes in sp.levels[:0:-1]:
                np.add.at(below, sp.parent[nodes], below[nodes])
            dependency = np.where(sp.reached, below - 1.0, 0.0)
            dependency[s] = 0.0
            total += dependency
        return total * self.n_voxels / len(sources)

    def max_flow(self, source_region, sink_region) -> dict:
        """Maximum flow from one region to another with each edge's
        probability as its capacity; each region voxel sends or receives at
        most 1.

        Returns ``value``, the flow into (``inflow``) and out of
        (``outflow``) every voxel, and ``sink_side``: the voxels on the sink
        side of the minimum cut.
        """
        src, dst = self.region_nodes(source_region), self.region_nodes(sink_region)
        n = self.n_voxels
        g = self.graph.copy()
        g.add_vertices(2)
        g.add_edges(np.vstack([np.column_stack([np.full(len(src), n), src]),
                               np.column_stack([dst, np.full(len(dst), n + 1)])]))
        capacity = np.concatenate([self.p, np.ones(len(src) + len(dst))])
        f = g.maxflow(n, n + 1, capacity=capacity.tolist())
        flow = np.asarray(f.flow)[: len(self.p)]
        return {
            "value": float(f.value),
            "inflow": np.bincount(self.target, flow, minlength=n),
            "outflow": np.bincount(self.source, flow, minlength=n),
            "sink_side": np.asarray(f.membership[:n]) == 1,
        }

    # -- output -----------------------------------------------------------

    def to_nifti(self, values, path) -> None:
        """Write one value (or vector) per voxel as a NIfTI image on the ODF grid."""
        values = np.asarray(values)
        vol = np.zeros(self.dims + values.shape[1:], dtype=np.float32)
        vol[tuple(self.ijk.T)] = values
        nib.Nifti1Image(vol, self.affine).to_filename(str(path))

    def streamlines(self, paths) -> list[np.ndarray]:
        """World (RAS mm) coordinates of the voxel centers along each path."""
        return [nib.affines.apply_affine(self.affine, self.ijk[np.asarray(p, dtype=np.int64)]) for p in paths]

    def save_streamlines(self, paths, filename) -> None:
        """Write paths as a ``.trk`` or ``.tck`` tractogram."""
        from nibabel.streamlines import Field, Tractogram

        tractogram = Tractogram(self.streamlines(paths), affine_to_rasmm=np.eye(4))
        header = {
            Field.VOXEL_TO_RASMM: self.affine,
            Field.DIMENSIONS: np.array(self.dims, dtype=np.int16),
            Field.VOXEL_SIZES: np.sqrt((self.affine[:3, :3] ** 2).sum(0)).astype(np.float32),
            Field.VOXEL_ORDER: "".join(nib.aff2axcodes(self.affine)),
        }
        nib.streamlines.save(tractogram, str(filename), header=header)

    def save(self, path) -> None:
        """Save to a ``.npz`` file."""
        extra = {} if self.atlas is None else {"atlas": self.atlas}
        np.savez_compressed(path, source=self.source, target=self.target, p=self.p, ijk=self.ijk,
                            dims=np.array(self.dims), affine=self.affine, null=self.null,
                            weights=np.str_(self.weights), **extra)

    @classmethod
    def load(cls, path) -> "VoxelGraph":
        z = np.load(path)
        vg = cls(z["source"], z["target"], z["p"], z["ijk"], z["dims"], z["affine"], z["null"], str(z["weights"]))
        if "atlas" in z.files:
            vg.atlas = z["atlas"]
        return vg
