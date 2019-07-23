#!/usr/bin/env python
import networkx as nx
import numpy as np
import os
from scipy.io.matlab import loadmat, savemat
import logging
from tqdm import tqdm
import nibabel as nib
import os.path as op
import networkit
from time import time
from .spatial import Spatial, hdr
try:
    from matplotlib import pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False
from mittens.utils import unit_vector

DISCONNECTED = 999999999
logger = logging.getLogger(__name__)


class VoxelGraph(Spatial):
    def __init__(
            self,
            matfile="",
            # Probability calculation data
            step_size=None,
            angle_max=None,
            odf_resolution=None,
            weighting_scheme=None,
            angle_weights=None,
            angle_weighting_power=None,
            normalize_doubleODF=None,
            # Spatial mapping data
            flat_mask=None,
            nvoxels=None,
            real_affine=None,
            ras_affine=None,
            voxel_size=None,
            volume_grid=None,
            voxel_coords=None,
            coordinate_lut=None,
            graph=None):
        """
        Represents a voxel graph.

        Parameters:
        ===========
        matfile:str
          Path to a MATLAB .mat file created by constructing a voxel graph.
        step_size:float
          Step size in voxel units. Used for calculating transition probabilities
        angle_max:float
          Maximum turning angle in degrees. Used for calculating transition
          probabilities.
        odf_resolution:str
          ODF tesselation used in DSI Studio. Options are {"odf4", "odf6", "odf8"}.
        angle_weights:str
          Angle weighting scheme used while calculating transition probabilities
        angle_weighting_power:float
          Parameter used when an exponential weighting scheme is selected
        normalize_doubleODF:bool
          Should the transition probabilities from doubleODF be forced to sum to 1?

        real_affine:np.ndarray
        flat_mask:np.ndarray
        ras_affine:np.ndarray
        voxel_size:np.ndarray
        volume_grid:np.ndarray
        nvoxels:int

        voxel_coords:np.ndarray
        coordinate_lut:dict

        """
        self.step_size = step_size
        self.angle_max = angle_max
        self.odf_resolution = odf_resolution
        self.weighting_scheme = weighting_scheme
        self.angle_weights = angle_weights
        self.normalize_doubleODF = normalize_doubleODF
        self.angle_weighting_power = angle_weighting_power

        self.flat_mask = flat_mask
        self.nvoxels = nvoxels
        self.voxel_size = voxel_size
        self.real_affine = real_affine
        self.ras_affine = ras_affine
        self.volume_grid = volume_grid

        self.voxel_coords = voxel_coords
        self.coordinate_lut = coordinate_lut
        self.graph = graph

        # These are computed later
        self.atlas_labels = None
        self.label_lut = None
        self.background_image = None
        self.null_graph = None

        # Nodes that do not correspond to a voxel (eg source or sink)
        self.nonvoxel_nodes = set()
        self.undirected_component_ids = None

        if matfile:
            logger.info("Loading voxel graph from matfile")
            self._load_matfile(matfile)

    # IO
    def _load_matfile(self, matfile):
        if not os.path.exists(matfile):
            logger.critical("No such file: %s", matfile)
            return

        #try:
        m = loadmat(matfile)
        #except Exception, e:
        #    logger.critical("Unable to load %s:\n %s", matfile, e)
        #    return

        # Transition prob details
        self.step_size = float(m['step_size'].squeeze())
        self.angle_max = float(m['angle_max'].squeeze())
        self.odf_resolution = str(m['odf_resolution'].squeeze())
        self.weighting_scheme = str(m['weighting_scheme'].squeeze())
        self.angle_weights = str(m['angle_weights'].squeeze())
        self.normalize_doubleODF = bool(m['normalize_doubleODF'].squeeze())
        self.angle_weighting_power = float(m['angle_weighting_power'].squeeze())

        # Spatial mappings
        self.flat_mask = m['flat_mask'].squeeze().astype(np.bool)
        masked_voxels = self.flat_mask.sum()
        assert masked_voxels == m['nvoxels']
        self.ras_affine = m['ras_affine']
        self.real_affine = m.get('real_affine', self.ras_affine).squeeze()
        self.voxel_size = m['voxel_size'].squeeze()
        self.volume_grid = m['volume_grid'].squeeze().astype(np.int)

        # Guaranteed to have a flat mask by now
        self.nvoxels = masked_voxels
        # These coordinates are LPS+ voxels
        self.voxel_coords = np.array(
            np.unravel_index(np.flatnonzero(self.flat_mask), self.volume_grid,
                             order="F")).T
        self.coordinate_lut = dict(
            [(tuple(coord), n) for n, coord in enumerate(self.voxel_coords)])

        logger.info("Loading graph from matfile")
        sparse_graph = nx.from_scipy_sparse_matrix(
            m['graph'], create_using=nx.DiGraph(), edge_attribute="w")
        self.graph = networkit.nxadapter.nx2nk(sparse_graph, weightAttr="w")

    def mask(self, mask_image):
        """
        remove voxels that aren't 0 in ``mask_image``.
        Voxels that aren't already in the voxel graph but are
        nonzero in the mask image will not be added.
        """
        external_flat_mask = self._oriented_nifti_data(mask_image) > 0
        if np.all(external_flat_mask):
            logger.info("Mask does not remove any voxels")
            return

        masked_num = external_flat_mask.sum()
        logger.info("Reducing number of nodes from %d to %d", self.nvoxels, masked_num)
        self.graph = self.get_subgraph(external_flat_mask)
        # Update the coordinate tables
        self.flat_mask[self.flat_mask] = external_flat_mask
        self.nvoxels = masked_num
        self.voxel_coords = np.array(
            np.unravel_index(np.flatnonzero(self.flat_mask), self.volume_grid,
                             order="F")).T
        self.coordinate_lut = dict(
            [(tuple(coord), n) for n, coord in enumerate(self.voxel_coords)])

    def get_subgraph(self, mask_array):
        masked_num = mask_array.sum()
        new_graph = networkit.Graph(n=masked_num, directed=True, weighted=True)
        # Map old (full) node ids to new (subgraph) node ids
        full_to_sub = {}
        for sub, full in enumerate(np.flatnonzero(mask_array)):
            full_to_sub[full] = sub

        # Insert edges
        def insert_new_edges(from_node, to_node, edgeweight, edgeid):
            new_from_node = full_to_sub.get(from_node, -99)
            new_to_node = full_to_sub.get(to_node, -99)
            if -99 in (new_from_node, new_to_node):
                return
            new_graph.addEdge(new_from_node, new_to_node, edgeweight)

        for node in full_to_sub.keys():
            self.graph.forEdgesOf(node, insert_new_edges)

        return new_graph

    def add_background_image(self, image_file):
        self.background_image = self._oriented_nifti_data(image_file)

    def save(self, matfile):
        if self.graph is None:
            raise ValueError("No graph to save")
        if self.graph.numberOfNodes() > self.nvoxels:
            logger.warn("Non-voxel nodes are present in your graph.")

        logger.info("Converting networkit Graph to csr matrix")
        m = {
            "flat_mask": self.flat_mask,
            "nvoxels": self.nvoxels,
            "voxel_size": self.voxel_size,
            "volume_grid": self.volume_grid,
            "weighting_scheme": self.weighting_scheme,
            "step_size": self.step_size,
            "odf_resolution": self.odf_resolution,
            "angle_max": self.angle_max,
            "angle_weights": self.angle_weights,
            "normalize_doubleODF": self.normalize_doubleODF,
            "angle_weighting_power": self.angle_weighting_power,
            "graph": networkit.algebraic.adjacencyMatrix(self.graph, matrixType="sparse"),
            "ras_affine": self.ras_affine,
            "real_affine": self.real_affine
        }

        savemat(matfile, m, do_compression=True)
        logger.info("Saved matfile to %s", matfile)

    def use_affine_from(self, affine_image):
        self._set_real_affine(affine_image)

    # Region of Interest functions
    def add_atlas(self, atlas_nifti, min_voxels=1, connect_to_voxels=False):
        """ Inserts bidirectional connections between a spaceless "region node"
        and all the voxels it inhabits. Also stores ROI information for querying.
        """
        #if self.label_lut is not None:
        #   raise NotImplementedError("Cannot add multiple atlases yet")
        self.atlas_labels = self._oriented_nifti_data(atlas_nifti)
        self.label_lut = {}

        if connect_to_voxels:
            for region in self.label_lut.keys():
                pass

    def _get_region(self, region):
        # Find what nodes are part of a region
        if type(region) is str:
            labels = self._oriented_nifti_data(region)
            return np.flatnonzero(labels), op.split(op.abspath(region))[-1]
        nodes = np.flatnonzero(self.atlas_labels == region)
        return nodes, "region_%05d" % region

    def set_source(self, g, from_id, weight=0):
        connected_nodes = np.flatnonzero(self.atlas_labels == from_id)
        if (from_id in self.label_lut):
            label_node = self.label_lut[from_id]
        else:
            g.addNode()
            label_node = g.numberOfNodes() - 1
            self.label_lut[from_id] = label_node
        for connected_node in connected_nodes:
            g.addEdge(label_node, connected_node, w=weight)
        return label_node

    def add_source_region(self, region, weight=0):
        connected_nodes, name = self._get_region(region)
        if len(connected_nodes) == 0:
            raise ValueError("No nonzero nodes in the VoxelGraph")
        self.graph.addNode()
        label_node = self.graph.numberOfNodes() - 1
        self.nonvoxel_nodes.add(label_node)
        for connected_node in connected_nodes:
            self.graph.addEdge(label_node, connected_node, w=weight)
        return label_node

    def add_sink_region(self, region):
        nodes, name = self._get_region(region)
        if len(nodes) == 0:
            raise ValueError("No nonzero nodes in the VoxelGraph")
        self.graph.addNode()
        label_node = self.graph.numberOfNodes() - 1
        self.nonvoxel_nodes.add(label_node)
        for connected_node in connected_nodes:
            self.graph.addEdge(connected_node, label_node, w=weight)
        return label_node

    def set_sink(self, g, to_id, weight=0):
        connected_nodes = np.flatnonzero(self.atlas_labels == to_id)
        if (to_id in self.label_lut):
            label_node = self.label_lut[to_id]
        else:
            g.addNode()
            label_node = g.numberOfNodes() - 1
            self.label_lut[to_id] = label_node
        for connected_node in connected_nodes:
            if (g.hasEdge(label_node, connected_node)):
                g.removeEdge(label_node, connected_node)
            g.addEdge(connected_node, label_node, w=weight)
        return label_node

    def region_voxels_to_region_query(self,
                                      from_region,
                                      to_region,
                                      write_trk="",
                                      write_prob="",
                                      write_nifti="",
                                      write_wm_maxprob_map=""):
        """
        Query paths from one region to another. Parameters ``from_region``
        and ``to_region`` can be a path to a nifti file or a region ID
        if an atlas has been added through the ``add_atlas`` function.
        The probability of being connected to the ``to_region`` is calculated
        for every voxel in the ``from_region``.

        Parameters:
        ===========

        from_region:str,int
          Region from which the probability will be calculated for
          every voxel along the shortest path to any voxel in the
          ``to_region``.

        to_region:str,int
          The "sink" region.

        write_trk:str
          Suffix that will be added to the trackvis output. If empty a
          trackvis file won't be written.

        write_prob:str
          suffix for a txt file that contains the probability for each
          path. These files can be loaded into DSI Studio to color the
          streamlines.

        write_nifti:str
          Write the probabilities to a NIfTI file. Each voxel in ``from_region``
          will contain its probability.

        write_wm_maxprob_map:str
          Writes a map where each voxel contains the probability of the maximum
          probability path that goes through it.
        """
        # Get lists of nodes for the query
        from_nodes, from_name = self._get_region(from_region)
        to_nodes, to_name = self._get_region(to_region)

        # Loop over all the voxels in the from_region
        sink_label_node = self.set_source(self.graph, to_region)

        # Find the connected components
        undirected_version = self.graph.toUndirected()
        components = networkit.components.ConnectedComponents(undirected_version)
        components.run()
        logger.info("Found %d components in the graph", components.numberOfComponents())
        target_component = components.componentOfNode(sink_label_node)

        n = networkit.distance.Dijkstra(self.graph, sink_label_node)
        t0 = time()
        n.run()
        t1 = time()
        logger.info("Computed shortest paths in %.05f sec", t1 - t0)

        trk_paths = []
        probs = np.zeros(self.nvoxels, dtype=np.float)
        maxprobs = np.zeros(self.nvoxels, dtype=np.float)
        for node in tqdm(from_nodes):
            if components.componentOfNode(node) == target_component:
                path = n.getPath(node)
                if not len(path): continue
                score = self.get_path_probability(self.graph, path)
                probs[node] = score
                if write_trk:
                    trk_paths.append(path[1:-1])
                if write_wm_maxprob_map:
                    path = np.array(path)
                    path = path[path < self.nvoxels]
                    maxprobs[path] = np.maximum(maxprobs[path], score)

        # Write outputs
        if write_prob:
            g = open("%s_to_%s_%s.txt" % (from_name, to_name, write_trk), "w")
            for prob in probs:
                g.write("%.9f\n" % prob)
            g.close()
        if write_trk:
            if write_trk.endswith(".trk") or write_trk.endswith(".trk.gz"):
                self._write_trk(trk_paths, write_trk)
            elif write_trk.endswith("txt"):
                self._write_trk_txt(trk_paths, write_trk)
        if write_nifti:
            self.save_nifti(probs, write_nifti)
        if write_wm_maxprob_map:
            self.save_nifti(maxprobs, write_wm_maxprob_map)

        return trk_paths, probs

    def _write_trk(self, paths, trk_file_name):
        if not len(paths):
            logger.warning("Empty paths, not writing %s", trk_file_name)
            return
        header = hdr.copy()
        header['voxel_size'] = self.voxel_size.astype("<f4")
        header['dim'] = self.volume_grid.astype('<i2')
        trk_paths = [(self.voxel_coords[np.array(path[1:-1])] * self.voxel_size, None,
                      None) for path in paths]
        nib.trackvis.write(trk_file_name, trk_paths, header)

    def _write_trk_txt(self, paths, txt_file_name):
        if not len(paths):
            logger.warning("Empty paths, not writing %s", trk_file_name)
            return

        str_paths = []
        for path in trk_paths:
            str_paths.append(
                " ".join(["%d %d %d" % tuple(self.voxel_coords[step]) \
                        for step in path[1:-1]]))

        with open(txt_file_name, "w") as f:
            f.write("\n".join(str_paths))

    # Utility functions
    def get_edge_weights(self):
        return np.array([self.graph.weight(e[0], e[1]) for e in self.graph.edges()])

    def to_undirected_graph(self):
        if self.graph.isDirected():
            self.graph = self.graph.toUndirected()
        else:
            logger.warn("Graph is already undirected")

    # Shortest paths
    def Dijkstra(self, g, source, sink):
        d = networkit.distance.Dijkstra(g, source, target=sink)
        d.run()
        path = d.getPath(sink)
        return path

    def BottleneckShortestPath(self, g, source, sink):
        d = networkit.graph.BottleneckSP(g, source, target=sink)
        d.run()
        path = d.getPath(sink)
        return path

    def get_prob(self, g, path):
        prob = 1
        for step in range(len(path) - 1):
            prob *= np.e**(-g.weight(path[step], path[step + 1]))
        return prob

    def get_path_probability(self, g, path):
        prob = self.get_prob(g, path)
        prob = prob**(1. / len(path))
        return prob

    def _get_path_weights(self, g, path):
        return np.array(
            [g.weight(path[step], path[step + 1]) for step in range(len(path) - 1)])

    def get_bottleneck_scores_for_path(self, g, path):
        """
        Extracts weights along a path in a graph and returns the
        cumulative weight and the mean weight along the path
        """
        if self.weighting_scheme is None:
            raise ValueError("Build a graph first")

        # Get weights out of the graph
        path_len = len(path)
        path_values = self._get_path_weights(g, path)
        if DISCONNECTED in path_values:
            return 0, 0

        # Negate and exponentiate the weights to get back probabilities
        if "negative_log" in self.weighting_scheme:
            prob = np.exp(-np.max(path_values))
            return prob, prob**(1. / path_len)
        # return the product and mean along the path
        return np.min(path_values), np.mean(path_values)

    def get_scores_for_path(self, g, path):
        """
        Extracts weights along a path in a graph and returns the
        cumulative weight and the mean weight along the path
        """
        if self.weighting_scheme is None:
            raise ValueError("Build a graph first")

        # Get weights out of the graph
        path_len = len(path)
        path_values = self._get_path_weights(g, path)
        if DISCONNECTED in path_values:
            return 0, 0

        # Negate and exponentiate the weights to get back probabilities
        if "negative_log" in self.weighting_scheme:
            prob = np.exp(np.sum(-path_values))
            return prob, prob**(1. / path_len)
        # return the product and mean along the path
        return np.prod(path_values), np.mean(path_values)

    def get_maximum_spanning_forest(self):
        forest = networkit.graph.UnionMaximumSpanningForest(self.graph)
        forest.run()
        return forest.getUMSF()

    def add_null_graph(self, null_graph):
        """
        Specifies a graph that can be used for comparisons in certain functions

        Parameters:
        ===========

        null_graph:mittens.VoxelGraph
          The VoxelGraph object that will be kept around for comparisons

        """

        if not null_graph.nvoxels == self.nvoxels and np.all(
                self.flat_mask == null_graph.flat_mask):
            raise ValueError("Null Graph must have the same number of voxels")

        self.null_graph = null_graph

    # Voxel value functions
    def backprop_prob_ratio_map(self, source_region, use_bottleneck=False):
        """
        Calculates the highest probability ratio of any shortest path going throungh each voxel

        Parameters:
        ===========
        source_region:str or int
          Either the integer ROI ID after ``add_atlas`` has been called, or a path
          to a NIfTI file where nonzero values are the source voxels.

        Returns:
        ========
        None

        """
        if self.null_graph is None:
            raise ValueError("A null graph must be set using VoxelGraph.add_null_graph()")

        starting_region, region_name = self._get_region(source_region)
        source_label_node = self.set_source_using_nifti(self.graph, starting_region)

        # Find the connected components
        undirected_version = self.graph.toUndirected()
        components = networkit.components.ConnectedComponents(undirected_version)
        components.run()
        #logger.info("Found %d components in the graph", components.numberOfComponents())
        target_component = components.componentOfNode(source_label_node)

        # Run the modified Dijkstra algorithm from networkit
        n = networkit.distance.Dijkstra(self.graph, source_label_node)
        n.run()
        #logger.info("Computed shortest paths")

        # Collect the shortest paths to each voxel and calculate a score
        raw_scores = np.zeros(self.nvoxels, dtype=np.float)
        null_scores = np.zeros(self.nvoxels, dtype=np.float)
        path_lengths = np.zeros(self.nvoxels, dtype=np.float)
        backprop = np.zeros(self.nvoxels, dtype=np.float)
        for node in tqdm(np.arange(self.nvoxels)):
            print(node)
            if components.componentOfNode(node) == target_component:
                path = n.getPath(node)
                if len(path):
                    raw_scores[node] = self.get_prob(self.graph, path)
                    null_scores[node] = self.get_prob(self.null_graph, path)
                    prob_ratio = raw_scores[node] / null_scores[node]
                    path = np.array(path)
                    path = path[path < self.nvoxels]
                    backprop[path] = np.maximum(backprop[path], prob_ratio)
                else:
                    logger.info("Shortest path map didn't find a path")
                path_lengths[node] = len(path)
        return raw_scores, null_scores, path_lengths, backprop

    def remove_nonvoxel_nodes(self):
        def rm_out(in_node, out_node, a, b):
            self.graph.removeEdge(in_node, out_node)

        def rm_in(out_node, in_node, a, b):
            self.graph.removeEdge(in_node, out_node)

        while len(self.nonvoxel_nodes):
            node = self.nonvoxel_nodes.pop()
            self.graph.forEdgesOf(node, rm_out)
            self.graph.forInEdgesOf(node, rm_in)
            self.graph.removeNode(node)

    def shortest_path_map(self,
                          source_region,
                          use_bottleneck=False,
                          back_propagate_scores=True):
        """
        Calculates the shortest paths and their scores from ``source_region`` to all
        other nodes (voxels) in the graph.

        Parameters:
        ===========
        source_region:str or int
          Either the integer ROI ID after ``add_atlas`` has been called, or a path
          to a NIfTI file where nonzero values are the source voxels.

        nifti_prefix:str
          Path to where outputs will be written. This prefix will have _something.nii.gz
          appended to it.

        Returns:
        ========
        raw_scores:np.ndarray
                nifti_prefix + "_shortest_path_score%s.nii.gz"%suffix)

        scores:np.ndarray
                nifti_prefix + "_shortest_path_mean_score%s.nii.gz"%suffix)

        path_lengths:np.ndarray
                nifti_prefix + "_shortest_path_length%s.nii.gz"%suffix)

        backprop_scores:np.ndarray
                nifti_prefix + "_shortest_path_backprop%s.nii.gz"%suffix)
        """
        self.remove_nonvoxel_nodes()
        starting_region, region_name = self._get_region(source_region)
        source_label_node = self.add_source_region(source_region)
        # Re-calculate components with the source node included
        self.update_component_ids()
        target_component = self.undirected_component_ids[source_label_node]

        # Run the modified Dijkstra algorithm from networkit
        if use_bottleneck:
            raise NotImplementedError()
        else:
            n = networkit.distance.Dijkstra(self.graph, source_label_node)
            score_func = self.get_scores_for_path
        t0 = time()
        n.run()
        t1 = time()
        logger.info("Computed shortest paths in %.05f sec", t1 - t0)

        # Collect the shortest paths to each voxel and calculate a score
        scores = np.zeros(self.nvoxels, dtype=np.float)
        raw_scores = np.zeros(self.nvoxels, dtype=np.float)
        path_lengths = np.zeros(self.nvoxels, dtype=np.float)
        backprop = np.zeros(self.nvoxels, dtype=np.float)
        for node in np.arange(self.nvoxels):
            if self.undirected_component_ids[node] == target_component:
                path = n.getPath(node)
                if len(path):
                    raw_scores[node], scores[node] = score_func(self.graph, path)
                    if back_propagate_scores:
                        path = np.array(path)
                        path = path[path < self.nvoxels]
                        backprop[path] = np.maximum(backprop[path], scores[node])
                else:
                    logger.info("Shortest path map didn't find a path")
                path_lengths[node] = len(path)

        # Clean up afterwards
        self.remove_nonvoxel_nodes()
        if back_propagate_scores:
            return raw_scores, scores, path_lengths, backprop
        return raw_scores, scores, path_lengths

    def update_component_ids(self):
        self.undirected_component_ids = self.get_undirected_node_component_ids()

    def get_undirected_node_component_ids(self):
        undir_graph = self.graph.toUndirected()
        components = networkit.components.ConnectedComponents(undir_graph)
        components.run()
        return np.array([components.componentOfNode(node) for \
                         node in range(self.graph.numberOfNodes())])

    def voxelwise_approx_betweenness(self, nSamples=500, normalized=True, parallel=True):
        btw = \
            networkit.centrality.ApproxBetweenness2(self.graph, nSamples=nSamples,
                    normalized=normalized,parallel=parallel)
        btw.run()
        return np.array(btw.scores())

    # Capacity
    def flow(self, mask_image, to_id, from_id, fname):
        self.build_graph(weighting_scheme="transition probability", doubleODF=True)
        self.add_atlas(mask_image)

        source_node = self.set_source(self.graph, to_id, weight=1)
        sink_node = self.set_sink(self.graph, from_id, weight=1)

        self.graph.indexEdges()

        f = networkit.flow.EdmondsKarp(self.graph, source_node, sink_node)
        f.run()

        max_flow = f.getMaxFlow()
        flow_vec = np.array(f.getFlowVector())
        scores_incoming = np.zeros(self.nvoxels)
        scores_outgoing = np.zeros(self.nvoxels)
        for i, edge in enumerate(self.graph.edges()):
            if (edge[0] >= self.nvoxels or edge[1] >= self.nvoxels):
                continue
            scores_incoming[edge[1]] += flow_vec[i]
            scores_outgoing[edge[0]] += flow_vec[i]
        self.save_nifti(scores_incoming, fname + "_incoming.nii.gz")
        self.save_nifti(scores_outgoing, fname + "_outoging.nii.gz")

        source_set = np.array(f.getSourceSet())
        red_nodes = source_set
        blue_nodes = np.setdiff1d(np.arange(self.nvoxels), red_nodes)

        outputs = np.zeros(self.nvoxels)
        outputs[red_nodes[red_nodes < self.nvoxels]] = 1
        outputs[blue_nodes[blue_nodes < self.nvoxels]] = 2

        self.save_nifti(outputs, fname + "_split.nii.gz")
        return max_flow

    def katz_centrality(self, alpha=5e-4, beta=0.1, tol=1e-8):
        """
        Works with directed or undirected graphs.

        Parameters:
        ===========
        alpha:float
          asdfadsf

        beta:float

        tol:float

        Returns:
        ========

        scores:np.ndarray
          Katz centrality scores

        ranks:
        """
        katz = networkit.centrality.KatzCentrality(self.graph)
        katz.run()
        scores = np.array(katz.scores())
        rankings = np.array(katz.ranking())
        ranks = np.zeros(self.nvoxels, dtype=np.float)
        ranks[rankings[:, 0].astype(np.int)] = rankings[:, 1]

        return scores, ranks

    def build_atlas_graph(self, full_region_ids=None):
        """
        After adding an atlas using VoxelGraph.add_atlas() this function
        calculates inter-regional connectivity.
        """
        if self.atlas_labels is None or not len(self.atlas_labels) == self.nvoxels:
            raise ValueError("No atlas labels are available")

        # Set up the label set
        available_labels = np.unique(self.atlas_labels[self.atlas_labels > 0])
        if full_region_ids is not None:
            full_region_ids.sort()
            labels_to_calculate = np.intersect1d(available_labels, full_region_ids)
            if not len(labels_to_calculate) == len(full_region_ids):
                logger.warning("full_region_ids does not match available labels,"
                               "this can result in a disconnected node")
        else:
            full_region_ids = available_labels
            labels_to_calculate = available_labels

        # Build the output graph objects
        asym_raw_prob_graph = networkit.Graph(
            n=len(full_region_ids), directed=True, weighted=True)
        asym_mean_prob_graph = networkit.Graph(
            n=len(full_region_ids), directed=True, weighted=True)
        asym_path_length_graph = networkit.Graph(
            n=len(full_region_ids), directed=True, weighted=True)
        conj_raw_prob_graph = networkit.Graph(
            n=len(full_region_ids), directed=False, weighted=True)
        conj_mean_prob_graph = networkit.Graph(
            n=len(full_region_ids), directed=False, weighted=True)

        # Calculate spaci maps for each region
        outgoing_spaci = {}
        for node_id, region_label in enumerate(full_region_ids):
            if region_label not in available_labels:
                continue
            raw_scores, scores, path_lengths = self.shortest_path_map(
                source_region=region_label, back_propagate_scores=False)
            outgoing_spaci[region_label] = {
                'raw_probs': raw_scores,
                'mean_probs': scores,
                'path_length': path_lengths
            }
        # Get masks for each region
        region_masks = {}
        for region_label in full_region_ids:
            region_masks[region_label] = self.atlas_labels == region_label

        # Add in the edge weights
        for from_node_id, from_region_label in tqdm(enumerate(full_region_ids)):
            from_mask = region_masks[from_region_label]
            from_spaci = outgoing_spaci[from_region_label]
            from_raw_probs = from_spaci['raw_probs']
            from_mean_probs = from_spaci['mean_probs']
            from_path_length = from_spaci['path_length']

            for to_node_id, to_region_label in enumerate(full_region_ids):
                to_mask = region_masks[to_region_label]
                # Extract values from voxels in 'to mask'
                asym_raw_prob_graph.addEdge(from_node_id, to_node_id,
                                            from_raw_probs[to_mask].max())
                asym_mean_prob_graph.addEdge(from_node_id, to_node_id,
                                             from_mean_probs[to_mask].max())
                asym_path_length_graph.addEdge(from_node_id, to_node_id,
                                               from_path_length[to_mask].min())

                # Get maps from the "to node"
                to_spaci = outgoing_spaci[to_region_label]
                to_raw_probs = to_spaci['raw_probs']
                to_mean_probs = to_spaci['mean_probs']
                # Get values for conjunction of "from node" and "to node"
                conj_raw_probs = from_raw_probs * to_raw_probs
                conj_raw_prob_graph.addEdge(from_node_id, to_node_id,
                                            conj_raw_probs[to_mask].max())
                conj_mean_probs = from_mean_probs * to_mean_probs
                conj_mean_prob_graph.addEdge(from_node_id, to_node_id,
                                             conj_mean_probs[to_mask].max())

        return asym_raw_prob_graph, asym_mean_prob_graph, asym_path_length_graph, conj_raw_prob_graph, conj_mean_prob_graph

    # Plotting
    def _plot_slice(self,
                    node_data,
                    slice_axis="y",
                    slice_func=np.max,
                    zero_ones=True,
                    invisible_zeros=True,
                    ax=None,
                    show_colorbar=True):

        if not has_matplotlib:
            logger.warning("Matplotlib not available")
            return

        if zero_ones:
            data = node_data.copy()
            data[data == 1] = 0.000001
        else:
            data = node_data

        if invisible_zeros:
            data[data == 0] = np.nan
        # Turn values into 3d
        as_mat = 1 / np.zeros_like(vgraph.flat_mask)
        as_mat[vgraph.flat_mask] = data
        as_mat = as_mat.reshape(vgraph.volume_grid, order="F")[::-1, ::-1]

        if ax is None:
            fig, ax = plt.subplots()

        if slice_axis == "x":
            data2d = slice_func(as_mat, axis=0)
        elif slice_axis == "y":
            data2d = slice_func(as_mat, axis=1)
        elif slice_axis == "z":
            data2d = slice_func(as_mat, axis=2)
        else:
            raise ValueError

        ax.imshow(
            data2d,
            interpolation="nearest",
            cmap="viridis",
            origin="lower",
            norm=LogNorm())
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_vectors(self,
                     background_data=None,
                     slice_axis="y",
                     slice_num=0,
                     ax=None,
                     show_colorbar=True,
                     arrow_zoom=2,
                     xlim=None,
                     ylim=None,
                     min_weight=0,
                     plot_voxel_boundaries=True,
                     quiveropts=dict(
                         headlength=0,
                         pivot='tail',
                         scale=1,
                         linewidth=.5,
                         units='xy',
                         width=.045,
                         headwidth=1,
                         cmap="viridis",
                         angles="xy",
                         scale_units='xy',
                         clim=(0, 0.35))):
        """
        Produces a quiver plot for a selected slice of the 3D volume. Probably not very fast.
        """
        if not has_matplotlib:
            logger.warning("Matplotlib not available")
            return

        if ax is None:
            fig, ax = plt.subplots()

        # If background data is provided, do an image plot in grayscale
        if background_data is not None:
            # Turn values into 3d
            as_mat = 1 / np.zeros_like(vgraph.flat_mask)
            as_mat[vgraph.flat_mask] = data
            as_mat = as_mat.reshape(vgraph.volume_grid, order="F")[::-1, ::-1]
            if slice_axis == "x":
                data2d = slice_func(as_mat, axis=0)
            elif slice_axis == "y":
                data2d = slice_func(as_mat, axis=1)
            elif slice_axis == "z":
                data2d = slice_func(as_mat, axis=2)
            else:
                raise ValueError
            ax.imshow(data2d, interpolation="nearest", cmap="gray", origin="lower")

        # Select only nodes within the slice
        if slice_axis == "x":
            selected_nodes = np.flatnonzero(self.voxel_coords[:, 0] == slice_num)
            selected_coords = self.voxel_coords[selected_nodes][:, np.array([1, 2])]
        elif slice_axis == "y":
            selected_nodes = np.flatnonzero(self.voxel_coords[:, 1] == slice_num)
            selected_coords = self.voxel_coords[selected_nodes][:, np.array([0, 2])]
        elif slice_axis == "z":
            selected_nodes = np.flatnonzero(self.voxel_coords[:, 2] == slice_num)
            selected_coords = self.voxel_coords[selected_nodes][:, np.array([0, 1])]
        else:
            raise ValueError

        max_scale_fac = quiveropts['clim'][1]
        coords = []
        vecs = []
        weights = []
        for from_node_index, from_node in enumerate(selected_nodes):
            for to_node_index, to_node in enumerate(selected_nodes):
                if from_node == to_node or not self.graph.hasEdge(from_node, to_node):
                    continue
                weight = np.exp(-self.graph.weight(from_node, to_node))
                if weight < min_weight: continue
                weights.append(weight)
                # Which direction should the arrow be pointing?
                dir_vec = unit_vector(selected_coords[to_node_index] -
                                      selected_coords[from_node_index])
                scale_fac = min(max_scale_fac, weight * arrow_zoom)
                coords.append(selected_coords[from_node_index])
                vecs.append(dir_vec * scale_fac)
        x, y = np.array(coords).T + 0.5  # center in the voxel
        u, v = np.array(vecs).T
        c = np.array(weights)

        if plot_voxel_boundaries:
            ax.set_xticks(np.arange(int(x.max() + 1)))
            ax.set_yticks(np.arange(int(y.max() + 1)))
            ax.grid(b=True, color="k", lw=0.5, linestyle="-")

        ax.quiver(x, y, u, v, c, **quiveropts)
        ax.grid(False)
        plt.axis("equal")
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
