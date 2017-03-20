#!/usr/bin/env python
from .fib_io import load_fibgz
import numpy as np
import re
import importlib
from .utils import (neighbor_names, get_transition_analysis_matrices, 
        lps_neighbor_shifts, weight_transition_probabilities_by_odf, 
        compute_weights_as_neighbor_voxels, ras_neighbor_shifts)
import logging
from tqdm import tqdm
from .distances import (kl_distance, aitchison_distance, kl_asymmetry, 
                        aitchison_asymmetry)
import nibabel as nib
import os.path as op
import networkit
from heapq import heappush, heappop
import pickle
logger = logging.getLogger(__name__)
opposites = [
    ("r","l"),
    ("a","p"),
    ("i","s"),
    ("ri","ls"),
    ("rs","li"),
    ("ra","lp"),
    ("rp","la"),
    ("as","pi"),
    ("ai","ps"),
    ("ras","lpi"),
    ("rps","lai"),
    ("rai","lps"),
    ("rpi","las")
    ]

hdr = np.array((b'TRACK', [ 98, 121, 121], [ 2.,  2.,  2.], [ 0.,  0.,  0.], 0, [b'', b'', b'', b'', b'', b'', b'', b'', b'', b''], 0, [b'', b'', b'', b'', b'', b'', b'', b'', b'', b''], [[-2.,  0.,  0.,  0.], [ 0., -2.,  0.,  0.], [ 0.,  0.,  2.,  0.], [ 0.,  0.,  0.,  1.]], b' A diffusion spectrum imaging scheme was used, and a total of 257 diffusion sampling were acquired. The maximum b-value was 4985 s/mm2. The in-plane resolution was 2 mm. The slice thickness was 2 mm. The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010) with a diffusion sampling length ratio of 1.25.\nA deterministic fiber tracking algorithm (Yeh et al., PLoS ONE 8(11): e80713', b'LPS', b'LPS', [ 1.,  0.,  0.,  0.,  1.,  0.], b'', b'', b'', b'', b'', b'', b'', 253, 2, 1000),
              dtype=[('id_string', 'S6'), ('dim', '<i2', (3,)), ('voxel_size', '<f4', (3,)), ('origin', '<f4', (3,)), ('n_scalars', '<i2'), ('scalar_name', 'S20', (10,)), ('n_properties', '<i2'), ('property_name', 'S20', (10,)), ('vox_to_ras', '<f4', (4, 4)), ('reserved', 'S444'), ('voxel_order', 'S4'), ('pad2', 'S4'), ('image_orientation_patient', '<f4', (6,)), ('pad1', 'S2'), ('invert_x', 'S1'), ('invert_y', 'S1'), ('invert_z', 'S1'), ('swap_xy', 'S1'), ('swap_yz', 'S1'), ('swap_zx', 'S1'), ('n_count', '<i4'), ('version', '<i4'), ('hdr_size', '<i4')])


class MITTENS(object):
    def __init__(self, fibgz_file="", nifti_prefix="", step_size=np.sqrt(3)/2. , angle_max=35,
            odf_resolution="odf8", angle_weights="flat",
            angle_weighting_power=1.):
        if fibgz_file == nifti_prefix == "":
            raise ValueError("Must provide either a DSI Studio fib file or prefix to "
                    "NIfTI1 images written out by a previous run")
        # These will get filled out from loading a fibgz or niftis
        self.flat_mask = None
        self.nvoxels = None
        self.voxel_coords = None
        self.coordinate_lut = None
        self.label_lut = None
        self.voxel_graph = None
        self.UMSF = None
        self.atlas_labels = None
        self.weighting_scheme = None
        # From args
        self.step_size = step_size
        self.odf_resolution = odf_resolution
        self.angle_max = angle_max
        self.orientation = None
        self.angle_weights = angle_weights
        self.angle_weighting_power = angle_weighting_power
        logger.info("\nUsing\n------\n  Step Size:\t\t%.4f Voxels \n  ODF Resolution:\t"
                "%s\n  Max Angle:\t\t%.2f Degrees\n  Orientation:\t\t%s\n"
                "  Angle Weights:\t%s\n  Angle weight power:\t%.1f",
                self.step_size, self.odf_resolution, self.angle_max, 
                self.orientation, self.angle_weights,self.angle_weighting_power)
        # Get matrices we'll need for analysis
        self.odf_vertices, self.prob_angles_weighted = \
                get_transition_analysis_matrices(self.odf_resolution, self.angle_max,
                        self.angle_weights, self.angle_weighting_power)
        self.n_unique_vertices = self.odf_vertices.shape[0]//2
        if fibgz_file:
            logger.info("Loading DSI Studio fib file")
            self._load_fibgz(fibgz_file)
        if nifti_prefix:
            logger.info("Loading output from pre-existing NIfTIs")
            self._load_niftis(nifti_prefix)

        self._initialize_nulls()

    def _initialize_nulls(self):

        # Note, only entries for unique vertices are created, but they 
        # are divided by the total number of vertices.
        self.isotropic = np.ones(self.n_unique_vertices,dtype=np.float64) / \
                self.odf_vertices.shape[0]
        
        # None-ahead Model
        self.singleODF_funcs = self.get_prob_funcs("singleODF")
        singleODF_null_probs = []
        for k in neighbor_names:
            singleODF_null_probs.append( self.singleODF_funcs[k](
                    self.isotropic, self.prob_angles_weighted))
        self.singleODF_null_probs = np.array(singleODF_null_probs)
        #if not self.singleODF_null_probs.sum() == 1.:
        #    raise ValueError("Null probailities do not add up to 1. Check Fortran")

        # One-ahead Model
        self.doubleODF_funcs = self.get_prob_funcs("doubleODF")
        doubleODF_null_probs = []
        isotropic_x2 = compute_weights_as_neighbor_voxels(
            self.isotropic[np.newaxis,:], self.prob_angles_weighted).squeeze()
        for k in neighbor_names:
            doubleODF_null_probs.append( self.doubleODF_funcs[k](
                    self.isotropic, isotropic_x2, self.prob_angles_weighted))
        self.doubleODF_null_probs = np.array(doubleODF_null_probs)
        self.doubleODF_null_probs[np.isnan(self.doubleODF_null_probs)]= 0 
        self.doubleODF_null_probs = self.doubleODF_null_probs / self.doubleODF_null_probs.sum()

    def _load_fibgz(self, path):
        logger.info("Loading %s", path)
        f = load_fibgz(path)
        logger.info("Loaded %s", path)
        self.orientation = "lps"
        # Check that this fib file matches what we expect
        fib_odf_vertices = f['odf_vertices'].T
        matches = np.allclose(self.odf_vertices, fib_odf_vertices)
        if not matches:
            logger.critical("ODF Angles in fib file do not match %s", self.odf_resolution)
            return

        # Extract the spacing info from the fib file
        self.volume_grid = f['dimension'].squeeze()
        aff = np.ones(4,dtype=np.float)
        aff[:3] = f['voxel_size'].squeeze()
        # DSI Studio stores data in LPS+
        #aff = aff * np.array([-1,-1,1,1])
        self.ras_affine = np.diag(aff)

        # Coordinate mapping information from fib file
        self.flat_mask = f["fa0"].squeeze() > 0
        self.nvoxels = self.flat_mask.sum()
        self.voxel_coords = np.array(np.unravel_index(
            np.flatnonzero(self.flat_mask), self.volume_grid, order="F")).T
        self.coordinate_lut = dict(
            [(tuple(coord), n) for n,coord in enumerate(self.voxel_coords)])

        # Create a contiguous ODF matrix, skipping all zero rows
        logger.info("Loading ODF data")
        odf_vars = [k for k in f.keys() if re.match("odf\\d+",k)]
        valid_odfs = []
        for n in range(len(odf_vars)):
            varname = "odf%d" % n
            odfs = f[varname]
            odf_sum = odfs.sum(0)
            odf_sum_mask = odf_sum > 0
            valid_odfs.append(odfs[:,odf_sum_mask].T)

        self.odf_values = np.row_stack(valid_odfs).astype(np.float64)
        self.odf_values = self.odf_values / self.odf_values.sum(1)[:,np.newaxis] * 0.5
        logger.info("Loaded ODF data: %s",str(self.odf_values.shape))

    def get_prob_funcs(self, order="singleODF"):

        requested_module = "%s_%s_ss%.2f_am%d" % (order, self.odf_resolution, 
                self.step_size, self.angle_max)
        requested_module = re.sub( "\.", "_", requested_module)
        module = importlib.import_module("mittens.fortran." + requested_module)

        # Function lookup for neighbors
        neighbor_functions = {}
        for nbr_name in neighbor_names:
            # Flip the x,y to match dsi studio if lps
            if self.orientation == "lps":
                if "r" in nbr_name:
                    rl = nbr_name.replace("r", "l")
                elif "l" in nbr_name:
                    rl = nbr_name.replace("l", "r")
                else:
                    rl = nbr_name

                if "a" in rl:
                    ap = nbr_name.replace("a","p")
                elif "p" in rl:
                    ap = nbr_name.replace("p", "a")
                else:
                    ap = rl
                neighbor_functions[ap] = getattr(module, ap+"_prob")
            else:
                neighbor_functions[nbr_name] = getattr(module, nbr_name+"_prob")

        return neighbor_functions

    def get_empty_outputs(self):
        return dict([(name,np.zeros(self.nvoxels,dtype=np.float)) for \
                name in neighbor_names])

    def save_nifti(self, data, fname):
        out_data = np.zeros(np.prod(self.volume_grid),dtype=np.float)
        out_data[self.flat_mask] = data
        out_data = out_data.reshape(self.volume_grid, order="F")[::-1,::-1,:]
        nib.Nifti1Image(out_data.astype(np.float32), self.ras_affine
                ).to_filename(fname)

    def estimate_singleODF(self, output_prefix=""):
        
        # Output matrix
        outputs = np.zeros((self.nvoxels,len(neighbor_names)),dtype=np.float)

        # Run it on each voxel
        for n, odf in enumerate(self.odf_values):
            prob_angles = weight_transition_probabilities_by_odf(
                    odf, self.prob_angles_weighted)
            if n%10000 == 0:
                logger.info("ODF %d/%d", n, self.nvoxels)
            for m, nbr_name in enumerate(neighbor_names):
                outputs[n,m] = self.singleODF_funcs[nbr_name](
                    odf, prob_angles)

        self.singleODF_results = outputs
        logger.info("Calculating None-Ahead CoDI")
        self.singleODF_codi = aitchison_distance(self.singleODF_null_probs, self.singleODF_results)
        logger.info("Calculating Order1 KL Distance")
        self.singleODF_kl = kl_distance(self.singleODF_null_probs, self.singleODF_results)

        # Write outputs if requested
        if output_prefix:
            logger.info("Writing singleODF results")
            for a in neighbor_names:
                outf = output_prefix + "_singleODF_%s_prob.nii.gz" % a
                logger.info("Writing %s", outf)
                self.save_nifti(
                        self.singleODF_results[:,neighbor_names.index(a)], outf)
            self.save_nifti(self.singleODF_kl, output_prefix + "_singleODF_KL.nii.gz")
            self.save_nifti(self.singleODF_codi, output_prefix + "_singleODF_CoDI.nii.gz")

    def estimate_doubleODF(self, output_prefix=""):
        
        logger.info("Pre-computing neighbor angle weights")
        Ypc = compute_weights_as_neighbor_voxels(
                self.odf_values, (self.prob_angles_weighted > 0).astype(np.float))

        # Output matrix
        outputs = np.zeros((self.nvoxels,len(neighbor_names)),dtype=np.float)
        if self.orientation == "lps":
            neighbor_shifts = lps_neighbor_shifts
        elif self.orientation == "ras":
            neighbor_shifts = ras_neighbor_shifts
        else:
            raise ValueError

        def check_coord(coord):
            return np.all(
                    [tuple(coord + shift) in self.coordinate_lut \
                           for shift in neighbor_shifts.values()  ])

        # Run it on each voxel
        for n, odf in enumerate(self.odf_values):
            if n%10000 == 0:
                logger.info("ODF %d/%d", n, self.nvoxels)
            coordinate = self.voxel_coords[n]
            if not check_coord(coordinate): continue
            prob_angles = weight_transition_probabilities_by_odf(
                    odf, self.prob_angles_weighted)
            for m, nbr_name in enumerate(neighbor_names):
                neighbor_coord_num = self.coordinate_lut[tuple(coordinate + \
                        neighbor_shifts[nbr_name])]
                outputs[n,m] = self.doubleODF_funcs[nbr_name](
                        odf, Ypc[neighbor_coord_num], prob_angles)
        self.doubleODF_results = outputs / np.nansum(outputs, 1)[:,np.newaxis]

        # Calculate the distances
        logger.info("Calculating One-Ahead CoDI")
        self.doubleODF_codi = aitchison_distance(self.doubleODF_null_probs, self.doubleODF_results)

        # Divide the Columns into two matrices, calculate asymmetry
        half1 = []
        half2 = []
        for a,b in opposites:
            half1.append(self.doubleODF_results[:,neighbor_names.index(a)])
            half2.append(self.doubleODF_results[:,neighbor_names.index(b)])
        half1 = np.column_stack(half1)
        half2 = np.column_stack(half2)
        logger.info("Calculating CoAsy")
        self.doubleODF_coasy = aitchison_asymmetry(half1, half2)

        # Write outputs if requested
        if output_prefix:
            logger.info("Writing One-Ahead results")
            for n,nbr  in enumerate(neighbor_names):
                outf = output_prefix + "_doubleODF_%s_prob.nii.gz" % (nbr)
                logger.info("Writing %s", outf)
                self.save_nifti(self.doubleODF_results[:,n], outf)
            self.save_nifti(self.doubleODF_codi, output_prefix + "_doubleODF_CoDI.nii.gz")
            self.save_nifti(self.doubleODF_coasy, output_prefix + "_doubleODF_CoAsy.nii.gz")
            self.save_nifti(self.doubleODF_results[:,-1], output_prefix + "_doubleODF_p_not_trackable.nii.gz")

    def _load_niftis(self,input_prefix):
        logger.info("Loading singleODF results")
        singleODF_data = []
        for a in neighbor_names:
            outf = input_prefix + "_singleODF_%s_prob.nii.gz" % a
            if not op.exists(outf):
                raise ValueError("Unable to load from niftis, can't find %s", outf)
            logger.info("Loading %s", outf)
            # Assumes niftis were written by MITTENS
            singleODF_data.append(nib.load(outf).get_data()[::-1,::-1,:].flatten(order="F"))
        singleODF_data = np.column_stack(singleODF_data)

        self.volume_grid = nib.load(outf).shape

        # Use the mask from the fib file 
        if self.flat_mask is None:
            logger.warn("Data mask estimated from NIfTI, not fib")
            self.flat_mask = singleODF_data.sum(1) > 0
            self.nvoxels = self.flat_mask.sum()
            self.voxel_coords = np.array(np.unravel_index(
                np.flatnonzero(self.flat_mask), self.volume_grid, order="F")).T
            self.coordinate_lut = dict(
                [(tuple(coord), n) for n,coord in enumerate(self.voxel_coords)])
        singleODF_data = singleODF_data[self.flat_mask]
        self.singleODF_results = singleODF_data

        def load_masked_nifti(nifti):
            return nib.load(nifti).get_data()[::-1,::-1,:].flatten(order="F")[self.flat_mask]

        self.singleODF_codi = load_masked_nifti( input_prefix + "_singleODF_CoDI.nii.gz")

        logger.info("Reading doubleODF results")
        doubleODF_data = np.zeros_like(self.singleODF_results)
        for n,nbr  in enumerate(neighbor_names):
            outf = input_prefix + "_doubleODF_%s_prob.nii.gz" % nbr
            if not op.exists(outf):
                raise ValueError("Unable to load from niftis, can't find %s", outf)
            logger.info("Loading %s", outf)
            doubleODF_data[:,n] = load_masked_nifti(outf)
        self.doubleODF_results = doubleODF_data
        self.doubleODF_codi = load_masked_nifti(input_prefix + "_doubleODF_CoDI.nii.gz")
        self.doubleODF_coasy = load_masked_nifti(input_prefix + "_doubleODF_CoAsy.nii.gz")

    def build_graph(self,doubleODF=True, weighting_scheme="minus_iso"):

        G = networkit.graph.Graph(self.nvoxels, weighted=True, directed=True)
        self.weighting_scheme = weighting_scheme
        if doubleODF:
            prob_mat = self.doubleODF_results
            null_p = self.doubleODF_null_probs
        else:
            prob_mat = self.singleODF_results
            null_p = self.singleODF_null_probs

        def weighting_func(probs):
            if weighting_scheme == "negative_log_p":
                probs = probs
            elif weighting_scheme == "minus_iso":
                probs = probs - null_p
                low = probs == 0
                probs[low] = 0 
                probs = probs/np.linalg.norm(probs)
            else:
                raise NotImplementedError("Unknown Weighting Scheme")
            '''elif weighting_scheme == "sharpen":
                scaled_probs = probs/null_p
                probs = scaled_probs/np.linalg.norm(scaled_probs)
            elif weighting_scheme == "ratio":
                probs = probs/max(null_p)
                high_array_indices = probs > 1
                probs[high_array_indices] = 1
            elif weighting_scheme == "normalized_ratio":
                probs = probs/max(null_p)
                high_array_indices = probs > 1
                probs[high_array_indices] = 1 
                probs = probs/np.linalg.norm(probs)
            elif weighting_scheme == "norm_ratio_no_cutoff":
                probs = probs/max(null_p)
                probs = probs/np.linalg.norm(probs)
            elif weighting_scheme == "norm_ratio_extreme":
                probs = probs/max(null_p)
                probs = probs**2
                probs = probs/np.linalg.norm(probs)'''
            return -np.log(probs)
                

        # Add the voxels and their 
        for j, starting_voxel in tqdm(enumerate(self.voxel_coords),total=self.nvoxels):
            probs = weighting_func(prob_mat[j])
            for i, name in enumerate(neighbor_names):
                coord = starting_voxel + ras_neighbor_shifts[name]
                if tuple(coord) in self.coordinate_lut and not np.isnan(probs[i]):
                    if (np.isfinite(probs[i])):
                        G.addEdge(j, int(self.coordinate_lut[tuple(coord)]), w = probs[i])
                    else:
                        G.addEdge(j, int(self.coordinate_lut[tuple(coord)]), w=10000)
        self.voxel_graph = G

    def add_atlas(self, atlas_nifti, min_voxels=1):
        """ Inserts bidirectional connections between a spaceless "region node"
        and all the voxels it inhabits. Also stores ROI information for querying.
        """
        if self.label_lut is not None:
            raise NotImplementedError("Cannot add multiple atlases yet")
        atlas_img = nib.load(atlas_nifti)
        if not atlas_img.shape[0] == self.volume_grid[0] and \
               atlas_img.shape[1] == self.volume_grid[1] and \
               atlas_img.shape[2] == self.volume_grid[2]:
            raise ValueError("%s does not match dMRI volume" % atlas_nifti)
        atlas_data = atlas_img.get_data().astype(np.int)
        # Convert to LPS+ to match internal coordinates
        if atlas_img.affine[0,0] > 0:
            atlas_data = atlas_data[::-1,:,:]
        if atlas_img.affine[1,1] > 0:
            atlas_data = atlas_data[:,::-1,:]
        if atlas_img.affine[2,2] < 0:
            atlas_data = atlas_data[:,:,::-1]
        #This will need to be accessed from the graph query function
        self.atlas_labels = atlas_data.flatten(order="F")[self.flat_mask]
        self.label_lut = {}
        # Add connections between a "label" node and 
        '''self.label_lut = {}
        for label in np.unique(atlas_labels):
            connected_nodes = np.flatnonzero(atlas_labels == label)
            if len(connected_nodes) < min_voxels: continue
            self.voxel_graph.addNode()
            label_node = self.voxel_graph.numberOfNodes() - 1
            self.label_lut[label] = label_node
            for connected_node in connected_nodes:
                self.voxel_graph.addEdge(label_node,connected_node,w=0)
                self.voxel_graph.addEdge(connected_node,label_node,w=0)'''
    def Dijkstra(self, g, source, sink):
        d = networkit.graph.Dijkstra(g, source, target = sink)
        d.run()
        path = d.getPath(sink)
        return path 

    def bottleneck(self, g, source, sink):
        queue = [(0, source)]
        dist = {}
        dist[source] = 0 
        prev = {}
        while queue:
            path_len, v = heappop(queue)
            if v == sink:
                break 
            for n in g.neighbors(v):
                alt = max(path_len, g.weight(v,n))
                if (not n in dist or alt < dist[n]):
                    dist[n] = alt
                    prev[n] = v
                    heappush(queue, (dist[n], n))
        #Now get the path to the sink 
        path = [sink]
        n = sink
        while (n != source):
            n = prev[n]
            path.append(n)
        return path 
            

    def set_sink(self,g, to_id):
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
            g.addEdge(connected_node,label_node, w=0)

    def get_prob(self, g, path):
        prob = 1 
        for step in range(len(path) - 1):
            prob*=np.e**(-g.weight(path[step], path[step+1]))
        return prob

    def get_weighted_score(self, g, path):
        prob = self.get_prob(g, path)
        prob = prob ** (1./len(path))
        return prob


    def voxel_to_region_connectivity(self, from_id, to_id, write_trk="", write_prob=""):
        if self.voxel_graph is None:
            raise ValueError("Construct a voxel graph first")
        if self.label_lut is None:
            raise ValueError("No atlas information")

 
        self.set_sink(self.voxel_graph,to_id)
        source_nodes = np.flatnonzero(self.atlas_labels == from_id)
        paths = []
        for node in tqdm(source_nodes):
            if self.voxel_graph.neighbors(node):
                path = self.Dijkstra(self.voxel_graph, node, self.label_lut[to_id])
                paths.append([path, self.get_weighted_score(self.voxel_graph, path)])
        g = open('%s_%s_%s_probs.txt'%(write_prob, from_id, to_id), 'w')
        for path in paths:
            trk_paths_region = []
            g.write(str(path[-1]) + '\n')
            trk_paths.append((self.voxel_coords[np.array(path[0][0:-1])]*2.0, None, None))
        g.close()
        nib.trackvis.write('%s_%s_%s.trk.gz'%(write_trk, from_id, to_id), trk_paths, hdr )
        return paths
    
    def corticol_ribbon_to_thalamus(self, cortical_nifti, write_trk ="", write_prob = ""):
        if self.voxel_graph is None:
            raise ValueError("Construct a voxel graph first")
        if self.label_lut is None:
            raise ValueError("No atlas information")

        cortical_img = nib.load(cortical_nifti)
        if not cortical_img.shape[0] == self.volume_grid[0] and \
               cortical_img.shape[1] == self.volume_grid[1] and \
               cortical_img.shape[2] == self.volume_grid[2]:
            raise ValueError("%s does not match dMRI volume" % cortical_nifti)
        cortical_data = cortical_img.get_data().astype(np.int)
        # Convert to LPS+ to match internal coordinates
        if cortical_img.affine[0,0] > 0:
            cortical_data = cortical_data[::-1,:,:]
        if cortical_img.affine[1,1] > 0:
            cortical_data = cortical_data[:,::-1,:]
        if cortical_img.affine[2,2] < 0:
            cortical_data = cortical_data[:,:,::-1]
        cortical_labels = cortical_data.flatten(order="F")[self.flat_mask]
        #for each region of the thalamus 
        thalamus_labels = np.unique(self.atlas_labels[self.atlas_labels >= 6000])
        thalamus_labels = [label for label in thalamus_labels if label < 7000]
        source_nodes = np.flatnonzero(cortical_labels == 1)
        trk_paths = []
        g = open("%s_cortical_to_thalamus_prob.txt"%(write_prob), "w")
        for thalamus_region in tqdm(thalamus_labels):
            paths = []
            trk_paths_region = []
            self.set_sink(self.voxel_graph, thalamus_region)
            for node in tqdm(source_nodes):
                if self.voxel_graph.neighbors(node):
                    path = self.Dijkstra(self.voxel_graph, node, self.label_lut[thalamus_region])
                    paths.append([path, self.get_weighted_score(self.voxel_graph, path)])
                    trk_paths.append((self.voxel_coords[np.array(path[0:-1])]*2.0, None, None))
                    trk_paths_region.append((self.voxel_coords[np.array(path[0:-1])]*2.0, None, None))
                    g.write(str(self.get_weighted_score(self.voxel_graph, path)) + '\n')
            f = open("cortical_paths_to_%s.pkl"%(thalamus_region), "wb")
            pickle.dump(paths, f)
            f.close()
            nib.trackvis.write("%s_cortical_paths_to_thalamus_region_%s.trk.gz"%(write_trk, thalamus_region), trk_paths_region, hdr)
        g.close()
        nib.trackvis.write("%s_cortical_paths_to_thalamus.trk.gz"%(write_trk), trk_paths, hdr)
            

    '''def query_region_pair(self, from_id, to_id, n_paths=1, write_trk="",
            write_nifti=""):
        if self.voxel_graph is None:
            raise ValueError("Please construct a voxel graph first")
        if self.label_lut is None:
            raise ValueError("No atlas information found")
        def set_source():
            connected_nodes = np.flatnonzero(self.atlas_labels == from_id)
            if (from_id in self.label_lut):
                label_node = self.label_lut[from_id]
            else:
                self.voxel_graph.addNode()
                label_node = self.voxel_graph.numberOfNodes() - 1 
                self.label_lut[from_id] = label_node
            for connected_node in connected_nodes:
                #be sure to remove any edges from when from_id was set to a sink
                if (self.voxel_graph.hasEdge(connected_node, label_node)):
                    self.voxel_graph.removeEdge(connected_node, label_node)
                self.voxel_graph.addEdge(label_node, connected_node, w=0)

        def set_sink():
            connected_nodes = np.flatnonzero(self.atlas_labels == to_id)
            if (to_id in self.label_lut):
                label_node = self.label_lut[to_id]
            else:
                self.voxel_graph.addNode()
                label_node = self.voxel_graph.numberOfNodes() - 1
                self.label_lut[to_id] = label_node
            for connected_node in connected_nodes:
                if (self.voxel_graph.hasEdge(label_node, connected_node)):
                    self.voxel_graph.removeEdge(label_node, connected_node)
                self.voxel_graph.addEdge(connected_node,label_node, w=0)

        def getCost(g, path):
            cost = 0 
            for i in range(len(path)-1):
                cost += g.weight(path[i], path[i+1])
            return cost

        def getProbability(g, path):
            prob = 1
            for i in range(len(path) -1):
                prob*=np.e**(-g.weight(path[i], path[i+1]))
            return prob 

        def YenKSP():
            foundPaths = []
            shortestPath = self.Dijkstra(self.voxel_graph, self.label_lut[from_id], self.label_lut[to_id]) 
            cost = getCost(self.voxel_graph, shortestPath)
            foundPaths.append(shortestPath)
            potentialPaths = set()
            graph_copy = networkit.graph.Graph(self.voxel_graph, weighted=True, directed=True)
            for k in range(1,n_paths):
                for i in range(0, len(foundPaths[-1]) - 2):
                    removedEdges = []
                    spurNode = foundPaths[k-1][i]
                    rootPath = foundPaths[k-1][0:i+1]
                    for path in foundPaths:
                        if (rootPath == path[0:i+1]):
                            if (graph_copy.hasEdge(path[i], path[i+1])):
                                w = graph_copy.weight(path[i], path[i+1])
                                graph_copy.removeEdge(path[i], path[i+1])
                                removedEdges.append([path[i], path[i+1],w])
                    def callbackOut(u, v, weight, edge_id):
                        removedEdges.append([u,v,weight])
                        graph_copy.removeEdge(u,v)
                    def callbackIn(u, v, weight, edge_id):
                        removedEdges.append([v,u,weight])
                        graph_copy.removeEdge(v, u)

                    for rootNode in rootPath:
                        if (rootNode != spurNode):
                            graph_copy.forEdgesOf(rootNode, callbackOut)
                            graph_copy.forInEdgesOf(rootNode, callbackIn)


                    spurPath = self.Dijkstra(graph_copy, spurNode, self.label_lut[to_id])
                    totalPath = []
                    for n in rootPath:
                        totalPath.append(n)
                    for n in spurPath[1:]:
                        totalPath.append(n)
                    cost = getCost(self.voxel_graph, totalPath)
                    potentialPaths.add(tuple(totalPath) + (cost,))
                    for e in removedEdges:
                        graph_copy.addEdge(e[0], e[1], w=e[2])
                if (not potentialPaths):
                    break
                potentialPaths = list(potentialPaths)
                potentialPaths.sort(key = lambda x:x[-1])
                foundPaths.append(potentialPaths[0][0:-1])
                potentialPaths.remove(potentialPaths[0])
                potentialPaths = set(potentialPaths)
            return foundPaths

        set_source()
        set_sink()
        paths = YenKSP()
        if (not (write_trk == "")):
            trk_paths = []
            for path in paths:
                trk_paths.append((self.voxel_coords[np.array(path[1:-1])]*2.0, None, None))
            nib.trackvis.write('%s_%s_%s_%s_%s.trk.gz'%(write_trk, from_id, to_id, self.weighting_scheme,n_paths), trk_paths, hdr )'''

    def get_maximum_spanning_forest(self):
        forest = networkit.graph.UnionMaximumSpanningForest(self.voxel_graph)
        forest.run()
        self.UMSF = forest.getUMSF()

    def voxel_region_bottleneck(self, from_id, to_id):
        self.set_sink(self.voxel_graph, to_id)
        source_nodes = np.flatnonzero(self.atlas_labels == from_id)
        paths = []
        for node in tqdm(source_nodes):
            if self.voxel_graph.neighbors(node):
                path = self.bottleneck(self.voxel_graph, node, self.label_lut[to_id])
                paths.append(path)
        pickle.dump(paths, open("bottleneck_paths_201_7002.pkl","wb"))
        return paths


    def calculate_connectivity_matrices(self,opts):
        """
        Calculate an asymmetric connectivity matrix.
        """
        pass

    def to_undirected_graph(self):
        if self.voxel_graph.isDirected():
            self.voxel_graph = self.voxel_graph.toUndirected()
        else:
            logger.warn("Graph is already undirected")

    def calculate_components(self):
        if self.connected_components is None:
            if self.voxel_graph.isDirected():
                self.connected_components = \
                        networkit.components.StronglyConnectedComponents(self.voxel_graph)
            else:
                self.connected_components = \
                        networkit.components.ConnectedComponents(self.voxel_graph)

            self.connected_components.run()

    def n_connected_components(self):
        if self.connected_components is None:
            self.calculate_components()
        return self.connected_components.numberOfComponents()

    def voxelwise_approx_betweenness(self,nSamples=500, normalized=True,
            parallel=True):
        self.betweenness = \
            networkit.centrality.ApproxBetweenness2(self.voxel_graph)
            



