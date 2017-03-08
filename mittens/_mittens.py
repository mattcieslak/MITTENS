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
        self.none_ahead_funcs = self.get_prob_funcs("none_ahead")
        none_ahead_null_probs = []
        for k in neighbor_names:
            none_ahead_null_probs.append( self.none_ahead_funcs[k](
                    self.isotropic, self.prob_angles_weighted))
        self.none_ahead_null_probs = np.array(none_ahead_null_probs)
        #if not self.none_ahead_null_probs.sum() == 1.:
        #    raise ValueError("Null probailities do not add up to 1. Check Fortran")

        # One-ahead Model
        self.one_ahead_funcs = self.get_prob_funcs("one_ahead")
        one_ahead_null_probs = []
        isotropic_x2 = compute_weights_as_neighbor_voxels(
            self.isotropic[np.newaxis,:], self.prob_angles_weighted).squeeze()
        for k in neighbor_names:
            one_ahead_null_probs.append( self.one_ahead_funcs[k](
                    self.isotropic, isotropic_x2, self.prob_angles_weighted))
        self.one_ahead_null_probs = np.array(one_ahead_null_probs)
        self.one_ahead_null_probs[np.isnan(self.one_ahead_null_probs)]= 0 
        self.one_ahead_null_probs = self.one_ahead_null_probs / self.one_ahead_null_probs.sum()

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

    def get_prob_funcs(self, order="none_ahead"):

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

    def estimate_none_ahead(self, output_prefix=""):
        
        # Output matrix
        outputs = np.zeros((self.nvoxels,len(neighbor_names)),dtype=np.float)

        # Run it on each voxel
        for n, odf in enumerate(self.odf_values):
            prob_angles = weight_transition_probabilities_by_odf(
                    odf, self.prob_angles_weighted)
            if n%10000 == 0:
                logger.info("ODF %d/%d", n, self.nvoxels)
            for m, nbr_name in enumerate(neighbor_names):
                outputs[n,m] = self.none_ahead_funcs[nbr_name](
                    odf, prob_angles)

        self.none_ahead_results = outputs
        logger.info("Calculating None-Ahead CoDI")
        self.none_ahead_codi = aitchison_distance(self.none_ahead_null_probs, self.none_ahead_results)
        logger.info("Calculating Order1 KL Distance")
        self.none_ahead_kl = kl_distance(self.none_ahead_null_probs, self.none_ahead_results)

        # Write outputs if requested
        if output_prefix:
            logger.info("Writing none_ahead results")
            for a in neighbor_names:
                outf = output_prefix + "_none_ahead_%s_prob.nii.gz" % a
                logger.info("Writing %s", outf)
                self.save_nifti(
                        self.none_ahead_results[:,neighbor_names.index(a)], outf)
            self.save_nifti(self.none_ahead_kl, output_prefix + "_none_ahead_KL.nii.gz")
            self.save_nifti(self.none_ahead_codi, output_prefix + "_none_ahead_CoDI.nii.gz")

    def estimate_one_ahead(self, output_prefix=""):
        
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
                outputs[n,m] = self.one_ahead_funcs[nbr_name](
                        odf, Ypc[neighbor_coord_num], prob_angles)
        self.one_ahead_results = outputs / np.nansum(outputs, 1)[:,np.newaxis]

        # Calculate the distances
        logger.info("Calculating One-Ahead CoDI")
        self.one_ahead_codi = aitchison_distance(self.one_ahead_null_probs, self.one_ahead_results)

        # Divide the Columns into two matrices, calculate asymmetry
        half1 = []
        half2 = []
        for a,b in opposites:
            half1.append(self.one_ahead_results[:,neighbor_names.index(a)])
            half2.append(self.one_ahead_results[:,neighbor_names.index(b)])
        half1 = np.column_stack(half1)
        half2 = np.column_stack(half2)
        logger.info("Calculating CoAsy")
        self.one_ahead_coasy = aitchison_asymmetry(half1, half2)

        # Write outputs if requested
        if output_prefix:
            logger.info("Writing One-Ahead results")
            for n,nbr  in enumerate(neighbor_names):
                outf = output_prefix + "_one_ahead_%s_prob.nii.gz" % (nbr)
                logger.info("Writing %s", outf)
                self.save_nifti(self.one_ahead_results[:,n], outf)
            self.save_nifti(self.one_ahead_codi, output_prefix + "_one_ahead_CoDI.nii.gz")
            self.save_nifti(self.one_ahead_coasy, output_prefix + "_one_ahead_CoAsy.nii.gz")
            self.save_nifti(self.one_ahead_results[:,-1], output_prefix + "_one_ahead_p_not_trackable.nii.gz")

    def _load_niftis(self,input_prefix):
        logger.info("Loading none_ahead results")
        none_ahead_data = []
        for a in neighbor_names:
            outf = input_prefix + "_none_ahead_%s_prob.nii.gz" % a
            if not op.exists(outf):
                raise ValueError("Unable to load from niftis, can't find %s", outf)
            logger.info("Loading %s", outf)
            # Assumes niftis were written by MITTENS
            none_ahead_data.append(nib.load(outf).get_data()[::-1,::-1,:].flatten(order="F"))
        none_ahead_data = np.column_stack(none_ahead_data)

        self.volume_grid = nib.load(outf).shape

        # Use the mask from the fib file 
        if self.flat_mask is None:
            logger.warn("Data mask estimated from NIfTI, not fib")
            self.flat_mask = none_ahead_data.sum(1) > 0
            self.nvoxels = self.flat_mask.sum()
            self.voxel_coords = np.array(np.unravel_index(
                np.flatnonzero(self.flat_mask), self.volume_grid, order="F")).T
            self.coordinate_lut = dict(
                [(tuple(coord), n) for n,coord in enumerate(self.voxel_coords)])
        none_ahead_data = none_ahead_data[self.flat_mask]
        self.none_ahead_results = none_ahead_data

        def load_masked_nifti(nifti):
            return nib.load(nifti).get_data()[::-1,::-1,:].flatten(order="F")[self.flat_mask]

        self.none_ahead_codi = load_masked_nifti( input_prefix + "_none_ahead_CoDI.nii.gz")

        logger.info("Reading one_ahead results")
        one_ahead_data = np.zeros_like(self.none_ahead_results)
        for n,nbr  in enumerate(neighbor_names):
            outf = input_prefix + "_one_ahead_%s_prob.nii.gz" % nbr
            if not op.exists(outf):
                raise ValueError("Unable to load from niftis, can't find %s", outf)
            logger.info("Loading %s", outf)
            one_ahead_data[:,n] = load_masked_nifti(outf)
        self.one_ahead_results = one_ahead_data
        self.one_ahead_codi = load_masked_nifti(input_prefix + "_one_ahead_CoDI.nii.gz")
        self.one_ahead_coasy = load_masked_nifti(input_prefix + "_one_ahead_CoAsy.nii.gz")

    def build_graph(self,one_ahead=True, weighting_scheme="negative_log_p"):
        G = networkit.graph.Graph(self.nvoxels, weighted=True, directed=True)
        if one_ahead:
            prob_mat = self.one_ahead_results
            null_p = self.one_ahead_null_probs
        else:
            prob_mat = self.none_ahead_results
            null_p = self.none_ahead_null_probs

        if weighting_scheme == "negative_log_p":
            weighting_func = lambda x: -np.log(x)
        else:
            weighting_func = lambda x: null_p - x

        # Add the voxels and their 
        for j, starting_voxel in tqdm(enumerate(self.voxel_coords),total=self.nvoxels):
            probs = weighting_func(prob_mat[j])
            for i, name in enumerate(neighbor_names):
                coord = starting_voxel + ras_neighbor_shifts[name]
                if tuple(coord) in self.coordinate_lut:
                    G.addEdge(j, int(self.coordinate_lut[tuple(coord)]), w = probs[i])
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
        atlas_labels = atlas_data.flatten(order="F")[self.flat_mask]

        # Add connections between a "label" node and 
        self.label_lut = {}
        for label in np.unique(atlas_labels):
            connected_nodes = np.flatnonzero(atlas_labels == label)
            if len(connected_nodes) < min_voxels: continue
            self.voxel_graph.addNode()
            label_node = self.voxel_graph.numberOfNodes() - 1
            self.label_lut[label] = label_node
            for connected_node in connected_nodes:
                self.voxel_graph.addEdge(label_node,connected_node,w=0)
                self.voxel_graph.addEdge(connected_node,label_node,w=0)
            
    def query_region_pair(self, from_id, to_id, n_paths=1, write_trk="",
            write_nifti=""):
        pass

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
            



