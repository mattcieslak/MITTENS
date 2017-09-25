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
from .distances import (kl_distance, aitchison_distance,  
                        aitchison_asymmetry)
import nibabel as nib
import os.path as op
import networkit
from .spatial import Spatial
from .voxel_graph import VoxelGraph


DISCONNECTED=999999999
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

class MITTENS(Spatial):
    def __init__(self, fibgz_file="", nifti_prefix="",
            real_affine_image="", mask_image="",
            step_size=np.sqrt(3)/2. , angle_max=35, odf_resolution="odf8", 
            angle_weights="flat", angle_weighting_power=1.,normalize_doubleODF=True):
        """
        Represents a voxel graph.  Can be constructed with a DSI Studio ``fib.gz``
        file or from NIfTI files or a voxel graph matfile. 

        Parameters:
        ===========

        fibgz_file:str
          Path to a dsi studio fib.gz file
        nifti_prefix:str
          Prefix used when calculating singleODF and/or doubleODF transition
          probabilities.
        real_affine_image:str
          Path to a NIfTI file that contains the real affine mapping for the
          data. DSI Studio does not preserve affine mappings. If provided, 
          all NIfTI outputs will be written with this affine. Otherwise the
          default affine from DSI Studio will be used.
        mask_image:str
          Path to a NIfTI file that has nonzero values in voxels that will be used
          as nodes in the graph.  If none is provided, the default mask estimated by
          DSI Studio is used.
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


        Note:
        =====
        The combination of odf_resolution, angle_max, angle_weights and angle_weighting
        power is only available if you have the corresponding Fortran extension module.
        If you're unable to initialize a MITTENS object with your desired combination,
        try downloading or generating/compiling the necessary Fortran modules.
        """
        if fibgz_file == nifti_prefix == "":
            raise ValueError("Must provide either a DSI Studio fib file or prefix to "
                    "NIfTI1 images written out by a previous run")
        # These will get filled out from loading a fibgz or niftis
        self.flat_mask = None
        self.nvoxels = None
        self.voxel_size = None
        self.voxel_coords = None
        self.coordinate_lut = None
        self.label_lut = None
        self.atlas_labels = None
        self.mask_image = mask_image
        # From args
        self.step_size = step_size
        self.odf_resolution = odf_resolution
        self.angle_max = angle_max
        self.orientation = None
        self.angle_weights = angle_weights
        self.normalize_doubleODF = normalize_doubleODF
        self.angle_weighting_power = angle_weighting_power
        logger.info("\nUsing\n------\n  Step Size:\t\t%.4f Voxels \n  ODF Resolution:\t"
                "%s\n  Max Angle:\t\t%.2f Degrees\n"
                "  Angle Weights:\t%s\n  Angle weight power:\t%.1f",
                self.step_size, self.odf_resolution, self.angle_max, 
                self.angle_weights,self.angle_weighting_power)
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
        self._set_real_affine(real_affine_image)

    def _initialize_nulls(self):

        # Note, only entries for unique vertices are created, but they 
        # are divided by the total number of vertices.
        self.isotropic = np.ones(self.n_unique_vertices,dtype=np.float64) / \
                self.odf_vertices.shape[0]
        
        # Single ODF Model
        self.singleODF_funcs = self.get_prob_funcs("singleODF")
        singleODF_null_probs = []
        for k in neighbor_names:
            singleODF_null_probs.append( self.singleODF_funcs[k](
                    self.isotropic, self.prob_angles_weighted))
        self.singleODF_null_probs = np.array(singleODF_null_probs)
        #if not self.singleODF_null_probs.sum() == 1.:
        #    raise ValueError("Null probailities do not add up to 1. Check Fortran")

        # Double ODF Model
        self.doubleODF_funcs = self.get_prob_funcs("doubleODF")
        doubleODF_null_probs = []
        isotropic_x2 = compute_weights_as_neighbor_voxels(
            self.isotropic[np.newaxis,:], self.prob_angles_weighted).squeeze()
        for k in neighbor_names:
            doubleODF_null_probs.append( self.doubleODF_funcs[k](
                    self.isotropic, isotropic_x2, self.prob_angles_weighted))
        self.doubleODF_null_probs = np.array(doubleODF_null_probs)
        self.doubleODF_null_probs[np.isnan(self.doubleODF_null_probs)]= 0 
        if self.normalize_doubleODF:
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
        self.voxel_size = aff[:3]

        # Coordinate mapping information from fib file
        self.flat_mask = f["fa0"].squeeze() > 0
        self.nvoxels = self.flat_mask.sum()
        self.voxel_coords = np.array(np.unravel_index(
            np.flatnonzero(self.flat_mask), self.volume_grid, order="F")).T
        self.coordinate_lut = dict(
            [(tuple(coord), n) for n,coord in enumerate(self.voxel_coords)])

        """
        This is not necessary. There will always be voxels without 26 
        outgoing edges. 

        # Which probabilities point to a voxel not in the mask?
        logger.info("Checking for voxels with 26 neighbors")
        all_neighbors_in_graph = np.ones((self.nvoxels), dtype=np.bool)
        for j, starting_voxel in enumerate(self.voxel_coords):
            for i, name in enumerate(neighbor_names):
                coord = tuple(starting_voxel + lps_neighbor_shifts[name])
                if not coord in self.coordinate_lut:
                    all_neighbors_in_graph[j] = False
                    break

        # Make new masks
        flat_output = np.zeros(np.prod(self.volume_grid))
        flat_output[self.flat_mask] = all_neighbors_in_graph
        self.flat_mask = flat_output > 0
        nvoxels = self.flat_mask.sum()
        logger.info("Removed %d/%d voxels with incomplete neighbors",
                self.nvoxels - nvoxels, self.nvoxels)
        self.nvoxels = nvoxels
        self.voxel_coords = np.array(np.unravel_index(
            np.flatnonzero(self.flat_mask), self.volume_grid, order="F")).T
        self.coordinate_lut = dict(
            [(tuple(coord), n) for n,coord in enumerate(self.voxel_coords)])

        """
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
        norm_factor = self.odf_values.sum(1)
        norm_factor[norm_factor == 0] = 1.
        self.odf_values = self.odf_values / norm_factor[:,np.newaxis] * 0.5
        logger.info("Loaded ODF data: %s",str(self.odf_values.shape))

    def _load_niftis(self,input_prefix):
        # If there is an external mask, use it before loading the niftis
        external_mask=False
        possible_mask = input_prefix + "_mask.nii.gz"
        if op.exists(self.mask_image):
            mask_path = self.mask_image
        elif op.exists(possible_mask):
            mask_path = possible_mask
        else:
            mask_path = ""
            
        if op.exists(mask_path):
            mask_img = nib.load(mask_path)
            self.volume_grid = mask_img.shape
            self.voxel_size = np.abs(np.diag(mask_img.affine)[:3])
            total_voxels = np.prod(mask_img.shape)
            self.flat_mask = np.ones(np.prod(total_voxels),dtype=np.bool)
            self.flat_mask = self._oriented_nifti_data(self.mask_image).astype(np.bool)
            masked_voxels = self.flat_mask.sum()
            logger.info("Used %s to mask from %d to %d voxels", 
                    self.mask_image, total_voxels, masked_voxels)
            external_mask = True
            
        logger.info("Loading singleODF results")
        singleODF_data = []
        for a in neighbor_names:
            outf = input_prefix + "_singleODF_%s_prob.nii.gz" % a
            if not op.exists(outf):
                raise ValueError("Unable to load from niftis, can't find %s", outf)
            logger.info("Loading NIfTI Image %s", outf)
            # Assumes niftis were written by MITTENS
            tmp_img = nib.load(outf)
            if any([tmp_img.affine[0,0] < 0, tmp_img.affine[1,1] < 0, 
                tmp_img.affine[2,2] < 0]):
                logger.warn("NIfTI may not have come from MITTENS.")
            if external_mask:
                singleODF_data.append(self._oriented_nifti_data(outf,warn=False))
            else:
                singleODF_data.append(tmp_img.get_data()[::-1,::-1,:].flatten(order="F"))
        singleODF_data = np.column_stack(singleODF_data)

        final_img = nib.load(outf)
        self.ras_affine = final_img.affine

        # Estimate the mask from the nifti file 
        if self.flat_mask is None:
            self.volume_grid = final_img.shape
            self.voxel_size = np.abs(np.diag(final_img.affine)[:3])
            logger.warn("Creating mask based on nonzero voxels in nifti files")
            self.flat_mask = singleODF_data.sum(1) > 0

        # Guaranteed to have a flat mask by now
        self.nvoxels = self.flat_mask.sum()
        self.voxel_coords = np.array(np.unravel_index(
            np.flatnonzero(self.flat_mask), self.volume_grid, order="F")).T
        self.coordinate_lut = dict(
            [(tuple(coord), n) for n,coord in enumerate(self.voxel_coords)])
        if not external_mask:
            singleODF_data = singleODF_data[self.flat_mask]
        self.singleODF_results = singleODF_data

        self.singleODF_codi = self._oriented_nifti_data( input_prefix + "_singleODF_CoDI.nii.gz")

        logger.info("Reading doubleODF results")
        doubleODF_data = np.zeros_like(self.singleODF_results)
        for n,nbr  in enumerate(neighbor_names):
            outf = input_prefix + "_doubleODF_%s_prob.nii.gz" % nbr
            if not op.exists(outf):
                raise ValueError("Unable to load from niftis, can't find %s", outf)
            logger.info("Loading %s", outf)
            doubleODF_data[:,n] = self._oriented_nifti_data(outf)
        self.doubleODF_results = doubleODF_data
        self.doubleODF_codi = self._oriented_nifti_data(input_prefix + "_doubleODF_CoDI.nii.gz")
        self.doubleODF_coasy = self._oriented_nifti_data(input_prefix + "_doubleODF_CoAsy.nii.gz")

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

    def calculate_transition_probabilities(self,output_prefix="mittens"):
        self.estimate_singleODF(output_prefix)
        self.estimate_doubleODF(output_prefix)
        self.save_nifti(self.flat_mask.astype(np.float), output_prefix + "_mask.nii.gz")

        # Write outputs if requested
        if output_prefix:
            #logger.info("Writing singleODF results")
            for a in neighbor_names:
                outf = output_prefix + "_singleODF_%s_prob.nii.gz" % a
                #logger.info("Writing %s", outf)
                self.save_nifti(
                        self.singleODF_results[:,neighbor_names.index(a)], outf)
            self.save_nifti(self.singleODF_codi, output_prefix + "_singleODF_CoDI.nii.gz")
            #logger.info("Writing Double ODF results")
            for n,nbr  in enumerate(neighbor_names):
                outf = output_prefix + "_doubleODF_%s_prob.nii.gz" % (nbr)
                #logger.info("Writing %s", outf)
                self.save_nifti(self.doubleODF_results[:,n], outf)
            self.save_nifti(self.doubleODF_codi, output_prefix + "_doubleODF_CoDI.nii.gz")
            self.save_nifti(self.doubleODF_coasy, output_prefix + "_doubleODF_CoAsy.nii.gz")
            self.save_nifti(self.doubleODF_results[:,-1], output_prefix + "_doubleODF_p_not_trackable.nii.gz")

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
                           for shift in lps_neighbor_shifts.values()  ])

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
                        lps_neighbor_shifts[nbr_name])]
                outputs[n,m] = self.doubleODF_funcs[nbr_name](
                        odf, Ypc[neighbor_coord_num], prob_angles)
        if self.normalize_doubleODF:
            self.doubleODF_results = outputs / np.nansum(outputs, 1)[:,np.newaxis]
        else:
            self.doubleODF_results = outputs

        # Calculate the distances
        logger.info("Calculating Double ODF CoDI")
        self.doubleODF_codi = aitchison_distance(self.doubleODF_null_probs, 
                self.doubleODF_results)

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

    def _voxel_graph(self):
        # Creates an appropriate VoxelGraph
        return VoxelGraph( 
                # Transition Prob details
                angle_max = self.angle_max, odf_resolution=self.odf_resolution, 
                angle_weights=self.angle_weights, step_size=self.step_size,
                angle_weighting_power=self.angle_weighting_power,
                normalize_doubleODF=self.normalize_doubleODF,
                # Spatial mapping
                real_affine=self.real_affine, flat_mask=self.flat_mask,
                ras_affine=self.ras_affine, voxel_size=self.voxel_size,
                volume_grid=self.volume_grid, nvoxels=self.nvoxels,
                # Pass it since we've got it
                voxel_coords=self.voxel_coords, coordinate_lut=self.coordinate_lut
                )

    def build_graph(self, doubleODF=True, weighting_scheme="minus_iso", 
                                                 require_all_neighbors=False):
        """
        Builds a ``networkit.graph.Graph`` from transition probabilities.
        
        Parameters:
        ===========
        
        doubleODF:bool
          If True, the transition probabilities from the doubleODF method will be used.
          If ``normalize_doubleODF`` was set to True when you constructed the object,
          the outgoing edges from every node will sum to 1. Default: `True`
          
        weighting_scheme:str
          One of {"negative_log_p", "minus_iso", "minus_iso_scaled", "minus_iso_negative_log",
          "minus_iso_scaled_negative_log", "transition probability"}. Determines how transition
          probabilities are used as edge weights. Default: ""
          
        require_all_neighbors:bool
          if a voxel does not have all 26 neighbors, remove all outgoing edges and replace them
          with a probability 1 self-edge
          
        Weighting Schemes:
        ===================
        
        Your weighting scheme choice needs to be compatible with how you want to use the graph.
        
        If you plan on calculating shortest paths, you want the edge weights to be **lower** 
        when the transition probability is higher. This can be accomplished by taking the negative
        log of a probability.
        
        If you are simulating 3D walks, you want the edge weights to be high when the transition
        probability is high.
        
        Schemes for walks:
        ------------------
        ``"minus_iso"``: 
          The isotropic probabilities are subtracted from the transition probabilities. No edge is added
          when transition probabilities are lower than the corresponding isotropic probability. This is 
          not ideal for simulating walks because each nodes outgoing edges do not sum to 1.
                         
        ``"minus_iso_scaled"``: 
          Same as ``"minus_iso"`` except probabilities are re-scaled so they sum to 1. This is a very 
          good idea for simulating walks.
                                
        ``"transition probability"``: 
          The transition probabilities are directly used as edge weights. These are the closest to 
          the input data, but if you use diffusion ODFs as input will likely not produce very specific
          paths.
        
        
        Schemes for shortest paths:
        ---------------------------
        
        ``"negative_log_p"``:
          Transition probabilities are log transformed and made negative.  This is similar to the
          Zalesky 2009 strategy.
          
        ``"minus_iso_negative_log"``:
          Isotropic probabilities are subtracted from transition probabilities. Edges are not added when
          transition probabilities are less than the isotropic probability.
          
        ``"minus_iso_scaled_negative_log"``:
          Same as ``"minus_iso_negative_log"`` except probabilities are re-scaled to sum to 1 *before*
          the log transform is applied. 
          
        """

        G = networkit.graph.Graph(int(self.nvoxels), weighted=True, directed=True)

        # Place the requested probabilities into `prob_mat` 
        if doubleODF:
            prob_mat = self.doubleODF_results
            null_p = self.doubleODF_null_probs
        else:
            prob_mat = self.singleODF_results
            null_p = self.singleODF_null_probs
            
        
        if require_all_neighbors:
            logger.info("Removing Voxels without all 26 neighbors connected")
            # Which probabilities point to a voxel not in the mask?
            not_in_graph = np.zeros((self.nvoxels, 26),dtype=np.bool)
            for j, starting_voxel in tqdm(enumerate(self.voxel_coords),total=self.nvoxels):
                for i, name in enumerate(neighbor_names):
                    coord = tuple(starting_voxel + lps_neighbor_shifts[name])
                    if not coord in self.coordinate_lut:
                        not_in_graph[j,i] = True
            incomplete_neighbor_nodes = not_in_graph.sum(1) > 0
            logger.info("Found %d nodes with incomplete neighbors",
                                            incomplete_neighbor_nodes.sum())
            prob_mat[incomplete_neighbor_nodes] = 0 # Remove the outgoing edges
            if weighting_scheme in ("negative_log_p","minus_iso_negative_log",
                                    "minus_iso_scaled_negative_log"):
                incomplete_weight = 0
            else:
                incomplete_weight = 1
            for incomplete_node in np.flatnonzero(incomplete_neighbor_nodes):
                # Add a self-edge
                G.addEdge(incomplete_node, incomplete_node, w = incomplete_weight)
            
        ## Determine how to weight edges. Changes `prob_mat`, which contains probabilities,
        ## to `prob_weights`, which contains edge weights.
        
        # Take the negative log of the transition probabilities
        if weighting_scheme == "negative_log_p":
            prob_weights = -np.log(prob_mat)
            # Change nans and infs to 0
            prob_weights[np.logical_not(np.isfinite(prob_weights))] = 0
            
        # Subtract the isotropic transition probs from the calculated transition probs
        elif weighting_scheme.startswith("minus_iso"):
            prob_weights = prob_mat - null_p
            low = prob_weights <= 0
            prob_weights[low] = 0
            
            if "scaled" in weighting_scheme:
                prob_weights = prob_weights / np.nansum(prob_weights, 1)[:,np.newaxis]
            
            if weighting_scheme.endswith("negative_log"):
                prob_weights = -np.log(prob_weights)
                prob_weights[np.logical_not(np.isfinite(prob_weights))] = DISCONNECTED
        
        # Use the transition probabilities as-is
        elif weighting_scheme == "transition probability":
            prob_weights = prob_mat.copy()

        else:
            raise NotImplementedError("Unknown weighting scheme")

        # Add the voxels and their probabilities
        for j, starting_voxel in tqdm(enumerate(self.voxel_coords),total=self.nvoxels):
            probs = prob_weights[j]
            for i, name in enumerate(neighbor_names):
                coord = tuple(starting_voxel + lps_neighbor_shifts[name])
                to_node = self.coordinate_lut.get(coord,-9999)
                if to_node == -9999:
                    continue
                if probs[i] > 0:
                    # Actually adds the edge to the graph
                    G.addEdge(j, to_node, w = probs[i])
                else:
                    if weighting_scheme == "transition probability":
                        G.addEdge(j, to_node, w = np.finfo(np.float).eps)
                    else:
                        G.addEdge(j, to_node, w = DISCONNECTED)

        vg = self._voxel_graph()    
        vg.weighting_scheme = weighting_scheme
        vg.graph = G
        return vg

    def build_null_graph(self, doubleODF=True, purpose="walks", 
                                            require_all_neighbors=False):
        """
        Builds a ``networkit.graph.Graph`` from null transition probabilities.
        
        Parameters:
        ===========
        
        doubleODF:bool
          If True, the transition probabilities from the doubleODF method will be used.
          If ``normalize_doubleODF`` was set to True when you constructed the object,
          the outgoing edges from every node will sum to 1. Default: `True`
          
        purpose:str
          One of {"walks", "shortest paths"}. Determines how transition
          probabilities are used as edge weights. Default: "walks"
          
        require_all_neighbors:bool
          if a voxel does not have all 26 neighbors, remove all outgoing edges and replace them
          with a probability 1 self-edge
          
        """

        G = networkit.graph.Graph(int(self.nvoxels), weighted=True, directed=True)

        # Place the requested probabilities into `prob_mat` 
        if doubleODF:
            null_p = self.doubleODF_null_probs
        else:
            null_p = self.singleODF_null_probs
            
        
        if require_all_neighbors:
            logger.info("Removing Voxels without all 26 neighbors connected")
            # Which probabilities point to a voxel not in the mask?
            not_in_graph = np.zeros((self.nvoxels, 26),dtype=np.bool)
            for j, starting_voxel in tqdm(enumerate(self.voxel_coords),total=self.nvoxels):
                for i, name in enumerate(neighbor_names):
                    coord = tuple(starting_voxel + lps_neighbor_shifts[name])
                    if not coord in self.coordinate_lut:
                        not_in_graph[j,i] = True
            incomplete_neighbor_nodes = not_in_graph.sum(1) > 0
            if purpose == "walks":
                incomplete_weight = 1
            elif purpose == "shortest paths":
                incomplete_weight = 0
            for incomplete_node in np.flatnonzero(incomplete_neighbor_nodes):
                # Add a self-edge
                G.addEdge(incomplete_node, incomplete_node, w = incomplete_weight)
            
        if purpose == "shortest paths":
            prob_weights = -np.log(null_p)
        elif purpose == "walks":
            prob_weights = null_p
        else:
            raise NotImplementedError("Unknown purpose")

        # Add the voxels and their probabilities. This can be sped up probably
        for j, starting_voxel in tqdm(enumerate(self.voxel_coords),total=self.nvoxels):
            for i, name in enumerate(neighbor_names):
                coord = tuple(starting_voxel + lps_neighbor_shifts[name])
                to_node = self.coordinate_lut.get(coord,-9999)
                if to_node == -9999:
                    continue
                # Actually adds the edge to the graph
                G.addEdge(j, to_node, w = prob_weights[i])
                    
        vg = self._voxel_graph()    
        vg.weighting_scheme = purpose
        vg.graph = G
        return vg

