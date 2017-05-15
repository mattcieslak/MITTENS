#!/usr/bin/env python
from .fib_io import load_fibgz
import numpy as np
from scipy.sparse import csr_matrix
import re
import importlib
from scipy.sparse.linalg import expm
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
    def __init__(self, fibgz_file="", nifti_prefix="", real_affine_image="", mask_image="",
            step_size=np.sqrt(3)/2. , angle_max=35, odf_resolution="odf8", 
            angle_weights="flat", angle_weighting_power=1.,normalize_doubleODF=True):
        """
        Represents a voxel graph.  Can be constructed with a DSI Studio ``fib.gz``
        file or from NIfTI files from a previous run. 

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
        self.voxel_graph = None
        self.null_voxel_graph = None
        self.UMSF = None
        self.atlas_labels = None
        self.weighting_scheme = None
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
        if real_affine_image:
            self._set_real_affine(real_affine_image)

    def _set_real_affine(self, affine_img):
        pass

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
        self.odf_values = self.odf_values / self.odf_values.sum(1)[:,np.newaxis] * 0.5
        logger.info("Loaded ODF data: %s",str(self.odf_values.shape))

    def _load_niftis(self,input_prefix):
        # If there is an external mask, use it before loading the niftis
        external_mask=False
        if op.exists(self.mask_image):
            mask_img = nib.load(self.mask_image)
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

    def save_nifti(self, data, fname):
        out_data = np.zeros(np.prod(self.volume_grid),dtype=np.float)
        out_data[self.flat_mask] = data
        out_data = out_data.reshape(self.volume_grid, order="F")[::-1,::-1,:]
        nib.Nifti1Image(out_data.astype(np.float32), self.ras_affine
                ).to_filename(fname)

    def analytical_probtrackx(self, source_region, seeds_per_voxel=1000, iters=200,
                             nifti_prefix="walks"):
        query_nodes, query_name = self._get_region(source_region)
        logger.info("Building Adjacency matrix")

        inputs = np.zeros(self.nvoxels)
        inputs[query_nodes] = seeds_per_voxel
        inputs = csr_matrix(inputs).T
        null_state = inputs.copy()
        real_state = inputs.copy()
        real_tally = inputs.copy()
        null_tally = inputs.copy()

        m_real = networkit.algebraic.adjacencyMatrix(self.voxel_graph)
        #assert np.allclose(m_real.sum(1),1)
        m_null = networkit.algebraic.adjacencyMatrix(self.null_voxel_graph)
        #assert np.allclose(m_null.sum(1),1)
        for iternum in tqdm(range(iters)):
            null_state = m_null.dot(null_state)
            null_tally = null_tally + null_state
            real_state = m_real.dot(real_state)
            real_tally = real_tally + real_state
        
        # Save the final state
        self.save_nifti(np.log(null_state.toarray().squeeze()),
                        "%s_null_state_%diters.nii.gz"%(nifti_prefix, iters))
        self.save_nifti(np.log(real_state.toarray().squeeze()),
                        "%s_real_state_%diters.nii.gz"%(nifti_prefix, iters))
        
        # Save the cumulative hits
        self.save_nifti(np.log(null_tally.toarray().squeeze()),
                        "%s_null_cumulative_%diters.nii.gz"%(nifti_prefix, iters))
        self.save_nifti(np.log(real_tally.toarray().squeeze()),
                        "%s_real_cumulative_%diters.nii.gz"%(nifti_prefix, iters))

    def calculate_transition_probabilities(self,output_prefix="mittens"):
        self.estimate_singleODF(output_prefix)
        self.estimate_doubleODF(output_prefix)

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

    def build_graph(self, doubleODF=True, weighting_scheme="minus_iso", require_all_neighbors=True):
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

        self.weighting_scheme = weighting_scheme
        G = networkit.graph.Graph(self.nvoxels, weighted=True, directed=True)

        # Place the requested probabilities into `prob_mat` 
        if doubleODF:
            prob_mat = self.doubleODF_results
            null_p = self.doubleODF_null_probs
        else:
            prob_mat = self.singleODF_results
            null_p = self.singleODF_null_probs
            
        
        if require_all_neighbors:
            # Which probabilities point to a voxel not in the mask?
            not_in_graph = np.zeros((self.nvoxels, 26),dtype=np.bool)
            for j, starting_voxel in tqdm(enumerate(self.voxel_coords),total=self.nvoxels):
                for i, name in enumerate(neighbor_names):
                    coord = tuple(starting_voxel + lps_neighbor_shifts[name])
                    if not coord in self.coordinate_lut:
                        not_in_graph[j,i] = True
            incomplete_neighbor_nodes = not_in_graph.sum(1) > 0
            #logger.info("Found %d nodes with incomplete neighbors",incomplete_neighbor_nodes.sum())
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
                prob_weights[np.logical_not(np.isfinite(prob_weights))] = 0
        
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
                if probs[i] > 0:
                    to_node = self.coordinate_lut.get(coord,-9999)
                    if to_node == -9999:
                        continue
                    # Actually adds the edge to the graph
                    G.addEdge(j, to_node, w = probs[i])
                    
        self.voxel_graph = G

    def build_null_graph(self, doubleODF=True, purpose="walks", require_all_neighbors=False):
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

        G = networkit.graph.Graph(self.nvoxels, weighted=True, directed=True)

        # Place the requested probabilities into `prob_mat` 
        if doubleODF:
            null_p = self.doubleODF_null_probs
        else:
            null_p = self.singleODF_null_probs
            
        
        if require_all_neighbors:
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
                    
        self.null_voxel_graph = G

    def add_atlas(self, atlas_nifti, min_voxels=1):
        """ Inserts bidirectional connections between a spaceless "region node"
        and all the voxels it inhabits. Also stores ROI information for querying.
        """
        if self.label_lut is not None:
            raise NotImplementedError("Cannot add multiple atlases yet")
        self.atlas_labels = self._oriented_nifti_data(atlas_nifti)
        self.label_lut = {}

    def Dijkstra(self, g, source, sink):
        d = networkit.graph.Dijkstra(g, source, target = sink)
        d.run()
        path = d.getPath(sink)
        return path 

    def set_source_using_nifti(self, g, connected_nodes):
        g.addNode()
        label_node = g.numberOfNodes()-1
        for node in connected_nodes:
            g.addEdge(label_node, node, w=0)
        return label_node
    
    def set_sink_using_nifti(self, g, connected_nodes):
        g.addNode()
        label_node = g.numberOfNodes()-1
        for node in connected_nodes:
            g.addEdge(node, label_node, w=0)
        return label_node

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

    def test_path_vs_null(self,path):
        """
        """
        if None in (self.voxel_graph, self.null_voxel_graph):
            raise AttributeError(
                    "Both a voxel graph and null voxel graph are required")

    def get_prob(self, g, path):
        prob = 1 
        for step in range(len(path) - 1):
            prob*=np.e**(-g.weight(path[step], path[step+1]))
        return prob

    def get_weighted_score(self, g, path):
        prob = self.get_prob(g, path)
        prob = prob ** (1./len(path))
        return prob

    def _get_path_weights(self, g, path):
        return np.array(
            [g.weight(path[step], path[step+1]) for step in range(len(path)-1)])

    def get_weighted_scores(self, g, path):
        """
        Extracts weights along a path in a graph and returns the 
        cumulative weight and the mean weight along the path
        """
        if self.weighting_scheme is None:
            raise ValueError("Build a graph first")
        
        # Get weights out of the graph
        path_len = len(path)
        path_values = self._get_path_weights(g,path)

        # Negate and exponentiate the weights to get back probabilities
        if "negative_log" in self.weighting_scheme:
            prob = np.exp(np.sum(-path_values))
            return prob, prob ** (1./path_len)
        # return the product and mean along the path
        return np.prod(path_values), np.mean(path_values)

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
        trk_paths = []
        for path in paths:
            g.write(str(path[-1]) + '\n')
            trk_paths.append((self.voxel_coords[np.array(path[0][0:-1])]*2.0, None, None))
        g.close()
        nib.trackvis.write('%s_%s_%s.trk.gz'%(write_trk, from_id, to_id), trk_paths, hdr )
        return paths

    def _oriented_nifti_data(self,nifti_file, is_labels=False, warn=False):
        """
        Loads a NIfTI file and extracts its data for each node in the graph.
        The NIfTI file must exist.

        Parameters:
        ===========
        nifti_file:str
          Path to NIfTI file
        is_labels:bool
          Does this file contain labels?
        warn:bool
          Show warnings when an image is flipped to match LPS+

        Returns:
        ========
        nifti_data_array:np.ndarray
          Array with a single value for each node in the voxel graph

        """

        if not op.exists(nifti_file):
            raise ValueError("%s does not exist" % nifti_file)
        if self.flat_mask is None:
            raise ValueError("No mask is available")

        # Check for compatible shapes
        logger.info("Loading NIfTI Image %s", nifti_file)
        img = nib.load(nifti_file)
        if not img.shape[0] == self.volume_grid[0] and \
               img.shape[1] == self.volume_grid[1] and \
               img.shape[2] == self.volume_grid[2]:
           raise ValueError("%s does not match dMRI volume" % nifti_file)

        # Convert to LPS+ to match internal coordinates
        dtype = np.int if is_labels else np.float
        data = img.get_data().astype(dtype)
        if img.affine[0,0] > 0:
            data = data[::-1,:,:]
            if warn: logger.info("Flipped X in %s", nifti_file)
        if img.affine[1,1] > 0:
            data = data[:,::-1,:]
            if warn: logger.info("Flipped Y in %s", nifti_file)
        if img.affine[2,2] < 0:
            data = data[:,:,::-1]
            if warn: logger.info("Flipped Z in %s", nifti_file)
        return data.flatten(order="F")[self.flat_mask]

    def _get_region(self,region):
        # Find what nodes are part of a region
        if type(region) is str:
            labels = self._oriented_nifti_data(region)
            return np.flatnonzero(labels), op.split(op.abspath(region))[-1]
        if type(region) in (int, float):
            nodes = np.flatnonzero(self.atlas_labels==region)
            return nodes, "region_%05d"%region

    def region_voxels_to_region_query(self, from_region, to_region, write_trk="", 
            write_prob="", write_nifti=""):
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
        """
        # Get lists of nodes for the query
        def get_region(region):
            # Find what nodes are part of a region
            if type(region) is str:
                labels = self._oriented_nifti_data(region)
                return np.flatnonzero(labels), op.split(op.abspath(region))[-1]
            if type(region) in (int, float):
                nodes = np.flatnonzero(self.atlas_labels==region)
                return nodes, "region_%05d"%region
        from_nodes, from_name = get_region(from_region)
        to_nodes, to_name = get_region(to_region)
        
        # Loop over all the voxels in the from_region
        sink_label_node = self.set_sink_using_nifti(self.voxel_graph, to_nodes)
        trk_paths = []
        probs = []
        used_voxels = []
        for node in tqdm(from_nodes):
            if self.voxel_graph.neighbors(node):
                path = self.Dijkstra(self.voxel_graph, node, sink_label_node)
                probs.append(self.get_weighted_score(self.voxel_graph, path))
                trk_paths.append(
                        (self.voxel_coords[np.array(path[0:-1])]*self.voxel_size, None, None))
                used_voxels.append(True)
            else:
                used_voxels.append(False)


        # Write outputs
        if write_prob:
            g = open("%s_to_%s_%s.txt"%(from_name, to_name, write_trk), "w")
            for prob in probs:
                g.write("%.9f\n"%prob)
            g.close()
        if write_trk:
            nib.trackvis.write('%s_to_%s_%s.trk.gz'%(from_name, to_name, write_trk), 
                    trk_paths, hdr )
        if write_nifti:
            used_voxels_mask = np.array(used_voxels, dtype=np.bool)
            node_probs = np.zeros(len(from_nodes),dtype=np.float)
            node_probs[used_voxels_mask] = np.array(probs)
  
            # Place in the whole volume
            output_probs = np.zeros(self.nvoxels, dtype=np.float)
            output_probs[from_nodes] = node_probs
            self.save_nifti(output_probs, write_nifti)

        return trk_paths, probs

    def get_maximum_spanning_forest(self):
        forest = networkit.graph.UnionMaximumSpanningForest(self.voxel_graph)
        forest.run()
        self.UMSF = forest.getUMSF()

    def shortest_path_map_null(self, source_region, nifti_prefix="shortest_path_null"):
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
        None

        Outputs:
        ========
        A number of 3D NIfTI files will be written to disk depending on the arguments. 
        At a minimum you will find

         * 

        """
        starting_region, region_name = self._get_region(source_region)
        source_label_node = self.set_source_using_nifti(self.voxel_graph, starting_region)

        # Find the connected components
        undirected_version = self.voxel_graph.toUndirected()
        components = networkit.components.ConnectedComponents(undirected_version)
        components.run()
        logger.info("Found %d components in the graph", components.numberOfComponents())
        target_component = components.componentOfNode(source_label_node)

        # Run the modified Dijkstra algorithm from networkit
        n = networkit.graph.Dijkstra(self.voxel_graph, source_label_node)
        n.run()
        logger.info("Computed shortest paths")
        
        # Same on the null graph
        null_source_label_node = self.set_source_using_nifti(self.null_voxel_graph, starting_region)
        null_n = networkit.graph.Dijkstra(self.null_voxel_graph,null_source_label_node)
        null_n.run()
        logger.info("Computed shortest paths in null graph")

        # Arrays to store results
        scores = np.zeros(self.nvoxels, dtype=np.float) # mean along-path probability
        null_scores = np.zeros(self.nvoxels, dtype=float)
        raw_scores = np.zeros(self.nvoxels, dtype=np.float) # product of step probabilities
        raw_null_scores = np.zeros(self.nvoxels, dtype=float)
        path_lengths = np.zeros(self.nvoxels, dtype=np.float) # Number of hops per path
        null_path_lengths = np.zeros(self.nvoxels, dtype=np.float)
        # Use the real path but scores from the null graph
        realnull_scores = np.zeros(self.nvoxels, dtype=np.float) # mean along-path probability
        raw_realnull_scores = np.zeros(self.nvoxels, dtype=np.float) # product of step probabilities
        # Collect the shortest paths to each voxel and calculate a score
        for node in tqdm(np.arange(self.nvoxels)):
            if components.componentOfNode(node) == target_component:
                path = n.getPath(node)
                raw_scores[node], scores[node] = self.get_weighted_scores(self.voxel_graph, path)
                path_lengths[node] = len(path)
                null_path = null_n.getPath(node)
                raw_null_scores[node], null_scores[node] = self.get_weighted_scores(
                                                            self.null_voxel_graph, null_path)
                null_path_lengths[node] = len(null_path)

                raw_realnull_scores[node], realnull_scores[node] = self.get_weighted_scores(
                                                            self.null_voxel_graph, path)

        # real data
        self.save_nifti(raw_scores, nifti_prefix + "_shortest_path_score.nii.gz")
        self.save_nifti(scores, nifti_prefix + "_shortest_path_mean_score.nii.gz")
        self.save_nifti(path_lengths, nifti_prefix + "_shortest_path_length.nii.gz")

        self.save_nifti(raw_null_scores, nifti_prefix + "_null_shortest_path_score.nii.gz")
        self.save_nifti(null_scores, nifti_prefix + "_null_shortest_path_mean_score.nii.gz")
        self.save_nifti(null_path_lengths, nifti_prefix + "_null_shortest_path_length.nii.gz")

        self.save_nifti(raw_realnull_scores, nifti_prefix + "_realnull_shortest_path_score.nii.gz")
        self.save_nifti(realnull_scores, nifti_prefix + "_realnull_shortest_path_mean_score.nii.gz")

    def shortest_path_map(self, source_region, nifti_prefix="shortest_path"):
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
        None

        Outputs:
        ========
        A number of 3D NIfTI files will be written to disk depending on the arguments. 
        At a minimum you will find

         * 

        """
        starting_region, region_name = self._get_region(source_region)
        source_label_node = self.set_source_using_nifti(self.voxel_graph, starting_region)

        # Find the connected components
        undirected_version = self.voxel_graph.toUndirected()
        components = networkit.components.ConnectedComponents(undirected_version)
        components.run()
        logger.info("Found %d components in the graph", components.numberOfComponents())
        target_component = components.componentOfNode(source_label_node)

        # Run the modified Dijkstra algorithm from networkit
        n = networkit.graph.Dijkstra(self.voxel_graph, source_label_node)
        n.run()
        logger.info("Computed shortest paths")

        # Collect the shortest paths to each voxel and calculate a score
        scores = np.zeros(self.nvoxels, dtype=np.float)
        raw_scores = np.zeros(self.nvoxels, dtype=np.float)
        path_lengths = np.zeros(self.nvoxels, dtype=np.float)
        for node in tqdm(np.arange(self.nvoxels)):
            if components.componentOfNode(node) == target_component and self.voxel_graph.neighbors(node):
                path = n.getPath(node)
                if len(path):
                    raw_scores[node], scores[node] = self.get_weighted_scores(self.voxel_graph, path)
                else:
                    logger.info("Shortest path map didn't find a path")
                path_lengths[node] = len(path)
        self.save_nifti(raw_scores, nifti_prefix + "_shortest_path_score.nii.gz")
        self.save_nifti(scores, nifti_prefix + "_shortest_path_mean_score.nii.gz")
        self.save_nifti(path_lengths, nifti_prefix + "_shortest_path_length.nii.gz")

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
        if self.voxel_graph.isDirected():
            self.connected_components = \
                    networkit.components.StronglyConnectedComponents(self.voxel_graph)
        else:
            self.connected_components = \
                    networkit.components.ConnectedComponents(self.voxel_graph)

        self.connected_components.run()
        return self.connected_components.numberOfComponents()

    def voxelwise_approx_betweenness(self,nSamples=500, normalized=True,
            parallel=True):
        self.betweenness = \
            networkit.centrality.ApproxBetweenness2(self.voxel_graph)
            



