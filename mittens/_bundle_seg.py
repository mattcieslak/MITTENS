#!/usr/bin/env python
from .fib_io import load_fibgz
import numpy as np
import re
import importlib
from .utils import (neighbor_names, get_transition_analysis_matrices,
        lps_neighbor_shifts, weight_transition_probabilities_by_odf,
        compute_weights_as_neighbor_voxels, ras_neighbor_shifts,
        pairwise_distances, angle_between, neighbor_names)
import logging
from tqdm import tqdm
from .distances import (kl_distance, aitchison_distance,
                        aitchison_asymmetry)
import nibabel as nib
import os.path as op
import networkit
from .spatial import Spatial
from .voxel_graph import VoxelGraph

logger = logging.getLogger(__name__)


class BundleSegmentation(Spatial):
    def __init__(self, fibgz_file="", real_affine_image="", mask_image=""):
        """
        Represents a bundle segmentation problem.

        Parameters:
        ===========

        fibgz_file:str
          Path to a dsi studio fib.gz file
        real_affine_image:str
          Path to a NIfTI file that contains the real affine mapping for the
          data. DSI Studio does not preserve affine mappings. If provided,
          all NIfTI outputs will be written with this affine. Otherwise the
          default affine from DSI Studio will be used.
        mask_image:str
          Path to a NIfTI file that has nonzero values in voxels that will be used
          as nodes in the graph.  If none is provided, the default mask estimated by
          DSI Studio is used.
        """
        if fibgz_file == "":
            raise ValueError("Must provide a DSI Studio fib file")
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
        if fibgz_file:
            logger.info("Loading DSI Studio fib file")
            self._load_fibgz(fibgz_file)
        self.n_unique_vertices = self.odf_vertices.shape[0]//2

        self._set_real_affine(real_affine_image)
        logger.info("Pre-computing angle distances")
        self.precompute_distances()

    def precompute_distances(self):
        angle_diffs = pairwise_distances(self.odf_vertices,
                                         metric=angle_between)
        twodir_keys = []
        for dir1 in range(self.n_unique_vertices):
            for dir2 in range(dir1+1, self.n_unique_vertices):
                twodir_keys.append((dir1,dir2))

        twodir_distances = {}
        for dir1_id in range(len(twodir_keys)):
            twodir_dir1 = twodir_keys[dir1_id]
            for dir2_id in twodir_keys[dir1_id]:
                twodir_dir2 = twodir_keys[dir2_id]
                twodir_distances[(twodir_dir1, twodir_dir2)] = max(
                                    min(angle_diffs[twodir_dir1[0], twodir_dir2[0]],
                                        angle_diffs[twodir_dir1[0], twodir_dir2[1]]),
                                    min(angle_diffs[twodir_dir1[1], twodir_dir2[0]],
                                        angle_diffs[twodir_dir1[1], twodir_dir2[1]])
                )
        self.angle_diffs = angle_diffs
        self.twodir_distances = twodir_distances

    def _load_fibgz(self, path):
        logger.info("Loading %s", path)
        f = load_fibgz(path)
        logger.info("Loaded %s", path)
        self.orientation = "lps"
        self.odf_vertices = f['odf_vertices'].T

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

        odf_values = np.row_stack(valid_odfs).astype(np.float64)
        norm_factor = odf_values.sum(1)
        norm_factor[norm_factor == 0] = 1.

        # Here we differ from MITTENS oject
        odf_values = self.odf_values / norm_factor[:,np.newaxis]

        # Pull out the peak values for each voxel
        peak_index_names = sorted([k for k in fib.keys() if k.startswith("index")])
        if len(peak_index_names) > 9:
            raise ValueError("Only works with less than 10 fibers")
        max_num_peaks = min(max_num_peaks,len(peak_index_names))

        peak_indices = np.column_stack([fib[index_name].squeeze()[flat_mask] \
                                        for index_name in peak_index_names[:max_num_peaks]])
        peak_adcs = np.zeros((nvoxels,max_num_peaks),dtype=np.float)


        for voxel_id in range(nvoxels):
            for peak_num, peak_index in enumerate(peak_indices[voxel_id]):
                adc_val = odf_values[voxel_id, peak_index]
                peak_adcs[voxel_id,peak_num] = adc_val

        self.peak_indices = peak_indices
        self.peak_adcs = peak_adcs

    def calculate_peaks(self, adc_min, max_num_peaks):
        self.n_fibers = np.sum(self.peak_adcs > adc_min, axis=1)

        peaks = []
        for indices, nfibers in zip(self.peak_indices, n_fibers):
            if nfibers == 1:
                peaks.append(indices[0])
            elif nfibers == 2:
                peaks.append(tuple(sorted(indices[:2])))
            else:
                peaks.append(tuple(sorted(indices)))

        self.peaks = peaks

    def grow_1dir(self, front_voxels, internal_voxels, excluded_voxels,
              included_angles, max_angle):

        # Grow into the neighbors of the voxels on the outside of the region
        next_voxels = set()
        for voxel in front_voxels:
            next_voxels.update(self.G1.neighbors(voxel))

        # Exclude voxels already visited (internal and excluded voxels)
        new_front_voxels = next_voxels - internal_voxels - excluded_voxels

        # If there are no new voxels to add, exit
        if len(new_front_voxels) == 0:
            return new_front_voxels, internal_voxels, excluded_voxels, included_angles

        # Get the angles contained in the new voxels
        voxels = list(new_front_voxels)
        new_angles = [self.peaks[voxel] for voxel in voxels]

        # Check if the angles contained are compatible with the included angles
        temp_new_included_voxels = []
        temp_new_included_angles = []
        new_included_angles = set()
        new_included_voxels = set()

        for voxel, new_angle in zip(voxels, new_angles):
            # If the angle is already included, it's ok
            if new_angle in included_angles:
                new_included_voxels.add(voxel)
                continue

            # It contains an unseen angle, calculate differences from included angles
            cluster_angle_diffs = np.array([angle_diffs[new_angle, included_angle] \
                                  for included_angle in included_angles])

            # If any are over the threshold, exclude the voxel
            if np.any(cluster_angle_diffs > max_angle):
                excluded_voxels.add(voxel)

            # This angle might be OK. Add it and its angle to be checked
            # at the end
            else:
                temp_new_included_voxels.append(voxel)
                temp_new_included_angles.append(new_angle)

        # Check if the potential new voxels are compatible with each other
        while len(temp_new_included_voxels):
            check_voxel = temp_new_included_voxels.pop()
            check_angle = temp_new_included_angles.pop()
            front_angle_diffs = np.array([angle_diffs[check_angle, front_angle] \
                                  for front_angle in temp_new_included_angles])
            if np.any(front_angle_diffs > max_angle):
                excluded_voxels.add(check_voxel)
            else:
                new_included_voxels.add(check_voxel)
                included_angles.add(check_angle)

        # Add the previous front voxels to internal voxels
        internal_voxels = internal_voxels | front_voxels
        if len(new_included_voxels) == 0:
            return new_included_voxels, internal_voxels, excluded_voxels, included_angles

        return grow_1dir(new_included_voxels, internal_voxels, excluded_voxels,
                         included_angles, max_angle)

    def grow_2dir(self, front_voxels, internal_voxels, excluded_voxels,
                  included_angles, max_angle):

        # Grow into the neighbors of the voxels on the outside of the region
        next_voxels = set()
        for voxel in front_voxels:
            next_voxels.update(self.G2.neighbors(voxel))

        # Exclude voxels already visited (internal and excluded voxels)
        new_front_voxels = next_voxels - internal_voxels - excluded_voxels

        # If there are no new voxels to add, exit
        if len(new_front_voxels) == 0:
            return new_front_voxels, internal_voxels, excluded_voxels, included_angles

        # Get the angles contained in the new voxels
        voxels = list(new_front_voxels)
        new_angles = [peaks[voxel] for voxel in voxels]

        # Check if the angles contained are compatible with the included angles
        temp_new_included_voxels = []
        temp_new_included_angles = []
        new_included_angles = set()
        new_included_voxels = set()

        for voxel, new_angle in zip(voxels, new_angles):
            # If the angle is already included, it's ok
            if new_angle in included_angles:
                new_included_voxels.add(voxel)
                continue

            # It contains an unseen angle, calculate differences from included angles
            cluster_angle_diffs = np.array([
                twodir_distances.get((new_angle,included_angle),
                                    twodir_distances[(included_angle,new_angle)]) \
                            for included_angle in included_angles])

            # If any are over the threshold, exclude the voxel
            if np.any(cluster_angle_diffs > max_angle):
                excluded_voxels.add(voxel)

            # This angle might be OK. Add it and its angle to be checked
            # at the end
            else:
                temp_new_included_voxels.append(voxel)
                temp_new_included_angles.append(new_angle)

        # Check if the potential new voxels are compatible with each other
        while len(temp_new_included_voxels):
            check_voxel = temp_new_included_voxels.pop()
            check_angle = temp_new_included_angles.pop()
            front_angle_diffs = np.array([
                twodir_distances.get((check_angle,front_angle),
                                    twodir_distances[(front_angle,check_angle)]) \
                            for front_angle in temp_new_included_angles])
            if np.any(front_angle_diffs > max_angle):
                excluded_voxels.add(check_voxel)
            else:
                new_included_voxels.add(check_voxel)
                included_angles.add(check_angle)

        # Add the previous front voxels to internal voxels
        internal_voxels = internal_voxels | front_voxels
        if len(new_included_voxels) == 0:
            return new_included_voxels, internal_voxels, excluded_voxels, included_angles

        return grow_2dir(new_included_voxels, internal_voxels, excluded_voxels,
                         included_angles, max_angle)

    def segment(self, adc_min, max_num_peaks=2, max_angle=20.):
        """
        Segments a 3D volume by growing regions that contain
        similar angles
        """

        # Pre-compute the angular distances between possible peaks
        # fills in self.angle_diffs, self.twodir_distances
        self.precompute_distances()

        # Set self.peaks, self.n_fibers to reflect adc_min and max_num_peaks
        self.calculate_peaks(adc_min, max_num_peaks)

        # Create a graph for 1-direction voxels
        onedir_indices = np.flatnonzero(self.n_fibers==1)
        onedir_voxels = set(onedir_indices)
        n_onedir = len(onedir_voxels)
        self.G1 = nk.graph.Graph(int(self.nvoxels), weighted=False, directed=False)
        for starting_index in onedir_indices:
            starting_voxel = self.voxel_coords[starting_index]
            for i, name in enumerate(neighbor_names):
                coord = tuple(starting_voxel + lps_neighbor_shifts[name])
                to_index = self.coordinate_lut.get(coord,-9999)
                if to_index == -9999:
                    continue
                if not to_index in onedir_voxels:
                    continue
                if not self.G1.hasEdge(starting_index, to_index):
                    self.G1.addEdge(starting_index, to_index)


        self.G2 = nk.graph.Graph(int(self.nvoxels), weighted=False, directed=False)
        for starting_index in onedir_indices:
            starting_voxel = self.voxel_coords[starting_index]
            for i, name in enumerate(neighbor_names):
                coord = tuple(starting_voxel + lps_neighbor_shifts[name])
                to_index = self.coordinate_lut.get(coord,-9999)
                if to_index == -9999:
                    continue
                if not to_index in onedir_voxels:
                    continue
                if not self.G2.hasEdge(starting_index, to_index):
                    self.G2.addEdge(starting_index, to_index)

        # An empty array for label assignments
        labels = np.zeros(self.nvoxels)
        labelnum = 1

        # Do the one-direction voxels
        while len(onedir_voxels):
            seed_voxel = random.sample(onedir_voxels, 1)[0]
            front_voxels, internal_voxels, excluded_voxels, included_angles = \
                self.grow_1dir(set([seed_voxel]), set([seed_voxel]), set([]),
                          set([self.peaks[seed_voxel]]), max_angle)

            # Remove labeled voxels from available voxels
            onedir_voxels = onedir_voxels - internal_voxels
            labels[np.array(list(internal_voxels))] = labelnum
            labelnum += 1

        # Do the two-direction voxels
        while len(twodir_voxels):
            seed_voxel = random.sample(available_voxels,1)[0]
            front_voxels, internal_voxels, excluded_voxels, included_angles = \
                self.grow_2dir(set([seed_voxel]), set([seed_voxel]), set([]),
                          set([self.peaks[seed_voxel]]), max_angle)

            # Remove labeled voxels from available voxels
            twodir_voxels = twodir_voxels - internal_voxels
            labels[np.array(list(internal_voxels))] = labelnum
            labelnum += 1

        return labels
