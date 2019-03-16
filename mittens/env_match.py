#!/usr/bin/env python
import numpy as np
import re
import logging
from tqdm import tqdm
import nibabel as nib
import os.path as op
import networkit
from .utils import neighbor_names, lps_neighbor_shifts
from .spatial import Spatial
from .voxel_graph import VoxelGraph
from .external import load_fixels_from_fib

logger = logging.getLogger(__name__)


def fixel_similarity(fixels1, fixels2):
    """Calculate the similarity between th sets of fixels in voxel 1 and voxel 2.

    At the moment this is a placeholder that returns the 1 / the difference in the
    number of fixels in the two voxels.

    Parameters
    ----------
        fixels1: np.ndarray (n x 3)
            Each row is a fixel vector from voxel 1.
        fixels2: np.ndarray (n x 3)
            Each row is a fixel vector from voxel 2.

    Returns
    -------
        similarity: float
            The RMSD of the two sets of fixels.
    """

    nfixels1 = fixels1.shape[0]
    nfixels2 = fixels2.shape[0]
    num_different_fixels = np.abs(nfixels1 - nfixels2)
    if num_different_fixels == 0:
        return 1.
    return 1. / num_different_fixels


class EnvironmentMatch(Spatial):
    def __init__(self, fixel_threshold=0, reconstruction="", nifti_prefix="",
                 real_affine_image="", mask_image="", cutoff_value=0.3):
        """
        Calculates similarity between pairs of ODF peak sets across voxels.

        Input Options:
        ==============

            reconstruction: str
              Path to a DSI Studio fib.gz or MRTRIX mif file containing the (f)ODFs
              on which to calculate transition probabilities.
            nifti_prefix: str
              Prefix used when for saving or loading similarity measurements.
            real_affine_image: str
              Path to a NIfTI file that contains the real affine mapping for the
              data. DSI Studio does not preserve affine mappings. If provided,
              all NIfTI outputs will be written with this affine. Otherwise the
              default affine from DSI Studio will be used.
            mask_image: str
              Path to a NIfTI file that has nonzero values in voxels that will be used
              as nodes in the graph. If none is provided, the default mask estimated from
              the ODFs is used.

        Local Environment options:
        ===========================

            cutoff_value: float

        """
        # Load input data and get spatial info
        self.label_lut = None
        self.atlas_labels = None
        self.mask_image = mask_image
        self.cutoff_value = cutoff_value
        self.real_affine = None

        if reconstruction.endswith(".fib") or reconstruction.endswith(".fib.gz"):
            self.fixels, self.flat_mask, self.volume_grid, \
                self.voxel_coords, self.coordinate_lut, self.voxel_size, \
                    self.odf_vertices = load_fixels_from_fib(
                        reconstruction, fixel_threshold=fixel_threshold)
            aff = np.ones(4, dtype=np.float)
            aff[:3] = self.voxel_size
            self.ras_affine = np.diag(aff)
        else:
            logger.critical("No valid inputs detected")

        self.nvoxels = self.flat_mask.sum()

        if op.exists(real_affine_image):
            self._set_real_affine(real_affine_image)


    def environment_similarity_graph(self, output_prefix="env_match"):
        """Calculate peak set similarity in each voxel and its neighbors"""
        G = networkit.graph.Graph(int(self.nvoxels), weighted=True, directed=False)

        for from_node, starting_voxel in tqdm(enumerate(self.voxel_coords),
                                              total=self.nvoxels):

            from_node_fixels = self.fixels[tuple(starting_voxel)]
            for nbr_name in neighbor_names:
                nbr_coord = tuple(starting_voxel + lps_neighbor_shifts[nbr_name])
                to_node = self.coordinate_lut.get(nbr_coord, -9999)
                if to_node == -9999:
                    continue
                to_node_fixels = self.fixels[nbr_coord]
                similarity = fixel_similarity(from_node_fixels, to_node_fixels)
                G.addEdge(from_node, to_node, w=similarity)

        vg = self._voxel_graph()
        vg.graph = G
        return vg

    def _voxel_graph(self):
        # Creates an appropriate VoxelGraph
        return VoxelGraph(
            # Spatial mapping
            real_affine=self.real_affine, flat_mask=self.flat_mask,
            ras_affine=self.ras_affine, voxel_size=self.voxel_size,
            volume_grid=self.volume_grid, nvoxels=self.nvoxels,
            # Pass it since we've got it
            voxel_coords=self.voxel_coords, coordinate_lut=self.coordinate_lut
            )
