import os.path as op
import nibabel as nib
import numpy as np
import logging
logger = logging.getLogger(__name__)

hdr = np.array((b'TRACK', [ 98, 121, 121], [ 2.,  2.,  2.], [ 0.,  0.,  0.], 0, [b'', b'', b'', b'', b'', b'', b'', b'', b'', b''], 0, [b'', b'', b'', b'', b'', b'', b'', b'', b'', b''], [[-2.,  0.,  0.,  0.], [ 0., -2.,  0.,  0.], [ 0.,  0.,  2.,  0.], [ 0.,  0.,  0.,  1.]], b' A diffusion spectrum imaging scheme was used, and a total of 257 diffusion sampling were acquired. The maximum b-value was 4985 s/mm2. The in-plane resolution was 2 mm. The slice thickness was 2 mm. The diffusion data were reconstructed using generalized q-sampling imaging (Yeh et al., IEEE TMI, ;29(9):1626-35, 2010) with a diffusion sampling length ratio of 1.25.\nA deterministic fiber tracking algorithm (Yeh et al., PLoS ONE 8(11): e80713', b'LPS', b'LPS', [ 1.,  0.,  0.,  0.,  1.,  0.], b'', b'', b'', b'', b'', b'', b'', 253, 2, 1000),
              dtype=[('id_string', 'S6'), ('dim', '<i2', (3,)), ('voxel_size', '<f4', (3,)), ('origin', '<f4', (3,)), ('n_scalars', '<i2'), ('scalar_name', 'S20', (10,)), ('n_properties', '<i2'), ('property_name', 'S20', (10,)), ('vox_to_ras', '<f4', (4, 4)), ('reserved', 'S444'), ('voxel_order', 'S4'), ('pad2', 'S4'), ('image_orientation_patient', '<f4', (6,)), ('pad1', 'S2'), ('invert_x', 'S1'), ('invert_y', 'S1'), ('invert_z', 'S1'), ('swap_xy', 'S1'), ('swap_yz', 'S1'), ('swap_zx', 'S1'), ('n_count', '<i4'), ('version', '<i4'), ('hdr_size', '<i4')])

class Spatial(object):
    def _set_real_affine(self, affine_img):
        if affine_img:
            img = nib.load(affine_img)
            self.real_affine = img.affine
            self.real_affine_image = affine_img
        else:
            self.real_affine = np.array([])

    def save_nifti(self, data, fname, real_affine=True, is_full_image=False):
        """
        Writes a value for each node into a 3D volume.

        Parameters:
        ===========

        data:np.ndarray
          A value for each voxel (node) in the graph.

        fname:str
          Path where the output nifti file goes. Must end with .nii or .nii.gz

        real_affine:bool
          Should the NIfTI file include a real affine (True) or use the default
          DSI Studio output space (False)
        """
        # Check that data is 1D
        if len(data.shape) > 1:
            sq = data.squeeze()
            if len(sq.shape) > 1:
                logger.critical("Can only write out 1D arrays")
                return
            data = sq
        # Are there extra nodes (eg from atlas regions)?
        if not is_full_image and len(data) > self.nvoxels:
            logger.warning("""
            Non-voxel nodes have values in this data. Only using the first %d""" % self.nvoxels)
            data = data[:self.nvoxels]

        # If the data only corresponds to nodes
        if not is_full_image:
            out_data = np.zeros(np.prod(self.volume_grid),dtype=np.float)
            out_data[self.flat_mask] = data
        else: # there is a value for each voxel
            out_data = data

        # Mimic the behavior of DSI Studio
        out_data = out_data.reshape(self.volume_grid, order="F")[::-1,::-1,:]
        if not real_affine:
            affine = self.ras_affine
        else:
            if not self.real_affine.size > 0:
                raise ValueError("No real affine is available")
            if np.sign(self.ras_affine[0,0]) != np.sign(self.real_affine[0,0]):
                out_data = out_data[::-1,:,:]
            if np.sign(self.ras_affine[1,1]) != np.sign(self.real_affine[1,1]):
                out_data = out_data[:,::-1,:]
            if np.sign(self.ras_affine[2,2]) != np.sign(self.real_affine[2,2]):
                out_data = out_data[:,:,::-1]
            affine = self.real_affine

        img = nib.Nifti1Image(out_data,affine)
        # Prevent annoying behavior in AFNI
        img.header.sform_code = 2
        img.header.qform_code = 2
        img.to_filename(fname)

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
        return oriented_array_to_lpsplus(data, img.affine, warn=warn
                                         ).flatten(order="F")[self.flat_mask]

def oriented_array_to_lpsplus(data, affine, warn=True):
    """ Takes a 3D+ array and returns a view of it
    such that the indices increase as their spatial locations
    become more left, posterior and superior
    """
    # Add dummy 4th dimension
    if data.ndim < 4:
        data = data[...,np.newaxis]

    if affine[0,0] > 0:
        data = data[::-1,:,:]
        if warn: logger.info("Flipped X")
    if affine[1,1] > 0:
        data = data[:,::-1,:]
        if warn: logger.info("Flipped Y")
    if affine[2,2] < 0:
        data = data[:,:,::-1]
        if warn: logger.info("Flipped Z")
    return data.squeeze()
