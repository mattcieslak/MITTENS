
def _load_odf_array(self, odf_array):
    self.volume_grid = odf_array.shape[:3]
    aff = np.ones(4,dtype=np.float)
    aff[:3] = self.real_affine[0][0]		
    self.ras_affine = np.diag(aff)
    numSamples = odf_array.shape[-1]//2
    odf_array = odf_array[::-1,::-1,:,:numSamples]
    odf_array = odf_array.reshape(np.prod(odf_array.shape[:3]),odf_array.shape[-1], order="F")
    odf_array[odf_array < 0] = 0
    odf_sum = odf_array.sum(1)
    odf_sum_mask = odf_sum > 0
    if op.exists(self.mask_image):
        mask = nib.load(self.mask_image)
        self.flat_mask = mask.get_data()
        self.flat_mask = self.flat_mask[::-1,::-1,:]
        self.flat_mask = self.flat_mask.flatten(order="F") > 0
    else:
        self.flat_mask = np.ones(self.volume_grid, dtype = np.bool).flatten()
    self.flat_mask = odf_sum_mask & self.flat_mask
    self.odf_values = odf_array[self.flat_mask,:].astype(np.float64)

    self.nvoxels = self.flat_mask.sum()
    self.voxel_coords = np.array(np.unravel_index(
        np.flatnonzero(self.flat_mask), self.volume_grid, order="F")).T
    self.coordinate_lut = dict(
        [(tuple(coord), n) for n,coord in enumerate(self.voxel_coords)])

    norm_factor = self.odf_values.sum(1)
    norm_factor[norm_factor == 0] = 1.
    self.odf_values = self.odf_values / norm_factor[:,np.newaxis] * 0.5
    logger.info("Loaded ODF data: %s",str(self.odf_values.shape))
    self.orientation = "lps"


    self.orientation = "lps"
    logger.info("Loaded %s", path)
    # DSI Studio stores data in LPS+
    #aff = aff * np.array([-1,-1,1,1])
    self.ras_affine = np.diag(aff)
    self.voxel_size = aff[:3]
