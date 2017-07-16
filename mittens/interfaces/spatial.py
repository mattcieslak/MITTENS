# -*- coding: utf-8 -*-
from __future__ import print_function, division, unicode_literals, absolute_import
from .base import MittensBaseInterface, IFLOGGER

import os.path as op
import numpy as np
from nipype.interfaces.base import (traits, File, isdefined,
                     BaseInterfaceInputSpec, TraitedSpec)
from glob import glob

class FixAffineInputSpec(BaseInterfaceInputSpec):
    dsi_studio_image = File(exists=True, usedefault=True, 
                        desc=('NIfTI image with DSI Studio affine'))
    real_affine_image = File(exists=True, mandatory=True, 
                        desc=('NIfTI image with real affine to use'))
    output_name = traits.Str("real_affine.nii.gz", mandatory=False, usedefault=True,
                        desc=('File name for output (ends with nii[.gz])'))


class FixAffineOutputSpec(TraitedSpec):
    fixed_affine_image = File(desc='Data from DSI Studio image with a real affine')

class FixAffine(MittensBaseInterface):

    """
    Replaces a DSI Studio affine with a real affine.
    """
    input_spec = FixAffineInputSpec
    output_spec = FixAffineOutputSpec

    def _run_interface(self, runtime):
        from os.path import abspath
        import nibabel as nib
        dsi_img = nib.load(self.inputs.dsi_studio_image)
        ants_img = nib.load(self.inputs.real_affine_image)
        dsi_affine = dsi_img.affine
        ants_affine = ants_img.affine
        data = dsi_img.get_data()

        if np.sign(dsi_affine[0,0]) != np.sign(ants_affine[0,0]):
            data = data[::-1,:,:]
        if np.sign(dsi_affine[1,1]) != np.sign(ants_affine[1,1]):
            data = data[:,::-1,:]
        if np.sign(dsi_affine[2,2]) != np.sign(ants_affine[2,2]):
            data = data[:,:,::-1]

        nib.Nifti1Image(data,ants_affine,header=ants_img.get_header()
                ).to_filename(self.inputs.output_name)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['fixed_affine_image'] = op.abspath(self.inputs.output_name)
        return outputs
