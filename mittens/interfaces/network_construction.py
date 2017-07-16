# -*- coding: utf-8 -*-
from __future__ import print_function, division, unicode_literals, absolute_import
from .base import MittensBaseInterface, IFLOGGER

import os.path as op
import numpy as np
from nipype.interfaces.base import (traits, File, isdefined,
                     BaseInterfaceInputSpec, TraitedSpec)
from glob import glob

class VoxelGraphConstructionInputSpec(BaseInterfaceInputSpec):
    fibgz_file = File(exists=True, mandatory=True, 
                        desc=('fib.gz (with ODF data) file from DSI Studio'))
    odf_resolution = traits.Enum(("odf8", "odf6", "odf4"), usedefault=True)
    nifti_prefix = traits.Str("mittens", usedefault=True, 
                        desc=('output prefix for file names'))
    real_affine_image = File(exists=True, mandatory=False, 
                        desc=('NIfTI image with real affine to use'))
    mask_image = File(exists=True, mandatory=False, 
                        desc=('Only include non-zero voxels from this mask'))
    step_size = traits.Float(np.sqrt(3)/2, usedefault=True, 
                        desc=('Step size (in voxels)'))
    angle_max = traits.Float(35., usedefault=True, 
                        desc=('Turning angle maximum (in degrees)'))
    angle_weights = traits.Enum(("flat", "weighted"), usedefault=True, 
                        desc=('How to weight sequential turning angles'))
    angle_weighting_power = traits.Float(1., usedefault=True, desc=(
      'Sharpness of conditional turning angle probability distribution(in degrees)'))
    normalize_doubleODF = traits.Bool(True,usedefault=True,desc=("This should be True"))

class VoxelGraphConstructionOutputSpec(TraitedSpec):
    singleODF_CoDI = File(desc='')
    doubleODF_CoDI = File(desc='')
    doubleODF_CoAsy = File(desc='')
    singleODF_probabilities = traits.List(desc=(''))
    doubleODF_probabilities = traits.List(desc=(''))
    nifti_prefix = traits.Str('')

class VoxelGraphConstruction(MittensBaseInterface):

    """
    Builds a voxel graph from a set of transition probability images

    .. [Cieslak2017] Cieslak, M., et al. NeuroImage 2017?.
      Analytic tractography: A closed-form solution for estimating local white matter
      connectivity with diffusion MRI


    Example
    -------

    >>> from mittens.interfaces import VoxelGraphConstruction
    >>> vgc = VoxelGraphConstruction()
    >>> vgc.inputs.nifti_prefix = 'mittens'
    >>> res = vgc.run() # doctest: +SKIP
    """
    input_spec  = VoxelGraphConstructionInputSpec
    output_spec = VoxelGraphConstructionOutputSpec

    def _run_interface(self, runtime):
        from mittens import MITTENS
        mitns = MITTENS(
                 nifti_prefix=self.inputs.nifti_prefix,
                 odf_resolution=self.inputs.odf_resolution,
                 real_affine_image = self.inputs.real_affine_image,
                 mask_image = self.inputs.mask_image,
                 step_size = self.inputs.step_size,
                 angle_max = self.inputs.angle_max,
                 angle_weights = self.inputs.angle_weights,
                 angle_weighting_power = self.inputs.angle_weighting_power,
                 normalize_doubleODF= self.inputs.normalize_doubleODF
                 )

        IFLOGGER.info('Calculating transition probabilities')
        mitns.calculate_transition_probabilities(output_prefix=self.inputs.nifti_prefix)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        prefix = op.abspath(self.inputs.nifti_prefix)
        outputs['singleODF_CoDI'] = prefix + '_singleODF_CoDI.nii.gz'
        outputs['doubleODF_CoDI'] = prefix + '_doubleODF_CoDI.nii.gz'
        outputs['doubleODF_CoAsy'] = prefix + '_doubleODF_CoAsy.nii.gz'
        outputs['singleODF_probabilities'] = glob(prefix+"*_singleODF_*_prob.nii.gz")
        outputs['doubleODF_probabilities'] = glob(prefix+"*_doubleODF_*_prob.nii.gz")
        return outputs
