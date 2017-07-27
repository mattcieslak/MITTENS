# -*- coding: utf-8 -*-
from __future__ import print_function, division, unicode_literals, absolute_import
from .base import MittensBaseInterface, IFLOGGER

import os.path as op
import numpy as np
from nipype.interfaces.base import (traits, File, isdefined,
                     BaseInterfaceInputSpec, TraitedSpec)
from glob import glob

class VoxelGraphInputSpec(BaseInterfaceInputSpec):
    # Traits for graph construction
    matfile = File( exists=True, 
            desc="Matfile containing voxel graph and metadata")
    real_affine_image = File(exists=True,
            desc="Image oriented how output images should be oriented")
    use_real_affine = traits.Bool(True, desc="Write output image(s) with a real affine")

class CentralityInputSpec(VoxelGraphInputSpec):
    atlas_image = File(exists=True, mandatory=False,
            desc="Atlas to add to the voxel graph before calculating centrality")
    nsamples = traits.Int(500, usedefault=True)
    normalized = traits.Bool(True, usedefault=True)
    parallel = traits.Bool(False, usedefault=True)
    output_image = traits.Str("centrality.nii.gz", usedefault=True)

class CentralityOutputSpec(TraitedSpec):
    output_image = File('')

class Centrality(MittensBaseInterface):
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
    input_spec  = CentralityInputSpec
    output_spec = CentralityOutputSpec

    def _run_interface(self, runtime):
        from mittens import VoxelGraph
        vg = VoxelGraph(matfile=self.inputs.matfile)
        if isdefined(self.inputs.atlas_image) and op.exists(self.inputs.atlas_image):
            vg.add_atlas(self.inputs.real_affine_image, connect_to_voxels=True)

        IFLOGGER.info("Running Betweenness")
        scores = vg.voxelwise_approx_betweenness(nSamples=self.inputs.nsamples, 
                    normalized=self.inputs.normalized, parallel=self.inputs.parallel)

        # Save with real affine
        if self.inputs.use_real_affine:
            if isdefined(self.inputs.real_affine_image):
                vg._set_real_affine(self.inputs.real_affine_image)
            vg.save_nifti(scores,self.inputs.output_image, real_affine=True)
        else:
            # Save with DSI Studio affine
            vg.save_nifti(scores,self.inputs.output_image)

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['output_image'] = op.abspath(self.inputs.output_image)
        return outputs
