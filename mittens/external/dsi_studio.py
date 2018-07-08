#!/usr/bin/env python
from __future__ import print_function
from scipy.io.matlab import loadmat
# 2 and 3 compatible
try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO
import subprocess 
import os
import gzip
import os.path as op
import numpy as np
import logging
import re
logger = logging.getLogger(__name__)

def fast_load_fibgz(fib_file):

    # Try to load a non-zipped file
    if not fib_file.endswith("gz"):
        try:
            m = loadmat(fib_file)
        except Exception:
            print("Unable to load", fib_file)
            return 
        return m

    # Load a zipped file quickly if possible
    def find_zcat():
        def is_exe(fpath):
            return os.path.exists(fpath) and os.access(fpath, os.X_OK)
        for program in ["zcat", "gzcat"]:
            for path in os.environ["PATH"].split(os.pathsep):
                path = path.strip('"')
                exe_file = os.path.join(path, program)
                if is_exe(exe_file):
                    return program
        return None

    # Check if a zcat is available on this system:
    zcatter = find_zcat()
    if zcatter is not None:
        p = subprocess.Popen([zcatter, fib_file],stdout=subprocess.PIPE)
        fh = StringIO(p.communicate()[0])
        return loadmat(fh)

    with gzip.open(fib_file,"r") as f:
        print( "Loading with python gzip. To load faster install zcat or gzcat.")
        try:
            m = loadmat(f)
        except Exception:
            print( "Unable to read", fib_file)
            return
        return m


def load_fib(fib_file, expected_vertices=None, 
             real_affine_image=None,sphere="odf8"):
    f = fast_load_fibgz(fib_file)
    volume_grid = f['dimension'].squeeze()
    voxel_size = f['voxel_size'].squeeze()
    
    # Check that this fib file matches what we expect
    fib_odf_vertices = f['odf_vertices'].T
    #matches = np.allclose(expected_vertices, fib_odf_vertices)
    #if not matches:
    #    logger.critical("ODF Angles in fib file do not match %s", self.odf_resolution)
    #    raise ValueError()
    
    # Create a contiguous ODF matrix, skipping all zero rows
    logger.info("Loading DSI Studio ODF data")
    odf_vars = [k for k in f.keys() if re.match("odf\\d+",k)]
    
    if len(odf_vars) == 0:
        raise ValueError("No ODF data present in %s", fib_file)
    
    valid_odfs = []
    flat_mask = f["fa0"].squeeze() > 0 
    for n in range(len(odf_vars)):
        varname = "odf%d" % n
        odfs = f[varname]
        odf_sum = odfs.sum(0)
        odf_sum_mask = odf_sum > 0
        valid_odfs.append(odfs[:,odf_sum_mask].T)
    odf_array = np.row_stack(valid_odfs).astype(np.float64)
    
    # Get the real affine
    if not op.exists(real_affine_image):
        logger.warning("Unable to load real affine image %s", 
                       real_affine_image)
        real_affine = np.array([])
    else:
        real_affine = nib.load(real_affine_image).affine
    
    return flat_mask, volume_grid, odf_array, real_affine, voxel_size