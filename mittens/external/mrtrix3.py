import subprocess
import os
import os.path as op
import shutil
import tempfile
import logging
import numpy as np
from dipy.core.geometry import cart2sphere
from mittens.utils import get_dsi_studio_ODF_geometry
from mittens.spatial import oriented_array_to_lpsplus
import nibabel as nib
import json

logger = logging.getLogger(__name__)

def which(program):
    import os
    def is_exe(fpath):
        return op.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = op.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = op.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None



required_executables = ["mrinfo", "dwi2mask", "sh2amp", 
                        "dwi2mask"]

mrtrix_paths = dict([(program, which(program)) for 
                      program in required_executables])
has_mrtrix = not None in mrtrix_paths

def mrinfo(image_file):
    new_file, filename = tempfile.mkstemp(suffix=".json")
    cmd = subprocess.Popen([mrtrix_paths["mrinfo"], '-force', image_file, "-json_all",
                            filename],stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out,err = cmd.communicate()
    with open(filename,"r") as f:
        data = json.load(f)
    os.remove(filename)
    return data
    
   
def popen_run(arg_list):
    cmd = subprocess.Popen(arg_list, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    out,err = cmd.communicate()
    logger.info(out)
    logger.info(err)
    
def load_mif(mif_file, temp_root=None, sphere="odf8",
                     mask=None):
    """Reads a mrtrix3 .mif file that contains spherical
    harmonics coefficients for FODs. The FODs are sampled
    on one of the spheres supported by MITTENS.     
    
    This function creates a temporary directory and runs
    some mrtrix3 commands.
    
    """
    if not has_mrtrix:
        raise OSError("Not all mrtrix3 executables found")
    
    # Where to make a temporary directory
    if temp_root is None:
        if tempfile.tempdir is not None:
            temp_root = tempfile.tempdir
        else:
            temp_root = os.getcwd()
    working_dir = tempfile.mkdtemp(dir=temp_root)
    logger.info("Using temporary directory %s", working_dir)
    
    # Write the requested sphere in polar coordinates
    verts, faces = get_dsi_studio_ODF_geometry(sphere)
    num_dirs, _ = verts.shape
    hemisphere = num_dirs // 2
    x,y,z = verts[:hemisphere].T
    # Convert to DSI Studio LPS+ from MRTRIX3 RAS+
    _, theta, phi = cart2sphere(x, y, z)
    dirs_txt = op.join(working_dir, "directions.txt")
    np.savetxt(dirs_txt, np.column_stack([phi,theta]))
    
    # Convert to amplitudes for each odf direction
    odf_amplitudes_nii = op.join(working_dir,"amplitudes.nii.gz")
    popen_run([mrtrix_paths["sh2amp"], "-nonnegative", mif_file,
        dirs_txt, odf_amplitudes_nii])
    
    if not op.exists(odf_amplitudes_nii):
        raise FileNotFoundError("Unable to create %s", 
                                odf_amplitudes_nii)
    amplitudes_img = nib.load(odf_amplitudes_nii)
    ampl_data = amplitudes_img.get_data()
    
    # Configure a mask.
    temp_mask = op.join(working_dir, "temp_mask.nii.gz")
    if mask is None:
        logger.warning("Creating mask from amplitudes")
        ampl_mask = ampl_data.sum(3) > 1e-6
        nib.Nifti1Image(ampl_mask.astype(np.float), 
                        amplitudes_img.affine).to_filename(temp_mask)
        
    # Convert to nifti if it's a mif
    elif mask.endswith(".mif"):
        popen_run(["mrconvert", mask, temp_mask])
        if not op.exists(temp_mask):
            raise FileNotFoundError("Unable to create %s from %s",
                                    temp_mask, mask)
    else:
        temp_mask = mask
        
    mask_img = nib.load(temp_mask)
    if not np.allclose(mask_img.affine, amplitudes_img.affine):
        raise ValueError("Differing orientation between mask and amplitudes")
    if not mask_img.shape == amplitudes_img.shape[:3]:
        raise ValueError("Differing grid between mask and amplitudes")
    real_affine = mask_img.affine
    volume_grid = mask_img.shape
    voxel_size = np.array(mask_img.header.get_zooms())
    
    # Get LPS+ flat mask
    flat_mask = oriented_array_to_lpsplus(mask_img.get_data(), 
                        real_affine).flatten(order="F") > 0
    
    # Get LPS+ flat odf data
    oriented_4d = oriented_array_to_lpsplus(ampl_data, real_affine)
    odf_array = oriented_4d.reshape(-1, oriented_4d.shape[3], order="F")
    odf_array = np.ascontiguousarray(odf_array[flat_mask,:],dtype=np.float64)
    
    return flat_mask, volume_grid, odf_array, real_affine, voxel_size
    
    
        
    
    
        
    
        
    
        
        
    
        
    
    
    