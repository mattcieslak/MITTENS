
# MITTENS

MITTENS is a python library that performs analytical tractography
on reconstructed diffusion-weighted MRI data. Currently the output
from DSI Studio is supported.  


## Installation

While this software can be installed like any other python package, 
it is possible to add custom compiled functions before installation.
Analytical tractography requires the specification of a set of geometric 
constraints for inter-voxel tract transition expectations can be
solved. MITTENS comes with the functions used to perform the analyses
described in [our paper], but nothing more.

You can add as many sets of geometric constraints as you like, but be 
warned that it can take awhile to compile them.  Remember you only
need to compile a solution __once__. Also beware that some
combinations of step size and turning angle maximum result in infinite
sets of turning angle sequences. Python to crash with a recursion error
in this case.

### Adding Geometric Constraints

Download the current version of MITTENS and enter its source tree:

```bash
$ git clone https://github.com/mattcieslak/MITTENS.git
$ cd mittens
```

Now, launch a python session in the mittens directory

```python
>>> from mittens.write_solution import write_solution
>>> write_solution("odf8", 35, 1.0)
```

This will write out two fortran files in the current directory. They are named
after the parameters you chose. This particular example will read data
reconstructed with 8-fold ODF tesselation, a turning angle maximum of 35
degrees and a step size of 1.0 voxel units (voxels are assumed to be
isotropic). 

After some calculations you will find 
``doubleODF_odf8_ss1_00_am35.f90`` and ``singleODF_odf8_ss1_00_am35.f90``
in the current directory. Move these files into the ``src/`` directory
and install the package with ``pip``. We recommend using an editable 
install, which will keep all these files in their current directory:

```bash
mv *.f90 src/
pip install -e .
```

This will compile everything in the ``src/`` directory into a 
python module. You can now use these geometric constraints to
calculate transition expectations.


## Preparing your data

DSI Studio can be used to reconstruct diffusion MRI using a variety
of methods.  You can use any method you like **except for DTI**. 
If you acquired DTI data, choose **GQI** reconstruction instead of 
DTI reconstruction.

![recon_opts](doc/img/recon_options.png)

DSI, GQI and QBI can be selected from this menu.  There are additional
options not listed on this GUI but are accessible through the commandline.
For example, ODF deconvolution and decomposition can be used. All
options in purple boxes can be changed, but it is required that ODF data
is written in the output file.  Critically, the ``ODF Tesselation`` 
option determines the angular resolution of your output ODFs. The choice 
here will result in ``"odf8"`` being the appropriate choice for 
the call to ``write_solutions()`` above.

Data from other diffusion MRI packages such as FSL and MRTRIX can be loaded
(theoretically) after being ![converted to DSI Studio
format](http://dsi-studio.labsolver.org/Manual/data-exchange-between-dsi-studio-and-mrtrix).

## Calculating intervoxel tract transition expectations

DSI Studio files are read directly by MITTENS:

```python
>>> from mittens import MITTENS
>>> mitns = MITTENS(input_fib="HCP.src.gz.odf8.f5rec.fy.gqi.1.25.fib.gz")
```

From here you can estimate none-ahead or one-ahead, where NIfTI-1 files are
saved for each neighbor direction:

```python
>>> mitns.estimate_singleODF("hcp")
>>> mitns.estimate_doubleODF("hcp")
```

You will find the output in the current working directory (unless you specified an 
absolute path as the argument). There is a single 3D file for each neighbor voxel
named ``hcp_doubleODF_r_prob.nii.gz``, ``hcp_singleODF_r_prob.nii.gz``, 
``hcp_doubleODF_lpi_prob.nii.gz``, etc.  There will also be CoDI and CoAsy output.
Instead of re-running the estimations again, you can create a MITTENS object 
by specifying the prefix of the NIfTI files written out during estimation.

```python
>>> nifti_mitns = MITTENS(nifti_prefix="hcp")
```

This is very fast.

## Voxel Graph Construction

MITTENS uses ``networkit`` to construct directed graphs of voxels, where edges
are weighted by the tract transition expectation from one voxel to another. 
A graph can build directly from MITTENS object.

```python
>>> mitns.build_graph(doubleODF=True, edge_weights="vs_null")
```




Credits
========
This source code was sponsored by a grant from the GE/NFL head health challenge. 
The content of the information does not necessarily reflect the position or
the policy of these companies, and no official endorsement should be inferred.

Authors
-------
 * Matt Cieslak
 * Tegan Brennan
