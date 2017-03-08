from __future__ import division, absolute_import, print_function
from glob import glob
import os
from numpy.distutils.misc_util import Configuration

here = os.path.abspath(os.path.dirname(__file__))
#here = os.getcwd()
README = open(os.path.join(here, 'README.md')).read()

def get_fortran_modules(f_path,top_path=None):
    config = Configuration(package_name="mittens",parent_name="",top_path=top_path)
    for fname in glob(os.path.join(f_path,"*.f90")):
        module_name = os.path.split(fname)[-1][:-4]
        config.add_extension(name = "fortran." + module_name,sources=[fname])
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(
          description       = "Mathematical solution for Intervoxel " \
                              "Tract Transition Expections requiring No Simulations",
          author            = "Matt Cieslak, Tegan Brennan",
          author_email      = "cieslak@psych.ucsb.edu",
          install_requires  = ["numpy", "scikit-learn", "tqdm", "networkit"],
          classifiers=[
              # Get strings from
              # http://pypi.python.org/pypi?%3Aaction=list_classifiers
              'Development Status :: 3 - Alpha',
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'Environment :: Console',
              'License :: OSI Approved :: BSD License',
              'Natural Language :: English',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3',
              'Topic :: Scientific/Engineering'
              ],
          **get_fortran_modules(os.path.join(here,"src"),top_path='').todict()
          )
