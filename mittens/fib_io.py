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

def load_fibgz(fib_file):

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



