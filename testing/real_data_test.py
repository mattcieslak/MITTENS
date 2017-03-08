#!/usr/bin/env python
from glob import glob
import logging
logging.basicConfig(level=logging.INFO)

def run_fib(fib_input):
    import re
    from MITTENS.mittens import MITTENS
    odf_order = re.match("^.*(odf\\d).*$", fib_input).groups()[0]
    print fib_input
    tester = MITTENS(fib_input,odf_resolution=odf_order)
    tester.estimate_first_order(fib_input)
    tester.estimate_second_order(fib_input)

import os.path as op
fib_inputs = glob(
    "/spark-data/ODFReeb/matt/topo_neuro/examples/real_data/*.fib.gz")
fib_inputs = [fib for fib in fib_inputs if not op.exists(fib+"_order2_l_prob.nii.gz")]
from ipyparallel import Client
import os

# Set up the engines
rc = Client()
dview = rc[:]

dview.map_sync(os.chdir, ["/spark-data/ODFReeb/matt/topo_neuro/release/"]*len(rc))
results =  dview.map_async( run_fib, fib_inputs)
results.wait_interactive()
