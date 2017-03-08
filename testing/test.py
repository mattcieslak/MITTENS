#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.INFO)
from mittens import MITTENS
from time import time
 
use_fib = False
if use_fib:
    fib_input = \
    "/spark-data/reconstructed/dsi-studio/100307/100307.src.gz.odf8.f5rec.fy.gqi.1.25.fib.gz"
    t0 = time()
    mitns = MITTENS(fib_input)
    t1 = time()
    mitns.estimate_one_ahead("tmp")
    t2 = time()
    mitns.estimate_none_ahead("tmp")
    t3 = time()


    print( "times:")
    print( "loading data", t1-t0)
    print( "None Ahead took ", t3-t2)
    print( "One Ahead took ", t2-t1)

else:
    mitns = MITTENS(nifti_prefix="tmp")

# Test the graph functions
mitns.build_graph()

# Add an atlas
mitns.add_atlas()


