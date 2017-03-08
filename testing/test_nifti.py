#!/usr/bin/env python
import logging
logging.basicConfig(level=logging.INFO)
from time import time

from MITTENS.mittens import MITTENS
 
fib_input = \
"/spark-data/reconstructed/dsi-studio/100307/100307.src.gz.odf8.f5rec.fy.gqi.1.25.fib.gz"
t0 = time()
tester = MITTENS(fib_input)
t1 = time()
tester.estimate_first_order("testn")
t2 = time()
tester.estimate_second_order("testn")
t3 = time()

gg = raw_input("Continue:")

print "times:"
print "loading data", t1-t0
print "first order", t2-t1
print "second order", t3-t2


nifti_tester = MITTENS(nifti_prefix="testn")

