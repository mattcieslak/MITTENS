#!/bin/bash
rm *nii* *mif *fib*
wget 'https://upenn.box.com/shared/static/25h8ozw929xhn1wmtj4wndrd29ym9xb7.tck'
mv 25h8ozw929xhn1wmtj4wndrd29ym9xb7.tck ismrm2015.tck
tckmap -vox 5 -tod 6 ismrm2015.tck TOD_5mm.nii.gz
3dresample -orient RAI -inset TOD_5mm.nii.gz -prefix TOD_5mm_LPS+.nii.gz
mrconvert TOD_5mm_LPS+.nii.gz TOD_5mm_LPS+.mif


3dcalc -a power_5mm.nii -expr 'step(a)' -prefix mask_5mm.nii.gz
tckmap -vox 2 -tod 6 ismrm2015.tck TOD_2mm.nii.gz
3dresample -orient RAI -inset TOD_2mm.nii.gz -prefix TOD_2mm_LPS+.nii.gz
mrconvert TOD_2mm_LPS+.nii.gz TOD_2mm_LPS+.mif

CONVERT=/Users/mcieslak/projects/qsiprep/qsiprep/cli/convertODFs.py
python $CONVERT \
    --mif TOD_2mm_LPS+.mif \
    --fib TOD_2mm.fib
