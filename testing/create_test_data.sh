#!/bin/bash
root=`pwd`
test_dirs="test_data DSI-Studio MRTRIX3 Dipy"

for test_dir in ${test_dirs}
do

   if [ -d ${test_dir} ]; then 
       rm -rf ${test_dir}
   fi
   mkdir ${test_dir}

done

wget 'http://tractometer.org/downloads/downloads/ismrm_challenge_2015/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2.zip'

unzip ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2.zip
rm ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2.zip
bval=${root}/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/NoArtifacts_Relaxation.bvals
bvec=${root}/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/NoArtifacts_Relaxation.bvecs
dwi=${root}/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/NoArtifacts_Relaxation.nii.gz

# First use mrtrix to make some masks 
dwi_mif=${root}/MRTRIX3/dwi.mif
mask_mif=${root}/MRTRIX3/dwi_mask.mif
mask=${root}/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/dwi_mask.nii.gz
dwi_las=${root}/ISMRM_2015_Tracto_challenge_ground_truth_dwi_v2/dwi_las.nii.gz
mrconvert  -strides -1,2,-3,4 ${dwi} ${dwi_las}
mrconvert ${dwi_las} -fslgrad ${bvec} ${bval} ${dwi_mif}
dwi2mask  ${dwi_mif} ${mask_mif}
mrconvert ${mask_mif} ${mask}

process_dsi_studio () {

    cd DSI-Studio
    SRCGZ=`pwd`/dwi.src.gz
    dsi_studio \
        --action=src \
        --mask=${mask} \
        --source=${dwi} \
        --bval=${bval} \
        --bvec=${bvec} \
        --output=${SRCGZ}

    dsi_studio \
        --action=rec \
        --method=4 \
        --mask=${mask} \
        --csf_calibration=1 \
        --record_odf=1 \
        --deconvolution=1 \
        --param2=0.5 \
        --source=${SRCGZ}

    cd ${root}
}
#FIBGZ=${root}/DSI-Studio/dwi.src.gz.odf8.f5.bal.csfc.de0.5.rdi.gqi.1.2.fib.gz


process_mrtrix () {

    cd MRTRIX3
    dwi2response tournier ${dwi_mif} dwi_response.txt
    dwi2fod csd ${dwi_mif} dwi_response.txt dwi_fod.mif -mask ${mask_mif}
    #tckgen dwi_fod.mif dwi_fod.tck -seed_image ${mask_mif} -mask ${mask_mif} -select 10000
    #mrview ${dwi_mif} -tractography.load dwi_fod.tck

}



process_dsi_studio
process_mrtrix




