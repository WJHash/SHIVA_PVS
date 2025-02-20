#!/bin/bash
MODEL_DIR=./PVS/v1/T1.PVS
IMAGE_DIR=./images
python ./predict_one_file.py \
    --verbose --gpu 0 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_0_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_1_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_2_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_3_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_4_model.h5 \
    -m $MODEL_DIR/20211030-162753_Unet3Dv2-10.7.2-1.8-T1.VRS_fold_1x6_pi_fold_5_model.h5 \
    -i $IMAGE_DIR/test_T1_Axial_resampled_111_cropped_intensity_normed.nii.gz \
    -o ./predicted/test.nii.gz


python predict_one_file.py -i mydata/0106_0000.nii.gz -o my_results/0106_000.nii.gz -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_0_bestvalloss -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_1_bestvalloss -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_2_bestvalloss -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_3_bestvalloss -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_4_bestvalloss

python predict_one_file.py -i mydata/preprocessed_t1/ -o my_results/ -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_0_bestvalloss -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_1_bestvalloss -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_2_bestvalloss -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_3_bestvalloss -m pvs_models/v2/T1.PVS.v2/ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_4_bestvalloss