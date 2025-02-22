@echo off
setlocal enabledelayedexpansion

set INPUT_DIR=mydata\preprocessed_t1
set OUTPUT_DIR=my_results
set MODEL_PATH=pvs_models\v2\T1.PVS.v2
set MODELS=-m %MODEL_PATH%\ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_0_bestvalloss ^
           -m %MODEL_PATH%\ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_1_bestvalloss ^
           -m %MODEL_PATH%\ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_2_bestvalloss ^
           -m %MODEL_PATH%\ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_3_bestvalloss ^
           -m %MODEL_PATH%\ResUnet3D-8.9.2-1.5-T1.VRS_fold_VRS_1x5_fold_4_bestvalloss

for %%f in (%INPUT_DIR%\*.nii.gz) do (
    set FILE_NAME=%%~nf
    echo Processing !FILE_NAME! ...
    python predict_one_file.py -i "%%f" -o "%OUTPUT_DIR%\!FILE_NAME!.nii.gz" %MODELS%
)

echo All files processed!
pause