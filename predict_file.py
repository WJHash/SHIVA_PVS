# # predict something from one multi-modal nifti images
# Tested with Python 3.7, Tensorflow 2.7
# @author : Philippe Boutinaud - Fealinx
import gc
import os
import time
import numpy as np
from pathlib import Path
import argparse
import nibabel
import tensorflow as tf


def _load_image(filename):
    dataNii = nibabel.load(filename)
    # load file and add dimension for the modality
    image = dataNii.get_fdata(dtype=np.float32)[..., np.newaxis]
    return image, dataNii.affine


# Script parameters
parser = argparse.ArgumentParser(
    description="Run inference with tensorflow models(s) on an image that may be built from several modalities"
)
parser.add_argument(
    "-i", "--input",
    type=Path,
    required=True,
    help="input directory containing .nii.gz files")

parser.add_argument(
    "-m", "--model",
    type=Path,
    action='append',
    help="(multiple) prediction models")

parser.add_argument(
    "-b", "--braimask",
    type=Path,
    help="brain mask image")

parser.add_argument(
    "-o", "--output",
    type=Path,
    required=True,
    help="output directory for the predictions")

parser.add_argument(
    "-g", "--gpu",
    type=int,
    default=0,
    help="GPU card ID, default 0; for CPU use -1")

parser.add_argument(
    "--verbose",
    help="increase output verbosity",
    action="store_true")

args = parser.parse_args()

_VERBOSE = args.verbose

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
if _VERBOSE:
    if args.gpu >= 0:
        print(f"Trying to run inference on GPU {args.gpu}")
    else:
        print("Trying to run inference on CPU")

# The tf model files for the predictors, the prediction will be averaged
predictor_files = args.model
if len(predictor_files) == 0:
    raise ValueError("ERROR : No model given on command line")

input_dir = args.input
output_dir = args.output
brainmask = args.braimask

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Get all .nii.gz files in the input directory
input_files = list(input_dir.glob("*.nii.gz"))

if len(input_files) == 0:
    raise ValueError(f"ERROR : No .nii.gz files found in {input_dir}")

# Load brainmask if given (and get the affine & shape from it)
affine = None
image_shape = None
if brainmask is not None:
    brainmask, aff = _load_image(brainmask)
    image_shape = brainmask.shape
    if affine is None:
        affine = aff

# Process each file
for input_file in input_files:
    if _VERBOSE:
        print(f"Processing file: {input_file}")

    # Load and/or build image from modalities
    images = []
    image, aff = _load_image(input_file)
    if affine is None:
        affine = aff
    if image_shape is None:
        image_shape = image.shape
    else:
        if image.shape != image_shape:
            raise ValueError(
                f'Images have different shape {image_shape} vs {image.shape} in {input_file}'  # noqa: E501
            )
    if brainmask is not None:
        image *= brainmask
    images.append(image)
    # Concat all modalities
    images = np.concatenate(images, axis=-1)
    # Add a dimension for a batch of one image
    images = np.reshape(images, (1,) + images.shape)

    chrono0 = time.time()
    # Load models & predict
    predictions = []
    for predictor_file in predictor_files:
        tf.keras.backend.clear_session()
        gc.collect()
        try:
            model = tf.keras.models.load_model(
                predictor_file,
                compile=False,
                custom_objects={"tf": tf})
        except Exception as err:
            print(f'\n\tWARNING : Exception loading model : {predictor_file}\n{err}')
            continue
        print('INFO : Predicting fold :', predictor_file.stem)
        prediction = model.predict(
            images,
            batch_size=1
            )
        if brainmask is not None:
            prediction *= brainmask
        predictions.append(prediction)

    # Average all predictions
    predictions = np.mean(predictions, axis=0)

    chrono1 = (time.time() - chrono0) / 60.
    if _VERBOSE:
        print(f'Inference time : {chrono1} sec.')

    # Save prediction
    output_file = output_dir / input_file.name
    nifti = nibabel.Nifti1Image(predictions[0], affine=affine)
    nibabel.save(nifti, output_file)

    if _VERBOSE:
        print(f'\nINFO : Done with predictions -> {output_file}\n')

# %%
