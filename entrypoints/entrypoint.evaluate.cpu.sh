#!/bin/bash
set -e

BASE_MODEL_NAME=$1
WEIGHTS_FILE=$2
N_CLASSES=$3
IMAGE_JSON=$4
IMAGE_DIR=$5
RESULTS_DIR=$6

# predict
python -m evaluater.evaluate \
--base-model-name $BASE_MODEL_NAME \
--weights-file $WEIGHTS_FILE \
--n-classes $N_CLASSES \
--image-json $IMAGE_JSON \
--image-dir $IMAGE_DIR \
--results-dir $RESULTS_DIR
