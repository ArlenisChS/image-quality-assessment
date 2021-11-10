#!/bin/bash
set -e

BASE_MODEL_NAME=$1
WEIGHTS_FILE=$2
N_CLASSES=$3
IMAGE_SOURCE=$4
RESULTS_DIR=$5

# predict
python -m evaluater.predict \
--base-model-name $BASE_MODEL_NAME \
--weights-file $WEIGHTS_FILE \
--n-classes $N_CLASSES \
--image-source $IMAGE_SOURCE \
--predictions-dir $RESULTS_DIR