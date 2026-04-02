#!/bin/bash

set -e  # stop script if any command fails
N=$1
CKPT=$2
SCALE=$3

echo "(1/3) Sample points and create filename txt files N=$N"
bash sample_scannet_npts.sh "$N"
echo ""
echo "(2/3) Inference for Scannet N=$N CHECKPOINT=$CKPT SCALE=$SCALE"
bash infer_scannet.sh "$N" "$CKPT" "$SCALE"
echo ""
echo "(3/3) Evaluation for Scannet N=$N CHECKPOINT=$CKPT SCALE=$SCALE"
bash eval_scannet.sh "$N" "$CKPT" "$SCALE"
echo ""
echo "N=$N finished Scannet CHECKPOINT=$CKPT SCALE=$SCALE"