#!/bin/bash

SCALES=(1.0)

for CONDITIONING_SCALE in "${SCALES[@]}"; do
  echo "Submitting Training with conditioning scale = $CONDITIONING_SCALE"
  sbatch --gpus=a100_80gb:1 \
  --time=72:00:00 \
  --mem-per-cpu=80G \
  --job-name=Train_Scale_${CONDITIONING_SCALE} \
  --wrap="bash train_controlnet_0909.sh $CONDITIONING_SCALE"
done