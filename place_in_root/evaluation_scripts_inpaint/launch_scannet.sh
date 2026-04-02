#!/bin/bash
#SAMPLES=(0 500 1000 5000 10000)
SAMPLES=(20000 40000 160000 200000) #80000
CKPTS=(20000) #(20000 50000)
SCALES=(1.0)

for N in "${SAMPLES[@]}"; do
  for CKPT in "${CKPTS[@]}"; do
    for SCALE in "${SCALES[@]}"; do
      echo "Submitting Scannet: N=$N, CHECKPOINT=$CKPT, SCALE=$SCALE"
      sbatch --gpus=1 \
           --time=2:00:00 \
           --mem-per-cpu=12G \
           --job-name=SCANNET_N${N}_C${CKPT}_S${SCALE}  \
           --wrap="bash experiment_scannet.sh $N $CKPT $SCALE"
    done
  done
done

# 
#