#!/bin/bash
#SAMPLES=(0 500 1000 5000 10000)
SAMPLES=(20000 40000 80000 160000 320000)
CKPTS=(0) #(20000 50000)
SCALES=(1.0)

for N in "${SAMPLES[@]}"; do
  for CHECKPOINT in "${CKPTS[@]}"; do
    for SCALE in "${SCALES[@]}"; do
      echo "Submitting VKITTI: N=$N, CHECKPOINT=$CHECKPOINT, SCALE=$SCALE"
      sbatch --gpus=rtx_3090:1 \
           --time=2:00:00 \
           --mem-per-cpu=24G \
           --job-name=VKITTI_N${N}_C${CHECKPOINT}_S${SCALE} \
           --wrap="bash experiment_vkitti.sh $N $CHECKPOINT $SCALE"
    done
  done
done

# --gpus=rtx_3090:1 \
#_C${CHECKPOINT} \