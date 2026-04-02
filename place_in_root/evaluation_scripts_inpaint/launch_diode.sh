#!/bin/bash
#SAMPLES=(0 500 1000 5000 10000)
SAMPLES=(80000) #20000 40000 160000 320000
CKPTS=(20000) #(0 20000 50000)
SCALES=(1.0)

for N in "${SAMPLES[@]}"; do
  for CKPT in "${CKPTS[@]}"; do
    for SCALE in "${SCALES[@]}"; do
      echo "Submitting Diode: N=$N, CHECKPOINT=$CKPT, SCALE=$SCALE"
      sbatch --gpus=rtx_3090:1 \
           --time=2:00:00 \
           --mem-per-cpu=12G \
           --job-name=DIODE_N${N}_C${CKPT}_S${SCALE} \
           --wrap="bash experiment_diode.sh $N $CKPT $SCALE"
    done
  done
done
#_C${CKPT} 
#--gpus=rtx_3090:1 \
           