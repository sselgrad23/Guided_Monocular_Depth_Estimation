#!/bin/bash
set -e

# Load necessary modules
#module load eth_proxy stack/2024-05 gcc/13.2.0 python_cuda/3.10.13 cudnn/9.2.0
#module load cmake/3.27.7 ninja/1.11.1 curl/8.4.0-mxgyalo

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cudnn/9.2.0
module load cmake/3.27.7 ninja/1.11.1 curl/8.4.0-s6dtj75

N=$1
CKPT=$2

HOME_DIRECTORY="/cluster/scratch/sselgrad"
WORKING_DIRECTORY="$HOME_DIRECTORY/diffusers"
VENV="$WORKING_DIRECTORY/venv_plz_work/diffusers"
export MODEL_DIR="GonzaloMG/stable-diffusion-e2e-ft-depth"
export CONTROLNET_DIR="$WORKING_DIRECTORY/examples/controlnet"
export CHECKPOINT_DIR="$CONTROLNET_DIR/checkpoints_both/SCALE_${SCALE}/depth.ckpt"
export HF_HOME="$HOME_DIRECTORY/huggingface_cache"
export EVAL_DATASET="$HOME_DIRECTORY/eval_dataset/nyuv2/nyu_labeled_extracted"

if [ ! -d "$VENV" ]; then
    echo "Venv not found, creating..."
    python -m venv "$VENV"
    source "$VENV/bin/activate"
else
    echo "Venv found, activating..."
    source "$VENV/bin/activate"
fi

pip install --upgrade pip
#pip install accelerate
#pip install -r examples/controlnet/requirements.txt
#pip install opencv-python matplotlib
#pip install torchmetrics
#pip install wandb
#wandb login        # paste your API key when prompted

#pip uninstall -y diffusers
#pip install -e .

#cd examples/controlnet
#ls requirements.txt
#pip install -r requirements.txt

echo "🎆 Sampling $N points"
python sample_nyuv2_npts.py \
     --base_dir $EVAL_DATASET \
     --split both \
     --num_points $N \
     --transform_order "sampling" \
     --seed 42

if [[ -f "$EVAL_DATASET/test_sparse_sampled_$N.txt" && -f "$EVAL_DATASET/train_sparse_sampled_$N.txt" ]]; then
    (cat "$EVAL_DATASET/test_sparse_sampled_$N.txt"; echo ""; cat "$EVAL_DATASET/train_sparse_sampled_$N.txt") > "$EVAL_DATASET/nyuv2_${N}.txt"
else
    echo "One or both input files do not exist." >&2
    exit 1
fi

echo "🎇 Sampling complete"