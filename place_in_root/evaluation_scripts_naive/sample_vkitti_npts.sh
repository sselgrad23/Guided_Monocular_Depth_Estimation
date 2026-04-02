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

DATASET_ROOT="$HOME_DIRECTORY/train_data"
VKITTI_ROOT="$DATASET_ROOT/vkitti_eval"
DEPTH_ROOT="$VKITTI_ROOT/depth"
RGB_ROOT="$VKITTI_ROOT/rgb"
OUT_ROOT="$VKITTI_ROOT/sparse_depth_${N}"
JSONL_FILE="metadata_eval.jsonl"
TXT_FILE="vkitti_${N}.txt"


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

echo "🎆 Sampling $N points: $VKITTI_ROOT"
python "$WORKING_DIRECTORY/vkitti_sample_7-2-1-1.py" \
    --depth_root "$DEPTH_ROOT" \
    --out_root "$OUT_ROOT" \
    --rgb_root "$RGB_ROOT" \
    --num_points "$N" \
    --save_png \
    --jsonl_name "$JSONL_FILE" \
    --txt_name "$TXT_FILE"

echo "🎇 Sampling complete"