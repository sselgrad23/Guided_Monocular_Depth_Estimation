#!/bin/bash
set -e

# Load necessary modules
#module load eth_proxy stack/2024-05 gcc/13.2.0 python_cuda/3.10.13 cudnn/9.2.0
#module load cmake/3.27.7 ninja/1.11.1 curl/8.4.0-mxgyalo

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cudnn/9.2.0
module load cmake/3.27.7 ninja/1.11.1 curl/8.4.0-s6dtj75

N=$1
CKPT=$2

HOME_DIRECTORY="/cluster/work/cvg/students/sselgrad"
SCRATCH_DIRECTORY="/cluster/scratch/sselgrad"
WORKING_DIRECTORY="$HOME_DIRECTORY/diffusers"
VENV="$WORKING_DIRECTORY/venv_plz_work/diffusers"
export MODEL_DIR="GonzaloMG/stable-diffusion-e2e-ft-depth"
export CONTROLNET_DIR="$WORKING_DIRECTORY/examples/controlnet"
export CHECKPOINT_DIR="$CONTROLNET_DIR/checkpoints_inpaint/SCALE_${SCALE}/depth.ckpt"
export CHECKPOINT_DIR_NO="checkpoint-${CKPT}/controlnet"
export HF_HOME="$SCRATCH_DIRECTORY/huggingface_cache"
export EVAL_DATASET="$SCRATCH_DIRECTORY/eval_dataset/diode/diode_val"
export FILENAME_LIST_ALL="$EVAL_DATASET/diode_val_all_filename_list.txt"


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
python sample_diode_rel.py \
     --data_root   "$EVAL_DATASET" \
     --txt_out    $FILENAME_LIST_ALL \
     --num_points $N \
     --transform_order "sampling" \
     --seed 42
echo "🎇 Sampling complete"

echo "🎆 Splitting file"
python split_diode_lists.py \
  --data_root /cluster/scratch/sselgrad/eval_dataset/diode/diode_val \
  --txt_in diode_val_all_filename_list_N1000.txt
echo "🎇 Splitting complete"
