#!/bin/bash
set -e


N="20000"
CKPT="0"

HOME_DIRECTORY="$HOME"
SCRATCH_DIRECTORY="$HOME"
WORKING_DIRECTORY="$HOME_DIRECTORY/diffusers"
VENV="$WORKING_DIRECTORY/venv_plz_work/diffusers"
export MODEL_DIR="GonzaloMG/stable-diffusion-e2e-ft-depth"
export CONTROLNET_DIR="$WORKING_DIRECTORY/examples/controlnet"
export CHECKPOINT_DIR="$CONTROLNET_DIR/checkpoints_inpaint/SCALE_${SCALE}/depth.ckpt"
export CHECKPOINT_DIR_NO="checkpoint-${CKPT}/controlnet"
export HF_HOME="$SCRATCH_DIRECTORY/huggingface_cache"
export EVAL_DATASET="$SCRATCH_DIRECTORY/eval_dataset/diode/diode_val"

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
python sample_diode_npts.py \
     --data_root   "$EVAL_DATASET" \
     --txt_out    "$EVAL_DATASET/diode_val_sampled_list.txt" \
     --num_points $N \
     --transform_order "sampling" \
     --seed 42
echo "🎇 Download complete"