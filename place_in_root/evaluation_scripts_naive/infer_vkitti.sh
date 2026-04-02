#!/bin/bash
set -e

# Load necessary modules
#module load eth_proxy stack/2024-05 gcc/13.2.0 python_cuda/3.10.13 cudnn/9.2.0
#module load cmake/3.27.7 ninja/1.11.1 curl/8.4.0-mxgyalo

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cudnn/9.2.0
module load cmake/3.27.7 ninja/1.11.1 curl/8.4.0-s6dtj75

N=$1
CKPT=$2
SCALE=$3

HOME_DIRECTORY="/cluster/work/cvg/students/sselgrad"
SCRATCH_DIRECTORY="/cluster/scratch/sselgrad"
WORKING_DIRECTORY="$HOME_DIRECTORY/diffusers"
VENV="$WORKING_DIRECTORY/venv_plz_work/diffusers"
export MODEL_DIR="GonzaloMG/stable-diffusion-e2e-ft-depth"
export CONTROLNET_DIR="$WORKING_DIRECTORY/examples/controlnet"
export CHECKPOINT_DIR="$CONTROLNET_DIR/checkpoints_0909/SCALE_${SCALE}/depth.ckpt"
export CHECKPOINT_DIR_NO="checkpoint-${CKPT}/controlnet"
export HF_HOME="$SCRATCH_DIRECTORY/huggingface_cache"
export EVAL_DATASET="$SCRATCH_DIRECTORY/train_data/vkitti_eval"
export OUTPUT_DIR="$CONTROLNET_DIR/0909/output_N_${N}_PROPER_CKPT_${CKPT}_SCALE_${SCALE}/vkitti_eval"

if [ ! -d "$VENV" ]; then
    echo "Venv not found, creating..."
    python -m venv "$VENV"
    source "$VENV/bin/activate"
else
    echo "Venv found, activating..."
    source "$VENV/bin/activate"
fi

#pip install --upgrade pip
#pip install accelerate
#pip install -r examples/controlnet/requirements.txt
#pip install opencv-python matplotlib
#pip install torchmetrics
#pip install wandb
#wandb login        # paste your API key when prompted

#pip uninstall -y diffusers
#pip install -e .

cd "$WORKING_DIRECTORY/examples/controlnet"
#ls requirements.txt
#pip install -r requirements.txt

echo "🎆 Inference kitti, ${N} points, Checkpoint ${CKPT}"
python infer_7-2-1-1.py \
     --pretrained_model_name_or_path "$MODEL_DIR" \
     --controlnet_model_name_or_path "$CHECKPOINT_DIR" \
     --resume_from_checkpoint "$CHECKPOINT_DIR_NO" \
     --split_file "$EVAL_DATASET/vkitti_${N}.txt" \
     --data_root "$EVAL_DATASET" \
     --output_dir "$OUTPUT_DIR" \
     --dataloader_num_workers=0 \
     --test_batch_size 1 \
     --dataset_name "vkitti" \
     --depth_min 1e-5 \
     --depth_max 80.0
echo "🎇 Inference vkitti, ${N} points, Checkpoint ${CKPT} complete"