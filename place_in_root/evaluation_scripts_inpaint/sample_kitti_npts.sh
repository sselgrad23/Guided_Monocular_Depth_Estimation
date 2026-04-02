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
export EVAL_DATASET="$SCRATCH_DIRECTORY/eval_dataset/kitti"
export EVAL_DATASET_1="$EVAL_DATASET/kitti_sampled_val_800/"
export EVAL_DATASET_2="$EVAL_DATASET/kitti_eigen_split_test/"

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

echo "🎆 Sampling $N points: $EVAL_DATASET_1"
python sample_kitti_npts.py \
     --depth_root $EVAL_DATASET_1 \
     --rgb_root $EVAL_DATASET_1 \
     --txt_out $HOME_DIRECTORY/eval_dataset/kitti/kitti_sampled_val_800.txt \
     --num_points $N \
     --transform_order "sampling" \
     --seed 42

python sample_kitti_npts.py \
     --depth_root $EVAL_DATASET_2 \
     --rgb_root $EVAL_DATASET_2 \
     --txt_out $HOME_DIRECTORY/eval_dataset/kitti/kitti_eigen_split_test.txt \
     --num_points $N \
     --transform_order "sampling" \
     --seed 42

#cat "$HOME_DIRECTORY/eval_dataset/kitti/kitti_sampled_val_800_$N.txt" "$HOME_DIRECTORY/eval_dataset/kitti/kitti_eigen_split_test_$N.txt" > "$HOME_DIRECTORY/eval_dataset/kitti/kitti_${N}.txt"
if [[ -f "$EVAL_DATASET/kitti_sampled_val_800_$N.txt" && -f "$EVAL_DATASET/kitti_eigen_split_test_$N.txt" ]]; then
    (cat "$EVAL_DATASET/kitti_sampled_val_800_$N.txt"; echo ""; cat "$EVAL_DATASET/kitti_eigen_split_test_$N.txt") > "$EVAL_DATASET/kitti_${N}.txt"
else
    echo "One or both input files do not exist." >&2
    exit 1
fi
echo "🎇 Sampling complete"