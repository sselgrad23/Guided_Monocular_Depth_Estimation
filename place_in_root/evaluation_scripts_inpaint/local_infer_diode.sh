#!/bin/bash
set -e


N="20000"
CKPT="20000"
SCALE="1.0"

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
export OUTPUT_DIR="$CONTROLNET_DIR/2009/output_N_${N}_PROPER_CKPT_${CKPT}_SCALE_${SCALE}/diode"

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

echo "🎆 Inference Diode, ${N} points, Checkpoint ${CKPT}"
python infer_inpaint.py \
     --pretrained_model_name_or_path "$MODEL_DIR" \
     --controlnet_model_name_or_path "$CHECKPOINT_DIR" \
     --resume_from_checkpoint "$CHECKPOINT_DIR_NO" \
     --split_file "$EVAL_DATASET/diode_val_sampled_list_${N}.txt" \
     --data_root "$EVAL_DATASET" \
     --output_dir "$OUTPUT_DIR" \
     --dataloader_num_workers=0 \
     --test_batch_size 1 \
     --dataset_name "diode" \
     --depth_min 0.6 \
     --depth_max 350.0
echo "🎇 Inference Diode, ${N} points, Checkpoint ${CKPT} complete"