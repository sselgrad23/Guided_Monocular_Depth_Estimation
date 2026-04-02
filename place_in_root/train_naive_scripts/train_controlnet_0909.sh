#!/bin/bash
set -e

CONDITIONING_SCALE=$1

# Load necessary modules
#module load eth_proxy stack/2024-05 gcc/13.2.0 python_cuda/3.10.13 cudnn/9.2.0
#module load cmake/3.27.7 ninja/1.11.1 curl/8.4.0-mxgyalo

module load eth_proxy stack/2024-06 gcc/12.2.0 python_cuda/3.11.6 cudnn/9.2.0
module load cmake/3.27.7 ninja/1.11.1 curl/8.4.0-s6dtj75

HOME_DIRECTORY="/cluster/work/cvg/students/sselgrad"
SCRATCH_DIRECTORY="/cluster/scratch/sselgrad"
WORKING_DIRECTORY="$HOME_DIRECTORY/diffusers"
VENV="$WORKING_DIRECTORY/venv_plz_work/diffusers"
DATASET_ROOT="$SCRATCH_DIRECTORY/train_data"
VKITTI_ROOT="$DATASET_ROOT/vkitti"
DEPTH_ROOT="$VKITTI_ROOT/depth"
RGB_ROOT="$VKITTI_ROOT/rgb"
VKITTI_SPARSE_DEPTH_NAME="sparse_depth_7-2-1-2"
OUT_ROOT="${VKITTI_ROOT}/${VKITTI_SPARSE_DEPTH_NAME}"
HYPERSIM_ROOT="$DATASET_ROOT/hypersim/processed/train/"
HYPERSIM_CSV_FILEPATH="$HYPERSIM_ROOT/filename_meta_train_40000_320000.csv"
export MODEL_DIR="GonzaloMG/stable-diffusion-e2e-ft-depth"
export OUTPUT_DIR="checkpoints_0410/SCALE_${CONDITIONING_SCALE}/depth.ckpt"
export HF_HOME="$SCRATCH_DIRECTORY/huggingface_cache"
export TRAIN_CODE_DIR="$WORKING_DIRECTORY/examples/controlnet/train_controlnet_naive.py"
JSONL_FILE="metadata_lu_20000_320000.jsonl"
#JSONL_PATH="$DATASET_ROOT/$JSONL_FILE"

#CONDITION_1="$OUT_ROOT/Scene01/morning/frames/depth/Camera_0/sparse_depth_00320.png"
#CONDITION_2="$OUT_ROOT/Scene01/morning/frames/depth/Camera_0/sparse_depth_00354.png"
#RGB_CONDITION_1="$RGB_ROOT/Scene01/morning/frames/rgb/Camera_0/rgb_00320.jpg"
#RGB_CONDITION_2="$RGB_ROOT/Scene01/morning/frames/rgb/Camera_0/rgb_00354.jpg"

CONDITIONS=(
  $OUT_ROOT/Scene01/morning/frames/depth/Camera_0/sparse_depth_00320.png
  $OUT_ROOT/Scene01/30-deg-right/frames/depth/Camera_0/sparse_depth_00401.png
  $OUT_ROOT/Scene02/clone/frames/depth/Camera_1/sparse_depth_00189.png
  $OUT_ROOT/Scene02/fog/frames/depth/Camera_1/sparse_depth_00053.png
  $HYPERSIM_ROOT/ai_001_001/sparse_depth_cam_00_fr0058.png
  $HYPERSIM_ROOT/ai_001_001/sparse_depth_cam_00_fr0080.png
  $HYPERSIM_ROOT/ai_001_001/sparse_depth_cam_00_fr0094.png
)
#END

RGB_CONDITIONS=(
  $RGB_ROOT/Scene01/morning/frames/rgb/Camera_0/rgb_00320.jpg
  $RGB_ROOT/Scene01/30-deg-right/frames/rgb/Camera_0/rgb_00401.jpg
  $RGB_ROOT/Scene02/clone/frames/rgb/Camera_1/rgb_00189.jpg
  $RGB_ROOT/Scene02/fog/frames/rgb/Camera_1/rgb_00053.jpg
  $HYPERSIM_ROOT/ai_001_001/rgb_cam_00_fr0058.png
  $HYPERSIM_ROOT/ai_001_001/rgb_cam_00_fr0080.png
  $HYPERSIM_ROOT/ai_001_001/rgb_cam_00_fr0094.png
)

# /home/sophiemjsd/train_data_smol/vkitti/sparse_depth_500/Scene01/Scene01/morning/frames/depth/Camera_0/sparse_depth_00031.png
if [ ! -d "$VENV" ]; then
    echo "Venv not found, creating..."
    python -m venv "$VENV"
    source "$VENV/bin/activate"
else
    echo "Venv found, activating..."
    source "$VENV/bin/activate"
fi

#pip install --upgrade pip
##pip install --upgrade diffusers[torch]
#pip install -e .
#pip install accelerate
#pip install git+https://github.com/huggingface/diffusers
#pip install -r examples/controlnet/requirements.txt
#pip install opencv-python matplotlib
#pip install torchmetrics
#pip install wandb
#pip install --no-deps xformers==0.0.25.post1
#pip install xformers
#wandb login        # paste your API key when prompted

#pip uninstall -y diffusers
#pip install -e .

# Run the sparse depth generator


echo "🎆 Kitti sample"
python kitti_sample_range_pts.py \
    --depth_root "$DEPTH_ROOT" \
    --out_root "$OUT_ROOT" \
    --rgb_root "$RGB_ROOT" \
    --min_points 20000 \
    --max_points 320000 \
    --save_png \
    --jsonl_name "$JSONL_FILE"


echo "🎆 Hypersim sample"
python examples/controlnet/hypersim_sample_range_pts.py \
  --root_dir $HYPERSIM_ROOT \
  --csv_in  $HYPERSIM_ROOT/filename_meta_train.csv \
  --csv_out $HYPERSIM_CSV_FILEPATH \
  --min_points 40000 --max_points 320000
echo "🎇 Hypersim sampling complete"

echo "🔍 Checking each validation path..."
for condition in "${CONDITIONS[@]}"; do
    if [ ! -f "$condition" ]; then
        echo "❌ MISSING: $condition"
    else
        echo "✅ FOUND:  $condition"
    fi
done

for rgb_condition in "${RGB_CONDITIONS[@]}"; do
    if [ ! -f "$rgb_condition" ]; then
        echo "❌ MISSING: $rgb_condition"
    else
        echo "✅ FOUND:  $rgb_condition"
    fi
done
#END

#: <<'END'
cd ../examples/controlnet
#ls requirements.txt
#pip install -r requirements.txt
accelerate config default

echo "🎆 Kitti training"
accelerate launch $TRAIN_CODE_DIR \
 --pretrained_model_name_or_path=$MODEL_DIR \
 --output_dir=$OUTPUT_DIR \
 --vkitti_data_dir=$VKITTI_ROOT \
 --hypersim_data_dir=$HYPERSIM_ROOT \
 --hypersim_csv_filepath=$HYPERSIM_CSV_FILEPATH \
 --validation_rgb_image "${RGB_CONDITIONS[@]}" \
 --validation_image "${CONDITIONS[@]}" \
 --validation_prompt "" \
 --train_batch_size=2 \
 --dataloader_num_workers=0 \
 --gradient_accumulation_steps=16 \
 --allow_tf32 \
 --report_to wandb \
 --tracker_project_name "controlnet-e2e-depth" \
 --validation_steps=2000 \
 --checkpointing_steps=1000 \
 --num_validation_images=1 \
 --learning_rate=3e-5 \
 --max_train_steps 20000 \
 --lr_total_iter_length 20000 \
 --lr_exp_warmup_steps 100 \
 --enable_xformers_memory_efficient_attention \
 --mixed_precision bf16 \
 --conditioning_scale=$CONDITIONING_SCALE \
 --vkitti_sparse_depth_name=$VKITTI_SPARSE_DEPTH_NAME \
 --validation_num_inference_steps 1 \
 --validation_show_progress

echo "🎇 Kitti training complete"
#For main training: 5 2 2
