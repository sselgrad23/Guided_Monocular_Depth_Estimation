# Guided_Monocular_Depth_Estimation
Stable Diffusion–based framework for monocular depth estimation and depth completion, integrating sparse depth inputs via ControlNet-style conditioning.

***************************************************
train_naive_scripts:
***************************************************

Things you may need to change:
HOME_DIRECTORY="$HOME"
WORKING_DIRECTORY="$HOME_DIRECTORY/diffusers"
VENV="$WORKING_DIRECTORY/venv_plz_work/diffusers"
DATASET_ROOT="$HOME_DIRECTORY/train_data_smol"
VKITTI_ROOT="$DATASET_ROOT/vkitti"
DEPTH_ROOT="$VKITTI_ROOT/depth"
RGB_ROOT="$VKITTI_ROOT/rgb"
VKITTI_SPARSE_DEPTH_NAME="sparse_depth"
OUT_ROOT="${VKITTI_ROOT}/${VKITTI_SPARSE_DEPTH_NAME}"
HYPERSIM_ROOT="$DATASET_ROOT/hypersim/processed/train"
HYPERSIM_CSV_FILEPATH="$HYPERSIM_ROOT/filename_meta_train_with_sparse.csv"
export MODEL_DIR="prs-eth/marigold-depth-v1-0"
export OUTPUT_DIR="checkpoints_1909/SCALE_${CONDITIONING_SCALE}/depth.ckpt"
export HF_HOME="$HOME_DIRECTORY/huggingface_cache"
export TRAIN_CODE_DIR="$WORKING_DIRECTORY/examples/controlnet/train_controlnet_naive.py"
JSONL_FILE="metadata_random.jsonl"



***************************************************
Where to place files
***************************************************

Files in "place_in_examples_slash_controlnet" need to be placed in the diffusers/examples/controlnet

Files in "place_in_root" need to be placed in diffusers/

Files in "place_in_src_diffusers_models_controlnets" need to be placed in diffusers/src/diffusers/models/controlnets/

Files in "place_in_src_diffusers_pipelines_controlnet" need to be placed in diffusers/src/pipelines/controlnet

***************************************************
When installing diffusers
***************************************************

When installing diffusers, make sure you install it in editable mode via:

git clone https://github.com/huggingface/diffusers.git
cd diffusers
pip install -e .

place_in_root/requirements.txt contains the libraries that worked for me in my venv environment. Library versions like those of torch etc may be different for you. 

*********************************************************************************
Important note regarding place_in_src_diffusers_models_controlnets/controlnet.py
*********************************************************************************
When running/training the naive version, make sure that on lines 79, 186, and 451, that they all have "conditioning_channels: int = 3,".

When running/training the inpainting version, make sure that on lines 79, 186, and 451, that they all have "conditioning_channels: int = 9,".

The code will not work otherwise as you will have too many or too few ControlNet conditioning channels and you will get an error.
