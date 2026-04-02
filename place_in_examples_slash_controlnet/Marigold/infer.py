# Last modified: 2024-03-30
# Copyright 2023 Bingxin Ke, ETH Zurich. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------------------------
# If you find this code useful, we kindly ask you to cite our paper in your work.
# Please find bibtex at: https://github.com/prs-eth/Marigold#-citation
# More information about the method can be found at https://marigoldmonodepth.github.io
# --------------------------------------------------------------------------

# @GonzaloMartinGarcia
# The following code is built upon Marigold's infer.py, and was adapted to include some new settings.
# All additions made are marked with # add.
# 
# Further modifications made by Sophie Selgrad.


import argparse
import logging
import os

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# add
import sys
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection
sys.path.append(os.getcwd())

from src.util.seed_all import seed_all
from src.dataset import (
    BaseDepthDataset,
    DatasetMode,
    get_dataset,
    get_pred_name,
)
import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import sys
import cv2
import pdb
import traceback


import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

import csv
from tqdm import tqdm

from packaging import version
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel, 
    DDPMScheduler,
    DDIMScheduler,
    StableDiffusionControlNetInpaintPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from torch.nn import Conv2d, Parameter



def edit_controlnet_for_latent_input(model: ControlNetModel) -> ControlNetModel:
    """
    This function modifies the ControlNet such that it accepts the conditional inputs in latent space 
    (64x64) rather than image space (512x512)
    """
    for mod in model.controlnet_cond_embedding.modules():
        if isinstance(mod, torch.nn.Conv2d):
            if mod.stride == (2, 2):
                mod.stride = (1, 1)

    return model

if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to trained controlnet model."
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument("--num_inference_steps", type=int, default=1)
    parser.add_argument("--processing_resolution", type=int, default=0)
    parser.add_argument("--output_type", type=str, default="pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=1234)

    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="GonzaloMG/stable-diffusion-e2e-ft-depth",
        help="Checkpoint path or hub name.",
    )

    # dataset setting
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to config file of evaluation dataset.",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        required=True,
        help="Path to base data directory.",
    )

    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="The scale for the controlnet conditioning.")


    args = parser.parse_args()

    # add

    dataset_config = args.dataset_config
    base_data_dir = args.base_data_dir
    output_dir = args.output_dir

    seed = args.seed

    # add
    # Save arguments in txt file
    print(f"arguments: {args}")
    parent_output_dir = os.path.dirname(args.output_dir)
    os.makedirs(parent_output_dir, exist_ok=True)
    args_dict = vars(args)
    args_str = '\n'.join(f"{key}: {value}" for key, value in args_dict.items())
    args_path = os.path.join(parent_output_dir, "arguments.txt")
    with open(args_path, 'w') as file:
        file.write(args_str)
    print(f"Arguments saved in {args_path}")

    # -------------------- Preparation --------------------
    # Print out config
    logging.info(
        f"seed = {seed}; "
        f"dataset config = `{dataset_config}`."
    )

    # Random seed
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"output dir = {output_dir}")

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Data --------------------
    cfg_data = OmegaConf.load(dataset_config)

    dataset: BaseDepthDataset = get_dataset(
        cfg_data, base_data_dir=base_data_dir, mode=DatasetMode.RGB_ONLY
    )

    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    # -------------------- Model --------------------
    # base and controlnet paths
    base_model_path = args.pretrained_model_name_or_path
    controlnet_path = args.controlnet_model_name_or_path
    checkpoint = args.resume_from_checkpoint

    # validate checkpoint directory
    if checkpoint is not None and os.path.isdir(os.path.join(controlnet_path, checkpoint)):
        complete_controlnet_path = os.path.join(controlnet_path, checkpoint)
    else:
        raise ValueError(
            f"Checkpoint {checkpoint} not found in {controlnet_path}. Please provide a valid checkpoint directory."
        )

    weight_dtype = torch.float32

    # Load ControlNet
    controlnet = ControlNetModel.from_pretrained(
        complete_controlnet_path, torch_dtype=weight_dtype
    )

    # Load tokenizer + submodules for SD backbone
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, revision=args.revision, use_fast=False
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_path, subfolder="tokenizer",
            revision=args.revision, use_fast=False
        )

    text_encoder = CLIPTextModel.from_pretrained(
        base_model_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        base_model_path, subfolder="vae",
        revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        base_model_path, subfolder="unet",
        revision=args.revision, variant=args.variant
    )

    # --- Scheduler ---
    val_scheduler = DDIMScheduler.from_pretrained(
        base_model_path,
        subfolder="scheduler",
        timestep_spacing="trailing",
        revision=args.revision,
        variant=args.variant,
    )

    # --- Modify ControlNet for latent input if needed ---
    controlnet = edit_controlnet_for_latent_input(controlnet)

    # --- Build pipeline ---
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        scheduler=val_scheduler,
    )

    pipe.set_progress_bar_config(disable=True)
    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()

    pipe.enable_model_cpu_offload()
    pipe.to(device)
    prompt = ""
    gen = torch.manual_seed(args.seed or 1234)


    inference_times = []
    csv_log_path = os.path.join(output_dir, "inference_times.csv")
    with open(csv_log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        writer.writerow(["sample", "inference_time_ms"])

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Inferencing on {dataset.disp_name}", leave=True):
                rgb = batch["rgb_int"].to(device=device, dtype=weight_dtype)
                control_image = batch["sparse_depth_linear"].to(device=device, dtype=weight_dtype)
                rgb_filename = batch["rgb_relative_path"][0]
                base_path = rgb_filename

                if torch.cuda.is_available():
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()

                pred = pipe(
                    rgb_image=rgb,
                    prompt="",
                    control_image=control_image,
                    num_inference_steps=args.num_inference_steps,
                    generator=gen,
                    output_type="pt",
                    processing_resolution=args.processing_resolution,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                ).prediction

                if torch.cuda.is_available():
                    end.record()
                    torch.cuda.synchronize()
                    elapsed = start.elapsed_time(end)
                else:
                    elapsed = None

                pred_cpu = pred.squeeze().detach().cpu().numpy()
                #save_to = os.path.join(output_dir, os.path.basename(rgb_filename)).replace(".jpg", ".npy")
                rel_dir = os.path.dirname(rgb_filename)                           # e.g., sceneXXXX_XX/color
                base_no_ext = os.path.splitext(os.path.basename(rgb_filename))[0] # no extension
                save_to = os.path.join(output_dir, rel_dir, base_no_ext + ".npy")
                os.makedirs(os.path.dirname(save_to), exist_ok=True)
                np.save(save_to, pred_cpu)
                os.makedirs(os.path.dirname(save_to), exist_ok=True)
                np.save(save_to, pred_cpu)

                if elapsed is not None:
                    writer.writerow([base_path, f"{elapsed:.2f}"])
                    log_file.flush()
                    inference_times.append(elapsed)

    if inference_times:
        avg = sum(inference_times) / len(inference_times)
        print(f"\n✅ Inference complete.")
        print(f"📊 Average Inference Time: {avg:.2f} ms ({1000 / avg:.2f} FPS)")
    else:
        print("\n⚠️ No inference times recorded.")