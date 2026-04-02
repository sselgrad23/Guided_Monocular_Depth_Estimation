#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
Training code for ControlNet + Stable Diffusion E2E FT (no inpainting)

This is referred to as the "naive" implementation in the report. 
"""

import argparse
import contextlib
import gc
import logging
import math
import os
import random
import shutil
from pathlib import Path
import cv2
import itertools

import accelerate
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel
from torchmetrics import MeanMetric
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader
from torch.serialization import add_safe_globals


import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from torch.nn import Conv2d, Parameter

from datasets import Image as HFImage
from loss import ScaleAndShiftInvariantLoss #, AngularLoss
from lr_scheduler import IterExponential
from load_metric import *
from unet_prep import replace_unet_conv_in

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

print("🥰"*22,"W&B key seen by Python:", os.getenv("WANDB_API_KEY") is not None)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid

def load_kitti_validation_image(validation_image):
    image = cv2.imread(validation_image, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth_m  = image.astype(np.float32) / 100.0                 # meters
    depth_m[depth_m == 0] = np.nan  # optional: keep NaNs

    # Compute per-image min/max, ignoring zeros (invalid/missing)
    valid_mask = depth_m > 0
    if np.any(valid_mask):
        img_min = depth_m[valid_mask].min()
        img_max = depth_m[valid_mask].max()
    else:
        img_min = 0.0
        img_max = 1.0  # fallback to avoid division by zero

    # Normalize per image
    denom = img_max - img_min + 1e-8
    norm = (depth_m - img_min) / denom
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)
    norm[np.isnan(norm)] = 0.0  # replace NaNs if needed

    processed_image=Image.fromarray(norm)

    return processed_image

def load_hypersim_validation_image(validation_image):
    image = Image.open(validation_image)
    depth_m  = np.array(image).astype(np.float32) / 1000                 # meters #TODO: Check how hypersim images should be loaded
    depth_m[depth_m == 0] = np.nan  # optional: keep NaNs

    # Compute per-image min/max, ignoring zeros (invalid/missing)
    valid = depth_m > 0
    img_min = float(depth_m[valid].min()) if np.any(valid) else 0.0
    img_max = float(depth_m[valid].max()) if np.any(valid) else 1.0

    # Normalize per image
    denom = img_max - img_min + 1e-8
    norm = (depth_m - img_min) / denom
    norm = np.clip(norm, 0.0, 1.0).astype(np.float32)

    norm[np.isnan(norm)] = 0.0  # replace NaNs if needed

    processed_image=Image.fromarray(norm)

    return processed_image

def _vis_uint8_pil(img):
    """
    Convert an arbitrary PIL/numpy image (F, I;16, etc.) to an 8-bit RGB PIL
    for safe logging/saving. Preserves your pipeline input by returning a copy.
    """
    if isinstance(img, Image.Image):
        arr = np.asarray(img)
    else:
        arr = img

    # Float images: assume [0,1] (your normalization); clamp & scale
    if arr.dtype in (np.float32, np.float64):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).astype(np.uint8)

    # 16-bit depth (cm). Map valid (<65535) to [0,255], keep 65535 as 0
    elif arr.dtype == np.uint16:
        mask = arr < 65535
        out = np.zeros_like(arr, dtype=np.uint8)
        if mask.any():
            lo = arr[mask].min()
            hi = arr[mask].max()
            scale = 255.0 / max(float(hi - lo), 1e-8)
            out[mask] = np.clip((arr[mask] - lo) * scale, 0, 255).astype(np.uint8)
        arr = out

    # shape → PIL
    if arr.ndim == 2:
        pil = Image.fromarray(arr).convert("RGB")
    elif arr.ndim == 3 and arr.shape[2] == 3 and arr.dtype == np.uint8:
        pil = Image.fromarray(arr, mode="RGB")
    else:
        pil = Image.fromarray(arr.squeeze()).convert("RGB")
    return pil

def _rel_to_any(p: Path, roots: list[Path]) -> tuple[tuple[str, ...] | None, Path | None]:
    """Try to make p relative to any root in roots. Return (parts, chosen_root) or (None, None)."""
    for r in roots:
        try:
            rel = p.relative_to(r)
            return rel.parts, r
        except Exception:
            pass
    return None, None

def _which_dataset(path_str, args):
    """
    Return 'vkitti' or 'hypersim' for a validation path.

    Supports Hypersim roots like:
      .../hypersim/processed
      .../hypersim/processed/train
    """
    p = Path(path_str)

    vk_roots = [Path(args.vkitti_data_dir)] if args.vkitti_data_dir else []
    hyp_root = Path(args.hypersim_data_dir) if args.hypersim_data_dir else None
    hyp_roots = [hyp_root, hyp_root / "train"] if hyp_root else []

    # precise root checks first
    if vk_roots:
        parts, _ = _rel_to_any(p, vk_roots)
        if parts is not None:
            return "vkitti"
    if hyp_roots:
        parts, _ = _rel_to_any(p, hyp_roots)
        if parts is not None:
            return "hypersim"

    # fallbacks: name heuristics
    low_parts = [s.lower() for s in p.parts]
    if any(s.startswith("scene") for s in low_parts):
        return "vkitti"
    if any(s.startswith("ai_") for s in low_parts):
        return "hypersim"
    return None

def _build_image_id(dataset_kind, validation_image_path, args):
    """
    Make a readable ID for filenames.
    Handles both vkitti and hypersim, including '.../processed/train/...'.
    """
    p = Path(validation_image_path)

    if dataset_kind == "vkitti":
        # try to relativize under vkitti root for robust parsing
        vk_roots = [Path(args.vkitti_data_dir)] if args.vkitti_data_dir else []
        parts, _ = _rel_to_any(p, vk_roots)
        parts = parts or p.parts
        scene   = next((s for s in parts if s.startswith("Scene")), "Scene")
        weather = next((s for s in parts
                        if s in {"morning","fog","rain","sunset","overcast"}), "wx")
        camera  = next((s for s in parts if s.startswith("Camera_")), "cam")
        return f"{scene}_{weather}_{camera}_{p.stem}"

    if dataset_kind == "hypersim":
        # accept either processed/… or processed/train/…
        hyp_root = Path(args.hypersim_data_dir) if args.hypersim_data_dir else None
        hyp_roots = [hyp_root, hyp_root / "train"] if hyp_root else []
        parts, _ = _rel_to_any(p, hyp_roots)
        parts = parts or p.parts

        # typical shapes:
        #   processed/train/ai_001_001/depth_plane_cam_00_fr0000.png
        #   processed/ai_001_001/depth_plane_cam_00_fr0000.png
        ai_id = next((s for s in parts if s.startswith("ai_")), "ai")
        # cam can be in the filename or a folder name (e.g., cam_00)
        stem_tokens = p.stem.split("_")
        cam = next((t for t in parts if t.startswith("cam_")), None)
        if cam is None:
            cam = next((t for t in stem_tokens if t.startswith("cam")), "cam")
        return f"{ai_id}_{cam}_{p.stem}"

    # fallback
    return p.stem

def log_validation(
    vae, text_encoder, tokenizer, unet, controlnet, args, accelerator, weight_dtype, step, scheduler, conditioning_scale, is_final_validation=False
):
    logger.info("Running validation... ")

    if not is_final_validation:
        controlnet = accelerator.unwrap_model(controlnet)
    else:
        print("THIS IS THE FINAL VALIDATION")
        cn_path = os.path.join(args.output_dir, "controlnet")
        controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=weight_dtype)

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        scheduler=scheduler,
        default_processing_resolution=0 # No downsampling of input images

    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(
        disable=not args.validation_show_progress,
        leave=False,
        desc="Validation diffusion"
    )

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
        validation_rgb_images = args.validation_rgb_image
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
        validation_rgb_images = args.validation_rgb_image * len(args.validation_prompt) 
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
        validation_rgb_images = args.validation_rgb_image 
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    inference_ctx = contextlib.nullcontext() if is_final_validation else torch.autocast("cuda")

    for validation_prompt, validation_image, validation_rgb_image in zip(validation_prompts, validation_images, validation_rgb_images):
        # turn the string into a Path
        dataset_kind = _which_dataset(validation_image, args)
        image_id = _build_image_id(dataset_kind, validation_image, args)

        # choose the right conditioning loader
        if dataset_kind == "hypersim":
            validation_image = load_hypersim_validation_image(validation_image)
        else:
            validation_image = load_kitti_validation_image(validation_image)
        validation_rgb_image = Image.open(validation_rgb_image).convert("RGB") 

        images = []

        # Only one (deterministic) validation output
        with inference_ctx:
            gen = None
            if args.seed is not None:
                gen = torch.Generator(device=accelerator.device).manual_seed(args.seed)
            image = pipeline(
                rgb_image=validation_rgb_image,
                prompt=validation_prompt,
                image=validation_image,
                num_inference_steps=args.validation_num_inference_steps,
                generator=gen,
                output_type="pil",
                controlnet_conditioning_scale=conditioning_scale, # Could try changing this to see the effects
            ).prediction
        images = [image]
        image_logs.append(
            {"validation_image": validation_image, "validation_rgb_image": validation_rgb_image, "images": images, "validation_prompt": validation_prompt}
        )
        # Save the single validation image
        single_out_path = os.path.join(
            args.output_dir,
            "validation_images",  # subdirectory
            f"validation_step_{step}_{image_id}.png"
        )
        # Ensure the subdirectory exists
        Path(single_out_path).parent.mkdir(parents=True, exist_ok=True)
        image.save(single_out_path)
        logger.info(f"Saved validation image to {single_out_path}")

        if len(images) > 1:
            # Save validation result (1×1 grid)
            grid = image_grid(images, 1, 1)
            grid_out_path = os.path.join(
                args.output_dir,
                "validation_images",  # subdirectory
                f"validation_step_{step}_{image_id}_grid.png"
            )
            grid.save(grid_out_path)
            logger.info(f"Saved validation grid to {grid_out_path}")

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                validation_rgb_image = log["validation_rgb_image"]

                vis_control = np.asarray(_vis_uint8_pil(validation_image))
                formatted_images = [vis_control] + [np.asarray(_vis_uint8_pil(img)) for img in images]
                formatted_images = np.stack(formatted_images)  # NHWC, uint8
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")
        elif tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                validation_rgb_image = log["validation_rgb_image"]

                # Only convert a COPY for logging
                vis_control = _vis_uint8_pil(validation_image)
                formatted_images.append(wandb.Image(vis_control, caption="Controlnet conditioning"))

                for image in images:
                    formatted_images.append(wandb.Image(_vis_uint8_pil(image), caption=validation_prompt))

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    gc.collect()
    torch.cuda.empty_cache()

    return image_logs


def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation

        return RobertaSeriesModelWithTransformation
    else:
        raise ValueError(f"{model_class} is not supported.")


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_rgb_image = log["validation_rgb_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"sparse depth prompt: {validation_prompt} rgb prompt: {validation_rgb_image} \n"
            images = [validation_image] + images
            image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# controlnet-{repo_id}

These are controlnet weights trained on {base_model} with new type of conditioning.
{img_str}
"""
    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "controlnet",
        "diffusers-training",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
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
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
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
    parser.add_argument(
        "--output_dir",
        type=str,
        default="controlnet-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
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
        "--gradient_accumulation_steps", # If I reduce the batch size by a factor of 4, I can increase the gradient accumulation steps to 4.
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_total_iter_length",
        type=int,
        default=20000,
    )
    parser.add_argument(
        "--lr_exp_warmup_steps",
        type=int,
        default=100,
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--set_grads_to_none",
        action="store_true",
        help=(
            "Save more memory by using setting grads to None instead of zero. Be aware, that this changes certain"
            " behaviors, so disable this argument if it causes any problems. More info:"
            " https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--vkitti_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the VKITTI training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--hypersim_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the Hypersim training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--hypersim_csv_filepath",
        type=str,
        default=None,
        help=(
            "Path to the CSV file containing the metadata for the Hypersim dataset."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnet conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument( 
        "--validation_rgb_image",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the *matching* RGB JPEGs for each sparse-depth PNG in --validation_image",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_num_inference_steps",
        type=int,
        default=1,
        help="Number of diffusion steps to use during validation/inference."
        "Currently doesn't work with more than 1 step.",
    )
    parser.add_argument(
        "--validation_show_progress",
        action="store_true",
        help="Show the pipeline progress bar during validation."
    )
    parser.add_argument(
        "--conditioning_scale",
        type=float,
        default=1.0,
        help="Scale for the controlnet conditioning. Defaults to 1.0.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="train_controlnet",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--vkitti_sparse_depth_name",
        type=str,
        default="sparse_depth",
        help="The name of the sparse depth folder for the VKITTI dataset."
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.vkitti_data_dir is None:
        raise ValueError("Specify either `--dataset_name` or `--vkitti_data_dir`")
    
    if args.hypersim_data_dir is None:
        raise ValueError("Specify `--hypersim_data_dir`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and args.validation_rgb_image is not None
        and len(args.validation_image) != 1
        and len(args.validation_rgb_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
        and len(args.validation_rgb_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, and 1 `--validation_rgb_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s and `--validation_rgb_image`s."
        )

    return args


# Apply VAE Encoder to image
def encode_image(vae, image):
    h = vae.encoder(image)
    moments = vae.quant_conv(h)
    latent, _ = torch.chunk(moments, 2, dim=1)
    return latent

# Apply VAE Decoder to latent
def decode_image(vae, latent):
    z = vae.post_quant_conv(latent)
    image = vae.decoder(z)
    return image

def main(args):
    print("Starting training!")
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # Move logger.info calls after Accelerator initialization
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load the tokenizer
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, revision=args.revision, use_fast=False)
    elif args.pretrained_model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="tokenizer",
            revision=args.revision,
            use_fast=False,
        )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    if args.controlnet_model_name_or_path:
        logger.info("Loading existing controlnet weights")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        logger.info("Initializing controlnet weights from unet")
        controlnet = ControlNetModel.from_unet(unet) # controlnet = ControlNetModel.from_unet(unet, conditioning_channels=3)

    # Taken from [Sayak Paul's Diffusers PR #6511](https://github.com/huggingface/diffusers/pull/6511/files)
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    print("Unwrapped models")

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                i = len(weights) - 1

                while len(weights) > 0:
                    weights.pop()
                    model = models[i]

                    sub_dir = "controlnet"
                    model.save_pretrained(os.path.join(output_dir, sub_dir))

                    i -= 1

        def load_model_hook(models, input_dir):
            while len(models) > 0:
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = ControlNetModel.from_pretrained(input_dir, subfolder="controlnet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if unet.config['in_channels'] != 8:
        replace_unet_conv_in(unet, repeat=2)
        logger.info("Unet conv_in layer is replaced for RGB-depth or RGB-normals input")

    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if args.gradient_checkpointing:
        controlnet.enable_gradient_checkpointing()

    # Check that all trainable models are in full precision
    low_precision_error_string = (
        " Please make sure to always have all model weights in full float32 precision when starting training - even if"
        " doing mixed precision training, copy of the weights should still be float32."
    )

    if unwrap_model(controlnet).dtype != torch.float32:
        raise ValueError(
            f"Controlnet loaded as datatype {unwrap_model(controlnet).dtype}. {low_precision_error_string}"
        )

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    # Optimizer creation
    params_to_optimize = controlnet.parameters()
    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # From diffusion-e2e-ft train.py
    # Learning rate scheduler
    lr_func      = IterExponential(total_iter_length = args.lr_total_iter_length*accelerator.num_processes, final_ratio = 0.01, warmup_steps = args.lr_exp_warmup_steps*accelerator.num_processes)
    lr_scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lr_func)

    # Training datasets
    train_dataset_hypersim = Hypersim(root_dir=args.hypersim_data_dir, csv_filepath=args.hypersim_csv_filepath, transform=True) # Hypersim dataset is already preprocessed, so no need to apply any transformations.
    train_dataset_vkitti   = VirtualKITTI2(root_dir=args.vkitti_data_dir, transform=True, sparse_depth_name=args.vkitti_sparse_depth_name) # Virtual KITTI 2 dataset is already preprocessed, so no need to apply any transformations.
    train_dataloader_vkitti   = torch.utils.data.DataLoader(train_dataset_vkitti,   shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)
    train_dataloader_hypersim = torch.utils.data.DataLoader(train_dataset_hypersim, shuffle=True, batch_size=args.train_batch_size, num_workers=args.dataloader_num_workers)
    train_dataloader = MixedDataLoader(train_dataloader_hypersim, train_dataloader_vkitti, split1=9, split2=1) # 9:1 split between hypersim and vkitti datasets

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    num_warmup_steps_for_scheduler = args.lr_warmup_steps * accelerator.num_processes
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    # Prepare everything with our `accelerator`.
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    # ─── NEW: create a MeanMetric to track non-zero losses ───
    epoch_loss_metric = MeanMetric().to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae, unet and text_encoder to device and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")
        tracker_config.pop("validation_rgb_image", None)

        # Debugging tracker_config
        print("Tracker config:", tracker_config)
        logger.info(f"Tracker config: {tracker_config}")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)
        # if we’re logging to W&B (or "all"), track gradients & weights
        if args.report_to in ("wandb", "all"):
            # unwrap_model gives you the raw nn.Module under accelerate
            wandb.watch(accelerator.unwrap_model(controlnet), log="all", log_freq=100)


    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset_vkitti)+len(train_dataset_hypersim)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # Allow-list NumPy objects used inside older optimizer pickles (PyTorch 2.6 weights_only=True)
            allow = [np.core.multiarray.scalar, np.dtype, np.float64]
            try:
                from numpy.dtypes import Float64DType  # NumPy 2.x dtype class
                allow.append(Float64DType)
            except Exception:
                pass
            add_safe_globals(allow)
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # --- Save checkpoint-0 (before any training has occurred) ---
    if accelerator.is_main_process and args.resume_from_checkpoint is None:
        # optional: enforce checkpoints_total_limit like later saves
        if args.checkpoints_total_limit is not None:
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
            if len(checkpoints) >= args.checkpoints_total_limit:
                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                for removing_checkpoint in checkpoints[:num_to_remove]:
                    shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

        save_path = os.path.join(args.output_dir, "checkpoint-0")
        accelerator.save_state(save_path)
        logger.info(f"Saved initial (step 0) state to {save_path}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    ssi_loss           = ScaleAndShiftInvariantLoss()

    # Pre-compute empty text CLIP encoding
    empty_token    = tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
    empty_token    = empty_token.to(accelerator.device)
    empty_encoding = text_encoder(empty_token, return_dict=False)[0]
    empty_encoding = empty_encoding.to(accelerator.device)

    # Get noise scheduling parameters for later conversion from a parameterized prediction into latent.
    alpha_prod = noise_scheduler.alphas_cumprod.to(accelerator.device, dtype=weight_dtype)
    beta_prod  = 1 - alpha_prod
    
    image_logs = None
    #cumulative_loss = 0.0  # Initialize cumulative loss
    should_stop = False
    
    for epoch in range(first_epoch, args.num_train_epochs):
        epoch_loss_metric.reset()
        logger.info(f"At Epoch {epoch}:")
        train_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Convert images to latent space
                # RGB latent
                rgb_latents = encode_image(vae, batch["rgb"].to(device=accelerator.device, dtype=weight_dtype))
                rgb_latents = rgb_latents * vae.config.scaling_factor

                depth_latents = encode_image(vae, batch["depth"].to(device=accelerator.device, dtype=weight_dtype))
                depth_latents = depth_latents * vae.config.scaling_factor

                # Validity mask
                val_mask = batch["val_mask"].bool().to(device=accelerator.device)
                controlnet_image = batch["sparse_depth"].to(device=accelerator.device, dtype=weight_dtype)
                
                # Set timesteps to the first denoising step
                timesteps = torch.ones((rgb_latents.shape[0],), device=rgb_latents.device) * (noise_scheduler.config.num_train_timesteps-1) # 999
                timesteps = timesteps.long()

                zeros_latents = torch.zeros_like(rgb_latents).to(accelerator.device)
                encoder_hidden_states = empty_encoding.repeat(len(batch["rgb"]), 1, 1)
                unet_input = torch.cat((rgb_latents, zeros_latents), dim=1).to(accelerator.device)

                down_block_res_samples, mid_block_res_sample = controlnet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    conditioning_scale=args.conditioning_scale,
                    return_dict=False,
                )

                # Predict the noise residual
                model_pred = unet(
                    unet_input,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=[
                        sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                    ],
                    mid_block_additional_residual=mid_block_res_sample.to(dtype=weight_dtype),
                    return_dict=False,
                )[0]

                loss = torch.tensor(0.0, device=accelerator.device, requires_grad=True)
                if val_mask.any():
                    # Convert parameterized prediction into latent prediction.
                    # Code is based on the DDIM code from diffusers,
                    # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py.
                    alpha_prod_t = alpha_prod[timesteps].view(-1, 1, 1, 1)
                    beta_prod_t = beta_prod[timesteps].view(-1, 1, 1, 1)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        current_latent_estimate = (alpha_prod_t**0.5) * zeros_latents - (beta_prod_t**0.5) * model_pred
                    elif noise_scheduler.config.prediction_type == "epsilon":
                        current_latent_estimate = (zeros_latents - beta_prod_t ** (0.5) * model_pred) / alpha_prod_t ** (0.5)
                    elif noise_scheduler.config.prediction_type == "sample":
                        current_latent_estimate = model_pred
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                    # clip and threshold prediction (only here for completeness, not used by SD2 or our models with v_prediction)
                    """
                    if noise_scheduler.config.thresholding:
                        pred_original_sample = noise_scheduler._threshold_sample(pred_original_sample)
                    elif noise_scheduler.config.clip_sample:
                        pred_original_sample = pred_original_sample.clamp(
                            -noise_scheduler.config.clip_sample_range, noise_scheduler.config.clip_sample_range
                        )
                    """
                    
                    # Decode latent prediction
                    current_latent_estimate = current_latent_estimate / vae.config.scaling_factor
                    current_estimate = decode_image(vae, current_latent_estimate)

                    # Post-process predicted images and retrieve ground truth
                    current_estimate = current_estimate.mean(dim=1, keepdim=True) 
                    current_estimate = torch.clamp(current_estimate,-1,1) 
                    ground_truth = batch["metric"].to(device=accelerator.device, dtype=weight_dtype)

                    # Compute task-specific loss
                    estimation_loss = 0
                    estimation_loss_ssi = ssi_loss(current_estimate, ground_truth, val_mask)
                    if not torch.isnan(estimation_loss_ssi).any():
                        estimation_loss = estimation_loss + estimation_loss_ssi
                    loss = loss + estimation_loss
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # ─── UPDATE METRIC ONLY FOR NON-ZERO LOSSES ───
                if avg_loss.item() > 0:
                    epoch_loss_metric.update(avg_loss.item())

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = controlnet.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
        
                # compute the mean loss for this batch (avg_loss) and the running epoch mean
                batch_loss = avg_loss.item()
                running_epoch_loss = epoch_loss_metric.compute().item()

                # single unified log
                logs = {
                    "loss": loss.detach().item(),  # this is the loss for the current step
                    "batch_loss": batch_loss,
                    "train_loss": train_loss,
                    "epoch_loss": running_epoch_loss,
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                train_loss = 0.0

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0: # ==1: 
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        # Lazy variant: create the trailing-spacing scheduler only once and reuse it
                        if "val_scheduler" not in globals():
                            global val_scheduler
                            val_scheduler = DDIMScheduler.from_pretrained(
                                args.pretrained_model_name_or_path,
                                subfolder="scheduler",
                                timestep_spacing="trailing",
                                revision=args.revision,
                                variant=args.variant,
                            )
                        image_logs = log_validation(
                            vae,
                            text_encoder,
                            tokenizer,
                            unet,
                            controlnet,
                            args,
                            accelerator,
                            weight_dtype,
                            global_step,
                            val_scheduler,
                            conditioning_scale=args.conditioning_scale,
                        )

            if global_step >= args.max_train_steps:
                should_stop = True
                break
        # ─── before end-of-epoch logging ───
        if should_stop:
            break
        # ─── number 5: End-of-epoch logging ───
        # Compute the true mean of all non-zero losses this epoch
        final_epoch_loss = epoch_loss_metric.compute().item()
        accelerator.log({"epoch_loss": final_epoch_loss}, step=global_step)
        logger.info(f"Epoch {epoch} average non-zero loss: {final_epoch_loss:.4f}")

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet = unwrap_model(controlnet)
        try:
            _final_scheduler = val_scheduler
        except NameError:
            _final_scheduler = DDIMScheduler.from_pretrained(
                args.pretrained_model_name_or_path, subfolder="scheduler", timestep_spacing="trailing",
                revision=args.revision, variant=args.variant,
            )
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
        scheduler=_final_scheduler
        )
        logger.info(f"Saving pipeline to {args.output_dir}")
        pipeline.save_pretrained(args.output_dir)
        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                vae=vae,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                unet=unet,
                controlnet=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
                scheduler=_final_scheduler,
                conditioning_scale=args.conditioning_scale,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
