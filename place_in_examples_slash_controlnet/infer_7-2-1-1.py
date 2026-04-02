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

from accelerate import Accelerator
#from accelerate.logging import get_logger
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

import diffusers
from diffusers import (
    AutoencoderKL,
    ControlNetModel, 
    DDIMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available, load_image
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module
from torch.nn import Conv2d, Parameter

from datasets import Image as HFImage
from loss import ScaleAndShiftInvariantLoss, AngularLoss
from lr_scheduler import IterExponential

#if is_wandb_available():
#    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

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

    parser.add_argument(
        "--test_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
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
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
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
        "--data_root",
        type=str,
        default=None,
        help=(
            "A folder containing the evaluation data. Both data_root and split_file must be specified."
        ),
    )
    parser.add_argument("--split_file", type=str, required=True, help="A file containing the split of the dataset to use for evaluation.")
    parser.add_argument("--output_dir", type=str, required=True, help="The output directory where model evaluation results will be written.")

    parser.add_argument("--no-npy", action="store_true", help="Skip saving .npy files")

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
        "--dataset_name",
        type=str,
        default="diode",
        choices=["diode", "kitti", "eth_3d", "nyu_depth_v2", "scannet", "vkitti"],
        help=(
            "The name of the dataset to use for evaluation."
        ),
    )
    parser.add_argument("--depth_min", type=float, default=0.6, help="Minimum depth in meters")
    parser.add_argument("--depth_max", type=float, default=350.0, help="Maximum depth in meters")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0, help="The scale for the controlnet conditioning.")
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.data_root is None:
        raise ValueError("Specify `--data_root`")

    return args

class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, split_file, data_root):
        with open(split_file, "r") as f:
            lines = f.read().strip().split("\n")
        self.data_root = data_root
        self.lines = lines

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        rgb, gt, sparse, mask = line.strip().split()

        rgb_path = os.path.join(self.data_root, rgb)
        depth_path = os.path.join(self.data_root, gt)
        sparse_depth_path = os.path.join(self.data_root, sparse)
        if mask is not None:
            mask_path = os.path.join(self.data_root, mask)
        else:
            mask_path = None

        return line, rgb_path, depth_path, sparse_depth_path, mask_path


def load_paths(split_file, data_root):
    with open(split_file, "r") as f:
        lines = f.read().strip().split("\n")
    rgb_paths, sparse_depth_paths = [], []
    for line in lines:
        try:
            rgb, gt, sparse, mask = line.strip().split("\t")
            rgb_paths.append(os.path.join(data_root, rgb))
            sparse_depth_paths.append(os.path.join(data_root, sparse))
        except ValueError:
            print(f"Skipping line due to ValueError: {line}", file=sys.stderr)
            continue
    return rgb_paths, sparse_depth_paths, lines

def kitti_benchmark_crop(input_img):
    KB_CROP_HEIGHT = 352
    KB_CROP_WIDTH = 1216
    height, width = input_img.shape[-2:] 
    top_margin = int(height - KB_CROP_HEIGHT)
    left_margin = int((width - KB_CROP_WIDTH) / 2)
    if 2 == len(input_img.shape): 
        out = input_img[
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    elif 3 == len(input_img.shape):
        out = input_img[
            :,
            top_margin : top_margin + KB_CROP_HEIGHT,
            left_margin : left_margin + KB_CROP_WIDTH,
        ]
    return out

def tensor_to_input_image(tensor):
    arr = tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
    norm = np.clip((arr + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
    return Image.fromarray(norm)

def mask_out_normalise_and_shift(sparse_depth_tensor, valid_sparse_mask, depth_min, depth_max, epsilon):
    if valid_sparse_mask.any():
        sparse_depth_tensor = sparse_depth_tensor.clone()
        sparse_depth_tensor[~valid_sparse_mask] = 0.0       # set invalid sparse depth to 0.0 - this will normalise to -1.0

        # Normalize to [0, 1]
        sparse_01 = (sparse_depth_tensor - depth_min) / (depth_max - depth_min)
        sparse_01 = torch.clamp(sparse_01, 0.0, 1.0)

        # Mask and shift
        sparse_01[valid_sparse_mask] = sparse_01[valid_sparse_mask] * (1 - epsilon) + epsilon
        sparse_01[~valid_sparse_mask] = 0.0

        return sparse_01
    else:
        raise RuntimeError("No valid sparse depth pixels found")

def infer_all_with_dataloader(pipe, dataset, data_root, dataset_name, prompt, gen, output_dir, depth_min, depth_max, controlnet_conditioning_scale=1.0, num_workers=4, batch_size=1):
    os.makedirs(output_dir, exist_ok=True)
    inference_times = []
    skipped = 0

    csv_log_path = os.path.join(output_dir, "inference_times.csv")
    skipped_log_path = os.path.join(output_dir, "skipped_samples.txt")

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    epsilon = 0.02

    with open(csv_log_path, "w", newline="") as log_file, open(skipped_log_path, "w") as skipped_log:
        writer = csv.writer(log_file)
        writer.writerow(["sample", "inference_time_ms"])

        for batch in tqdm(dataloader, desc="ControlNet-E2E Inference", unit="batch"):
            lines, rgb_paths, gt_paths, sparse_depth_paths, mask_paths = batch
            mask_paths = mask_paths or [None] * len(lines)

            for line, rgb_path, depth_path, sparse_depth_path, mask_path in zip(lines, rgb_paths, gt_paths, sparse_depth_paths, mask_paths):
                #base_path = line.split("\t")[0]
                base_path = line.strip().split()[0] # For spaces, hopefully this is the right way to split
                base_name = os.path.splitext(os.path.basename(base_path))[0]
                subdir = os.path.dirname(base_path)
                relative_subdir = os.path.relpath(subdir, start=data_root)
                if relative_subdir.startswith(".."):
                    relative_subdir = os.path.basename(subdir)
                print(f"relative_subdir: {relative_subdir}")
                out_path_npy = os.path.join(output_dir, relative_subdir, f"{base_name}.npy")
                #out_path_vis = os.path.join(output_dir, relative_subdir, f"{base_name}.png")

                try:
                    if not os.path.exists(rgb_path) or not os.path.exists(sparse_depth_path):
                        raise FileNotFoundError("Missing input file")

                    print(rgb_path, "\n", sparse_depth_path, "\n", depth_path, "\n", mask_path)
                    rgb_image = Image.open(rgb_path).convert("RGB") ###

                    # Load sparse depth
                    if dataset_name == "diode":
                        sparse_depth_array = np.load(sparse_depth_path).astype(np.float32)
                        sparse_depth_tensor = torch.from_numpy(sparse_depth_array)
                        valid_sparse_mask = (sparse_depth_tensor > depth_min) & (sparse_depth_tensor < depth_max)
                        sparse_depth = mask_out_normalise_and_shift(sparse_depth_tensor, valid_sparse_mask, depth_min, depth_max, epsilon)
                    elif dataset_name == "kitti":
                        rgb_tensor = to_tensor(rgb_image)
                        rgb_tensor = kitti_benchmark_crop(rgb_tensor)
                        rgb_image = to_pil(rgb_tensor)
                        sparse_depth_image = cv2.imread(sparse_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        sparse_depth_m = sparse_depth_image.astype(np.float32) / 256.0
                        sparse_depth_tensor = torch.from_numpy(sparse_depth_m)
                        sparse_depth_tensor = kitti_benchmark_crop(sparse_depth_tensor)
                        valid_sparse_mask = (sparse_depth_tensor > depth_min) & (sparse_depth_tensor < depth_max)
                        sparse_depth = mask_out_normalise_and_shift(sparse_depth_tensor, valid_sparse_mask, depth_min, depth_max, epsilon)
                    elif dataset_name in ["nyu_depth_v2", "scannet"]:
                        sparse_depth_image = cv2.imread(sparse_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        if sparse_depth_image is None:
                            raise RuntimeError(f"cv2 failed to read sparse depth: {sparse_depth_path}")
                        sparse_depth_image = sparse_depth_image.astype(np.float32) / 1000.0
                        sparse_depth_tensor = torch.from_numpy(sparse_depth_image)
                        valid_sparse_mask = (sparse_depth_tensor > depth_min) & (sparse_depth_tensor < depth_max)
                        sparse_depth = mask_out_normalise_and_shift(sparse_depth_tensor, valid_sparse_mask, depth_min, depth_max, epsilon)
                    elif dataset_name == "vkitti":
                        rgb_tensor = to_tensor(rgb_image)
                        rgb_tensor = kitti_benchmark_crop(rgb_tensor)
                        rgb_image = to_pil(rgb_tensor)
                        
                        sparse_depth_image = cv2.imread(sparse_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                        if sparse_depth_image is None:
                            raise RuntimeError(f"cv2 failed to read sparse depth: {sparse_depth_path}")
                        sparse_depth_image = sparse_depth_image.astype(np.float32) / 100.0
                        sparse_depth_tensor = torch.from_numpy(sparse_depth_image)
                        sparse_depth_tensor = kitti_benchmark_crop(sparse_depth_tensor)
                        valid_sparse_mask = (sparse_depth_tensor > depth_min) & (sparse_depth_tensor < depth_max)
                        sparse_depth = mask_out_normalise_and_shift(sparse_depth_tensor, valid_sparse_mask, depth_min, depth_max, epsilon)
                    
                    control = sparse_depth.clamp(0, 1).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                    #control = control.expand(-1, 3, -1, -1)  # (1,3,H,W)

                    
                    if torch.cuda.is_available():
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        start.record()

                    try:
                        pred = pipe(
                            rgb_image=rgb_image,
                            prompt=prompt,
                            image=control,
                            num_inference_steps=1,
                            generator=gen,
                            output_type="pt",
                            controlnet_conditioning_scale=controlnet_conditioning_scale,
                        ).prediction
                    except ValueError as e:
                        raise RuntimeError(
                            f"🚫 PIPELINE REJECTED SAMPLE {base_path} — {e}"
                        )

                    if torch.cuda.is_available():
                        end.record()
                        torch.cuda.synchronize()
                        elapsed = start.elapsed_time(end)
                    else:
                        elapsed = None

                    os.makedirs(os.path.dirname(out_path_npy), exist_ok=True)

                    pred_cpu = pred.squeeze().detach().cpu().numpy()  # Shape: (H, W)
                    np.save(out_path_npy, pred_cpu)

                    if elapsed is not None:
                        writer.writerow([base_path, f"{elapsed:.2f}"])
                        log_file.flush()
                        inference_times.append(elapsed)

                except Exception as e:
                    traceback.print_exc()
                    skipped += 1
                    tb = traceback.TracebackException.from_exception(e)
                    err_frame = tb.stack[-1]
                    skipped_log.write(
                        f"{base_path} - error in {err_frame.filename}:{err_frame.lineno}: {e}\n"
                    )
                    continue

    print(f"\n✅ Inference complete. Skipped {skipped} samples.")
    if inference_times:
        avg = sum(inference_times) / len(inference_times)
        print(f"📊 Average Inference Time: {avg:.2f} ms ({1000 / avg:.2f} FPS)")

def main(args):
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    base_model_path = args.pretrained_model_name_or_path
    controlnet_path = args.controlnet_model_name_or_path
    checkpoint = args.resume_from_checkpoint
    if checkpoint is not None and os.path.isdir(os.path.join(controlnet_path, checkpoint)):
        complete_controlnet_path = os.path.join(controlnet_path, checkpoint)
    else:
        raise ValueError(
            f"Checkpoint {checkpoint} not found in {controlnet_path}. Please provide a valid checkpoint directory."
        )
    weight_dtype = torch.float32

    controlnet = ControlNetModel.from_pretrained(complete_controlnet_path, torch_dtype=weight_dtype)

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

    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
    )

    val_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        timestep_spacing="trailing",
        revision=args.revision,
        variant=args.variant,
    )

    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
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
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    # memory optimization.
    pipeline.enable_model_cpu_offload()
    prompt = ""

    gen = torch.manual_seed(1234)

    dataset = InferenceDataset(args.split_file, args.data_root)
    infer_all_with_dataloader(
    pipe=pipeline,
    dataset=dataset,
    data_root=args.data_root,
    dataset_name=args.dataset_name,
    prompt=prompt,
    gen=gen,
    output_dir=args.output_dir,
    num_workers=args.dataloader_num_workers,
    batch_size=args.test_batch_size,
    depth_min=args.depth_min,
    depth_max=args.depth_max,
    controlnet_conditioning_scale=args.controlnet_conditioning_scale
)

if __name__ == "__main__":
    args = parse_args()
    main(args)   # ← this was missing