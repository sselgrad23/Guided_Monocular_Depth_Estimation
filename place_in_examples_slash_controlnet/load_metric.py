# @GonzaloMartinGarcia
# This file houses our dataset mixer and training dataset classes.
# Modified to replace normals with sparse depth

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import numpy as np
import random
import pandas as pd
import cv2
from torch.utils.data import Sampler #, ConcatDataset, DataLoader, RandomSampler
import itertools
import torch.nn.functional as F
from functools import partial

#################
# Dataset Mixer
#################

class MixedDataLoader:
    def __init__(self, loader1, loader2, split1=9, split2=1):
        self.loader1 = loader1
        self.loader2 = loader2
        self.split1 = split1
        self.split2 = split2
        self.frac1, self.frac2 = self.get_split_fractions()
        self.randchoice1=None

    def __iter__(self):
        self.loader_iter1 = iter(self.loader1)
        self.loader_iter2 = iter(self.loader2)
        self.randchoice1 = self.create_split()
        self.indx = 0
        return self
    
    def get_split_fractions(self):
        size1 = len(self.loader1)
        size2 = len(self.loader2)
        effective_fraction1 = min((size2/size1) * (self.split1/self.split2), 1) 
        effective_fraction2 = min((size1/size2) * (self.split2/self.split1), 1) 
        print("Effective fraction for loader1: ", effective_fraction1)
        print("Effective fraction for loader2: ", effective_fraction2)
        return effective_fraction1, effective_fraction2

    def create_split(self):
        randchoice1 = [True]*int(len(self.loader1)*self.frac1) + [False]*int(len(self.loader2)*self.frac2)
        np.random.shuffle(randchoice1)
        return randchoice1

    def __next__(self):
        if self.indx == len(self.randchoice1):
            raise StopIteration
        if self.randchoice1[self.indx]:
            self.indx += 1
            return next(self.loader_iter1)
        else:
            self.indx += 1
            return next(self.loader_iter2)
        
    def __len__(self):
        return int(len(self.loader1)*self.frac1) + int(len(self.loader2)*self.frac2)
    

#################
# Transforms 
#################

def resize_tensor(img: torch.Tensor, size, mode="bilinear"):
    # accept (H,W), (C,H,W), or (N,C,H,W)
    squeeze_back = False
    if img.dim() == 2:                 # (H,W)
        img = img.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)
        squeeze_back = True
    elif img.dim() == 3:               # (C,H,W)
        img = img.unsqueeze(0)               # -> (1,C,H,W)

    align = False if mode in ("bilinear", "bicubic") else None
    out = F.interpolate(img, size=size, mode=mode, align_corners=align)

    if squeeze_back:                   # return (H,W)
        return out.squeeze(0).squeeze(0)
    return out.squeeze(0)              # return (C,H,W)

# Hyperism
class SynchronizedTransform_Hyper:
    def __init__(self, H, W):
        self.resize          = transforms.Resize((H,W))
        self.resize_depth = partial(resize_tensor, size=(H,W), mode="nearest")
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.to_tensor       = transforms.ToTensor()

    def __call__(self, rgb_image, depth_tensor, sparse_depth_tensor):
        # h-flip
        if random.random() > 0.5:
            rgb_image = self.horizontal_flip(rgb_image)
            depth_tensor = depth_tensor.flip(-1)
            sparse_depth_tensor = sparse_depth_tensor.flip(-1)
        # resize
        rgb_image   = self.resize(rgb_image)
        depth_tensor = self.resize_depth(depth_tensor)
        sparse_depth_tensor = self.resize_depth(sparse_depth_tensor)
        # to tensor
        rgb_tensor = self.to_tensor(rgb_image)
        return rgb_tensor, depth_tensor, sparse_depth_tensor

# Virtual KITTI 2
class SynchronizedTransform_VKITTI:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
        self.horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)

    # KITTI benchmark crop from Marigold:
    # https://github.com/prs-eth/Marigold/blob/62413d56099d36573b2de1eb8c429839734b7782/src/dataset/kitti_dataset.py#L75
    @staticmethod
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

    def __call__(self, rgb_image, depth_tensor, sparse_depth_tensor):
        # h-flip
        if random.random() > 0.5:
            rgb_image = self.horizontal_flip(rgb_image)
            depth_tensor = depth_tensor.flip(-1)
            sparse_depth_tensor = sparse_depth_tensor.flip(-1)
        # to tensor
        rgb_tensor = self.to_tensor(rgb_image)      
        # kitti benchmark crop
        rgb_tensor = self.kitti_benchmark_crop(rgb_tensor)
        depth_tensor = self.kitti_benchmark_crop(depth_tensor)
        sparse_depth_tensor = self.kitti_benchmark_crop(sparse_depth_tensor)
        # return
        return rgb_tensor, depth_tensor, sparse_depth_tensor

#####################
# Training Datasets
#####################

# Hypersim   
class Hypersim(Dataset):
    def __init__(self, root_dir, csv_filepath, transform=True, near_plane=1e-5, far_plane=65.0):
        self.root_dir   = root_dir
        self.split_path = csv_filepath
        self.near_plane = near_plane
        self.far_plane  = far_plane
        self.pairs = self._find_pairs()
        self.transform =  SynchronizedTransform_Hyper(H=480, W=640) if transform else None

    def _find_pairs(self):
        df = pd.read_csv(self.split_path)
        pairs = []
        for _, row in df.iterrows():
            if row['included_in_public_release'] and (row['split_partition_name'] == "train"):
                rgb_path = os.path.join(self.root_dir, row['rgb_path']) 
                depth_path = os.path.join(self.root_dir, row['depth_path']) 
                sparse_depth_path = os.path.join(self.root_dir, row['sparse_depth_path']) 
                if os.path.exists(rgb_path) and os.path.exists(depth_path) and os.path.exists(sparse_depth_path):
                    pair_info = {'rgb_path': rgb_path, 'depth_path': depth_path, 'sparse_depth_path': sparse_depth_path}    
                    pairs.append(pair_info)
        return pairs

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pairs = self.pairs[idx]

        # get RGB
        rgb_path   = pairs['rgb_path']
        rgb_image  = Image.open(rgb_path).convert('RGB')
        # get depth
        depth_path  = pairs['depth_path']

        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_image = depth_image.astype(np.float32)/1000.0    # mm to meters
        depth_tensor = torch.from_numpy(depth_image).float()

        # get sparse depth
        sparse_depth_path = pairs['sparse_depth_path']

        sparse_depth_image = cv2.imread(sparse_depth_path, cv2.IMREAD_UNCHANGED)
        sparse_depth_image = sparse_depth_image.astype(np.float32)/1000.0    # mm to meters
        sparse_depth_tensor = torch.from_numpy(sparse_depth_image).float()

        # transform
        if self.transform is not None:
            rgb_tensor, depth_tensor, sparse_depth_tensor = self.transform(rgb_image, depth_tensor, sparse_depth_tensor)
        else:
            rgb_tensor    = transforms.ToTensor()(rgb_image)

        # get valid depth mask
        valid_depth_mask = (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)
        valid_sparse_mask = (sparse_depth_tensor > self.near_plane) & (sparse_depth_tensor < self.far_plane)


        # Process RGB 
        rgb_tensor  = rgb_tensor*2.0 - 1.0 # [-1,1]

        # Process depth
        if valid_depth_mask.any():
            epsilon = 0.02
            depth_tensor = depth_tensor.clone()
            sparse_depth_tensor = sparse_depth_tensor.clone()    
            depth_tensor[~valid_depth_mask] = 0.0                    # set invalid depth to 0.0 - this will normalise to -1.0
            sparse_depth_tensor[~valid_sparse_mask] = 0.0
            metric_tensor = depth_tensor.clone()                           # keep metric depth for e2e loss ft

            # Normalize to [0, 1]
            depth_01 = (depth_tensor - self.near_plane) / (self.far_plane - self.near_plane)
            depth_01 = torch.clamp(depth_01, 0.0, 1.0)

            # Mask and shift
            depth_01[valid_depth_mask] = depth_01[valid_depth_mask] * (1 - epsilon) + epsilon
            depth_01[~valid_depth_mask] = 0.0
            # Convert to [-1.0, 1.0]
            depth_tensor = torch.clamp(depth_01 * 2.0 - 1.0, -1.0, 1.0)

            # Normalize to [0, 1]
            sparse_01 = (sparse_depth_tensor - self.near_plane) / (self.far_plane - self.near_plane)
            sparse_01 = torch.clamp(sparse_01, 0.0, 1.0)
    
            # Mask and shift
            sparse_01[valid_sparse_mask] = sparse_01[valid_sparse_mask] * (1 - epsilon) + epsilon
            sparse_01[~valid_sparse_mask] = 0.0
            # Convert to [-1.0, 1.0]
            sparse_depth_tensor = torch.clamp(sparse_01 * 2.0 - 1.0, -1.0, 1.0)
        else:
            depth_tensor = torch.zeros_like(depth_tensor)
            sparse_depth_tensor = torch.zeros_like(depth_tensor)
            metric_tensor = torch.zeros_like(depth_tensor)
        depth_tensor   = torch.stack([depth_tensor, depth_tensor, depth_tensor]).squeeze() # stack depth map for VAE encoder
        sparse_depth_tensor = torch.stack([sparse_depth_tensor, sparse_depth_tensor, sparse_depth_tensor]).squeeze() # stack sparse depth map for VAE encoder

        return {"rgb": rgb_tensor, 
                "depth": depth_tensor, 
                'metric': metric_tensor, 
                'sparse_depth': sparse_depth_tensor, 
                "val_mask": valid_depth_mask, 
                "domain": "indoor"}

    
# Virtual KITTI 2.0
class VirtualKITTI2(Dataset):
    def __init__(self, root_dir, sparse_depth_name=None, transform=None, near_plane=1e-5, far_plane=80.0):
        self.root_dir = root_dir
        self.near_plane = near_plane
        self.far_plane  = far_plane
        self.sparse_depth_name = sparse_depth_name
        self.transform = SynchronizedTransform_VKITTI() if transform else None
        self.pairs = self._find_pairs()

    def _find_pairs(self):
        scenes = ["Scene01", "Scene02", "Scene06", "Scene18", "Scene20"]
        weather_conditions = ["morning", "fog", "rain", "sunset", "overcast"]
        cameras = ["Camera_0", "Camera_1"]
        vkitti2_rgb_path = os.path.join(self.root_dir, "rgb") # vkitti_2.0.3_rgb
        vkitti2_depth_path =  os.path.join(self.root_dir, "depth") # vkitti_2.0.3_depth
        if self.sparse_depth_name is not None:
            vkitti2_sparse_depth_path = os.path.join(self.root_dir, self.sparse_depth_name) # vkitti_2.0.3_sparse_depth
        else:
            vkitti2_sparse_depth_path = os.path.join(self.root_dir, "sparse_depth") # vkitti_2.0.3_sparse_depth
        pairs = []
        for scene in scenes:
            for weather in weather_conditions:
                for camera in cameras:
                    rgb_dir = os.path.join(vkitti2_rgb_path, scene, weather, "frames", "rgb" ,camera)
                    depth_dir = os.path.join(vkitti2_depth_path, scene, weather, "frames","depth" , camera)
                    sparse_depth_dir = os.path.join(vkitti2_sparse_depth_path, scene, weather, "frames", "depth", camera)
                    if os.path.exists(rgb_dir) and os.path.exists(depth_dir):
                        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".jpg")]
                        rgb_files  = [file[3:] for file in rgb_files]
                        for file in rgb_files:
                            rgb_file = "rgb" + file
                            depth_file = "depth" + file.replace('.jpg', '.png')
                            sparse_depth_file = "sparse_depth" + file.replace('.jpg', '.png')
                            rgb_path = os.path.join(rgb_dir, rgb_file)
                            depth_path = os.path.join(depth_dir, depth_file)
                            sparse_depth_path = os.path.join(sparse_depth_dir, sparse_depth_file)
                            pairs.append((rgb_path, depth_path, sparse_depth_path))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path, sparse_depth_path = self.pairs[idx]

        # get RGB
        rgb_image   = Image.open(rgb_path).convert('RGB')
        # get depth
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth_image = depth_image.astype(np.float32)/100.0    # cm to meters
        depth_tensor = torch.from_numpy(depth_image).float()

        # get sparse depth
        sparse_depth_image = cv2.imread(sparse_depth_path, cv2.IMREAD_UNCHANGED)
        sparse_depth_image = sparse_depth_image.astype(np.float32)/100.0    # cm to meters
        sparse_depth_tensor = torch.from_numpy(sparse_depth_image).float()

        # transform
        if self.transform is not None:
            rgb_tensor, depth_tensor, sparse_depth_tensor = self.transform(rgb_image, depth_tensor, sparse_depth_tensor)
        else:
            rgb_tensor    = transforms.ToTensor()(rgb_image)

        # get valid depth mask
        valid_depth_mask =  (depth_tensor > self.near_plane) & (depth_tensor < self.far_plane)
        valid_sparse_mask = (sparse_depth_tensor > self.near_plane) & (sparse_depth_tensor < self.far_plane)

        # Process RGB
        rgb_tensor = rgb_tensor*2.0 - 1.0 # [-1,1]

        # Process depth
        if valid_depth_mask.any():
            #else:
            epsilon = 0.02
            depth_tensor = depth_tensor.clone()
            sparse_depth_tensor = sparse_depth_tensor.clone()
            depth_tensor[~valid_depth_mask] = 0.0               # set invalid depth to 0.0 - this will normalise to -1.0
            sparse_depth_tensor[~valid_sparse_mask] = 0.0       # set invalid sparse depth to 0.0 - this will normalise to -1.0
            metric_tensor = depth_tensor.clone()                           # keep metric depth for e2e loss ft

            # Normalize to [0, 1]
            depth_01 = (depth_tensor - self.near_plane) / (self.far_plane - self.near_plane)
            depth_01 = torch.clamp(depth_01, 0.0, 1.0)

            # Mask and shift
            depth_01[valid_depth_mask] = depth_01[valid_depth_mask] * (1 - epsilon) + epsilon
            depth_01[~valid_depth_mask] = 0.0
            # Convert to [-1.0, 1.0]
            depth_tensor = torch.clamp(depth_01 * 2.0 - 1.0, -1.0, 1.0)

            # Normalize to [0, 1]
            sparse_01 = (sparse_depth_tensor - self.near_plane) / (self.far_plane - self.near_plane)
            sparse_01 = torch.clamp(sparse_01, 0.0, 1.0)

            # Mask and shift
            sparse_01[valid_sparse_mask] = sparse_01[valid_sparse_mask] * (1 - epsilon) + epsilon
            sparse_01[~valid_sparse_mask] = 0.0
            # Convert to [-1.0, 1.0]
            sparse_depth_tensor = torch.clamp(sparse_01 * 2.0 - 1.0, -1.0, 1.0)

        else:
            depth_tensor = torch.zeros_like(depth_tensor)
            sparse_depth_tensor = torch.zeros_like(depth_tensor)
            metric_tensor = torch.zeros_like(depth_tensor)
        depth_tensor   = torch.stack([depth_tensor, depth_tensor, depth_tensor]).squeeze() # stack depth map for VAE encoder
        sparse_depth_tensor = torch.stack([sparse_depth_tensor, sparse_depth_tensor, sparse_depth_tensor]).squeeze() # stack sparse depth map for VAE encoder
        


        return {"rgb": rgb_tensor, 
                "depth": depth_tensor, 
                'metric': metric_tensor, 
                'sparse_depth': sparse_depth_tensor, 
                "val_mask": valid_depth_mask, 
                "domain": "outdoor"}