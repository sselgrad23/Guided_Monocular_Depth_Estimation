import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import json
from numpy.random import default_rng

def center_crop(img, target_height, target_width):
    h, w = img.shape[:2]
    top = max((h - target_height) // 2, 0)
    left = max((w - target_width) // 2, 0)
    return img[top:top+target_height, left:left+target_width]


def create_sparse_depth(depth_img, num_points, exclude_max, rng):
    if exclude_max:
        mask = (depth_img > 0) & (depth_img < 65535)
    else:
        mask = depth_img > 0

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros_like(depth_img)

    sampled_indices = rng.choice(len(coords), min(num_points, len(coords)), replace=False)
    sampled_coords = coords[sampled_indices]

    sparse = np.zeros_like(depth_img)
    for y, x in sampled_coords:
        sparse[y, x] = depth_img[y, x]

    return sparse

def build_sparse_path(rel_path: Path, ext: str, out_root: str) -> str:
    sparse_filename = f"sparse_{rel_path.stem}{ext}"
    sparse_path = Path(out_root).name / rel_path.parent / sparse_filename
    return sparse_path.as_posix()

def build_rgb_path(rel_path: Path, ext: str, rgb_root: str) -> str:
    # rel_path = Scene01/morning/frames/depth/Camera_0/depth_00257.png
    parts = list(rel_path.parts)

    # Replace 'depth' folder with 'rgb'
    if 'depth' in parts:
        depth_idx = parts.index('depth')
        parts[depth_idx] = 'rgb'

    # Replace filename from depth_XXXX.png → rgb_XXXX.jpg
    filename = parts[-1].replace("depth", "rgb").replace(".png", ext)
    parts[-1] = filename

    rgb_path = Path(rgb_root).name / Path(*parts)
    return rgb_path.as_posix()


def process_all(
    depth_root,
    out_root,
    rgb_root,
    num_points=None,
    min_points=None,
    max_points=None,
    jsonl_name="metadata.jsonl",
    save_png=True,
    save_npy=False,
    exclude_max=False,
    rng=None
):
    metadata_png = []
    metadata_npy = []

    if rng is None:
        rng = default_rng()

    for path in tqdm(list(Path(depth_root).rglob("*.png"))):
        rel_path = path.relative_to(depth_root)
        out_path_prefix = Path(out_root) / rel_path.parent / f"sparse_{rel_path.name}"
        out_path_prefix.parent.mkdir(parents=True, exist_ok=True)

        depth_img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth_img is None:
            print(f"Skipping {path} — could not load.")
            continue
        if len(depth_img.shape) > 2:
            print(f"Skipping {path} — not a single-channel depth image.")
            continue
        if depth_img.dtype != np.uint16:
            print(f"Skipping {path} — depth image is not 16-bit. Got {depth_img.dtype}")
            continue

        #depth_img = center_crop(depth_img, 352, 1216)
        # Decide how many points to sample for THIS image
        if (min_points is not None) and (max_points is not None):
            k = int(rng.integers(min_points, max_points + 1))
        else:
            k = num_points
        sparse = create_sparse_depth(depth_img, k, exclude_max, rng)

        if save_png:
            sparse_png_path = out_path_prefix.with_suffix('.png')
            cv2.imwrite(str(sparse_png_path), sparse.astype(np.uint16))
            metadata_png.append({
                "image": str(Path("depth") / rel_path).replace(os.sep, "/"),
                "text": "",
                "conditioning_image": build_sparse_path(rel_path, ".png", out_root),
                "rgb_image": build_rgb_path(rel_path, ".jpg", rgb_root),
                "file_name": str(Path("depth") / rel_path).replace(os.sep, "/")  # <- ADD THIS
            })

        if save_npy:
            sparse_npy_path = out_path_prefix.with_suffix('.npy')
            np.save(str(sparse_npy_path), sparse)
            metadata_npy.append({
                "image": str(Path("depth") / rel_path).replace(os.sep, "/"),
                "text": "",
                "conditioning_image": build_sparse_path(rel_path, ".npy", out_root),
                "rgb_image": build_rgb_path(rel_path, ".jpg", rgb_root),
                "file_name": str(Path("depth") / rel_path).replace(os.sep, "/")  # <- ADD THIS
           })

    Path(out_root).mkdir(parents=True, exist_ok=True)

    if save_png:
        with open(Path(out_root).parent / jsonl_name, "w") as f:
            for entry in metadata_png:
                f.write(json.dumps(entry) + "\n")

    if save_npy:
        with open(Path(out_root).parent / "metadata_npy.jsonl", "w") as f:
            for entry in metadata_npy:
                f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_root", type=str, required=True, help="Path to original vkitti/depth")
    parser.add_argument("--rgb_root", type=str, required=True, help="Path to RGB vkitti data")
    parser.add_argument("--out_root", type=str, required=True, help="Where to save sparse depth")
    parser.add_argument("--num_points", type=int, default=500, help="(Deprecated if --min_points/--max_points set) Fixed number of points to keep")
    parser.add_argument("--min_points", type=int, default=None, help="Minimum points to sample per image")
    parser.add_argument("--max_points", type=int, default=None, help="Maximum points to sample per image (inclusive)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for RNG (omit for OS-entropy)")
    parser.add_argument("--save_png", action="store_true", help="Save 16-bit PNGs")
    parser.add_argument("--save_npy", action="store_true", help="Save .npy arrays")
    parser.add_argument("--exclude_max_depth", action="store_true", help="Skip 65535-valued pixels (far/sky)")
    parser.add_argument("--jsonl_name", type=str, default="metadata.jsonl", help="Custom name for JSONL output")
    args = parser.parse_args()

    rng = default_rng(args.seed)
    process_all(
        args.depth_root,
        args.out_root,
        args.rgb_root,
        num_points=args.num_points,
        min_points=args.min_points,
        max_points=args.max_points,
        jsonl_name=args.jsonl_name,
        save_png=args.save_png,
        save_npy=args.save_npy,
        exclude_max=args.exclude_max_depth,
        rng=rng
    )