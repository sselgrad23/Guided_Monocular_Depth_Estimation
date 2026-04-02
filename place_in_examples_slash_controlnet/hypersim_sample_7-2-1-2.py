#!/usr/bin/env python3
import os
import argparse
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from numpy.random import default_rng
import math

def create_sparse_depth(depth_img: np.ndarray, num_points: int, exclude_max: bool, rng) -> np.ndarray:
    if depth_img is None or depth_img.ndim != 2 or depth_img.dtype != np.uint16:
        raise ValueError("depth_img must be a single-channel uint16 image")

    # Valid = >0, and optionally < max (e.g., excludes 65535 'far plane')
    if exclude_max:
        maxv = np.iinfo(depth_img.dtype).max
        mask = (depth_img > 0) & (depth_img < maxv)
    else:
        mask = (depth_img > 0)

    coords = np.argwhere(mask)
    if len(coords) == 0:
        return np.zeros_like(depth_img, dtype=np.uint16)

    k = min(int(num_points), len(coords))
    idx = rng.choice(len(coords), k, replace=False)
    sampled_coords = coords[idx]

    sparse = np.zeros_like(depth_img, dtype=np.uint16)
    # vectorized assignment
    ys, xs = sampled_coords[:, 0], sampled_coords[:, 1]
    sparse[ys, xs] = depth_img[ys, xs]
    return sparse

def choose_k(num_points: int, min_points: int, max_points: int, rng, sampling: str = "uniform") -> int:
    if min_points is None or max_points is None:
        return int(num_points)

    if max_points < min_points:
        raise ValueError("--max_points must be >= --min_points")

    lo, hi = int(min_points), int(max_points)
    if sampling == "uniform":
        return int(rng.integers(lo, hi + 1))

    if sampling == "loguniform":
        if lo < 1:
            raise ValueError("For log-uniform sampling, --min_points must be >= 1.")
        u = rng.uniform(math.log(lo), math.log(hi))
        k = int(round(math.exp(u)))
        return max(lo, min(hi, k))

    raise ValueError(f"Unknown sampling mode: {sampling}")


def derive_sparse_rel_path(row, prefer_from="depth") -> str:
    if prefer_from == "depth" and isinstance(row.get("depth_path"), str):
        rel = Path(row["depth_path"])
        name = rel.name
        if name.startswith("depth_plane_"):
            new_name = name.replace("depth_plane_", "sparse_depth_", 1)
        elif name.startswith("depth_"):
            new_name = name.replace("depth_", "sparse_depth_", 1)
        else:
            new_name = "sparse_" + name
        return (rel.parent / new_name).as_posix()

    if isinstance(row.get("rgb_path"), str):
        rel = Path(row["rgb_path"])
        name = rel.name
        if name.startswith("rgb_"):
            new_name = name.replace("rgb_", "sparse_depth_", 1).rsplit(".", 1)[0] + ".png"
        else:
            stem = Path(name).stem
            new_name = f"sparse_depth_{stem}.png"
        return (rel.parent / new_name).as_posix()

    return "sparse_depth.png"

def main(args):
    rng = default_rng(args.seed)

    root = Path(args.root_dir)
    if not root.exists():
        raise FileNotFoundError(f"--root_dir does not exist: {root}")

    csv_in = Path(args.csv_in)
    if not csv_in.exists():
        raise FileNotFoundError(f"--csv_in does not exist: {csv_in}")

    df = pd.read_csv(csv_in)

    if "depth_path" not in df.columns or "rgb_path" not in df.columns:
        raise ValueError("Input CSV must contain 'depth_path' and 'rgb_path' columns.")

    if "sparse_depth_path" not in df.columns:
        df["sparse_depth_path"] = ""

    missing_depth, written, skipped_existing = 0, 0, 0
    for i in tqdm(range(len(df)), desc="Generating sparse depth"):
        row = df.iloc[i].to_dict()

        depth_rel = row.get("depth_path", "")
        if not isinstance(depth_rel, str) or depth_rel.strip() == "":
            missing_depth += 1
            continue

        depth_abs = root / depth_rel
        if not depth_abs.exists():
            missing_depth += 1
            continue

        sparse_rel = derive_sparse_rel_path(row, prefer_from="depth")
        sparse_abs = root / sparse_rel

        if sparse_abs.exists() and not args.overwrite:
            df.at[i, "sparse_depth_path"] = sparse_rel
            skipped_existing += 1
            continue

        sparse_abs.parent.mkdir(parents=True, exist_ok=True)

        depth_img = cv2.imread(str(depth_abs), cv2.IMREAD_UNCHANGED)
        if depth_img is None:
            missing_depth += 1
            continue
        if depth_img.ndim != 2 or depth_img.dtype != np.uint16:
            if depth_img.ndim == 3:
                depth_img = depth_img[:, :, 0]
            depth_img = depth_img.astype(np.uint16)

        k = choose_k(args.num_points, args.min_points, args.max_points, rng, args.points_sampling)

        sparse = create_sparse_depth(depth_img, k, args.exclude_max_depth, rng)

        ok = cv2.imwrite(str(sparse_abs), sparse)
        if not ok:
            print(f"Warning: failed to write {sparse_abs}")
            continue

        df.at[i, "sparse_depth_path"] = sparse_rel
        written += 1

    out_csv = Path(args.csv_out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    print(f"\nDone.")
    print(f"  Rows processed: {len(df)}")
    print(f"  Sparse written: {written}")
    print(f"  Existing kept : {skipped_existing}")
    print(f"  Missing depth : {missing_depth}")
    print(f"  Output CSV   : {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate sparse depth PNGs alongside Hypersim RGB/Depth and add 'sparse_depth_path' to a CSV copy.")
    p.add_argument("--root_dir", required=True, help="Root directory containing Hypersim 'train' folders (e.g., .../processed/train)")
    p.add_argument("--csv_in", required=True, help="Path to the input CSV (e.g., filename_meta_train.csv)")
    p.add_argument("--csv_out", required=True, help="Path to write the OUTPUT CSV (copy + sparse_depth_path column)")
    p.add_argument("--num_points", type=int, default=500, help="Fixed number of points to sample (ignored if both --min_points and --max_points are set)")
    p.add_argument("--min_points", type=int, default=None, help="Minimum number of points (uniform)")
    p.add_argument("--max_points", type=int, default=None, help="Maximum number of points (uniform, inclusive)")
    p.add_argument("--exclude_max_depth", action="store_true", help="Exclude 65535-valued pixels (far plane) from sampling")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing sparse files if they already exist")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    p.add_argument(
        "--points_sampling",
        type=str,
        choices=["uniform", "loguniform"],
        default="uniform",
        help="How to sample k when --min_points/--max_points are set. Default: uniform"
        )
    args = p.parse_args()
    main(args)
