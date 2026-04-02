import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
import sys


def create_sparse_depth(depth_img, num_points):
    mask = depth_img > 0
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return np.zeros_like(depth_img), mask

    sampled = coords[np.random.choice(len(coords), min(num_points, len(coords)), replace=False)]
    sparse = np.zeros_like(depth_img)
    for y, x in sampled:
        sparse[y, x] = depth_img[y, x]

    return sparse, mask


def add_gaussian_to_sparse(sparse, noise_divisor):
    """
    Add Gaussian noise to non-zero entries of a sparse depth map.
    Noise stddev = (max - min) / noise_divisor, applied in meters then converted back to mm.
    """
    depth_m = sparse.astype(np.float32) / 1000.0  # mm → m
    valid = sparse > 0
    if not np.any(valid):
        return sparse

    min_val = depth_m[valid].min()
    max_val = depth_m[valid].max()
    stddev = (max_val - min_val) / noise_divisor
    noise = np.zeros_like(depth_m)
    noise[valid] = np.random.normal(0.0, stddev, size=valid.sum())
    noisy = depth_m + noise
    noisy[~valid] = 0.0
    return (noisy * 1000).astype(sparse.dtype)


def add_outliers_to_sparse(sparse, outlier_prob):
    """
    Inject uniform outliers into non-zero entries of a sparse depth map.
    Outlier values sampled uniformly between min and max of valid entries.
    """
    depth_f = sparse.astype(np.float32)
    valid = sparse > 0
    if not np.any(valid):
        return sparse

    min_d = depth_f[valid].min()
    max_d = depth_f[valid].max()
    mask_out = (np.random.rand(*depth_f.shape) < outlier_prob) & valid
    depth_f[mask_out] = np.random.uniform(min_d, max_d, size=mask_out.sum())
    return depth_f.astype(sparse.dtype)


def sample_kitti(depth_root, rgb_root, out_txt, num_points, noise_divisor, outlier_prob, order, seed):
    if seed is not None:
        np.random.seed(seed)

    depth_root = Path(depth_root)
    rgb_root = Path(rgb_root)

    # build suffix in specified order
    suffix_tokens = [str(num_points)]
    for op in order:
        if op == 'gaussian':
            suffix_tokens.append(f"gaussian_{int(noise_divisor)}")
        elif op == 'outliers':
            suffix_tokens.append(f"outliers_{int(outlier_prob * 100)}")
    suffix = "_".join(suffix_tokens)

    # preserve input and output paths separately
    txt_in = Path(out_txt)
    out_txt = txt_in.with_name(f"{txt_in.stem}_N{num_points}{txt_in.suffix}")

    # read input filename list instead of scanning all files
    with open(txt_in, "r") as f:
        lines = [l.strip().split() for l in f.readlines() if l.strip()]

    entries = []
    for l in tqdm(lines, desc="Processing KITTI", dynamic_ncols=True):
        # skip lines with 'None' in any column
        if any(x.strip().lower() == "none" for x in l):
            continue

        # expect at least RGB and depth columns
        if len(l) < 2:
            continue

        rgb_rel, depth_rel = l[0], l[1]

        rgb_file = rgb_root / rgb_rel
        depth_file = depth_root / depth_rel

        if not depth_file.exists() or not rgb_file.exists():
            continue

        # load depth
        depth = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            continue

        # sampling and transforms
        current, mask = None, None
        for op in order:
            if op == 'sampling':
                current, mask = create_sparse_depth(depth, num_points)
            elif op == 'gaussian':
                current = add_gaussian_to_sparse(current, noise_divisor)
            elif op == 'outliers':
                current = add_outliers_to_sparse(current, outlier_prob)

        # output directories
        sparse_dir = depth_root / f"sparse_depth_N{num_points}"
        mask_dir   = depth_root / f"mask_N{num_points}"
        sparse_path = sparse_dir / depth_rel
        mask_path   = mask_dir / (Path(depth_rel).stem + f"_mask_N{num_points}.npy")

        sparse_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(sparse_path), current.astype(np.uint16))
        np.save(str(mask_path), mask)

        # relative paths for output (sparse depth last)
        rel_rgb = rgb_file.relative_to(rgb_root)
        rel_depth = depth_file.relative_to(depth_root)
        rel_mask = mask_path.relative_to(depth_root)
        rel_sparse = sparse_path.relative_to(depth_root)
        entries.append(f"{rel_rgb} {rel_depth} {rel_mask} {rel_sparse}")


    print(f"Writing output to {out_txt}")  # Debug print statement
    with open(out_txt, 'w') as f:
        f.write("\n".join(entries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_root", type=str, required=True, help="Root KITTI depth folder")
    parser.add_argument("--rgb_root", type=str, required=True, help="Root KITTI RGB folder")
    parser.add_argument("--txt_out", type=str, required=True, help="Base output .txt path (without suffix)")
    parser.add_argument("--num_points", type=int, default=500, help="Number of points to sample")
    parser.add_argument("--noise_divisor", type=float, default=None, help="Divisor for Gaussian noise stddev")
    parser.add_argument("--outlier_prob", type=float, default=None, help="Probability for uniform outliers")
    parser.add_argument("--transform_order", type=str, default=None, help="Comma-separated order: sampling,gaussian,outliers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    # determine transform order
    if args.transform_order:
        order = [t.strip() for t in args.transform_order.split(',') if t.strip()]
    else:
        order = ['sampling']
        if args.noise_divisor is not None:
            order.append('gaussian')
        if args.outlier_prob is not None:
            order.append('outliers')

    # validate required params for each transform
    for op in order:
        if op == 'gaussian' and args.noise_divisor is None:
            parser.error("--noise_divisor must be set when 'gaussian' is in --transform_order")
        if op == 'outliers' and args.outlier_prob is None:
            parser.error("--outlier_prob must be set when 'outliers' is in --transform_order")

    print(f"Sampling KITTI with order: {order} (npts={args.num_points}, noise_div={args.noise_divisor}, outlier_prob={args.outlier_prob})...")
    sample_kitti(
        args.depth_root, args.rgb_root, args.txt_out, args.num_points,
        args.noise_divisor, args.outlier_prob, order, args.seed
    )
