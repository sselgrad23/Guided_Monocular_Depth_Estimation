import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from itertools import chain

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
    depth_m = sparse.astype(np.float32) / 1000.0  # convert to meters
    valid = sparse > 0
    if not np.any(valid):
        return sparse

    min_val = depth_m[valid].min()
    max_val = depth_m[valid].max()
    stddev = (max_val - min_val) / noise_divisor
    noise = np.zeros_like(depth_m)
    noise[valid] = np.random.normal(0.0, stddev, size=valid.sum())
    noisy = depth_m + noise
    noisy[~valid] = 0.0  # preserve zeros
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
    outlier_mask = (np.random.rand(*depth_f.shape) < outlier_prob) & valid
    depth_f[outlier_mask] = np.random.uniform(min_d, max_d, size=outlier_mask.sum())
    return depth_f.astype(sparse.dtype)

def sample_scannet(root_dir, output_txt, num_points, noise_divisor, outlier_prob, order, seed):
    # reproducibility
    if seed is not None:
        np.random.seed(seed)

    root_dir = Path(root_dir).resolve()
    txt_in = Path(output_txt)
    output_txt = txt_in.with_name(f"{txt_in.stem}_N{num_points}{txt_in.suffix}")

    # build suffix for folder naming
    suffix_tokens = [f"N{num_points}"]
    for op in order:
        if op == 'gaussian':
            suffix_tokens.append(f"gaussian_{int(noise_divisor)}")
        elif op == 'outliers':
            suffix_tokens.append(f"outliers_{int(outlier_prob * 100)}")
    suffix = "_".join(suffix_tokens)

    # read filename list
    with open(txt_in, "r") as f:
        lines = [l.strip().split() for l in f.readlines() if l.strip()]

    entries = []
    for l in tqdm(lines, desc="Processing ScanNet", dynamic_ncols=True):
        # skip invalid or incomplete lines
        if len(l) < 2 or any(x.strip().lower() == "none" for x in l):
            continue

        rgb_rel, depth_rel = l
        rgb_path = root_dir / rgb_rel
        depth_path = root_dir / depth_rel

        if not rgb_path.exists() or not depth_path.exists():
            continue

        # Load depth map
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            continue

        # Create sparse depth and GT mask
        current, mask = create_sparse_depth(depth, num_points)

        # Apply optional transforms
        for op in order:
            if op == 'gaussian':
                current = add_gaussian_to_sparse(current, noise_divisor)
            elif op == 'outliers':
                current = add_outliers_to_sparse(current, outlier_prob)

        # Define output paths
        sparse_dir = root_dir / f"sparse_depth_N{num_points}"
        mask_dir   = root_dir / f"mask_N{num_points}"
        sparse_path = sparse_dir / depth_rel
        mask_path   = mask_dir / depth_rel.replace(".png", ".npy")

        sparse_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        cv2.imwrite(str(sparse_path), current.astype(np.uint16))
        np.save(str(mask_path), mask)  # GT mask (not sampled)

        # Relative output entries
        rel_rgb = rgb_path.relative_to(root_dir)
        rel_depth = depth_path.relative_to(root_dir)
        rel_mask = mask_path.relative_to(root_dir)
        rel_sparse = sparse_path.relative_to(root_dir)

        entries.append(f"{rel_rgb} {rel_depth} {rel_mask} {rel_sparse}")

    print(f"Writing output to {output_txt}")
    with open(output_txt, "w") as f:
        f.write("\n".join(entries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True, help="Path to Scannet scenes folder")
    parser.add_argument("--output_txt", type=str, required=True, help="Output .txt file path prefix")
    parser.add_argument("--num_points", type=int, default=500, help="Number of points to sample (sampling)")
    parser.add_argument("--noise_divisor", type=float, default=None, help="Divisor for Gaussian noise stddev (e.g. 10)")
    parser.add_argument("--outlier_prob", type=float, default=None, help="Probability of injecting outliers (e.g. 0.05)")
    parser.add_argument("--transform_order", type=str, default=None, help="Comma-separated transforms: sampling,gaussian,outliers")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    # parse transform order or default sequence
    if args.transform_order:
        order = [t.strip() for t in args.transform_order.split(',') if t.strip()]
    else:
        order = ['sampling']
        if args.noise_divisor is not None:
            order.append('gaussian')
        if args.outlier_prob is not None:
            order.append('outliers')

    # validate that required parameters are provided
    for op in order:
        if op == 'gaussian' and args.noise_divisor is None:
            parser.error("--noise_divisor must be set when including 'gaussian' in --transform_order")
        if op == 'outliers' and args.outlier_prob is None:
            parser.error("--outlier_prob must be set when including 'outliers' in --transform_order")

    sample_scannet(
        args.root_dir, args.output_txt, args.num_points,
        args.noise_divisor, args.outlier_prob, order, args.seed
    )
