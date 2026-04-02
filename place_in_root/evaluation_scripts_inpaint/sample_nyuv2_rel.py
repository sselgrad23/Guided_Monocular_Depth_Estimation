import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import os

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
    depth_m = sparse.astype(np.float32) / 1000.0
    valid = sparse > 0
    if not np.any(valid):
        return sparse
    min_val, max_val = depth_m[valid].min(), depth_m[valid].max()
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
    min_d, max_d = depth_f[valid].min(), depth_f[valid].max()
    outlier_mask = (np.random.rand(*depth_f.shape) < outlier_prob) & valid
    depth_f[outlier_mask] = np.random.uniform(min_d, max_d, size=outlier_mask.sum())
    return depth_f.astype(sparse.dtype)

def sample_from_filelist(base_dir, file_list, output_txt, num_points, noise_divisor, outlier_prob, order, seed):
    if seed is not None:
        np.random.seed(seed)

    base_dir = Path(base_dir)
    txt_in = Path(file_list)
    out_txt = Path(output_txt)
    out_txt = out_txt.with_name(f"{out_txt.stem}_N{num_points}{out_txt.suffix}")

    entries = []

    with open(txt_in, "r") as f:
        lines = [l.strip().split() for l in f.readlines() if l.strip()]

    for l in tqdm(lines, desc=f"Processing {txt_in.stem}"):
        # skip invalid or incomplete lines
        if len(l) < 3 or any(x.strip().lower() == "none" for x in l):
            continue

        rgb_rel, depth_rel, filled_rel = l
        rgb_path = base_dir / rgb_rel
        depth_path = base_dir / depth_rel
        filled_path = base_dir / filled_rel

        if not rgb_path.exists() or not depth_path.exists():
            continue

        # Load depth map
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            continue

        # Create sparse depth + GT mask
        current, mask = create_sparse_depth(depth, num_points)

        # Apply optional noise/outliers
        for op in order:
            if op == 'gaussian':
                current = add_gaussian_to_sparse(current, noise_divisor)
            elif op == 'outliers':
                current = add_outliers_to_sparse(current, outlier_prob)

        # Output directories
        sparse_dir = base_dir / f"sparse_depth_N{num_points}"
        mask_dir = base_dir / f"mask_N{num_points}"
        sparse_path = sparse_dir / depth_rel
        mask_path = mask_dir / depth_rel.replace(".png", ".npy")

        sparse_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        # Save results
        cv2.imwrite(str(sparse_path), current.astype(np.uint16))
        np.save(str(mask_path), mask)

        # Relative output entries (keep all oRriginal columns + new ones)
        rel_rgb = rgb_path.relative_to(base_dir)
        rel_depth = depth_path.relative_to(base_dir)
        rel_filled = filled_path.relative_to(base_dir)
        rel_mask = mask_path.relative_to(base_dir)
        rel_sparse = sparse_path.relative_to(base_dir)

        entries.append(f"{rel_rgb} {rel_depth} {rel_mask} {rel_sparse} {rel_filled}")

    print(f"Writing output to {out_txt}")
    with open(out_txt, "w") as f:
        f.write("\n".join(entries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="Path to nyu_labeled_extracted")
    parser.add_argument("--split", type=str, choices=["train", "test", "both"], help="Predefined split option")
    parser.add_argument("--file_list", type=str, help="Path to custom file list")
    parser.add_argument("--output_txt", type=str, help="Path to output txt file prefix")
    parser.add_argument("--num_points", type=int, default=500)
    parser.add_argument("--noise_divisor", type=float, default=None, help="Divisor for Gaussian noise stddev")
    parser.add_argument("--outlier_prob", type=float, default=None, help="Probability for uniform outliers")
    parser.add_argument("--transform_order", type=str, default=None, help="Comma-separated transforms: sampling,gaussian,outliers")
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

    # validate required parameters
    for op in order:
        if op == 'gaussian' and args.noise_divisor is None:
            parser.error("--noise_divisor must be set when including 'gaussian' in --transform_order")
        if op == 'outliers' and args.outlier_prob is None:
            parser.error("--outlier_prob must be set when including 'outliers' in --transform_order")

    base_dir = Path(args.base_dir)

    if args.split:
        split_map = {
            "train": "filename_list_train.txt",
            "test":  "filename_list_test.txt",
        }
        splits = [args.split] if args.split != "both" else ["train", "test"]
        for split in splits:
            filelist = base_dir / split_map[split]
            out_txt = base_dir / f"filename_list_{split}.txt"
            sample_from_filelist(base_dir, filelist, out_txt, args.num_points,
                                 args.noise_divisor, args.outlier_prob, order, args.seed)
    else:
        if not args.file_list or not args.output_txt:
            parser.error("Must provide both --file_list and --output_txt if --split is not used.")
        sample_from_filelist(base_dir, Path(args.file_list), Path(args.output_txt),
                             args.num_points, args.noise_divisor, args.outlier_prob, order, args.seed)
