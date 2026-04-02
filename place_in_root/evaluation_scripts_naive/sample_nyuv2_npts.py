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

    # build suffix based on transform order
    suffix_tokens = [str(num_points)]
    for op in order:
        if op == 'gaussian':
            suffix_tokens.append(f"gaussian_{int(noise_divisor)}")
        elif op == 'outliers':
            suffix_tokens.append(f"outliers_{int(outlier_prob*100)}")
    suffix = "_".join(suffix_tokens)

    # update output txt name with suffix
    output_txt = Path(output_txt)
    output_txt = output_txt.with_name(output_txt.stem + f"_{suffix}" + output_txt.suffix)

    entries = []
    with open(file_list, "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc=f"Processing {Path(file_list).stem}"):
        rgb_rel, depth_rel, _ = line.strip().split()
        rgb_path = base_dir / rgb_rel
        depth_path = base_dir / depth_rel

        if not rgb_path.exists() or not depth_path.exists():
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            continue

        current = None
        mask = None
        for op in order:
            if op == 'sampling':
                current, mask = create_sparse_depth(depth, num_points)
            elif op == 'gaussian':
                current = add_gaussian_to_sparse(current, noise_divisor)
            elif op == 'outliers':
                current = add_outliers_to_sparse(current, outlier_prob)

        scene = depth_path.parent
        stem = depth_path.stem
        sparse_path = scene / f"{stem}_sparse_{suffix}.png"
        mask_path = scene / f"{stem}_mask_{suffix}.npy"

        cv2.imwrite(str(sparse_path), current.astype(np.uint16))
        np.save(str(mask_path), mask)

        entries.append(f"{rgb_path.resolve()} {depth_path.resolve()} {sparse_path.resolve()} {mask_path.resolve()}")

    print(f"Writing output to {output_txt}")  # Debug print statement
    with open(output_txt, "w") as f:
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
            out_txt = base_dir / f"{split}_sparse_sampled.txt"
            sample_from_filelist(base_dir, filelist, out_txt, args.num_points,
                                 args.noise_divisor, args.outlier_prob, order, args.seed)
    else:
        if not args.file_list or not args.output_txt:
            parser.error("Must provide both --file_list and --output_txt if --split is not used.")
        sample_from_filelist(base_dir, Path(args.file_list), Path(args.output_txt),
                             args.num_points, args.noise_divisor, args.outlier_prob, order, args.seed)
