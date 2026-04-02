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
    output_txt = Path(output_txt)

    # build suffix for naming in specified order
    suffix_tokens = [str(num_points)]
    for op in order:
        if op == 'gaussian':
            suffix_tokens.append(f"gaussian_{int(noise_divisor)}")
        elif op == 'outliers':
            suffix_tokens.append(f"outliers_{int(outlier_prob*100)}")
    suffix = "_".join(suffix_tokens)

    # update output txt name
    output_txt = output_txt.with_name(output_txt.stem + f"_{suffix}" + output_txt.suffix)

    entries = []
    all_color_paths = list(chain.from_iterable(
        sorted((scene / "color").glob("*.jpg"))
        for scene in sorted(root_dir.iterdir())
        if (scene / "color").exists() and (scene / "depth").exists()
    ))

    for color_path in tqdm(all_color_paths, desc="Processing ScanNet", dynamic_ncols=True):
        scene_dir = color_path.parent.parent
        frame = color_path.stem
        depth_path = scene_dir / "depth" / f"{frame}.png"
        if not depth_path.exists():
            continue

        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            continue

        current = None
        mask = None
        # apply transforms in order
        for op in order:
            if op == 'sampling':
                sparse, mask = create_sparse_depth(depth, num_points)
                current = sparse
            elif op == 'gaussian':
                current = add_gaussian_to_sparse(current, noise_divisor)
            elif op == 'outliers':
                current = add_outliers_to_sparse(current, outlier_prob)

        # save results
        sparse_dir = scene_dir / f"sparse_depth_{suffix}"
        mask_dir = scene_dir / f"mask_{suffix}"
        sparse_path = sparse_dir / f"{frame}.png"
        mask_path = mask_dir / f"{frame}_mask_{suffix}.npy"

        sparse_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sparse_path), current.astype(np.uint16))
        np.save(str(mask_path), mask)

        entries.append(f"{color_path.resolve()} {depth_path.resolve()} {sparse_path.resolve()} {mask_path.resolve()}")

    print(f"Writing output to {output_txt}")  # Debug print statement
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
