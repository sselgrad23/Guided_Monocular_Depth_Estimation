import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

def create_sparse_depth(depth, mask, num_points):
    valid = (depth > 0) & mask
    coords = np.argwhere(valid)
    if len(coords) == 0:
        return np.zeros_like(depth)
    sampled = coords[np.random.choice(len(coords), min(num_points, len(coords)), replace=False)]
    sparse = np.zeros_like(depth)
    for y, x in sampled:
        sparse[y, x] = depth[y, x]
    return sparse


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


def sample_diode_points(data_root, txt_out_base, num_points,
                        noise_divisor, outlier_prob, order, seed):
    if seed is not None:
        np.random.seed(seed)

    data_root = Path(data_root)

    # build suffix tokens in specified order
    suffix_tokens = [str(num_points)]
    for op in order:
        if op == 'gaussian':
            suffix_tokens.append(f"gaussian_{int(noise_divisor)}")
        elif op == 'outliers':
            suffix_tokens.append(f"outliers_{int(outlier_prob*100)}")
    suffix = "_".join(suffix_tokens)

    # update output txt name
    txt_out = Path(txt_out_base)
    txt_out = txt_out.with_name(f"{txt_out.stem}_{suffix}{txt_out.suffix}")

    entries = []
    sparse_folder = f"sparse_depth_{suffix}"
    mask_folder = f"mask_{suffix}"

    # gather files for progress bar
    depth_files = list(data_root.rglob("*_depth.npy"))
    for npy_file in tqdm(depth_files, desc="Processing Diode", dynamic_ncols=True):
        rel = npy_file.relative_to(data_root)
        mask_file = data_root / rel.with_name(rel.stem + "_mask.npy")
        rgb_file  = data_root / rel.with_name(rel.stem.replace("_depth","") + ".png")

        if not mask_file.exists() or not rgb_file.exists():
            continue

        depth = np.load(npy_file)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = np.squeeze(depth, axis=-1)
        mask  = np.load(mask_file).astype(bool)

        current = None
        for op in order:
            if op == 'sampling':
                current = create_sparse_depth(depth, mask, num_points)
            elif op == 'gaussian':
                current = add_gaussian_to_sparse(current, noise_divisor)
            elif op == 'outliers':
                current = add_outliers_to_sparse(current, outlier_prob)

        sparse_path = data_root / rel.parent / sparse_folder / (rel.stem + "_sparse.npy")
        mask_path   = data_root / rel.parent / mask_folder   / (rel.stem + f"_mask_{suffix}.npy")

        sparse_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(sparse_path, current)
        np.save(mask_path,   mask)

        entries.append(f"{rgb_file.resolve()} {npy_file.resolve()} {sparse_path.resolve()} {mask_path.resolve()}")

    print(f"Writing output to {txt_out}")  # Debug print statement
    with open(txt_out, "w") as f:
        f.write("\n".join(entries))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory containing depth, mask, and RGB files")
    parser.add_argument("--txt_out",   type=str, required=True,
                        help="Base output .txt path (without suffix)")
    parser.add_argument("--num_points", type=int,   default=500,
                        help="Number of points to sample")
    parser.add_argument("--noise_divisor", type=float, default=None,
                        help="Divisor for Gaussian noise stddev")
    parser.add_argument("--outlier_prob",   type=float, default=None,
                        help="Probability for uniform outliers")
    parser.add_argument("--transform_order", type=str, default=None,
                        help="Comma-separated transforms: sampling,gaussian,outliers")
    parser.add_argument("--seed",            type=int, default=None,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # parse transform order or default
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

    sample_diode_points(
        args.data_root, args.txt_out, args.num_points,
        args.noise_divisor, args.outlier_prob, order, args.seed
    )
